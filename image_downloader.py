from datetime import datetime
import requests as r
from tqdm import tqdm
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

from config import copernicus_config
from bbox import Bbox


def find_images(system, instrument_short_name, start_date, end_date, bbox, products, cloud_cover):
    url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    data_filter = f"startswith(Name,'{system}') " + \
                  "and Attributes/OData.CSC.StringAttribute/any(" + \
                  "att:att/Name eq 'instrumentShortName' " + \
                  f"and att/OData.CSC.StringAttribute/Value eq '{instrument_short_name}'" + \
                  ") " + \
                  "and Attributes/OData.CSC.DoubleAttribute/any(" + \
                  "att:att/Name eq 'cloudCover' " + \
                  f"and att/OData.CSC.DoubleAttribute/Value le {cloud_cover}" + \
                  ") " + \
                  "and (" + \
                  ' or '.join("contains(Name,'" + el + "')" for el in products) + \
                  ") " + \
                  "and OData.CSC.Intersects(" + \
                  f"area=geography'SRID=4326;{bbox.wkt()}'" + \
                  ") " + \
                  "and Online eq true " + \
                  f"and ContentDate/Start ge {start_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')} " + \
                  f"and ContentDate/Start lt {end_date.strftime('%Y-%m-%dT%H:%M:%S.999Z')}"
    params = {
        "$filter": data_filter,
        "$orderby": "ContentDate/Start desc",
        "$expand": ["Attributes", "Assets"],
        "$count": "True",
        "$top": "50",
        "$skip": "0"
    }
    g = r.get(url, params=params)
    return g.json()


# Find Sentinel-2 images
# start_date type datetime
# end_date type datetime
# bbox type Bbox
# products type list of Strings, all possible values in default
# cloud_cover type int, maximum cloud cover on images in percent
def find_sentinel2(start_date, end_date, bbox, products=('L1C', 'L2A', 'L2AP'), cloud_cover=20):
    return find_images('S2', 'MSI', start_date, end_date, bbox, products, cloud_cover)


def get_oauth_session():
    print(copernicus_config.client_id[0])
    print(copernicus_config.client_secret)
    client = BackendApplicationClient(client_id=copernicus_config.client_id[0])
    oauth = OAuth2Session(client=client)

    token = oauth.fetch_token(
        token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
        client_secret=copernicus_config.client_secret
    )
    return token


# Download Copernicus image
# identifier type str is an identifier of image
def download_image(identifier):
    url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({identifier})/$value"
    print(url)
    token = get_oauth_session()
    session = r.Session()
    session.headers.update({'Authorization': f'Bearer {token}'})
    with session.get(url) as g:
        with tqdm(unit="B", unit_scale=True, disable=False) as progress:
            with open(f"images/{identifier}", 'wb') as f:
                for chunk in g.iter_content(chunk_size=2**20):
                    if chunk:
                        f.write(chunk),
                        progress.update(len(chunk))


images = find_sentinel2(
    datetime(2023, 9, 16, 0, 0, 0),
    datetime(2023, 10, 16, 23, 59, 59),
    Bbox(
        52.116097999238534,
        20.732102394104004,
        52.12079094588992,
        20.744298934936527
    ),
    products=['L2A']
)
print([el['Id'] for el in images['value']])
download_image(images['value'][2]['Id'])
