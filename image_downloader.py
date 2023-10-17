import requests as r
from tqdm import tqdm

from config import copernicus_token_data


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
    print(g.url)
    return g.json()


# Find Sentinel-2 images
# start_date type datetime
# end_date type datetime
# bbox type Bbox
# products type list of Strings, all possible values in default
# cloud_cover type int, maximum cloud cover on images in percent
def find_sentinel2(start_date, end_date, bbox, products=('L1C', 'L2A', 'L2AP'), cloud_cover=20):
    return find_images('S2', 'MSI', start_date, end_date, bbox, products, cloud_cover)


def get_token():
    token = r.post(
        'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
        data=copernicus_token_data
    ).json()
    return token["access_token"]


# Download Copernicus image
# identifier type str is an identifier of image
def download_image(identifier):
    token = get_token()
    url = f'https://zipper.creodias.eu/download/{identifier}?token={token}'
    print(f'downloading: {identifier}')
    with r.get(url, stream=True, timeout=100) as g:
        with tqdm(unit="B", unit_scale=True, disable=False) as progress:
            with open(f"images/{identifier}.zip", 'wb') as f:
                for chunk in g.iter_content(chunk_size=2**20):
                    if chunk:
                        f.write(chunk),
                        progress.update(len(chunk))
