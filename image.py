import datetime
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import pickle
import rasterio
import rasterio.mask
from scipy import interpolate
from shapely import Polygon
import shutil
from zipfile import ZipFile
from copy import deepcopy

import matplotlib.pyplot as plt

import requests as r
from shapely import wkt
from shapely.geometry import box
from tqdm import tqdm

from config import copernicus_token_data, workspace
import bbox


def extract_xml_tag(xml, tag):
    return xml.split(f"<{tag}", 1)[1].split(">", 1)[1].split(f"</{tag}>", 1)[0]


def extract_many_xml_tags(xml, tag):
    return [el.split(">", 1)[1].split(f"</{tag}>", 1)[0] for el in xml.split(f"<{tag}")[1:]]


def array_from_xml_values(xml):
    return np.array([[float(el2) for el2 in el.split(" ")] for el in extract_many_xml_tags(xml, "VALUES")])


def is_channel_file(path):
    if 'L2A' in path and 'IMG_DATA' in path:
        if ('R10m' in path and ('B02' in path or 'B03' in path or 'B04' in path or 'B08' in path) or
                'R20m' in path and ('B01' in path or 'B05' in path or 'B06' in path or 'B07' in path
                                    or 'B8A' in path or 'B11' in path or 'B12' in path) or
                'R60m' in path and 'B09' in path):
            return True
    elif 'L1C' in path and 'IMG_DATA' in path:
        if 'B' in path and path[-4:] == '.jp2':
            return True
    if 'MTD_TL.xml' in path:
        return True
    return False


def download_image(url, target):
    with r.get(url, stream=True, timeout=100) as g:
        with tqdm(unit="B", unit_scale=True, disable=False) as progress:
            with open(target, 'wb') as f:
                for chunk in g.iter_content(chunk_size=2 ** 20):
                    if chunk:
                        f.write(chunk),
                        progress.update(len(chunk))


class Sentinel2Image:
    def __init__(self, image_info):
        self.identifier = image_info['Id']
        self.name = image_info['Name']
        self.bbox = bbox.from_shapely_geom(wkt.loads(image_info['Footprint'].split(";", 1)[1]), crs='EPSG:4326')
        for att in image_info['Attributes']:
            if att['Name'] == 'cloudCover':
                self.cloud_cover = att['Value']
            elif att['Name'] == 'processingLevel':
                self.processing_level = att['Value']
            elif att['Name'] == 'beginningDateTime':
                self.date_time = datetime.datetime.strptime(att['Value'].split(".", 1)[0], '%Y-%m-%dT%H:%M:%S')

    def save(self):
        pickle.dump(self, open(os.path.join(workspace, "sentinel2images", self.name), 'wb'))

    def unzip(self):
        zip_path = os.path.join(workspace, 'images', f"{self.identifier}.zip")
        if not os.path.exists(zip_path):
            token = get_token()
            print(f'downloading: {self.identifier}')
            download_image(
                f'https://zipper.creodias.eu/download/{self.identifier}?token={token}',
                zip_path
            )
        target_path = os.path.join(workspace, 'temp', self.identifier)
        if not os.path.exists(target_path):
            os.mkdir(target_path)
        with ZipFile(zip_path, 'r') as archive:
            channels = [el for el in archive.namelist() if is_channel_file(el)]
            for member in channels:
                filename = os.path.basename(member)
                if not filename:
                    continue
                source = archive.open(member)
                target = open(os.path.join(target_path, filename), "wb")
                with source, target:
                    shutil.copyfileobj(source, target)

    def get_cloud_mask(self, buffer_size, dem_path):
        self.unzip()
        path = os.path.join(workspace, 'temp', self.identifier)
        metadata_file = open(os.path.join(path, "MTD_TL.xml")).read()
        sun_angles_grid = extract_xml_tag(metadata_file, "Sun_Angles_Grid")

        sun_zenith_angles = array_from_xml_values(
            extract_xml_tag(extract_xml_tag(sun_angles_grid, "Zenith"), "Values_List"))
        s = sun_zenith_angles.shape[0]
        r = np.array(range(s))
        f = interpolate.interp2d(r, r, sun_zenith_angles, kind='cubic')
        xnew = np.linspace(0, s - 1, 1830)
        sun_zenith_angles = f(xnew, xnew)

        sun_azimuth_angles = array_from_xml_values(
            extract_xml_tag(extract_xml_tag(sun_angles_grid, "Azimuth"), "Values_List"))
        f = interpolate.interp2d(r, r, sun_azimuth_angles, kind='cubic')
        sun_azimuth_angles = f(xnew, xnew)

        view_zenith_angles = []
        view_azimuth_angles = []
        for v in extract_many_xml_tags(metadata_file, "Viewing_Incidence_Angles_Grids"):
            view_zenith_angles.append(
                array_from_xml_values(extract_xml_tag(extract_xml_tag(v, "Zenith"), "Values_List")))
            view_azimuth_angles.append(
                array_from_xml_values(extract_xml_tag(extract_xml_tag(v, "Azimuth"), "Values_List")))
        view_zenith_angles = np.nanmean(np.array(view_zenith_angles), 0)
        view_azimuth_angles = np.nanmean(np.array(view_azimuth_angles), 0)

        f = interpolate.interp2d(r, r, view_zenith_angles, kind='cubic')
        view_zenith_angles = f(xnew, xnew)

        f = interpolate.interp2d(r, r, view_azimuth_angles, kind='cubic')
        view_azimuth_angles = f(xnew, xnew)

        scattering_angle = -np.cos(sun_zenith_angles) * np.cos(view_zenith_angles) - np.sin(sun_zenith_angles) * np.sin(
            view_zenith_angles) * np.cos(view_azimuth_angles - sun_azimuth_angles)
        t_brr422 = 0.03 + 0.03 * np.cos(scattering_angle) ** 2
        t_white = 0.9
        t_brightwhite = 1.5
        t_land = 0.9
        t_ndvi = 0.5
        t_b3b11 = 1.0
        t_ndsi = 0.6
        t_tc1 = 0.36
        t_tc4cirrus_1 = -0.1
        t_tc4cirrus_2 = -0.11
        t_tc4cirrus_3 = -0.085
        t_tc4 = -0.08
        t_ndwi = 0.04
        t_visbright = 0.12
        t_b10_1 = 0.0035
        t_b10_2 = 0.01

        layers = [os.path.join(path, el) for el in os.listdir(path) if '_B' in el and el[-4:] == '.jp2']
        array = np.zeros((13, 1830, 1830))
        i = 0
        for file_name in layers:
            print(file_name)
            with rasterio.open(file_name) as channel:
                data = (channel.read(1)) / 10000 - 0.1
                if i == 0:
                    xmin, ymin, xmax, ymax = channel.bounds
                    profile = channel.profile
                    cols, rows = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]))
                    xs, ys = rasterio.transform.xy(channel.transform, rows, cols)
                    df = pd.DataFrame(
                        np.array(
                            [
                                np.array(xs).reshape((1830 * 1830,)),
                                np.array(ys).reshape((1830 * 1830,))
                            ]
                        ).transpose()
                    )
                    latitude = gpd.GeoDataFrame(
                        df,
                        geometry=gpd.points_from_xy(df[0], df[1]),
                        crs=32643
                    ).to_crs(4326).geometry.y.to_numpy().reshape((1830, 1830))
                if data.shape[0] == 1830:
                    array[i] = data
                elif data.shape[0] == 1830 * 3:
                    for j in range(3):
                        for k in range(3):
                            array[i] += data[j::3, k::3]
                    array[i] /= 9
                else:
                    for j in range(6):
                        for k in range(6):
                            array[i] += data[j::6, k::6]
                    array[i] /= 36
                i += 1

        with rasterio.open(dem_path) as file:
            elevation, transform = rasterio.mask.mask(file, [
                Polygon([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]])], crop=True)
            elevation = elevation[0]
        ndsi = (array[2] * 1.0 - array[10] * 1.0) / (array[2] * 1.0 + array[10] * 1.0)
        ndwi = (array[12] * 1.0 - array[10] * 1.0) / (array[12] * 1.0 + array[10] * 1.0)
        b3b11 = array[2] / array[10]
        vis_bright = np.mean(array[1:4], 0)
        tc1 = 0.3029 * array[1] + 0.2786 * array[2] + 0.4733 * array[3] + 0.5599 * array[12] + 0.508 * array[
            10] + 0.1872 * array[11]
        tc4 = -0.8239 * array[1] + 0.0849 * array[2] + 0.4396 * array[3] - 0.058 * array[12] + 0.2013 * array[
            10] - 0.2773 * array[11]
        tc4cirrus = tc4 - array[9]

        invalid = np.min(array, 0) < 0

        cloud_sure = (b3b11 > t_b3b11) & ((tc4cirrus < t_tc4cirrus_1) | (tc4 < t_tc4) & (ndwi < t_ndwi))
        snow = (b3b11 <= t_b3b11) & (ndsi > t_ndsi) & (tc1 >= t_tc1) & ((np.abs(latitude) >= 30) | (elevation > 3000))
        cloud_sure = cloud_sure | (
                ~snow & (b3b11 <= t_b3b11) & (tc4cirrus < t_tc4cirrus_2) & (vis_bright > t_visbright))
        cloud_ambiguous = ~snow & ~cloud_sure & (b3b11 <= t_b3b11) & (tc4cirrus < t_tc4cirrus_3) & (
                vis_bright > t_visbright)
        cirrus_sure = ~cloud_sure & ~snow & (b3b11 <= t_b3b11) & (array[9] > t_b10_2) & (elevation < 2000)
        cirrus_ambiguous = ~cloud_sure & ~snow & ~cirrus_sure & (b3b11 <= t_b3b11) & (array[9] > t_b10_1) & (
                    elevation < 2000)

        filter_size = 7
        r8a_7 = np.ones((array.shape[1] - filter_size + 1, array.shape[2] - filter_size + 1, filter_size ** 2))
        for i in range(filter_size):
            for j in range(filter_size):
                r8a_7[:, :, i * filter_size + j] = array[12, i:i + r8a_7.shape[0], j:j + r8a_7.shape[0]] - array[6,
                                                                                                           i:i +
                                                                                                             r8a_7.shape[
                                                                                                                 0],
                                                                                                           j:j +
                                                                                                             r8a_7.shape[
                                                                                                                 0]]
        r8a_7 = np.std(r8a_7, 2) ** 2
        r8a_8 = np.ones((array.shape[1] - filter_size + 1, array.shape[2] - filter_size + 1, filter_size ** 2))
        for i in range(filter_size):
            for j in range(filter_size):
                r8a_8[:, :, i * filter_size + j] = array[12, i:i + r8a_8.shape[0], j:j + r8a_8.shape[0]] - array[7,
                                                                                                           i:i +
                                                                                                             r8a_8.shape[
                                                                                                                 0],
                                                                                                           j:j +
                                                                                                             r8a_8.shape[
                                                                                                                 0]]
        r8a_8 = np.std(r8a_8, 2) ** 2
        cdi = -np.ones((1830, 1830))
        cdi[filter_size // 2:1830 - filter_size // 2, filter_size // 2:1830 - filter_size // 2] = (r8a_7 - r8a_8) / (
                r8a_7 + r8a_8)
        cloud_sure = (cdi < -0.5) & cloud_sure
        cloud_sure_buffered = deepcopy(cloud_sure)
        for i in range(-buffer_size, buffer_size + 1):
            for j in range(-buffer_size, buffer_size + 1):
                cloud_sure_buffered[
                max(0, -i):cloud_sure.shape[0] - max(0, i),
                max(0, -j):cloud_sure.shape[0] - max(0, j)
                ] |= cloud_sure[
                     max(0, i):cloud_sure.shape[0] - max(0, -i),
                     max(0, j):cloud_sure.shape[0] - max(0, -j)
                     ]
        cloud_ambiguous = (cdi < -0.5) & cloud_ambiguous
        cloud_ambiguous_buffered = deepcopy(cloud_ambiguous)
        for i in range(-buffer_size, buffer_size + 1):
            for j in range(-buffer_size, buffer_size + 1):
                cloud_ambiguous_buffered[
                max(0, -i):cloud_ambiguous.shape[0] - max(0, i),
                max(0, -j):cloud_ambiguous.shape[0] - max(0, j)
                ] |= cloud_ambiguous[
                     max(0, i):cloud_ambiguous.shape[0] - max(0, -i),
                     max(0, j):cloud_ambiguous.shape[0] - max(0, -j)
                     ]

        profile.update(dtype=rasterio.uint8)
        cloud = cloud_sure_buffered * 8 + cloud_ambiguous_buffered * 4 + cirrus_sure * 2 + cirrus_ambiguous * 1
        with rasterio.open(
                os.path.join(
                    workspace,
                    "cloud_masks",
                    f"{self.name.rsplit('_', 1)[0]}.tif"
                ),
                "w",
                **profile
        ) as dst:
            dst.write(cloud.astype(rasterio.uint8), 1)
        """cth_max = 0.5 * (90 - np.abs(latitude))**2 + 25 * (90 - np.abs(latitude)) + 5000
        d_proj_x = (cth_max - elevation.min()) * np.tan(sun_zenith_angles) * np.cos(sun_azimuth_angles) / 60
        d_proj_y = (cth_max - elevation.min()) * np.tan(sun_zenith_angles) * np.sin(sun_azimuth_angles) / 60
        potential_shadow = np.zeros((1830, 1830))"""

    def draw_composition(self, output_file, bbox=None):
        self.unzip()
        path = os.path.join(workspace, "temp", self.identifier)
        layers = [os.path.join(path, el) for el in os.listdir(path) if '_B' in el and el[-4:] == '.jp2']
        features = []
        first = True
        for el in layers:
            with rasterio.open(el) as channel:
                if first:
                    first = False
                    bbox = bbox.to_crs(channel.crs)
                    bbox.adjust_to_grid(60)
                if bbox is None:
                    data, transform = rasterio.mask.mask(channel, [box(*channel.bounds)], crop=True)
                else:
                    data, transform = rasterio.mask.mask(channel, [bbox.as_shapely()], crop=True)
                if '10m' in el:
                    profile = channel.profile
                    profile.update(
                        transform=transform,
                        height=data.shape[1],
                        width=data.shape[2]
                    )
                elif '20m' in el:
                    data2 = np.zeros((1, data.shape[1] * 2, data.shape[2] * 2), dtype=data.dtype)
                    for i in range(2):
                        for j in range(2):
                            data2[:, i::2, j::2] = data
                    data = data2
                elif '60m' in el:
                    data2 = np.zeros((1, data.shape[1] * 6, data.shape[2] * 6), dtype=data.dtype)
                    for i in range(6):
                        for j in range(6):
                            data2[:, i::6, j::6] = data
                    data = data2
                features.append(data[0])
        profile.update(count=10)
        with rasterio.open(os.path.join(workspace, "compositions", f"{output_file}.tif"), "w", **profile) as dst:
            for i in range(10):
                dst.write(features[i], i + 1)


def load_image(name):
    return pickle.load(open(os.path.join(workspace, "sentinel2images", name), 'rb'))


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
    return [Sentinel2Image(el) for el in g.json()['value']]


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
