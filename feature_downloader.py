import geopandas as gpd
import pandas as pd
import requests as r
from bbox import Bbox
from shapely import wkt
from shapely.ops import unary_union
import sys
from matplotlib import pyplot as plt


def key_value(key, value):
    if value != "":
        return f'["{key}"="{value}"]'
    else:
        return f'["{key}"]'


# Download from osm, parameters:
# bbox type Bbox8
# tags type dict collecting key-value data for query
def osm_data(bbox, tags=None):
    if tags is None:
        tags = {}
    url = "http://overpass-api.de/api/interpreter"
    tag_queries = "".join(key_value(key, tags[key]) for key in tags)
    data = f"""[out:json][timeout:25];wr{tag_queries}{bbox.format_osm()};out geom;"""
    g = r.get(url, data=data)
    objects = []
    for el in g.json()["elements"]:
        objects.append(el["tags"])
        objects[-1]["id"] = el["id"]
        objects[-1]["geometry"] = osm_geom_proc(el)
    df = pd.DataFrame.from_records(objects)
    return gpd.GeoDataFrame(df, geometry=gpd.GeoSeries(df["geometry"]))


def coord_series(geometry):
    return ','.join(f'{coord["lon"]} {coord["lat"]}' for coord in geometry)


def osm_way(geometry):
    return wkt.loads(f"POLYGON(({coord_series(geometry)}))")


def compare_nodes(node1, node2):
    return node1["lat"] == node2["lat"] and node1["lon"] == node2["lon"]


def is_way_closed(way):
    return compare_nodes(way[0], way[-1])


def get_rings(ways):
    used = set([])
    geoms = []
    for i in range(len(ways)):
        if i not in used:
            used.add(i)
            geometry = ways[i]['geometry']
            if is_way_closed(geometry):
                geoms.append(osm_way(geometry))
            else:
                rings = [geometry]
                while not compare_nodes(rings[0][0], rings[-1][-1]):
                    for j in range(i, len(ways)):
                        if j not in used:
                            geometry = ways[i]['geometry']
                            if compare_nodes(rings[-1][-1], geometry[0]):
                                used.add(j)
                                rings.append(geometry)
                                continue
                            elif compare_nodes(rings[-1][-1], geometry[-1]):
                                used.add(j)
                                rings.append(geometry[::-1])
                                continue
                    print("Invalid multipolygon!")
                    sys.exit(1)
                geoms.append(wkt.loads(f"POLYGON(({','.join(coord_series(way) for way in rings)}))"))
    return geoms


def osm_relation(geometry):
    outers = [el for el in geometry if el["role"] == "outer"]
    inners = [el for el in geometry if el["role"] == "inner"]
    outer_multipolygon = unary_union(get_rings(outers))
    inner_multipolygon = unary_union(get_rings(inners))
    return outer_multipolygon.difference(inner_multipolygon)


def osm_geom_proc(el):
    osm_geom_funcs = {'way': osm_way, 'relation': osm_relation}
    osm_geom_source = {'way': "geometry", 'relation': "members"}
    return osm_geom_funcs[el["type"]](el[osm_geom_source[el["type"]]])


"""gdf = osm_data(
    Bbox(
        52.116097999238534,
        20.732102394104004,
        52.12079094588992,
        20.744298934936527
    ),
    {"landuse": ""}
)"""