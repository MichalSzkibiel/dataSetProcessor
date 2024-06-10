import json
import math

from shapely.wkt import loads as shapely_wkt
from shapely.ops import transform
import pyproj


def from_shapely_geom(geom, crs):
    minx = geom.exterior.xy[0][0]
    maxx = geom.exterior.xy[0][0]
    for x in geom.exterior.xy[0][1:]:
        if minx > x:
            minx = x
        elif maxx < x:
            maxx = x
    miny = geom.exterior.xy[1][0]
    maxy = geom.exterior.xy[1][0]
    for y in geom.exterior.xy[1][1:]:
        if miny > y:
            miny = y
        elif maxy < y:
            maxy = y
    return Bbox(minx, miny, maxx, maxy, crs)


def from_shapely_bounds(bounds, crs):
    return Bbox(
        bounds[0],
        bounds[1],
        bounds[2],
        bounds[3],
        crs
    )


class Bbox:
    def __init__(self, minx, miny, maxx, maxy, crs):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        self.crs = crs

    def format_osm(self):
        return f"({self.minx},{self.miny},{self.maxx},{self.maxy})"

    def wkt(self):
        return f"POLYGON(({self.maxx} {self.miny}, " + \
            f"{self.minx} {self.miny}, " + \
            f"{self.minx} {self.maxy}, " + \
            f"{self.maxx} {self.maxy}, " + \
            f"{self.maxx} {self.miny}))"

    def to_geojson(self):
        return json.dumps({
            "type": "Polygon",
            "crs": {
                "properties": {
                    "name": str(self.crs)
                }
            },
            "coordinates": [
                [
                    [self.maxx, self.miny],
                    [self.minx, self.miny],
                    [self.minx, self.maxy],
                    [self.maxx, self.maxy],
                    [self.maxx, self.miny]
                ]
            ]
        })

    def to_crs(self, crs):
        geom = shapely_wkt(self.wkt())
        origin = pyproj.CRS(self.crs)
        dest = pyproj.CRS(crs)
        project = pyproj.Transformer.from_crs(origin, dest).transform
        reprojected_geom = transform(project, geom)
        return from_shapely_geom(reprojected_geom, crs)

    def as_shapely(self):
        return shapely_wkt(self.wkt())

    def adjust_to_grid(self, grid_size):
        self.minx = self.minx // grid_size * grid_size
        self.miny = self.miny // grid_size * grid_size
        self.maxx = math.ceil(self.maxx / grid_size) * grid_size
        self.maxy = math.ceil(self.maxy / grid_size) * grid_size
