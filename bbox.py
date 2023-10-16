class Bbox:
    def __init__(self, lat_south, lon_west, lat_north, lon_east):
        self.lat_south = lat_south
        self.lon_west = lon_west
        self.lat_north = lat_north
        self.lon_east = lon_east

    def format_osm(self):
        return f"({self.lat_south},{self.lon_west},{self.lat_north},{self.lon_east})"

    def wkt(self):
        return f"POLYGON(({self.lon_east} {self.lat_south}, " + \
            f"{self.lon_west} {self.lat_south}, " + \
            f"{self.lon_west} {self.lat_north}, " + \
            f"{self.lon_east} {self.lat_north}, " + \
            f"{self.lon_east} {self.lat_south}))"
