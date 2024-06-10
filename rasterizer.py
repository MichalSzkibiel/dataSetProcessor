import numpy as np


def draw_ring(edges_horizontal, ring, bbox, resolution):
    xx, yy = ring.xy
    xx = [(el - bbox.minx)/resolution[0] for el in xx]
    yy = [(el - bbox.miny)/resolution[1] for el in yy]
    aa = [yy[i] - yy[i + 1] for i in range(len(yy) - 1)]
    bb = [xx[i + 1] - xx[i] for i in range(len(xx) - 1)]
    cc = [aa[i]*xx[i] + bb[i]*yy[i] for i in range(len(aa))]
    xx = [int(el + 0.5) for el in xx]
    yy = [int(el + 0.5) for el in yy]
    for i in range(len(aa)):
        startx = xx[i]
        starty = yy[i]
        destx = xx[i + 1]
        desty = yy[i + 1]
        a = aa[i]
        b = bb[i]
        c = cc[i]
        curx = startx
        cury = starty
        stepx = int(np.sign(b))
        stepy = -int(np.sign(a))
        if stepx == 0:
            if stepy > 0:
                edges_horizontal[startx, starty:desty] = ~edges_horizontal[startx, starty:desty]
            else:
                edges_horizontal[startx, desty:starty] = ~edges_horizontal[startx, desty:starty]
            continue
        if stepy == 0:
            continue
        while curx != destx or cury != desty:
            cdown = a*(curx + stepx) + b*cury
            cright = a*curx + b*(cury + stepy)
            if abs(cdown - c) < abs(cright - c):
                curx += stepx
            else:
                if stepy > 0:
                    edges_horizontal[curx, cury] = not edges_horizontal[curx, cury]
                else:
                    edges_horizontal[curx, cury - 1] = not edges_horizontal[curx, cury - 1]
                cury += stepy
    return edges_horizontal


def process_polygon(polygon, bbox, resolution):
    output = np.zeros(
        (int((bbox.maxx - bbox.minx) // resolution[0]), int((bbox.maxy - bbox.miny) // resolution[1])),
        dtype=bool
    )
    edges_horizontal = np.zeros((output.shape[0] + 1, output.shape[1] + 1), dtype=bool)
    edges_horizontal = draw_ring(
        edges_horizontal,
        polygon.exterior,
        bbox,
        resolution
    )
    for interior in polygon.interiors:
        edges_horizontal = draw_ring(
            edges_horizontal,
            interior,
            bbox,
            resolution
        )

    for j in range(output.shape[1]):
        interior = False
        for k in range(output.shape[0]):
            if edges_horizontal[k, j]:
                interior = not interior
            if interior:
                output[k, j] = True
    return output[:, ::-1].T
