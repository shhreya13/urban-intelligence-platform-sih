import cv2
import numpy as np
from math import hypot

def center_xy(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def bottom_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)

def normalize_polygon(poly, w, h):
    return np.array([[int(x * w), int(y * h)] for x, y in poly], dtype=np.int32)

def inside_polygon(point, polygon_px):
    return cv2.pointPolygonTest(polygon_px, point, False) >= 0

def distance(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])
