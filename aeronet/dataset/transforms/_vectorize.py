import cv2
import numpy as np
from collections import defaultdict
from rasterio.transform import IDENTITY, xy
from shapely.geometry import shape, Polygon, MultiPolygon, GeometryCollection

from ..vector import Feature, FeatureCollection

cv_approx = {'simple': cv2.CHAIN_APPROX_SIMPLE,
             'tc89_kcos': cv2.CHAIN_APPROX_TC89_KCOS,
             'tc89_l1': cv2.CHAIN_APPROX_TC89_L1}


def polygonize(sample, epsilon=0.1, properties=None, approx='tc89_kcos', upscale=1.0):
    """ Transform the raster mask to vector polygons.
    The pixels in the raster mask are treated as belonging to the object if their value is non-zero, and zero values are background.
    All the objects are transformed to the vector form (polygons).

    The algorithm is OpenCV
    `cv2.findContours
    <https://docs.opencv.org/3.1.0/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a>`_

    This method is used as it does process the hierarchy of inlayed contours correctly.
    It also makes polygon simplification, which produces more smooth and lightweight polygons, but they do not match
    the raster mask exactly, which should be taken into account.

    Args:
        sample: BandSample to be vectorized
        epsilon: the epsilon parameter for the cv2.approxPolyDP, which specifies the approximation accuracy.
        This is the maximum distance between the original curve and its approximation
        properties: (dict) Properties to be added to the resulting FeatureCollection
        approx: Approximation parameter for cv2:findContours
        upscale (float): scale image for better precision of the polygon. The polygon is correctly downscaled back.
            works only when `upscale > 1`, default value means no upscaling

    Returns:
        FeatureCollection:
            Polygons in the CRS of the sample, that represent non-black objects in the image

    """

    approx = cv_approx[approx.lower()]
    geoms = _vectorize(sample.numpy(), epsilon=epsilon, approx=approx, transform=sample.transform)
    # remove all the geometries except for polygons
    polys = _extract_polygons(geoms)

    if properties is None:
        properties = {}
    features = ([Feature(geometry, properties=properties, crs=sample.crs)
                 for geometry in polys])
    return FeatureCollection(features, crs=sample.crs)


def _extract_polygons(geometries):
    """
    Makes the consistent polygon-only geometry list of valid polygons
    It ignores
    :return: list of shapely Polygons
    """
    shapes = []
    for geom in geometries:
        if not geom['type'] in ['Polygon', 'MultiPolygon']:
            continue
        else:
            new_shape = shape(geom).buffer(0)
            if isinstance(new_shape, Polygon):
                shapes.append(new_shape)
            elif isinstance(new_shape, MultiPolygon):
                shapes += new_shape
            elif isinstance(new_shape, GeometryCollection):
                for sh in new_shape:
                    if isinstance(sh, Polygon):
                        shapes.append(sh)
                    elif isinstance(sh, MultiPolygon):
                        shapes += sh
    return shapes


def _vectorize(binary_image,
               epsilon=0., min_area=1., approx=cv2.CHAIN_APPROX_TC89_KCOS,
               transform=IDENTITY, upscale=1.0):
    """
    Vectorize binary image, returns a 4-level list of floats [[[[X,Y]]]]

    Args:
        binary_image (numpy.ndarray): image in grayscale HxW or HxWx1
        epsilon (float): approximates a contour shape to another shape with less
            number of vertices depending upon the precision we specify, %
        transform (list of float):  affine transform matrix (first two lines)
            list with 6 params - [GSDx, ROTy, ROTx, GSDy, X, Y]
        upscale (float): scale image for better precision of the polygon. The polygon is correctly downscaled back

    """
    # remove possible 1-sized dimension
    binary_image = np.squeeze(binary_image)
    if binary_image.ndim != 2:
        raise ValueError('Input image must have 2 non-degenerate dimensions')

    if upscale > 1:
        h, w = binary_image.shape[:2]
        binary_image = cv2.resize(binary_image, (int(w * upscale), int(h * upscale)), cv2.INTER_NEAREST)

    # search for all contours
    contours_result = cv2.findContours(
        binary_image,
        cv2.RETR_CCOMP, approx)
    # For compatibility with opencv3, where contours_result[0]=image, so we should throw it away
    contours, hierarchy = contours_result if len(contours_result) == 2 else contours_result[1:]

    # approximate contours with less number of points
    if epsilon > 0.:
        contours = [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]

    if not contours:
        return []

    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])

    geometries = []
    for idx, cnt in enumerate(contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            # take a contour as a primary shape
            coords = [[tuple(xy(transform, p[1], p[0])) for p in cnt[:, 0, :].tolist()]]
            coords[0].append(coords[0][0])
            # add all children as secondary contours (holes)
            for c in cnt_children.get(idx, []):
                if cv2.contourArea(c) >= min_area:
                    c_list = c[:, 0, :].tolist()
                    c_list.append(c_list[0])
                    coords.append([tuple(xy(transform, p[1], p[0])) for p in c_list])

            # append geometry
            geometries.append({
                'type': 'Polygon',
                'coordinates': coords,
            })
    return geometries
