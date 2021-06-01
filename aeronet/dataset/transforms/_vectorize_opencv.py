import cv2
import warnings
import numpy as np
from collections import defaultdict
from rasterio.transform import IDENTITY, xy


def vectorize_opencv(binary_image, transform=IDENTITY, epsilon=0., min_area=1., upscale=1, **kwargs):
    """
    Vectorize binary image, returns a 4-level list of floats [[[[X,Y]]]]

    Args:
        binary_image (numpy.ndarray): image in grayscale HxW or HxWx1
        epsilon (float): approximates a contour shape to another shape with less
            number of vertices depending upon the precision we specify, %
        min_area (float): polygons and holes with area less than this threshold will be removed
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
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
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