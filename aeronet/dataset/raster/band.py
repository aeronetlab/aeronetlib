import os
import numpy as np
import warnings

import rasterio
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.coords import BoundingBox
from rasterio.warp import calculate_default_transform, reproject, Resampling

from ..coords import get_utm_zone
from .geoobject import GeoObject
from ._utils import band_shape_guard, random_name

TMP_DIR = '/tmp/raster'


class Band(GeoObject):
    """Filesystem object `Band` - Rasterio DatasetReader wrapper.

    The `Band` provides access to a georeferenced raster file placed in the filesystem.
    On creation the Band opens the file for reading and
    stores all the necessary metadata and allows to read the raster data on request.

    The majority of properties are inherited from
    rasterio `DatasetReader
    <https://rasterio.readthedocs.io/en/latest/api/rasterio.io.html#rasterio.io.DatasetReader>`_.

    Any file format supported by GDAL drivers can be read.

    Args:
        fp: full path to the raster file

    """
    def __init__(self, fp):
        super().__init__()
        self._band = rasterio.open(fp)
        self._tmp_file = False

    def __del__(self):
        fp = self._band.name
        self._band.close()
        if self._tmp_file:
            os.remove(fp)

    # ======================== PROPERTY BLOCK ========================
    @property
    def crs(self):
        """
            :obj:`CRS` - coordindate reference system of the band;  in the file
        """
        return self._band.crs

    @property
    def transform(self):
        """
        Transform matrix as the `affine.Affine
        <https://github.com/sgillies/affine>`_  object.
        This transform maps pixel row/column coordinates to coordinates in the dataset’s coordinate reference system.

        affine.identity is returned if if the file does not contain transform
        """
        return self._band.transform

    @property
    def nodata(self):
        """
        Band nodata value, type depends on the image dtype; None if the nodata value is not specified
        """
        return self._band.nodata

    @property
    def res(self):
        """
        Spatial resolution (x_res, y_res) of the Band in X and Y directions of the georeferenced coordinate system,
        derived from tranaform. Normally is equal to (transform.a, - transform.e)
        """
        return self._band.res

    @property
    def width(self):
        return self._band.width

    @property
    def height(self):
        return self._band.height

    @property
    def count(self):
        """
        By design of the aeronetlib, should be always 1. A Band can be created from image of any channel count,
        but only the first band can be read. If you need to work with multi-channel image, use
        :meth:`aeronet.converters.split.split` to get the one-channel images.

        Returns:
            (int) number of the bands in the image.
        """
        return self._band.count

    @property
    def shape(self):
        """
        The raster dimension as a Tuple (height, width)
        """
        return self.height, self.width

    @property
    def name(self):
        """
        Name of the file associated with the Band, without extension and the directory path
        """
        return os.path.basename(self._band.name).split('.')[0]

    @property
    def bounds(self):
        """
        Georeferenced bounds - bounding box in the CRS of the image, based on transform and shape

        Returns:
            `BoundingBox object
            <https://rasterio.readthedocs.io/en/latest/api/rasterio.coords.html#rasterio.coords.BoundingBox>`_:
            (left, bottom, right, top)
        """
        return self._band.bounds

    @property
    def meta(self):
        """
        The basic metadata of the associated rasterio DatasetReader
        """
        return self._band.meta

    @property
    def dtype(self):
        """
        Numerical type of the data stored in raster, according to numpy.dtype
        """
        return self._band.dtypes[0]

    # ======================== METHODS BLOCK ========================

    def numpy(self):
        """
        Read all the raster data into memory as a numpy array

        Returns:
            numpy array containing the whole Band raster data
        """
        return self.sample(0, 0, self.height, self.width).numpy()

    def same(self, other):
        """Compare if samples have same resolution, crs and shape.

        This means that the samples represent the same territory (like different spectral channels of the same image)
        and can be processed together as collection.

        Args:
            other: GeoObject to compare with

        Returns:
            True if the objects match in shape, crs, transform, False otherwise
        """
        res = True
        res = res and (self.crs == other.crs)
        res = res and (self.transform == other.transform)
        res = res and (self.height == other.height)
        res = res and (self.width == other.width)
        return res

    def _same_extent(self, other: GeoObject):
        """
        Compares the spatial extent of the current and other Bands based on their CRSes and transforms.
        The extent is treated as 'same' if the boundaries differ not more than half of the biggest pixel
        Args:
            other: GeoObject to compare the extent to
        Returns:
            bool: True if the rasters are compatible, False otherwise
        """

        # explicitly calculate the other Band's dimensions and resolution in the current crs
        other_bounds = rasterio.warp.transform_bounds(other.crs, self.crs, *other.bounds)
        other_res = [abs(other_bounds[0] - other_bounds[2])/other.width, abs(other_bounds[1] - other_bounds[3])/other.height]
        max_pixel = [max(self.res[0], other_res[0]), max(self.res[1], other_res[1])]

        # check every bound to be different not more than half of the bigger pixel
        if abs(other_bounds[0] - self.bounds[0]) > 0.5*max_pixel[0] or \
           abs(other_bounds[1] - self.bounds[1]) > 0.5 * max_pixel[1] or \
           abs(other_bounds[2] - self.bounds[2]) > 0.5 * max_pixel[0] or \
           abs(other_bounds[3] - self.bounds[3]) > 0.5 * max_pixel[1]:
            return False
        else:
            return True

    def sample(self, y, x, height, width, **kwargs):
        """ Read sample of the Band to memory.

        The sample is defined by its size and position in the raster, without respect to the georeference.
        In case if the sample coordinates spread out of the image boundaries, the image is padded with nodata value.

        Args:
            x: pixel horizontal coordinate of left top corner of the sample
            y: pixel vertical coordinate of left top corner of the sample
            width: spatial dimension of sample in pixels
            height: spatial dimension of sample in pixels

        Returns:
             a new :obj:`BandSample` containing the specified spatial subset of the band
        """

        coord_x = self.transform.c + x * self.transform.a
        coord_y = self.transform.f + y * self.transform.e

        dst_crs = self.crs
        dst_name = os.path.basename(self.name)
        dst_nodata = self.nodata if self.nodata is not None else 0
        dst_transform = Affine(self.transform.a, self.transform.b, coord_x,
                               self.transform.d, self.transform.e, coord_y)

        dst_raster = self._band.read(window=((y, y + height), (x, x + width)),
                                     boundless=True, fill_value=dst_nodata)

        sample = BandSample(dst_name, dst_raster, dst_crs, dst_transform, dst_nodata)

        return sample

    def resample(self, dst_res, fp=None, interpolation='nearest'):
        """ Change spatial resolution of the band. It does not alter the existing file,
        and creates a new file either in the specified location or a temporary file

        It is based on `rasterio.warp.reproject
        <https://rasterio.readthedocs.io/en/latest/api/rasterio.warp.html#rasterio.warp.reproject>`_,
        see for more variants of interpolation.

        Args:
            dst_res (Tuple[float, float]): new resoluton, georeferenced pixel size for the new band
            fp (str): a filename for the new resampled band. If none, a temporary file is created
            interpolation: interpolation type as in rasterio,  `nearest`, `bilinear`, `cubic`, `lanzsos` or others

        Returns:
            a new resampled Band.
        """
        # get temporary filepath if such is not provided
        tmp_file = False if fp is not None else True
        if fp is None:
            fp = '{tmp}/resampled/{directory}/{name}.tif'.format(
                tmp=TMP_DIR, directory=random_name(), name=self.name)

        os.makedirs(os.path.dirname(fp), exist_ok=True)

        transform = Affine(dst_res[0], self.transform.b, self.transform.c,
                           self.transform.d, - dst_res[1], self.transform.f)
        width = round(self.width / (dst_res[0]/self.res[0]))
        height = round(self.height / (dst_res[1]/self.res[1]))

        kwargs = self.meta.copy()
        kwargs.update({
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(fp, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(self._band, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=self.transform,
                    src_crs=self.crs,
                    dst_transform=transform,
                    dst_crs=self.crs,
                    resampling=getattr(Resampling, interpolation))

        # new band
        band = Band(fp)
        band._tmp_file = tmp_file  # file will be automatically removed when `Band` instance will be deleted

        return band

    def reproject(self, dst_crs, fp=None, interpolation='nearest'):
        """ Change coordinate system (projection) of the band.
        It does not alter the existing file, and creates a new file either in the specified location or a temporary file.

        The band ground sampling distance is not changed, however the resolution may change due to the new coordinate system
        It is based on `rasterio.warp.reproject
        <https://rasterio.readthedocs.io/en/latest/api/rasterio.warp.html#rasterio.warp.reproject>`_,
        see for more variants of interpolation.

        Args:
            dst_crs: new CRS, may be in any form acceptable by rasterio, for example as EPSG code, string, CRS object; if dst_crs == `utm`, the appropriate UTM zone is used according to the center of the image
            fp (str): a filename for the new resampled band. If none, a temporary file is created
            interpolation: interpolation type as in rasterio,  `nearest`, `bilinear`, `cubic`, `lanzsos` or others

        Returns:
            a new reprojected Band
        """
        if dst_crs == 'utm':
            dst_crs = get_utm_zone(self.crs, self.transform, (self.height, self.width))
        else:
            dst_crs = dst_crs if isinstance(dst_crs, CRS) else CRS.from_user_input(dst_crs)

        # Old rasterio compatibility: a separate check for validity
        if not dst_crs.is_valid:
            raise rasterio.errors.CRSError('Invalid CRS {} given'.format(dst_crs))

        # get temporary filepath if such is not provided
        tmp_file = False if fp is not None else True
        if fp is None:
            fp = '{tmp}/reprojected_{crs}/{directory}/{name}.tif'.format(
                tmp=TMP_DIR, crs=dst_crs, directory=random_name(), name=self.name)
        os.makedirs(os.path.dirname(fp), exist_ok=True)

        # calculate params of new reprojected Band
        transform, width, height = calculate_default_transform(
            self.crs, dst_crs, self.width, self.height, *self.bounds)
        kwargs = self.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        # reproject
        with rasterio.open(fp, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(self._band, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=self.transform,
                    src_crs=self.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=getattr(Resampling, interpolation))

        # new band
        band = Band(fp)
        band._tmp_file = tmp_file # file will be automatically removed when `Band` instance will be deleted
        return band

    def reproject_to(self, other: GeoObject, fp=None, interpolation='nearest'):
        """
        Reprojects and resamples the band to match exactly the `other`.

        This function ensures that the raster size, crs and transform will be the same,
        allowing them to be merged into one BandCollection. If the intial raster
        exceeds the other in coverage, it will be cut, and if it is insufficient or displaced, it will be zero-padded.

        It aims to overpass the rounding problem which may cause an image to
        be misaligned with itself after a different series of transforms.

        If the images are far from each other, the warning will be shown,
        because the raster may be zero due to severe misalignment.

        Args:
            other(GeoObject): the Band with the parameters to fit to
            fp (str): a filename for the new resampled band. If none, a temporary file is created
            interpolation: interpolation type as in rasterio,  `nearest`, `bilinear`, `cubic`, `lanzsos` or others.
        Returns:
            a new reprojected and resampled Band
        """
        if not self._same_extent(other):
            warnings.warn('You are trying to match two bands that are not even approxiamtely aligned. '
                          'The resulting raster may be empty')

        # get temporary filepath if such is not provided
        tmp_file = False if fp is not None else True
        if fp is None:
            fp = '{tmp}/reprojected_{crs}/{directory}/{name}.tif'.format(
                tmp=TMP_DIR, crs=other.crs, directory=random_name(), name=self.name)
        os.makedirs(os.path.dirname(fp), exist_ok=True)

        kwargs = self.meta.copy()
        kwargs.update({
            'crs': other.crs,
            'transform': other.transform,
            'width': other.width,
            'height': other.height
        })

        # reproject - as in rio.warp --like
        with rasterio.open(fp, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(self._band, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=self.transform,
                    src_crs=self.crs,
                    dst_transform=other.transform,
                    dst_crs=other.crs,
                    resampling=getattr(Resampling, interpolation))

        # new band
        band = Band(fp)
        band._tmp_file = tmp_file # file will be automatically removed when `Band` instance will be deleted
        return band

    def reproject_to_utm(self, fp=None, interpolation='nearest'):
        """
        Alias of :obj:`Band.reproject` method with automatic Band utm zone determining
        """
        dst_crs = get_utm_zone(self.crs, self.transform, (self.height, self.width))
        return self.reproject(dst_crs, fp=fp, interpolation=interpolation)

    def generate_samples(self, width, height):
        """
        A generator for sequential sampling of the whole band, used for the windowed reading of the raster.
        It allows to handle and process large files without reading them at once in the memory.

        Args:
            width (int): dimension of sample in pixels and step along `X` axis
            height (int): dimension of sample in pixels and step along `Y` axis

        Yields:
            BandSample: sequential samples of the specified dimensions
        """
        for x in range(0, self.width, width):
            for y in range(0, self.height, height):
                yield self.sample(y, x, height, width)


class BandSample(GeoObject):
    """ A wrapper over numpy array representing an in-memory georeferenced raster image.

    It implements all the interfaces of the GeoObject, and stores the raster data in memory

    Args:
        name (str): a name of the sample, which is used as a defaule name for saving to file
        raster (np.array): the raster data
        crs: geographical coordinate reference system, as :obj:`CRS` or string representation
        transform (Affine): affine transform for the
        nodata: the pixels with this value in raster should be ignored
    """

    def __init__(self, name, raster, crs, transform, nodata=0):
        """

        """

        super().__init__()

        self._name = name
        self._raster = band_shape_guard(raster)
        self._nodata = nodata
        self._transform = Affine(*transform) if not isinstance(transform, Affine) else transform
        self._crs = CRS.from_user_input(crs) if not isinstance(crs, CRS) else crs

        # Old rasterio compatibility: a separate check for validity
        if not self._crs.is_valid:
            raise rasterio.errors.CRSError('Invalid CRS {} given'.format(crs))

    def __eq__(self, other):
        res = np.allclose(self.numpy(), other.numpy())
        res = res and (self.crs == other.crs)
        res = res and np.allclose(np.array(self.transform), np.array(other.transform))
        return res

    def __repr__(self):
        return '<BandSample: name={}, shape={}, dtype={}>'.format(self.name,
                                                                  self.shape,
                                                                  self.dtype)

    # ======================== PROPERTY BLOCK ========================
    @property
    def width(self):
        return self._raster.shape[1]

    @property
    def height(self):
        return self._raster.shape[0]

    @property
    def count(self):
        return 1

    @property
    def shape(self):
        """
        The raster dimension as a Tuple (height, width)
        """
        return self.height, self.width

    @property
    def dtype(self):
        """
        Data type of the associated numpy array
        """
        return self._raster.dtype

    @property
    def res(self):
        return abs(self.transform.a), abs(self.transform.e)

    @property
    def transform(self):
        return self._transform

    @property
    def crs(self):
        return self._crs

    @property
    def nodata(self):
        return self._nodata

    @property
    def bounds(self):
        """
        Georeferenced bounds - bounding box in the CRS of the image, based on transform and shape

        Returns:
            `BoundingBox object
            <https://rasterio.readthedocs.io/en/latest/api/rasterio.coords.html#rasterio.coords.BoundingBox>`_:
            (left, bottom, right, top)
        """
        left = self.transform.c
        top = self.transform.f
        right = left + self.transform.a * self.width
        bottom = top + self.transform.e * self.height
        return BoundingBox(left, bottom, right, top)

    @property
    def name(self):
        """
        name of the sample, is used as a base filename when saving to file
        """
        return self._name

    # ======================== METHODS BLOCK ========================

    @classmethod
    def from_file(cls, fp):
        """
        Reads the raster data directly from the file.
        File must have only one channel. If you need to read multi-channel file,
        use :meth:`aeronet.converters.split.split` first

        Args:
            fp: full path to the file

        Returns:
            a new `BandSample` object
        """
        band = Band(fp)
        return band.sample(0, 0, band.width, band.height)

    def same(self, other):
        """Compare if samples have same resolution, crs and shape.

        This means that the samples represent the same territory (like different spectral channels of the same image)
        and can be processed together as collection.

        Args:
            other: GeoObject to compare with

        Returns:
            True if the objects match in shape, crs, transform, False otherwise
        """

        res = True
        res = res and (self.crs == other.crs)
        res = res and (self.transform == self.transform)
        res = res and (self.height == self.height)
        res = res and (self.width == self.width)
        return res

    def save(self, directory, ext='.tif', **kwargs):
        """
        Saves the raster data to a new geotiff file; the filename is derived from this `BandSample` name.
        If file exists, it will be overwritten.

        Args:
            directory: folder to save the file
            ext: file extension; as now only GTiff driver is used, it should match `tif`, `tiff`, `TIF` or `TIFF`.
            kwargs: other keywords arguments to be passed to `rasterio.open <https://rasterio.readthedocs.io/en/latest/api/rasterio.html#rasterio.open>`_ .
        """
        fp = os.path.join(directory, self._name + ext)
        with rasterio.open(fp, mode='w', driver='GTiff', width=self.width,
                           height=self.height, count=1, crs=self.crs.get('init'),
                           transform=self.transform, nodata=self.nodata,
                           dtype=self.dtype, **kwargs) as dst:
            dst.write(self._raster.squeeze(), 1)

    def sample(self, y, x, height, width):
        """ Subsample of the Sample with specified dimensions and position within the raster:

        Args:
            x (int): horizontal pixel coordinate of left top corner
            y (int): vertical pixel coordinate of left top corner
            width (int): spatial x-dimension of sample in pixels
            height (int): spatial y-dimension of sample in pixels

        Return:
            a new `BandSample` object
        """

        coord_x = self.transform.c + x * self.transform.a
        coord_y = self.transform.f + y * self.transform.e

        dst_transform = Affine(self.transform.a, self.transform.b, coord_x,
                               self.transform.d, self.transform.e, coord_y)
        dst_raster = self._raster[y:y+height, x:x+width]

        return BandSample(self.name, dst_raster, self.crs, dst_transform, self.nodata)

    def reproject(self, dst_crs, interpolation='nearest'):
        """ Change coordinate system (projection) of the band.
        It returns a new BandSample and does not alter the current object

        It is based on `rasterio.warp.reproject
        <https://rasterio.readthedocs.io/en/latest/api/rasterio.warp.html#rasterio.warp.reproject>`_,
        see for more variants of interpolation.

        Args:
            dst_crs: new CRS, may be in any form acceptable by rasterio, for example as EPSG code, string, CRS object; if dst_crs == `utm`, the appropriate UTM zone is used according to the center of the image
            interpolation: interpolation type as in rasterio,  `nearest`, `bilinear`, `cubic`, `lanzsos` or others

        Returns:
            BandSample: a new instance with changed CRS.
        """
        if isinstance(dst_crs, str) and dst_crs == 'utm':
            dst_crs = get_utm_zone(self.crs, self.transform, (self.height, self.width))
        else:
            dst_crs = dst_crs if isinstance(dst_crs, CRS) else CRS.from_user_input(dst_crs)

        # Old rasterio compatibility: a separate check for validity
        if not dst_crs.is_valid:
            raise rasterio.errors.CRSError('Invalid CRS {} given'.format(dst_crs))

        dst_transform, dst_width, dst_height = calculate_default_transform(
            self.crs, dst_crs, self.width, self.height, *self.bounds)

        new_raster = np.empty(shape=(1, dst_height, dst_width), dtype=self.dtype)

        reproject(
            self._raster, new_raster,
            src_transform=self.transform,
            dst_transform=dst_transform,
            src_crs=self.crs,
            dst_crs=dst_crs,
            resampling=getattr(Resampling, interpolation))

        return BandSample(self.name, new_raster, dst_crs, dst_transform, self.nodata)

    def reproject_to_utm(self, interpolation='nearest'):
        """
        Alias of :obj:`BandSample.reproject` method with automatic Band utm zone determining
        """
        dst_crs = get_utm_zone(self.crs, self.transform, (self.height, self.width))
        return self.reproject(dst_crs, interpolation=interpolation)

    def resample(self, dst_res=None, dst_shape=None, interpolation='nearest'):
        """ Change spatial resolution of the sample, resizing the raster according to the new resolution.
        dst_res should be specified, otherwise the destination transform will be equal to the source.
        If dst_shape is not specified, it is calculated from dst_res,
        but it can be specified to override it and get the desired output shape

        It is based on `rasterio.warp.reproject
        <https://rasterio.readthedocs.io/en/latest/api/rasterio.warp.html#rasterio.warp.reproject>`_,
        see for more variants of interpolation.

        Args:
            dst_res (Tuple[float, float]): new resoluton, georeferenced pixel size for the new band
            dst_shape: new shape of the resampled raster, can override calculated new shape
            interpolation: interpolation type as in rasterio,  `nearest`, `bilinear`, `cubic`, `lanzsos` or others

        Returns:
            a new resampled BandSample.
        """
        transform = self.transform if dst_res is None else Affine(dst_res[1],
                                                                  self.transform.b,
                                                                  self.transform.c,
                                                                  self.transform.d,
                                                                  - dst_res[0],
                                                                  self.transform.f)

        if dst_res is not None and dst_shape is None:
            target_height = int(self.height * self.res[0] / dst_res[0])
            target_width = int(self.width * self.res[1] / dst_res[1])
        elif dst_shape is not None:
            target_height = dst_shape[1]
            target_width = dst_shape[2]
        else:
            target_height = self.height
            target_width = self.width

        new_raster = np.empty(shape=(1, target_height, target_width), dtype=self.dtype)

        reproject(
            self._raster, new_raster,
            src_transform=self.transform,
            dst_transform=transform,
            src_crs=self.crs,
            dst_crs=self.crs,
            resampling=getattr(Resampling, interpolation))

        return BandSample(self._name, new_raster, self.crs, transform, self.nodata)

    def numpy(self):
        """
        A numpy representation of the raster, without metadata

        Returns: Sample's raster data as a numpy array
        """
        return self._raster

    def generate_samples(self, width, height):
        """
        A generator for sequential sampling of the whole sample, similar to Band,
        used for the windowed processing of the raster data.

        Args:
            width (int): dimension of sample in pixels and step along `X` axis
            height (int): dimension of sample in pixels and step along `Y` axis

        Yields:
            BandSample: sequential samples of the specified dimensions
        """
        for x in range(0, self.width, width):
            for y in range(0, self.height, height):
                yield self.sample(y, x, height, width)
