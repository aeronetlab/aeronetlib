import os
from abc import ABC

import numpy as np
from .band import Band

from .geoobject import GeoObject
from ..coords import get_utm_zone


class BandCollection(GeoObject):
    """
    A collection of :obj:`Band` s with the same crs, transform and shape.

    It is intended to contain a multi-band image
    and associated raster masks. All the bands must have the same resolution, so if the initial images have
    different resolution, they must be resampled previously.
    Every band must be stored in a separate file; if the initial data is in one file,
    use :meth:`aeronet.converters.split.split` to split it to separate files

    Args:
        bands: list of `Band` or list of file paths
    """

    def __init__(self, bands):

        super().__init__()

        self._bands = [band if isinstance(band, Band) else Band(band) for band in bands]
        if not self.is_valid:
            del self
            raise ValueError('Bands are not suitable for collection. '
                             'CRS, transform or shape are different!')

    def __repr__(self):
        names = [b.name for b in self._bands]
        return '<BandCollection: {}>'.format(names)

    def __getitem__(self, item):
        return self._bands[item]

    def __len__(self):
        return self.count

    # ======================== PROPERTY BLOCK ========================

    @property
    def crs(self):
        return self._bands[0].crs

    @property
    def transform(self):
        return self._bands[0].transform

    @property
    def nodata(self):
        return self._bands[0].nodata

    @property
    def height(self):
        return self._bands[0].height

    @property
    def width(self):
        return self._bands[0].width

    @property
    def count(self):
        return len(self._bands)

    @property
    def bounds(self):
        return self._bands[0].bounds

    @property
    def shape(self):
        return self.count, self._bands[0].height, self._bands[0].width

    @property
    def res(self):
        return self._bands[0].res

    @property
    def is_valid(self):
        """Check if all bands have the same resolution, shape and coordinate system"""
        if len(self._bands) < 2:
            res = True
        else:
            first = self._bands[0]
            rest = self._bands[1:]
            res = all(first.same(other) for other in rest)

        return res

    # ======================== PRIVATE METHODS ========================

    def _get_band(self, name):
        for b in self._bands:
            if b.name == name:
                return b
            elif len(b.name) > len(name):  # legacy datasets support
                if b.name.endswith('_{name}'.format(name=name)):
                    return b
            # in all other cases raise error
        raise NameError('No sample with name {name}.'.format(name=name))

    # ======================== PUBLIC METHODS  ========================

    def append(self, other):
        """Add a band to collection, checking it to be compatible by shape, transform and crs"""
        if all(other.same(band) for band in self._bands):
            self._bands.append(other)
        else:
            raise ValueError('Band is not suitable for collection. '
                             'CRS, transform or shape are different!')
        return

    def sample(self, y, x, height, width):
        """
        Sample memory object from BandCollection.

        Args:
            x: int, top left corner `X` offset in pixels
            y: int, top left corner `Y` offset in pixels
            width: int, width of samples in pixels
            height: int, height of samples in pixels

        Returns:
            a new :obj:`BandCollectionSample` containing the specified spatial subset of the BandCollection
        """
        samples = [band.sample(y, x, height, width) for band in self._bands]
        return BandCollectionSample(samples)

    def ordered(self, *names):
        """
        Creates a new object, containing the specified bands in the specific order.

        Args:
            *names: order of names

        Returns:
            reordered `BandCollection`
        """
        ordered_bands = [self._get_band(name) for name in names]
        return BandCollection(ordered_bands)

    def reproject(self, dst_crs, directory=None, interpolation='nearest'):
        """
        Reprojects every Band of the collection, see :meth:`Band.reproject` and returns a new reprojected BandCollection
        """
        if directory:
            os.makedirs(directory, exist_ok=True)
        r_bands = []
        for band in self:
            fp = os.path.join(directory, band.name + '.tif') if directory else None
            r_band = band.reproject(dst_crs, fp=fp, interpolation=interpolation)
            r_bands.append(r_band)
        return BandCollection(r_bands)

    def reproject_to_utm(self, directory=None, interpolation='nearest'):
        """
        Alias of `reproject` method with automatic utm zone determining
        """
        dst_crs = get_utm_zone(self.crs, self.transform, (self.height, self.width))
        return self.reproject(dst_crs, directory=directory, interpolation=interpolation)

    def resample(self, dst_res, directory=None, interpolation='nearest'):
        """
        Resamples every Band of the collection, see :meth:`Band.resample` and returns a new reprojected BandCollection
        """

        if directory:
            os.makedirs(directory, exist_ok=True)
        r_bands = []
        for band in self:
            fp = os.path.join(directory, band.name + '.tif') if directory else None
            r_band = band.resample(dst_res, fp=fp, interpolation=interpolation)
            r_bands.append(r_band)
        return BandCollection(r_bands)

    def generate_samples(self, height, width):
        """
        A generator for sequential sampling of the whole BandCollection, used for the windowed reading of the raster.
        It allows to handle and process large files without reading them at once in the memory.

        Args:
            width (int): dimension of sample in pixels and step along `X` axis
            height (int): dimension of sample in pixels and step along `Y` axis

        Yields:
            BandCollectionSample: sequential samples of the specified dimensions
        """

        for x in range(0, self.width, width):
            for y in range(0, self.height, height):
                yield self.sample(y, x, height, width)

    def numpy(self):
        return self.sample(0, 0, self.height, self.width).numpy()


class BandCollectionSample(GeoObject):
    """
    A collection of :obj:`BandSample` , which are also
    """

    def __init__(self, samples):
        super().__init__()
        # todo: add is_valid check
        self._samples = samples

    def __repr__(self):
        names = [b.name for b in self._samples]
        return '<BandCollectionSample: {}>'.format(names)

    def __getitem__(self, item):
        return self._samples[item]

    def __len__(self):
        return self.count

    # ======================== PROPERTY BLOCK ========================

    @property
    def crs(self):
        return self._samples[0].crs

    @property
    def transform(self):
        return self._samples[0].transform

    @property
    def res(self):
        return self._samples[0].res

    @property
    def width(self):
        return self._samples[0].width

    @property
    def height(self):
        return self._samples[0].height

    @property
    def count(self):
        """ Number of bands (layers) in the collection"""
        return len(self._samples)

    @property
    def shape(self):
        return self.count, self.height, self.width

    @property
    def nodata(self):
        return self._samples[0].nodata

    @property
    def bounds(self):
        return self._samples[0].bounds

    @property
    def is_valid(self):
        """
        Check if all bands have the same resolution, shape and coordinate system
        """
        if len(self._samples) < 2:
            res = True
        else:
            first = self._samples[0]
            rest = self._samples[1:]
            res = all(first.same(other) for other in rest)

        return res

    # ======================== PRIVATE METHODS ========================

    def _get_sample(self, name):
        for s in self._samples:
            if s.name == name:
                return s
            elif len(s.name) > len(name):  # legacy datasets support
                if s.name.endswith('_{name}'.format(name=name)):
                    return s
            # in all other cases raise error
        raise NameError('No sample with name {name}.'.format(name=name))

    # ======================== PUBLIC METHODS  ========================

    def append(self, obj):
        """
        Add sample to collection, checking it to be compatible by shape, transform and crs
        """
        if all(obj.same(band) for band in self._samples):
            self._samples.append(obj)
        else:
            raise ValueError('Band is not suitable for collection. '
                             'CRS, transform or shape are different!')
        return

    def sample(self, y, x, height, width):
        samples = [obj.sample(y, x, height, width) for obj in self._samples]
        return BandCollectionSample(samples)

    def reproject(self, dst_crs, interpolation='nearest'):
        """
        Reprojects every BandSample of the collection, see :meth:`BandSample.reproject`
        and returns a new reprojected BandCollectionSample
        """
        reprojected_samples = [s.reproject(dst_crs, interpolation) for s in self._samples]
        return BandCollectionSample(reprojected_samples)

    def reproject_to_utm(self, interpolation='nearest'):
        """
        Alias of `reproject` method with automatic utm zone determining
        """
        dst_crs = get_utm_zone(self.crs, self.transform, (self.height, self.width))
        return self.reproject(dst_crs, interpolation=interpolation)

    def resample(self, dst_res=None, dst_shape=None, interpolation='nearest'):
        """
        Reprojects every BandSample of the collection, see :meth:`BandSample.reproject`
        and returns a new reprojected BandCollectionSample
        """
        resamples = [s.resample(dst_res, dst_shape, interpolation) for s in self._samples]
        return BandCollectionSample(resamples)

    def numpy(self):
        return np.stack([x.numpy() for x in self._samples], 0)

    def ordered(self, *names):
        """
        Creates a new object, containing the specified bands in the specific order.

        Args:
            *names: order of names

        Returns:
            reordered `BandCollectionSample`
        """
        ordered_bands = [self._get_sample(name) for name in names]
        return BandCollectionSample(ordered_bands)

    def save(self, directory, extension='.tif', **kwargs):
        """
        Saves every BandSample in the specified deirectory, see :meth:`BandSample.save`
        """
        os.makedirs(directory, exist_ok=True)
        for s in self._samples:
            s.save(directory, extension, **kwargs)

    def generate_samples(self, height, width):
        """
        A generator for sequential sampling of the BandCollectionSample similar to BandCollection,
        used for the windowed processing of the raster data.

        Args:
            width (int): dimension of sample in pixels and step along `X` axis
            height (int): dimension of sample in pixels and step along `Y` axis

        Yields:
            BandCollectionSample: sequential samples of the specified dimensions
        """
        for x in range(0, self.width, width):
            for y in range(0, self.height, height):
                yield self.sample(y, x, height, width)
