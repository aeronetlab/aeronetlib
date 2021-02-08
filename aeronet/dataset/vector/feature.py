import json
import rtree
import warnings
import shapely
import shapely.geometry

from rasterio.warp import transform_geom
from rasterio.crs import CRS
from rasterio.errors import CRSError

from ..coords import _utm_zone, CRS_LATLON


class Feature:
    """
    Proxy class for shapely geometry, include crs and properties of feature
    """

    def __init__(self, geometry, properties=None, crs=CRS_LATLON):
        self.crs = crs
        self._geometry = self._valid(
            shapely.geometry.shape(geometry))
        self.properties = properties

    def __repr__(self):
        print('CRS: {}\nProperties: {}'.format(self.crs, self.properties))
        return repr(self._geometry)

    def __getattr__(self, item):
        return getattr(self._geometry, item)
    
    def __setstate__(self, state):
        self.__dict__ = state

    def __getstate__(self):
        return self.__dict__

    def _valid(self, shape):
        if not shape.is_valid:
            shape = shape.buffer(0)
        return shape

    def apply(self, func):
        return Feature(func(self._geometry), properties=self.properties, crs=self.crs)

    @property
    def shape(self):
        return self._geometry

    @property
    def geometry(self):
        return shapely.geometry.mapping(self._geometry)

    @property
    def geojson(self):

        if self.crs != CRS_LATLON:
            f = self.reproject(CRS_LATLON)
        else:
            f = self

        data = {
            'type': 'Feature',
            'geometry': f.geometry,
            'properties': f.properties
        }
        return data

    def reproject(self, dst_crs):
        new_geometry = transform_geom(
            src_crs=self.crs,
            dst_crs=dst_crs,
            geom=self.geometry,
        )
        return Feature(new_geometry, properties=self.properties, crs=dst_crs)
    
    def reproject_to_utm(self):
        lon1, lat1, lon2, lat2 = self.shape.bounds
        utm_zone = _utm_zone((lat1 + lat2)/2 , (lon1 + lon2)/2)
        return self.reproject(utm_zone)


class FeatureCollection:
    """A set of Features with the same CRS"""

    def __init__(self, features, crs=CRS_LATLON):
        self.crs = crs
        self.features = self._valid(features)

        # create indexed set for faster processing
        self.index = rtree.index.Index()
        for i, f in enumerate(self.features):
            self.index.add(i, f.bounds, f.shape)

    def __getitem__(self, item):
        return self.features[item]

    def __len__(self):
        return len(self.features)

    def _valid(self, features):
        valid_features = []
        for f in features:
            if not f.geometry.get('coordinates'): # remove possible empty shapes
                warnings.warn('Empty geometry detected. This geometry have been removed from collection.',
                              RuntimeWarning)
            else:
                valid_features.append(f)
        return valid_features

    def apply(self, func):
        """ Applies a given function to all the Features of this FeatureColletion

        Args:
            func: A function to be applied to the Features. Must take and return shapely.geometry

        Returns:
            A new FeatureCollection with modified Features
        """
        new_features = [f.apply(func) for f in self.features]
        return FeatureCollection(new_features, crs=self.crs)

    def filter(self, func):
        features = [x for x in self if func(x)]
        return FeatureCollection(features, crs=self.crs)

    def extend(self, fc):
        for i, f in enumerate(fc):
            self.index.add(i + len(self), f.bounds)
        self.features.extend(fc.features)

    def append(self, feature):
        self.index.add(len(self) + 1, feature.bounds)
        self.features.append(feature)

    def bounds_intersection(self, feature):
        idx = self.index.intersection(feature.bounds)
        features = [self.features[i] for i in idx]
        return FeatureCollection(features, self.crs)

    def intersection(self, feature):
        proposed_features = self.bounds_intersection(feature)
        features = []
        for pf in proposed_features:
            if pf.intersection(feature).area > 0:
                features.append(pf)
        return FeatureCollection(features, self.crs)

    @staticmethod
    def _read_crs(collection):

        # if there is no defined CRS in geojson file, we folloe the standard, which says that it must be lat-lon
        if 'crs' not in collection.keys():
            return CRS_LATLON

        crs_raw = collection.get('crs', CRS_LATLON)
        crs = CRS()

        try:
            if isinstance(crs_raw, str):
                crs = CRS.from_user_input(crs_raw)
            elif isinstance(crs_raw, dict):
                if 'type' in crs_raw.keys() and 'properties' in crs_raw.keys():
                    if crs_raw['type'] == 'name':
                        crs = CRS.from_user_input(crs_raw['properties']['name'])
            # Old rasterio compatibility: a separate check for validity
            if not crs.is_valid:
                message = 'CRS {} is not supported by rasterio,' \
                          'May cause an error in further reprojection or rasterization'.format(crs)
                warnings.warn(message, RuntimeWarning)
            return crs
            # Really invalid CRS will throw CRSError
        except CRSError:
            message = 'CRS was not imported correctly, assuming EPSG:4326 (lat-lon). ' \
                      'May cause an error in further reprojection or rasterization if it is not so.'
            warnings.warn(message, RuntimeWarning)
            return CRS_LATLON

    @classmethod
    def read(cls, fp):
        r"""Reading the FeatureCollection from a geojson file.
            Args:
                fp: file identifier to open and read the data
            Returns:
                new FeatureCollection with all the polygon data in the file
           """
        with open(fp, 'r', encoding='utf-8') as f:
            collection = json.load(f)

        '''
        We want the CRS to be specified in rasterio-compatible way so that we could reproject the collection
        If it is not specified, it is OK and assumed by the geojson standard that it is CRS_LATLON
        Else it must be read from the file, and if it does not meet any known scheme, we have 2 options:
        - either throw an exception (predictable, but can fail unnecessarily)
        - or ignore the CRS data with a warning (will work better normally, but can give unexpected results)
        '''
        crs = cls._read_crs(collection)
        features = []
        for i, feature in enumerate(collection['features']):
            
            try:
                feature_ = Feature(
                    geometry=feature['geometry'], 
                    properties=feature['properties'],
                    crs=crs,
                )
                features.append(feature_)
            except (KeyError, IndexError) as e:
                message = 'Feature #{} have been removed from collection. Error: {}'.format(i, str(e))
                warnings.warn(message, RuntimeWarning)
                
        return cls(features)

    def save(self, fp):
        r"""Saving the feature collection as geojson file
            The features will be written in lat-lon CRS
            Args:
                fp: file identifier to open and save the data
           """
        with open(fp, 'w') as f:
            json.dump(self.geojson, f)

    @property
    def geojson(self):
        data = {
            'type': 'FeatureCollection',
            'crs': CRS_LATLON.to_dict(),
            'features': [f.geojson for f in self.features]
        }
        return data

    def reproject(self, dst_crs):
        """
        Reprojects all the features to the new crs
        Args:
            dst_crs: rasterio.CRS or any acceptable by your rasterio version input (str, dict, epsg code), ot 'utm'

        Returns:
            new reprojected FeatureCollection
        """
        if isinstance(dst_crs, str) and dst_crs == 'utm':
            lon1, lat1, lon2, lat2 = self.index.bounds
            dst_crs = _utm_zone((lat1 + lat2)/2, (lon1 + lon2)/2)
        else:
            dst_crs = dst_crs if isinstance(dst_crs, CRS) else CRS.from_user_input(dst_crs)

        # Old rasterio compatibility: a separate check for validity
        if not dst_crs.is_valid:
            raise CRSError('Invalid CRS {} given'.format(dst_crs))

        features = [f.reproject(dst_crs) for f in self.features]
        return FeatureCollection(features, dst_crs)

    def reproject_to_utm(self):
        """
        Alias of `reproject` method with automatic Band utm zone determining
        The utm zone is determined according to the center of the bounding box of the collection.
        Does not suit to large area geometry, that would not fit into one zone (about 6 dergees in longitude)
        """
        return self.reproject(dst_crs='utm')
