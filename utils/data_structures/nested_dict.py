'''
Created on Aug 9, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import collections
import datetime
import os.path
import warnings

import h5py
import numpy as np
import six



def get_chunk_size(shape, num_elements):
    """ tries to determine an optimal chunk size for an array with a given 
    shape by chunking the longest axes first """
    chunks = list(shape)
    while np.prod(chunks) > num_elements:
        dim_long = np.argmax(chunks)  # get longest dimension
        chunks[dim_long] = 1  # temporary set to one for np.prod
        chunks[dim_long] = max(1, num_elements // np.prod(chunks))
    return tuple(chunks)
    
    

class LazyLoadError(RuntimeError):
    """ exception that can be thrown if lazy-loading failed """
    pass



class LazyValue(object):
    """ base class that represents a value that is only loaded when it is
    accessed """
    def load(self):
        raise NotImplementedError
    


class LazyHDFValue(LazyValue):
    """ class that represents a value that is only loaded from HDF when it is
    accessed """
    chunk_elements = True  # 10000
    compression = None
    

    def __init__(self, data_cls, key, hdf_filename):
        self.data_cls = data_cls
        self.key = key
        self.hdf_filename = hdf_filename
        

    def __repr__(self):
        return '%s(data_cls=%s, key="%s", hdf_filename="%s")' % (
                    self.__class__.__name__, self.data_cls.__name__,
                    self.key, self.hdf_filename)


    def __eq__(self, other):
        return self.__dict__ == other.__dict__        

        
    def set_hdf_folder(self, hdf_folder):
        """ replaces the folder of the hdf file """
        hdf_name = os.path.basename(self.hdf_filename)
        self.hdf_filename = os.path.join(hdf_folder, hdf_name)
        
        
    def get_yaml_string(self):
        """ returns a representation of the object as a single string, which
        is useful for referencing the object in YAML """
        hdf_name = os.path.basename(self.hdf_filename)
        return '@%s:%s' % (hdf_name, self.key)
        
        
    @classmethod
    def create_from_yaml_string(cls, value, data_cls, hdf_folder):
        """ create an instance of the class from the yaml string and additional
        information """

        # consistency check
        if not value.startswith('@'):
            raise ValueError('Item with lazy loading does not start with `@`')
        
        # read the link
        data_str = value[1:]  # strip the first character, which should be an @
        hdf_name, key = data_str.split(':')
        hdf_filename = os.path.join(hdf_folder, hdf_name)
        return cls(data_cls, key, hdf_filename)
        
    
    @classmethod    
    def create_from_data(cls, key, data, hdf_filename):
        """ store the data in a HDF file and return the storage object """
        data_cls = data.__class__
        with h5py.File(hdf_filename, 'a') as hdf_file:
            # delete possible previous key to have a clean storage
            if key in hdf_file:
                del hdf_file[key]
                
            # save actual data as an array
            data_array = np.asarray(data.to_array())

            # determine the chunk size if necessary            
            if isinstance(cls.chunk_elements, int): 
                chunks = get_chunk_size(data_array.shape, cls.chunk_elements)
            else:
                chunks = cls.chunk_elements

            # create the dataset with the given options
            hdf_file.create_dataset(key, data=data_array, track_times=True,
                                    chunks=chunks, compression=cls.compression)
                
            # add attributes to describe data 
            hdf_file[key].attrs['written_on'] = str(datetime.datetime.now())
            if hasattr(data_cls, 'hdf_attributes'):        
                for attr_key, attr_value in \
                        six.iteritems(data_cls.hdf_attributes):
                    hdf_file[key].attrs[attr_key] = attr_value
            
        return cls(data_cls, key, hdf_filename)
    
        
    def load(self):
        """ load the data and return it """
        # open the associated HDF5 file and read the data
        with h5py.File(self.hdf_filename, 'r') as hdf_file:
            data = hdf_file[self.key][:]  # copy data into RAM
            result = self.data_cls.from_array(data)
        
        # create object
        return result



class LazyHDFCollection(LazyHDFValue):
    """ class that represents a collection of values that are only loaded when 
    they are accessed """

    @classmethod    
    def create_from_data(cls, key, data, hdf_filename):
        """ store the data in a HDF file and return the storage object """
        data_cls = data.__class__

        # save a collection of objects to hdf
        with h5py.File(hdf_filename, 'a') as hdf_file:
            # reset the whole structure if it is there
            if key in hdf_file:
                del hdf_file[key]
                
            # create group in case data is empty
            hdf_file.create_group(key)

            # write all objects as individual datasets            
            key_format = '{}/%0{}d'.format(key, len(str(len(data))))
            for index, obj in enumerate(data):
                obj.save_to_hdf5(hdf_file, key_format % index)
    
            hdf_file[key].attrs['written_on'] = str(datetime.datetime.now())
            if hasattr(data_cls, 'hdf_attributes'):        
                for attr_key, attr_value in \
                        six.iteritems(data_cls.hdf_attributes):
                    hdf_file[key].attrs[attr_key] = attr_value

        return cls(data_cls, key, hdf_filename)
    
        
    def load(self):
        """ load the data and return it """
        # open the associated HDF5 file and read the data
        item_cls = self.data_cls.item_class
        with h5py.File(self.hdf_filename, 'r') as hdf_file:
            # iterate over the data and create objects from it
            data = hdf_file[self.key]
            if data:
                result = self.data_cls(item_cls.from_array(data[index][:])
                                       for index in sorted(data.keys()))
                # here, we have to use sorted() to iterate in the correct order 
            else:  # empty dataset
                result = self.data_cls()
                
        return result



class NestedDict(collections.MutableMapping):
    """ special dictionary class representing nested dictionaries.
    This class allows easy access to nested properties using a single key:
    
    d = NestedDict({'a': {'b': 1}})
    
    d['a/b']
    >>>> 1
    
    d['a/c'] = 2
    
    d
    >>>> {'a': {'b': 1, 'c': 2}}
    """
    
    def __init__(self, data=None, sep='/', dict_class=dict):
        """ initialize the NestedDict object
        `data` is a dictionary that is used to fill the current object
        `sep` determines the separator used for accessing different levels of
            the structure
        `dict_class` is the dictionary class that will handle items under the
            hood and for instance determines how items are iterated over
        """

        # store details about the dictionary
        self.sep = sep
        self.dict_class = dict_class

        # set data
        self.data = self.dict_class()
        if data is not None:
            self.from_dict(data)


    def get_item(self, key):
        """ returns the item identified by `key`.
        If load_data is True, a potential LazyValue gets loaded """
        try:
            if isinstance(key, six.string_types) and self.sep in key:
                # sub-data is accessed
                child, grandchildren = key.split(self.sep, 1)
                try:
                    value = self.data[child].get_item(grandchildren)
                except AttributeError:
                    raise KeyError(key)
            else:
                value = self.data[key]
        except KeyError:
            raise KeyError(key)

        return value

    
    def __getitem__(self, key):
        return self.get_item(key)
        
        
    def __setitem__(self, key, value):
        """ writes the item into the dictionary """
        if isinstance(key, six.string_types) and self.sep in key:
            # sub-data is written
            child, grandchildren = key.split(self.sep, 1)
            try:
                self.data[child][grandchildren] = value
            except KeyError:
                # create new child if it does not exists
                child_node = self.create_dict()
                child_node[grandchildren] = value
                self.data[child] = child_node
            except TypeError:
                raise TypeError('Can only use Xpath assignment if all children '
                                'are NestedDict instances.')
                
        else:
            self.data[key] = value
    
    
    def __delitem__(self, key):
        """ deletes the item identified by key """
        try:
            if isinstance(key, six.string_types) and self.sep in key:
                # sub-data is deleted
                child, grandchildren = key.split(self.sep, 1)
                try:
                    del self.data[child][grandchildren]
                except TypeError:
                    raise KeyError(key)
    
            else:
                del self.data[key]
        except KeyError:
            raise KeyError(key)


    def __contains__(self, key):
        """ returns True if the key is contained in the data """
        if isinstance(key, six.string_types) and self.sep in key:
            child, grandchildren = key.split(self.sep, 1)
            try:
                return child in self.data and grandchildren in self.data[child]
            except TypeError:
                return False

        else:
            return key in self.data


    # Miscellaneous dictionary methods are just mapped to data
    def __len__(self): return len(self.data)
    def __iter__(self): return self.data.__iter__()
    def keys(self): return self.data.keys()
    def values(self): return self.data.values()
    def items(self): return self.data.items()
    def clear(self): self.data.clear()


    def itervalues(self, flatten=False):
        """ an iterator over the values of the dictionary
        If flatten is true, iteration is recursive """
        for value in six.itervalues(self.data):
            if flatten and isinstance(value, NestedDict):
                # recurse into sub dictionary
                for v in value.itervalues(flatten=True):
                    yield v
            else:
                yield value 
                
                
    def iterkeys(self, flatten=False):
        """ an iterator over the keys of the dictionary
        If flatten is true, iteration is recursive """
        if flatten:
            for key, value in six.iteritems(self.data):
                if isinstance(value, NestedDict):
                    # recurse into sub dictionary
                    try:
                        prefix = key + self.sep
                    except TypeError:
                        raise TypeError('Keys for NestedDict must be strings '
                                        '(`%s` is invalid)' % key)
                    for k in value.iterkeys(flatten=True):
                        yield prefix + k
                else:
                    yield key
        else:
            for key in six.iterkeys(self.data):
                yield key


    def iteritems(self, flatten=False):
        """ an iterator over the (key, value) items
        If flatten is true, iteration is recursive """
        for key, value in six.iteritems(self.data):
            if flatten and isinstance(value, NestedDict):
                # recurse into sub dictionary
                try:
                    prefix = key + self.sep
                except TypeError:
                    raise TypeError('Keys for NestedDict must be strings '
                                    '(`%s` is invalid)' % key)
                for k, v in value.iteritems(flatten=True):
                    yield prefix + k, v
            else:
                yield key, value 

            
    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, repr(self.data))


    def create_dict(self, data=None):
        """ creates an empty NestedDict with the same properties as this one """
        return self.__class__(data, sep=self.sep, dict_class=self.dict_class)


    def create_child(self, key, values=None):
        """ creates a child dictionary and fills it with values """
        self[key] = self.create_dict(values)
        return self[key]


    def insert(self, key, value):
        """ insert new data into the location at key. If the `value` is a
        subclass of collections.Mapping it is turned into a NestedDict, so it
        can be accessed nicely """
        if isinstance(value, collections.Mapping):
            self[key] = self.create_dict(value)
        else:
            self[key] = value


    def update_recursive(self, other):
        """ update the dictionary recursively """
        if other:
            for k, v in six.iteritems(other):
                try:
                    self[k].update_recursive(v)
                except (AttributeError, KeyError):
                    self[k] = v


    def copy(self):
        """ makes a deepcopy of the dictionary (but a shallow copy of the actual
        values stored in the dictionary. Use copy.deepcopy if the values also
        need to be copied """
        res = self.create_dict()
        for key, value in six.iteritems(self):
            if isinstance(value, (dict, NestedDict)):
                value = value.copy()
            res[key] = value
        return res


    def from_dict(self, data):
        """ fill the object with data from a dictionary """
        for key, value in six.iteritems(data):
            if isinstance(value, dict):
                if key in self and isinstance(self[key], NestedDict):
                    # extend existing NestedDict instance
                    self[key].from_dict(value)
                else:
                    # create new NestedDict instance
                    self[key] = self.create_dict(value)
            else:
                # store simple value
                self[key] = value

            
    def to_dict(self, flatten=False):
        """ convert object to a nested dictionary structure.
        If flatten is True a single dictionary with complex keys is returned.
        If flatten is False, a nested dictionary with simple keys is returned
        """
        res = self.dict_class()
        for key, value in self.iteritems():
            if isinstance(value, NestedDict):
                value = value.to_dict(flatten=flatten)
                if flatten:
                    for k, v in six.iteritems(value):
                        try:
                            res[key + self.sep + k] = v
                        except TypeError:
                            raise TypeError('Keys for NestedDict must be '
                                            'strings (`%s` or `%s` is invalid)'
                                            % (key, k))
                else:
                    res[key] = value
            else:
                res[key] = value
        return res

    
    def pprint(self, *args, **kwargs):
        """ pretty print the current structure as nested dictionaries """
        from pprint import pprint
        pprint(self.to_dict(), *args, **kwargs)



class LazyNestedDict(NestedDict):
    """ special dictionary class representing nested dictionaries.
    This class allows easy access to nested properties using a single key.
    Additionally, this class supports loading lazy values if they are accessed
    """
    
    def get_item(self, key, load_data=True):
        """ returns the item identified by `key`.
        If load_data is True, a potential LazyValue gets loaded """
        try:
            if isinstance(key, six.string_types) and self.sep in key:
                # sub-data is accessed
                child, grandchildren = key.split(self.sep, 1)
                try:
                    value = self.data[child].get_item(grandchildren, load_data)
                except AttributeError:
                    raise KeyError(key)
            else:
                value = self.data[key]
        except KeyError:
            raise KeyError(key)

        # load lazy values
        if load_data and isinstance(value, LazyValue):
            try:
                value = value.load()
            except KeyError as err:
                # we have to relabel KeyErrors, since they otherwise shadow
                # KeyErrors raised by the item actually not being in the
                # NestedDict. This then allows us to distinguish between items
                # not found in NestedDict (raising KeyError) and items not being
                # able to load (raising LazyLoadError)
                msg = ('Cannot load item `%s`.\nThe original error was: %s'
                       % (key, str(err)))
                six.raise_from(LazyLoadError(msg), err)
            self.data[key] = value  # replace loader with actual value
            
        return value
    
    

def normalize_dict(data, flatten=False):
    """ converts the NestedDict `data` into a normal dict by calling its
    `to_dict` method. If data is already a dict it is returned unchanged.
    
    `flatten` determines whether the NestedDict is flattened into a dictionary
        with a single level and complex keys
    """
    try:
        return data.to_dict(flatten=flatten)
    except AttributeError:
        return data
    
    

def prepare_data_for_yaml(data, _key=None):
    """ recursively converts some special types to close python equivalents """
    if _key is None:
        _key = []
        
    if isinstance(data, np.ndarray):
        return data.tolist()
    
    elif isinstance(data, np.floating):
        return float(data)
    
    elif isinstance(data, np.integer):
        return int(data)
    
    elif isinstance(data, collections.MutableMapping):
        return {k: prepare_data_for_yaml(v, _key + [k])
                for k, v in six.iteritems(data)}
        
    elif isinstance(data, (list, tuple, set)):
        return [prepare_data_for_yaml(v, _key + [n])
                for n, v in enumerate(data)]
    
    elif isinstance(data, LazyHDFValue):
        return data.get_yaml_string()
    
    elif (data is None or 
          isinstance(data, (bool, int, float, six.string_types))):
        return data
    
    else:
        warnings.warn('Encountered unknown instance of `%s` at `%s` in YAML '
                      'preparation' % (data.__class__, _key))
    return data    
