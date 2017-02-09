'''
Created on Dec 19, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

This module contains several data structures and functions for manipulating them
'''

from __future__ import division

import collections
from contextlib import contextmanager
import logging
import yaml



def transpose_list_of_dicts(data, missing=None):
    """ turns a list of dictionaries into a dictionary of lists, filling in
    `missing` for items that are not available in some dictionaries """
    result = {}
    result_len = 0
    keys = set()
    
    # iterate through the whole list and add items one by one
    for item in data:
        # add the items to the result dictionary
        for k, v in item.items():
            try:
                result[k].append(v)
            except KeyError:
                keys.add(k)
                result[k] = [missing] * result_len + [v]
                
        # add missing items
        for k in keys - set(item.keys()):
            result[k].append(missing)
                
        result_len += 1
    
    return result



def save_dict_to_csv(data, filename, first_columns=None, **kwargs):
    """ function that takes a dictionary of lists and saves it as a csv file """
    if first_columns is None:
        first_columns = []

    # sort the columns 
    sorted_index = {c: k for k, c in enumerate(sorted(data.keys()))}
    def column_key(col):
        """ helper function for sorting the columns in the given order """
        try:
            return first_columns.index(col)
        except ValueError:
            return len(first_columns) + sorted_index[col]
    sorted_keys = sorted(data.keys(), key=column_key)
        
    # create a data table and indicated potential units associated with the data
    # in the header
    table = collections.OrderedDict()
    for key in sorted_keys:
        value = data[key]
        
        # check if value has a unit
        if hasattr(value, 'units'):
            # value is single item with unit
            key += ' [%s]' % value.units
            value = value.magnitude
            
        elif len(value) > 0 and any(hasattr(v, 'units') for v in value):
            # value is a list with at least one unit attached to it
            
            try:
                # get list of units ignoring empty items
                units = set(str(item.units)
                            for item in value
                            if item is not None)
            except AttributeError:
                # one item did not have a unit
                for k, item in enumerate(value):
                    if not hasattr(item, 'units'):
                        logging.info([val[k] for val in data.values()])
                        raise AttributeError('Value `%s = %s` does not have '
                                             'any units' % (key, item))
                raise
            
            # make sure that the units are all the same
            assert len(units) == 1
            
            # construct key and values
            key += ' [%s]' % units.pop()
            value = [item.magnitude if item is not None else None
                     for item in value]
            
        table[key] = value

    # create a pandas data frame to save data to CSV
    import pandas as pd
    pd.DataFrame(table).to_csv(filename, **kwargs)



class OmniContainer(object):
    """ helper class that acts as a container that contains everything """
    
    def __bool__(self):
        return True
    
    def __nonzero__(self):
        return True
    
    def __contains__(self, key):
        return True
    
    def __delitem__(self, key):
        pass
    
    def __repr__(self):
        return '%s()' % self.__class__.__name__
    
    

@contextmanager
def yaml_database(filename, default_flow_style=False, factory=dict,
                  allow_classes=False):
    """ a context manager that opens a yaml file and yields its content. When
    the context manager is left, the data is written back on the disk. This is
    useful to modify simple configuration files or databases:
    
    with yaml_database('config.yaml') as config:
        config['user'] = 'name'
    
    Now, the file `config.yaml` will read "user: name".
    
    `default_flow_style` sets the style used for writing the yaml file. See the
        docstring of the `yaml.dump` function for details.
    `factory` defines how the database should be initialized in case the file is
        not present or is empty.
    `allow_classes` allows dumping classes using `yaml.safe` when enabled
    """
    # read the database
    try:
        with open(filename, 'r') as fp:
            database = yaml.load(fp)
    except IOError:
        # file does not seem to exists
        database = None
        
    # initialize an empty database
    if database is None and factory is not None:
        database = factory()
            
    yield database
    
    # write the database back to file
    with open(filename, 'w') as fp:
        if allow_classes:
            yaml.dump(database, fp, default_flow_style=default_flow_style)
        else:
            yaml.safe_dump(database, fp, default_flow_style=default_flow_style)



# TODO: write context manager for multiple database structures (yaml, pickle)
