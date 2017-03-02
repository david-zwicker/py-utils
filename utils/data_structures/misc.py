'''
Created on Dec 19, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

This module contains several data structures and functions for manipulating them
'''

from __future__ import division

import collections
import logging
import yaml
import warnings



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
    
    
                
class SimpleDatabase(object):
    
    formats = {'yaml': {'default_flow_style': False,
                        'allow_classes': False}}    

    def __init__(self, filename, locking=True, factory=dict, db_format='yaml',
                 format_parameters=None):
        """ a context manager that opens a database file and yields its content.
        When the context manager is left, the data is written back on the disk.
        This is useful to modify simple configuration files or databases:
         
        with SimpleDatabase('config.yaml', db_format='yaml') as config:
            config['user'] = 'name'
         
        Now, the file `config.yaml` will read "user: name".
         
        `filename` gives the name of the database file
        `locking` determines whether the file will be locked the entire time so
            other processes cannot use it. This is turned on by default to
            prevent data corruption and race conditions.
        `factory` defines how the database should be initialized in case the
            file is not present or is empty.
        `db_format` is the file format that is used to read and write the data.
            So far, only `yaml` has been implemented.
        `format_parameters` is a dictionary with additional parameters that
            influence how the database is written to the file.
        """
    
        # save some options for using this decorator
        self._filename = filename
        self._locking = locking
        self.factory = factory
        
        self.format = db_format
        try:
            self.format_parameters = self.formats[self.format]
        except KeyError:
            raise ValueError('The format `%s` is not supported' % db_format)
        if format_parameters is not None:
            self.format_parameters.update(format_parameters)
            
        self._database_fh = None
        self._database = None


    def __enter__(self):
        """ read the database or initialize it if it was not setup """
        # read the 
        if self._locking:
            # open database file with locking
            try:
                import portalocker
            except ImportError:
                logging.error('The `portalocker` module must be installed to '
                              'support locking of files on all platforms.')
                raise
    
            logging.debug('Open and lock file `%s` to work with YAML database',
                          self._filename)
            try:
                self._database_fh = open(self._filename, 'r+')
            except IOError:
                # file did not exist => create it
                self._database_fh = open(self._filename, 'w+')
                
            # try locking the file
            portalocker.lock(self._database_fh,
                             portalocker.LOCK_EX | portalocker.LOCK_NB)
            
        else:
            # open database file without locking
            logging.debug('Open file `%s` to read entire YAML database',
                          self._filename)
            try:
                self._database_fh = open(self._filename, 'r') 
            except IOError:
                self._database_fh = None  # file does not seem to exists

        # read the database if file handle is available
        if self._database_fh is None:
            self._database = None
        elif self.format == 'yaml':
            self._database = yaml.load(self._database_fh)
        else:
            raise NotImplementedError('Unsupported format `%s`' % self.format)

        # close database file it is not locked for the entire time
        if not self._locking and self._database_fh is not None:
            self._database_fh.close()
            
        # initialize the database if it was empty
        if self._database is None and self.factory is not None:
            self._database = self.factory()
        
        return self._database


    def __exit__(self, *args):
        """ write back the database after it was potentially changed """
        # prepare the database file
        if self._locking:
            # rewind the database file
            logging.debug('Rewind file `%s` to update YAML database',
                          self._filename)
            self._database_fh.seek(0)
        else:
            # reopen the database file
            logging.debug('Open file `%s` to write entire YAML database',
                          self._filename)
            self._database_fh = open(self._filename, 'w')
            
        if self.format == 'yaml':
            # dump the database as yaml
            default_flow_style = self.format_parameters['default_flow_style']
            if self.format_parameters['allow_classes']:
                yaml.dump(self._database, self._database_fh,
                          default_flow_style=default_flow_style)
            else:
                yaml.safe_dump(self._database, self._database_fh,
                               default_flow_style=default_flow_style)
                
        else:
            raise NotImplementedError('Unsupported format `%s`' % self.format)

        # close file and release lock if it was acquired
        self._database_fh.close()
        self._database_fh = None



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
    
    # raise warning that this interface is deprecated
    warnings.warn("Calling the deprecated function `yaml_database`.",
                  category=DeprecationWarning, stacklevel=1)
    
    # translate this call to the newer interface
    return SimpleDatabase(
        filename, locking=False, factory=dict, db_format='yaml',
        format_parameters={'allow_classes': allow_classes,
                           'default_flow_style': default_flow_style}
    )

