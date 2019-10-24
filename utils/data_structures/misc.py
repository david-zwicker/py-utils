'''
Created on Dec 19, 2014

@author: David Zwicker <dzwicker@seas.harvard.edu>

This module contains several data structures and functions for manipulating them
'''

from __future__ import division

import collections
import logging
import os
import warnings

import six
import yaml

from ..math import homogenize_unit_array



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
                logger = logging.getLogger(__name__)
                for k, item in enumerate(value):
                    if not hasattr(item, 'units'):
                        logger.info([val[k] for val in data.values()])
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



def read_comsol_table(filename, ret_header=False, **kwargs):
    """ read tabular data exported from comsol.
    
    `filename` gives the filename where the data has been exported
    `ret_header` determines whether the header is also returned
    
    All other keyword arguments are forwarded to the `pandas.read_csv` call
    """
    import pandas as pd  # lazy load so its not a requirement for the module
    
    # determine the separator based on the filename
    if filename.endswith('.csv'):
        sep = ','
    elif filename.endswith('.tsv'):
        sep = '\t'
    else:
        sep = ' '
        
    kwargs.setdefault('skipinitialspace', True)
    
    # reformat the data such that pandas can read it
    buf = six.StringIO()
    header, in_header = [], True
    with open(filename) as fp:
        for line in fp:
            if line.startswith('%'):
                # ignore comments, but keep the first block, which is the
                # header
                if in_header:
                    header.append(line[2:])  # remove comment character
            else:
                if header and in_header:
                    buf.write(header.pop())  # write column names from header
                    in_header = False
                buf.write(line)

    buf.flush()
    buf.seek(0)
    
    # read the data using pandas
    data = pd.read_csv(buf, sep=sep)
    
    if ret_header:
        return data, header
    else:
        return data
    
    
    
def write_comsol_parameters(path, data):
    """ function for writing parameters to a file that comsol can read
    
    `path` gives the path to the file where the data is stored
    `data` is a dictionary with the name of the parameter as the key and a list
        (potentially with units) of the parameter values
    """    
    
    # unit translation table
    UNITS = {'kilogram': 'kg',
             'millimeter': 'mm',
             'millimeter ** 2': 'mm^2',
             '1 / second': '1/s'}

    with open(path, 'w') as fp:
        # iterate through the given data
        for name, values in data.iteritems():
            values = homogenize_unit_array(values)
            try:
                values_s = ', '.join('%g' % v.magnitude for v in values)
            except AttributeError:
                values_s = ', '.join('%g' % v for v in values)
                s = '{name} {{{values}}}'.format(name=name, values=values_s)
            else:
                unit = str(values.units)
                unit = UNITS.get(unit, unit)
                s = '{name} {{{values}}} [{unit}]'.format(
                                        name=name, values=values_s, unit=unit)

            fp.write(s + '\n')
            


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
    
    
                
class PeristentObject(object):
    """ context manager for simple persistent storage """
    
    
    formats = {'yaml': {'default_flow_style': False,
                        'allow_classes': False}}
                            

    def __init__(self, filename, locking=None, factory=dict, db_format='yaml',
                 format_parameters=None):
        """ a context manager that opens a database file and yields its content.
        When the context manager is left, the data is written back on the disk.
        This is useful to modify simple configuration files or databases:
         
        with PeristentObject('config.yaml', db_format='yaml') as config:
            config['user'] = 'name'
         
        Now, the file `config.yaml` will read "user: name".
         
        `filename` gives the name of the database file
        `locking` determines whether the file will be locked the entire time so
            other processes cannot use it. This should be turned on by default
            to prevent data corruption and race conditions. The default is to
            turn it on when the module `portalocker` is installed, otherwise it
            is disabled.
        `factory` defines how the database should be initialized in case the
            file is not present or is empty.
        `db_format` is the file format that is used to read and write the data.
            So far, only `yaml` has been implemented.
        `format_parameters` is a dictionary with additional parameters that
            influence how the database is written to the file.
        """
    
        # save some options for using this decorator
        self._filename = filename
        self.logger = logging.getLogger(__name__) 
        self.factory = factory

        if locking is None:
            try:
                import portalocker
            except ImportError:
                warnings.warn('Locking is not supported since python module '
                              '`portalocker` is not available.')
                self._locking = False
            else:
                self._locking = True
        else:
            self._locking = locking

        
        self.format = db_format
        try:
            self.format_parameters = self.formats[self.format]
        except KeyError:
            raise ValueError('The format `%s` is not supported' % db_format)
        if format_parameters is not None:
            self.format_parameters.update(format_parameters)
            
        self._database_fh = None
        self._database = None


    def unlock_file(self):
        """ unlocks the potentially locked file. """
        if self._locking and os.path.isfile(self._filename):
            import portalocker
            portalocker.unlock(open(self._filename, 'r'))


    def __enter__(self):
        """ read the database or initialize it if it was not setup """
        # read the 
        if self._locking:
            # open database file with locking
            try:
                import portalocker
            except ImportError:
                self.logger.error('The `portalocker` module must be installed '
                                  'to support locking of files on all '
                                  'platforms.')
                raise
    
            self.logger.debug('Open and lock file `%s` to work with YAML '
                              'database', self._filename)
            try:
                self._database_fh = open(self._filename, 'r+')
            except IOError:
                # file did not exist => create it
                self._database_fh = open(self._filename, 'w+')
                
            # try locking the file exclusively and without blocking
            portalocker.lock(self._database_fh,
                             portalocker.LOCK_EX | portalocker.LOCK_NB)
            
        else:
            # open database file without locking
            self.logger.debug('Open file `%s` to read entire YAML database',
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
            # clear the database file
            self.logger.debug('Clear file `%s` to update YAML database',
                              self._filename)
            self._database_fh.seek(0)
            self._database_fh.truncate(0)
            
        else:
            # reopen the database file
            self.logger.debug('Open file `%s` to write entire YAML database',
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



class ConstValueDict(dict):
    """ represents a dictionary where values cannot be set to a new value when
    they have been set once. An `AssertionError` is raised if a value is
    attempted to be changed. """
    
    def __repr__(self):
        name = self.__class__.__name__
        content = super(ConstValueDict, self).__repr__()
        return '{name}({content})'.format(name=name, content=content)
    
    
    def __setitem__(self, key, value):
        if key in self and self[key] != value:
            raise AssertionError('Values for key `%s` are inconsistent '
                                 '(%s != %s).' % (key, self[key], value))
        else:
            super(ConstValueDict, self).__setitem__(key, value)
    
            
    def update(self, *args, **kwargs):
        for key, value in six.iteritems(dict(*args, **kwargs)):
            self[key] = value



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
    return PeristentObject(
        filename, locking=False, factory=dict, db_format='yaml',
        format_parameters={'allow_classes': allow_classes,
                           'default_flow_style': default_flow_style})

