'''
Created on Nov 28, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import sys

from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesNSImpl



class XMLStreamWriter(object):
    """ class for writing an xml file iteratively """


    def __init__(self, file_handle=None, header='<?xml version="1.0" ?>'):
        """ initializes the writer with a stream to write to. If
        `filehandle=None`, the output is written to sys.stdout """
        if file_handle is None:
            file_handle = sys.stdout
        self.file_handle = file_handle
        
        if header:
            self.file_handle.write(header)
            
        self._generator = XMLGenerator(file_handle, 'utf-8')
        self._tags = []


    def start_tag(self, name, attr=None, body=None, namespace=None):
        """ start tag `name` with attributes `attr` and body `body` """
        attr_vals = {}
        attr_keys = {}
        if attr is not None:
            for key, val in attr.items():
                key_tuple = (namespace, key)
                attr_vals[key_tuple] = str(val)
                attr_keys[key_tuple] = key
    
        attr_obj = AttributesNSImpl(attr_vals, attr_keys)
        self._generator.startElementNS((namespace, name), name, attr_obj)
        
        if body:
            self._generator.characters(str(body))
            
        self._tags.append(name)
    
    
    def end_tag(self, name=None, namespace=None):
        """ and tag `name` """
        close_tag = self._tags.pop()
        
        if name is not None:
            if name != close_tag:
                raise ValueError('Cannot close tag `%s`, since the last opened '
                                 'tag was `%s`' % (name, close_tag))
                
        
        self._generator.endElementNS((namespace, name), name)
    
    
    def tag(self, name, attr=None, body=None, namespace=None):
        """ write tag `name` """
        self.start_tag(name, attr, body, namespace)
        self.end_tag(name, namespace)     
