'''
Created on Nov 28, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesNSImpl



class XMLStreamWriter(object):
    """ class for writing an xml file iteratively """


    def __init__(self, out=None):
        """ initializes the writer with a stream to write to. If `out=None`, the
        output is writen to sys.stdout """
        self._generator = XMLGenerator(out, 'utf-8')


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
    
    
    def end_tag(self, name, namespace=None):
        """ and tag `name` """
        self._generator.endElementNS((namespace, name), name)
    
    
    def tag(self, name, attr=None, body=None, namespace=None):
        """ write tag `name` """
        self.start_tag(name, attr, body, namespace)
        self.end_tag(name, namespace)     
