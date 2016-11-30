'''
Created on Aug 25, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

from six import StringIO

from .. import xml_tools



class TestXml(unittest.TestCase):


    _multiprocess_can_split_ = True  # let nose know that tests can run parallel


    def _assert_xml_write(self, intent, tag_name, attr=None, body=None):
        s = StringIO()
        writer = xml_tools.XMLStreamWriter(s, header="")
        writer.tag(tag_name, attr, body)
        self.assertEqual(s.getvalue(), intent)
        s.close()


    def test_xml(self):
        """ test the XML writer """
        self._assert_xml_write('<a></a>', 'a')
        self._assert_xml_write('<a b="b"></a>', 'a', {'b': "b"})
        self._assert_xml_write('<a b="1"></a>', 'a', {'b': 1})
        self._assert_xml_write('<a>c</a>', 'a', body="c")
        self._assert_xml_write('<a b="b">c</a>', 'a', {'b': "b"}, "c")
        
        
    def test_xml_fail(self):
        """ test error handling of the XML writer """
        def init_writer():
            s = StringIO()
            writer = xml_tools.XMLStreamWriter(s)
            writer.start_tag('a')
            return writer
        
        writer = init_writer()
        self.assertRaises(ValueError, lambda: writer.end_tag('b'))

        writer = init_writer()
        writer.end_tag('a')
        self.assertRaises(IndexError, lambda: writer.end_tag('b'))
        


if __name__ == "__main__":
    unittest.main()
