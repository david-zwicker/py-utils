'''
Created on Oct 31, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import unittest

from ..parameter_mixin import ParameterMixin 



class TestParameterMixin(unittest.TestCase):


    _multiprocess_can_split_ = True  # let nose know that tests can run parallel


    def test_trivial(self):
        """ test the class directly """
        class Trivial(ParameterMixin):
            pass
        
        p = {'a': 1}
        self.assertEqual(Trivial().parameters, {})
        self.assertRaises(ValueError, lambda: Trivial(p))
        self.assertEqual(Trivial(p, False).parameters, p)
        

    def test_simple(self):
        """ test the simple case """
        class Simple(ParameterMixin):
            parameters_default = {'a': 1}
            
        
        self.assertEqual(Simple().parameters, {'a': 1})
        self.assertEqual(Simple({'a': 2}).parameters, {'a': 2})
        self.assertRaises(ValueError, lambda: Simple({'b': 2}))
        self.assertEqual(Simple({'b': 2}, False).parameters, {'a': 1, 'b': 2})


    def test_inheritence(self):
        """ test the simple case """
        class Parent(ParameterMixin):
            parameters_default = {'a': 1}
            
        class Child(Parent):
            parameters_default = {'b': 2}
            
        
        self.assertEqual(Child().parameters, {'a': 1, 'b': 2})
        self.assertEqual(Child({'a': 3}).parameters, {'a': 3, 'b': 2})
        self.assertEqual(Child({'b': 3}).parameters, {'a': 1, 'b': 3})
        self.assertRaises(ValueError, lambda: Child({'c': 3}))
        self.assertEqual(Child({'c': 3}, False).parameters,
                         {'a': 1, 'b': 2, 'c': 3})



if __name__ == "__main__":
    unittest.main()

