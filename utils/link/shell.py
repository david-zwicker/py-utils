'''
Created on Aug 9, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division

import pipes



def shellquote(s):
    """ Quotes characters problematic for most shells """
    return pipes.quote(s)
