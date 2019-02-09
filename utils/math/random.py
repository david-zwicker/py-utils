'''
Created on Aug 9, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''

from __future__ import division, absolute_import

import itertools
import random

import numpy as np
from six.moves import range

try:
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb



def log_uniform(v_min, v_max, size):
    """ returns random variables that a distributed uniformly in log space """
    log_min, log_max = np.log(v_min), np.log(v_max)
    res = np.random.uniform(log_min, log_max, size)
    return np.exp(res)




def _take_random_combinations_gen(data, r, num, repeat=False):
    """ a generator yielding `num` random combinations of length `r` of the 
    items in `data`. If `repeat` is False, none of the combinations is yielded
    twice. Note that the generator will be caught in a infinite loop if there
    are less then `num` possible combinations. """
    count, seen = 0, set()
    while True:
        # choose a combination
        s = tuple(sorted(random.sample(data, r)))
        # check whether it has been seen already
        if s in seen:
            continue
        # return the combination
        yield s
        # keep track of what combinations we have already seen
        if not repeat:
            seen.add(s)
        # check how many we have produced
        count += 1
        if count >= num:
            break
                
                
                
def take_combinations(iterable, r, num='all'):
    """ returns a generator yielding at most `num` random combinations of
    length `r` of the items in `iterable`. """
    if num == 'all':
        # yield all combinations
        return itertools.combinations(iterable, r)
    else:
        # check how many combinations there are
        data = list(iterable)
        num_combs = comb(len(data), r, exact=True)
        if num_combs <= num:
            # yield all combinations
            return itertools.combinations(data, r)
        elif num_combs <= 10 * num:
            # yield a chosen sample of the combinations
            choosen = set(random.sample(range(num_combs), num))
            gen = itertools.combinations(data, r)
            return (v for k, v in enumerate(gen) if k in choosen)
        else:
            # yield combinations at random
            return _take_random_combinations_gen(data, r, num)
        

        
def take_product(data, r, num='all'):
    """ returns a generator yielding at most `num` random instances from the
    product set of `r` times the `data` """ 
    if num == 'all':
        # yield all combinations
        return itertools.product(data, repeat=r)
    else:
        # check how many combinations there are
        num_items = len(data)**r
        if num_items <= num:
            # yield all combinations
            return itertools.product(data, repeat=r)
        else:
            # yield a chosen sample of the combinations
            choosen = set(random.sample(range(num_items), num))
            gen = itertools.product(data, repeat=r)
            return (v for k, v in enumerate(gen) if k in choosen)

