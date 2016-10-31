'''
Created on Oct 31, 2016

@author: David Zwicker <dzwicker@seas.harvard.edu>
'''


import copy


class ParameterMixin(object):
    """ a mixin which manages a dictionary of parameters assigned to classes
    """
    

    parameters_default = {}


    def __init__(self, parameters=None, check_validity=True):
        """ initialize the object with optional parameters that overwrite the
        default behavior
        
        `parameters`      a dictionary of parameters overwriting the defaults
        `check_validity`  determines whether an error is raised if there are
                          keys in parameters that are not in the defaults
        """
        # initialize parameters with default ones from all parent classes
        self.parameters = {}
        for cls in reversed(self.__class__.__mro__):
            if hasattr(cls, 'parameters_default'):
                # we need to make a deep copy to copy nested dictionaries
                self.parameters.update(copy.deepcopy(cls.parameters_default))
                
        # update parameters with the supplied ones
        if parameters is not None:
            if check_validity and any(key not in self.parameters
                                      for key in parameters):
                for key in parameters:
                    if key not in self.parameters:
                        raise ValueError('Parameter `{}` was provided in '
                                         'instance specific parameters but is '
                                         'not defined for the class `{}`'
                                         .format(key, self.__class__.__name__))
            
            self.parameters.update(parameters)
