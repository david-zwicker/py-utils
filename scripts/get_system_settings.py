#!/usr/bin/env python

import os, sys

print("Host name:")
os.system( 'hostname' )


print("-------------------------------------------")
print("| PROCESSOR")
print("-------------------------------------------")

print("Byte order: %s" % sys.byteorder)
try:
    print("Float info: %s" % sys.float_info)
except AttributeError:
    print("Float info not available")
print("Platform: %s" % sys.platform)

print("-------------------------------------------")
print("| PATHS")
print("-------------------------------------------")

print("System Path ($PATH):")
os.system( 'echo $PATH' )
print("Python Path (sys.path):")
print(sys.path)

print("-------------------------------------------")
print("| VERSIONS")
print("-------------------------------------------")

print("Python Version:")
print(sys.version)
print(sys.version_info)

print("Numpy Version:")
try:
    import numpy
    print(numpy.version.version)
except ImportError:
    print("Not available")

print("Scipy Version:")
try:
    import scipy
    print(scipy.version.version)
except ImportError:
    print("Not available")

print("Matplotlib Version:")
try:
    import matplotlib
    print(matplotlib.__version__)
except ImportError:
    print("Not available")

print("OpenOpt Version:")
try:
    import openopt
    print(openopt.__version__)
except ImportError:
    print("Not available")


print("-------------------------------------------")
print("| ENVIRONMENT")
print("-------------------------------------------")
for k,v in os.environ.items():
    print(( "%s: %s" % (k,v) ))
