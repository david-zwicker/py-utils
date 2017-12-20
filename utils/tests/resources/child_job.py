#!/usr/bin/env python2

import time
import sys


# determine how many lines we write
if len(sys.argv) > 1:
    count = int(sys.argv[1])
else:
    count = 5


for i in range(1, count + 1):
    time.sleep(0.01)  # slow down a bit
    
    if i % 2:
        # write numbers to stdout on odd lines
        s = ''.join(chr(c + 49) for c in range(i))
        sys.stdout.write(s + '\n')
        
    else:
        # write letters to stderr on even lines
        s = ''.join(chr(c + 65) for c in range(i))
        sys.stderr.write(s + '\n')
