"""
Some utilities
"""

import scipy as sc
from scipy import array
import time
import sys

class ProgressBar:
    """Progress Bar"""

    def __init__(self, width = 40):
        """Initialise with some width"""
        self.width = width
        self.state = 0
        self.total = 0

    def start(self, total_iterations):
        """Set up a scaling factor for total iterations"""
        # setup toolbar
        self.total = total_iterations
        sys.stdout.write("[%s]" % (" " * self.width))
        sys.stdout.flush()
        # return to start of line, after '['
        sys.stdout.write("\b" * (self.width+1)) 

        self.state = 0

    def update(self, n):
        """Update the ticker"""
        state_ = (self.width * n)/self.total
        if state_ == self.state:
            pass
        elif self.state < self.width and state_ > self.state:
            # update the bar
            sys.stdout.write("-")
            sys.stdout.flush()
            self.state += 1

    def stop(self):
        """End the ticker"""
        sys.stdout.write("\n")
        self.state = 0
        self.total = 0

def chunked_update( out, n, blocksize, fn, *params ):
    """Run @fn to produce samples @blocksize at a time. This
    function tries to balance efficient use of the numpy sampler,
    while writing to a memmapped/HDF array to reduce memory overhead.

    Assumes a named argument "size" for number of samples
    
    """

    pbar = ProgressBar()
    pbar.start( n )

    # Draw "blocksize" number of samples for 1 less than required blocks
    blocks = int( sc.floor( float(n) / blocksize ) )
    for block in xrange( blocks ):
        out[block * blocksize : (block+1)*blocksize ] = fn(
                *params, size=blocksize ) 
        pbar.update( block ** blocksize )

    # Draw the remaining number of samples.
    out[ blocks * blocksize : ] = fn( *params, size = n - blocks
            * blocksize ) 
    pbar.stop()

    return out

