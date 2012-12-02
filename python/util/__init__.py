"""
Some utilities
"""

import scipy as sc
from scipy import array
import time
import sys

from util.DataLogger import DataLogger
from util.ProgressBar import ProgressBar

def chunked_update( fn, start, step, stop, show_pbar = False ):
    """Run @fn with arguments @start to @stop in @step sized blocks."""
    if show_pbar:
        pbar = ProgressBar()

    iters = int( (stop - start)/step )
    if show_pbar:
        pbar.start( iters )
    for i in xrange( iters ):
        fn( start, start + step )
        start += step
        if show_pbar:
            pbar.update( i )
    if start < stop:
        fn( start, stop )
    if show_pbar:
        pbar.stop()

def slog( x ):
    """Safe log - preserve 0"""
    if type(x) == sc.ndarray:
        y = sc.zeros( x.shape )
        y[ x > 0 ] = sc.log( x[ x > 0 ] )
    else:
        y = 0.0 if x == 0 else sc.log(x)

    return y

#def chunked_update( out, offset, n, blocksize, fn, *params ):
#    """Run @fn to produce samples @blocksize at a time. This
#    function tries to balance efficient use of the numpy sampler,
#    while writing to a memmapped/HDF array to reduce memory overhead.
#
#    Assumes a named argument "size" for number of samples
#    
#    """
#
#    pbar = ProgressBar()
#
#    # Draw "blocksize" number of samples for 1 less than required blocks
#    blocks = int( sc.floor( float(n) / blocksize ) )
#
#    pbar.start( blocks )
#    for block in xrange( blocks ):
#        out[offset + block * blocksize : offset + (block+1)*blocksize ] = fn(
#                *params, size=blocksize ) 
#        pbar.update( block )
#
#    # Draw the remaining number of samples.
#    n_ = n - blocks * blocksize
#    if n_ > 0:
#        out[ offset + blocks * blocksize : offset + n ] = fn( *params, size = n_ )
#    pbar.stop()
#
#    return out
#
