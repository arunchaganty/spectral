# Cython Pairs function
# ------------------------

from __future__ import division
import numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.double
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.double_t DTYPE_t
# "def" can type its arguments but not have a return type. The type of the
# arguments for a "def" function is checked at run-time when entering the
# function.

def Pairs(np.ndarray[DTYPE_t, ndim=2] x1, np.ndarray[DTYPE_t, ndim=2] x2):
    """Compute E[x1 \ctimes x2]"""

    assert x1.dtype == DTYPE and x2.dtype == DTYPE

    cdef unsigned int N = x1.shape[0]
    cdef unsigned int d = x1.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] pairs = np.zeros( (d,d), dtype=DTYPE )
    cdef unsigned int n, i, j

    # Compute one element of Pairs at a time
    for n in range( N ):
        for j in range( d ):
            for i in range( d ):
                pairs[i,j] += (x1[n,i] * x2[n,j] - pairs[i,j])/(n+1)
    return pairs

def Triples(np.ndarray[DTYPE_t, ndim=2] x1, np.ndarray[DTYPE_t, ndim=2]
        x2, np.ndarray[DTYPE_t, ndim=2] x3, np.ndarray[DTYPE_t,
            ndim=1] eta):
    """Compute E[x1 \ctimes x2 <eta, x3> ]"""
    assert x1.dtype == DTYPE and x2.dtype == DTYPE and x3.dtype == DTYPE and eta.dtype == DTYPE

    cdef unsigned int N = x1.shape[0]
    cdef unsigned int d = x1.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] triples = np.zeros( (d,d), dtype=DTYPE )
    cdef unsigned int n, i, j

    # Compute one element of Pairs at a time
    for n in range( N ):
        for j in range( d ):
            for i in range( d ):
                # Compute <x3, eta> - this seems faster because it is non-pythonic
                nm = 0
                for k in range( d ):
                    nm += x3[n,k] * eta[k]
                triples[i,j] += (x1[n,i] * x2[n,j] * nm - triples[i,j])/(n+1)
    return triples

