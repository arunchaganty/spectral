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
DTYPE = np.float32
LONG = np.long
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float32_t DTYPE_t

ctypedef np.long_t LONG_t
# "def" can type its arguments but not have a return type. The type of the
# arguments for a "def" function is checked at run-time when entering the
# function.

def count_frequency(np.ndarray[LONG_t, ndim=2] X, unsigned int d):
    cdef unsigned int N = X.shape[0]
    cdef unsigned int W = X.shape[1]

    cdef np.ndarray[LONG_t, ndim=2] Y = np.zeros( (N,d), dtype=LONG )

    for n in range( N ):
        for w in range( W ):
            i = X[n,w]
            Y[n,i] += 1

    return Y

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
        x2, np.ndarray[DTYPE_t, ndim=2] x3):
    """Compute E[x1 \ctimes x2 \ctimes x3 ]"""
    assert x1.dtype == DTYPE and x2.dtype == DTYPE and x3.dtype == DTYPE

    cdef unsigned int N = x1.shape[0]
    cdef unsigned int d = x1.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=3] triples = np.zeros( (d,d,d), dtype=DTYPE )
    cdef unsigned int n, i, j, k

    # Compute one element of Triples at a time
    for n in range( N ):
        for k in range( d ):
            for j in range( d ):
                for i in range( d ):
                    triples[i,j,k] += (x1[n,i] * x2[n,j] * x3[n,k] - triples[i,j,k])/(n+1)
    return triples


def apply_shuffle( np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[LONG_t, ndim=1] perm ):
    assert X.dtype == DTYPE 

    cdef unsigned int N = X.shape[0]
    cdef unsigned int d = X.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=1] buf = np.zeros( (d,), dtype=DTYPE )
    for i in range( N-1 ):
        j = perm[i]
        buf[:] = X[i,:]
        X[i,:] = X[i+j,:]
        X[i+j,:] = buf

    return X

