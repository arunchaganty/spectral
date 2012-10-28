"""Data Logger"""

import scipy as sc
import scipy.linalg

from scipy.linalg import norm, svdvals
from spectral.linalg import condition_number, eigen_sep, \
        column_aerr, column_rerr

import time

class DataLogger:
    def __init__(self, prefix=None):
        if prefix is None:
            prefix = time.strftime("%Y%m%d+%H:%M:%S")
        self.fname = "%s.log" % prefix  
        self.store = {}

    def add( self, key, value ):
        """Add an entry with key, value"""
        self.store[key] = value
        # Save each time to prevent data loss 
        self.save()

    def save( self ):
        """Save to file"""
        sc.savez( self.fname, **self.store )

    def add_err( self, key, A, A_, ntype=None ):
        """Print the error between two objects"""

        if ntype is None:
            self.add( "aerr_%s" % key, norm( A - A_ ) )
            self.add( "rerr_%s" % key, norm( A - A_ )/norm(A) )
        elif ntype == "col":
            self.add( "aerr_%s_col" % key, column_aerr( A, A_ ) )
            self.add( "rerr_%s_col" % key,  column_rerr( A, A_ ) )
        else:
            self.add( "aerr_%s_%d" % (key, ntype), norm( A - A_ ) )
            self.add( "rerr_%s_%d" % (key, ntype), norm( A - A_, ntype )/norm(A, ntype) )

    def add_consts( self, key, A, k=-1, ntype=None ):
        """Print the error between two objects"""

        if ntype is None:
            self.add( "norm_%s" % key, norm( A ) )
        else:
            self.add( "norm_%s_%d" % (key, ntype), norm( A, ntype ) )

        if ntype == 2:
            if k > 0:
                self.add( "s_k_%s" % key, svdvals(A)[k-1]  )
            else:
                self.add( "s_k_%s" % key, svdvals(A)[-1]  )
            self.add( "K_%s" % key, condition_number( A, k ) )
            if A.shape[0] == A.shape[1]:
                self.add( "D_%s" % key, eigen_sep( A, k ) )

