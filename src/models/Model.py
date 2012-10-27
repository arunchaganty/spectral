"""
The parameters are stored as is in an npz file, and the data is stored
in a memmappable npy file, as objects, while the data is compressed.
"""
import scipy as sc
import os

class Model:
    """Generic mixture model that contains a bunch of weighted means"""

    def __init__( self, fname, **params ):
        """Create a mixture model for components using given weights"""
        self.fname = fname
        self.params = params
        if "data" not in self.params:
            self.params["data"] = []

    def add_parameter( self, name, values ):
        """Add a parameter with values as a whole object. No compression"""
        self.params[ name ] = values

    def get_parameter( self, name ):
        """Read the parameter value from the store"""
        return self.params[ name ]

    def _allocate_samples( self, name, shape ):
        """Allocate for (shape) samples"""
        # Save samples in a mem-mapped array, fname_name
        # save in the metadata "params"
        self.params[ "data" ].append( name )
        arr = sc.memmap( "%s_%s.npy" % ( self.fname, name ), mode="w+", dtype=sc.float32, shape=shape )
        return arr

    def get_samples( self, name ):
        """Get samples from the store if they exist"""
        arr = sc.memmap( "%s_%s.npy" % ( self.fname, name ), mode="r+", dtype=sc.float32, shape=shape )
        return arr

    def save(self):
        """Flush to disk"""
        sc.savez( self.fname, **self.params )

    def delete(self):
        """Flush to disk"""
        for name in self.params["data"]:
            os.remove( "%s_%s.npy" %( self.fname, name ) )
        os.remove( "%s.npz" % self.fname )

    @staticmethod
    def from_file( fname ):
        """Load model from a HDF file"""

        params = dict( sc.load( fname ).items() )
        return Model( **params )

