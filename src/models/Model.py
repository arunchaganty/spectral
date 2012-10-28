"""
The parameters are stored as is in an npz file, and the data is stored
in a memmappable npy file, as objects, while the data is compressed.
"""
import scipy as sc
import os

class Model:
    """Generic mixture model that contains a bunch of weighted means"""

    def __init__( self, prefix, **params ):
        """Create a mixture model for components using given weights"""
        self.prefix = prefix
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
        # Save samples in a mem-mapped array, prefix
        # save in the metadata "params"
        self.params[ "data" ].append( name )
        arr = sc.memmap( "%s_%s.npy" % ( self.prefix, name ), mode="w+", dtype=sc.float32, shape=shape )
        return arr

    def get_samples( self, name, d):
        """Get samples from the store if they exist"""
        arr = sc.memmap( "%s_%s.npy" % ( self.prefix, name ), mode="r+", dtype=sc.float32 )
        N = len(arr)/d
        arr = arr.reshape( (N,d ) )
        return arr

    def save(self):
        """Flush to disk"""
        sc.savez( self.prefix, **self.params )

    def delete(self):
        """Flush to disk"""
        for name in self.params["data"]:
            path = "%s_%s.npy" %( self.prefix, name )
            if os.path.exists( path ):
                os.remove( path )
        path = "%s.npz" %( self.prefix )
        if os.path.exists( path ):
            os.remove( path )

    @staticmethod
    def from_file( prefix ):
        """Load model from a HDF file"""

        fname = "%s.npz" % prefix 

        params = dict( sc.load( fname ).items() )
        return Model( prefix, **params )

