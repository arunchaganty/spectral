"""
A thin wrapper around HDF table stores to store a model.
The parameters are stored as is, as objects, while the data is compressed.
"""
import scipy as sc

# Store the model and its parameters in a HDF table
import tables

class Model:
    """Generic mixture model that contains a bunch of weighted means"""

    def __init__( self, store ):
        """Create a mixture model for components using given weights"""
        # Save the model in a HDF file 
        self.store = store

    def add_parameter( self, name, values ):
        """Add a parameter with values as a whole object. No compression"""
        self.store.createArray( "/params/", name, values )

    def get_parameter( self, name ):
        """Read the parameter value from the store"""
        try: 
            x = self.store.getNode( "/params/%s" %(name) )
            # Parameters are always small enough that they can be
            # trivially handled in memory
            return x.read()
        except tables.NoSuchNodeError:
            raise IndexError( "That parameter is not defined" ) 

    def _allocate_samples( self, name, shape ):
        """Allocate for (shape) samples"""
        # Allocate the number of samples (needs to be compressible)
        atom = tables.Float32Atom()
        arr = self.store.createCArray( "/data/", name, atom, shape )
        return arr

    def get_samples( self, name ):
        """Get samples from the store if they exist"""
        try: 
            # Get the memmapped version
            return self.store.getNode( "/data/%s" % name )
        except tables.NoSuchNodeError:
            raise IndexError( "No samples exist" ) 

    def save(self):
        """Flush to disk"""
        self.store.flush()

    def close(self):
        """Flush to disk"""
        self.store.close()

    @staticmethod
    def load_from_file( fname ):
        """Load model from a HDF file"""
        hdf = tables.openFile( fname, "r+" )
        return Model( hdf )

    @staticmethod
    def create( fname, complevel = 9, complib="blosc" ):
        """Create a new HDF file"""
        hdf = tables.openFile( fname, "w" )
        hdf.filters = tables.Filters( complevel, complib )
        hdf.createGroup( "/", "params" )
        hdf.createGroup( "/", "data" )
        return Model( hdf )

