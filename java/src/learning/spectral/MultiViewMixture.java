/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.spectral;

import learning.utils.MatrixOps;
import learning.utils.SimpleMatrixFactory;
import learning.utils.RandomFactory;
import learning.utils.SimpleMatrixOps;
import learning.utils.SimpleTensor;
import learning.utils.Tensor;

import org.ejml.simple.SimpleEVD;
import org.ejml.simple.SimpleMatrix;

import fig.basic.LogInfo;

/**
 * Implementation of the technique described in Anandkumar, 
 * Hsu, Kakade, "A Method of Moments for Mixture Models and
 * Hidden Markov Models" (2012).
 */
public class MultiViewMixture extends MomentMethod {
	public class RecoveryFailure extends Exception {
		/**
		 * 
		 */
		private static final long serialVersionUID = -2899249561379935402L;
		/**
		 * 
		 */
		public RecoveryFailure() {
			super();
		}
		public RecoveryFailure(String message) {
			super(message);
		}
		
	}
	
	protected class NumericalException extends Exception {
		/**
		 * 
		 */
		private static final long serialVersionUID = 471085729902573029L;
		public NumericalException() {
			super();
		}
		public NumericalException(String message) {
			super(message);
		}
		
	}
	
	/**
	 * Attempt to recover the eigen value matrix for a particular Theta
	 * @param U1
	 * @param U2
	 * @param U3
	 * @param P12
	 * @param P123
	 * @param Theta
	 * @return
	 * @throws NumericalException 
	 */
	private SimpleMatrix attemptRecovery( int k, SimpleMatrix U1T, SimpleMatrix U2, 
			SimpleMatrix U3, SimpleMatrix P12, Tensor P123, 
			SimpleMatrix Theta ) throws NumericalException {
		
		SimpleMatrix L = SimpleMatrixFactory.zeros( k, k );
		SimpleMatrix R = SimpleMatrixFactory.zeros( k, k );
		SimpleMatrix R_;
		
		Theta = U3.mult( Theta );
		
		SimpleMatrix theta = SimpleMatrixOps.col( Theta, 0 );
		SimpleMatrix P123T = P123.project(2, theta);
		assert( P123T.svd().rank() >= k );
		
		// B123 = (U1' P123 U2) (U1' P12 U2)^-1
		SimpleMatrix U1P12U2 = (U1T.mult(P12).mult(U2));
		SimpleMatrix U1P12U2_1 = U1P12U2.invert();
		SimpleMatrix B123 = U1T.mult( P123T )
				.mult( U2 ).mult( U1P12U2_1 );
		
		LogInfo.begin_track("simultaneously diagonalization");
		try {
			@SuppressWarnings("unchecked")
			SimpleEVD<SimpleMatrix> EVD = B123.eig();
			// Get the eigenvector matrix
			for( int i = 0; i<k; i++ )
			{
				if( !EVD.getEigenvalue(i).isReal() )
				{
					LogInfo.error("Non-real eigen value at index " + i);
					LogInfo.end_track("simultaneously diagonalization");
					throw new NumericalException();
				}
				L.set( 0, i, EVD.getEigenvalue(i).real );
				SimpleMatrixOps.setRow( R, i, EVD.getEigenVector(i));
			}
			R_ = R.invert();
		} catch( RuntimeException e ) {
			LogInfo.end_track("simultaneously diagonalization");
			throw new NumericalException( e.getMessage() );
		}
		
		// Simultaneously diagonalize all the other matrices
		for( int i = 1; i<k; i++ )
		{
			SimpleMatrix theta_ = SimpleMatrixOps.col( Theta, i );
			SimpleMatrix B123_ = U1T.mult( P123.project(2, theta_ ) )
					.mult( U2 ).mult( U1P12U2_1 );
			SimpleMatrixOps.setRow( L, i, SimpleMatrixFactory.diag( R_.mult(B123_).mult(R) ) );
		}
		LogInfo.end_track("simultaneously diagonalization");
		
		return L;
	}
		
	
	/**
	 * Recover component means up to a permutation from 
	 * three independent views of the data, X1, X2, X3.
	 * @param k - Number of components
	 * @param P12 - Second moment of X1 and X2
	 * @param P13 - Second moment of X1 and X3
	 * @param P123 - Third moment of X1, X2 and X3
	 * @return - Component means (M3)
	 * @throws RecoveryFailure 
	 */
	public SimpleMatrix recoverM3( int k, SimpleMatrix P12, SimpleMatrix P13, Tensor P123 ) throws RecoveryFailure {
		LogInfo.begin_track("spectral");
		// Get U1, U2, U3
		SimpleMatrix[] U1DU2 = SimpleMatrixOps.svdk(P12, k);
		SimpleMatrix U1T = U1DU2[0].transpose();
		SimpleMatrix U2 = U1DU2[2];
		SimpleMatrix[] U1DU3 = SimpleMatrixOps.svdk(P13, k);
		SimpleMatrix U3 = U1DU3[2];
		
		LogInfo.logsForce( "Subspace computation done." );
		
		// Try to project onto theta 
		for( int i = 0; i < 100; i++ )
		{
			try{
				LogInfo.logs( "Attempt %d", i );
				// Project onto an orthogonal basis set
				SimpleMatrix Theta = RandomFactory.orthogonal( k );
				SimpleMatrix L = attemptRecovery(k, U1T, U2, U3, P12, P123, Theta);
				
				// Reconstruct
				LogInfo.end_track("spectral");
				return U3.mult( Theta.transpose().invert() ).mult( L );
			}
			catch( NumericalException e )
			{
				continue;
			}
		}
		
		LogInfo.end_track("spectral");
		throw new RecoveryFailure();
	}
	
	/**
	 * Generate the exact moments of the data and extract M3 
	 * by running the algorithm
	 * @param k - number of clusters
	 * @param w - weights
	 * @param M1 - mean of first view
	 * @param M2
	 * @param M3
	 * @return
	 * @throws RecoveryFailure 
	 */
	public SimpleMatrix exactRecovery( int k, SimpleMatrix w, SimpleMatrix M1, SimpleMatrix M2, SimpleMatrix M3 ) throws RecoveryFailure {
		assert( M1.svd().rank() >= k );
		assert( M2.svd().rank() >= k );
		assert( M3.svd().rank() >= k );
		
		int d = M1.numRows();
		
		SimpleMatrix P12 = M1.mult( SimpleMatrixFactory.diag( w ) ).mult( M2.transpose() );
		SimpleMatrix P13 = M1.mult( SimpleMatrixFactory.diag( w ) ).mult( M3.transpose() );
		SimpleMatrix Z = SimpleMatrixFactory.vectorStack(d, w);
		Z = Z.scale( k ); // Scale by k so that the averaging doesn't hurt.
		
		SimpleTensor P123 = new SimpleTensor( M1.transpose(), M2.transpose(), M3.elementMult( Z ).transpose() );
		
		SimpleMatrix M3_ = recoverM3( k, P12, P13, P123 );
		
		return M3_;
	}
	
	/**
	 * Compute the moments from observed data X and try to recover M3
	 * @param k - number of components
	 * @param X1 - Data from first view. Each data point is a row
	 * @param X2
	 * @param X3
	 * @return
	 * @throws RecoveryFailure 
	 */
	public SimpleMatrix sampleRecovery( int k, double[][] X1, double[][] X2, double[][] X3 ) throws RecoveryFailure {
		
		SimpleMatrix P12 = new SimpleMatrix( MatrixOps.Pairs(X1, X2) );
		SimpleMatrix P13 = new SimpleMatrix( MatrixOps.Pairs(X1, X3) );
		assert( P12.svd().rank() >= k );
		assert( P13.svd().rank() >= k );
		SimpleTensor P123 = new SimpleTensor( X1, X2, X3 );
		
		SimpleMatrix M3_ = recoverM3( k, P12, P13, P123 );
		
		return M3_;
	}

}
