/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.spectral;

import learning.utils.MatrixFactory;
import learning.utils.SimpleTensor;

import org.ejml.simple.SimpleEVD;
import org.ejml.simple.SimpleMatrix;

/**
 * Implementation of the technique described in Anandkumar, 
 * Hsu, Kakade, "A Method of Moments for Mixture Models and
 * Hidden Markov Models" (2012).
 */
public class MultiViewMixture extends MomentMethod {
	
	protected class NumericalException extends Exception {
		/**
		 * 
		 */
		private static final long serialVersionUID = 471085729902573029L;
		
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
			SimpleMatrix U3, SimpleMatrix P12, SimpleTensor P123, 
			SimpleMatrix Theta ) throws NumericalException {
		
		SimpleMatrix L = MatrixFactory.zeros( k, k );
		SimpleMatrix R = MatrixFactory.zeros( k, k );
		
		Theta = U3.mult( Theta );
		
		SimpleMatrix theta = MatrixFactory.col( Theta, 0 );
		
		// B123 = (U1' P123 U2) (U1' P12 U2)^-1
		SimpleMatrix U1P12U2_1 = U1T.mult(P12).mult(U2).invert();
		SimpleMatrix P123T = P123.project(2, theta );
		SimpleMatrix B123 = U1T.mult( P123T )
				.mult( U2 ).mult( U1P12U2_1 );
		
		@SuppressWarnings("unchecked")
		SimpleEVD<SimpleMatrix> EVD = B123.eig();
		
		// Simultaneously diagonalize
		for( int i = 0; i<k; i++ )
		{
			if( !EVD.getEigenvalue(i).isReal() )
				throw new NumericalException();
			L.set( 0, i, EVD.getEigenvalue(i).real );
			MatrixFactory.setRow( R, i, EVD.getEigenVector(i));
		}
		SimpleMatrix R_ = R.invert();
		
		// Simultaneously diagonalize all the other matrices
		for( int i = 1; i<k; i++ )
		{
			SimpleMatrix theta_ = MatrixFactory.col( Theta, i );
			SimpleMatrix B123_ = U1T.mult( P123.project(2, theta_ ) )
					.mult( U2 ).mult( U1P12U2_1 );
			MatrixFactory.setRow( L, i, MatrixFactory.diag( R_.mult(B123_).mult(R) ) );
		}
		
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
	 */
	protected SimpleMatrix recoverM3( int k, SimpleMatrix P12, SimpleMatrix P13, SimpleTensor P123 ) {
		int d = P12.numCols();
		
		// Get U1, U2, U3
		SimpleMatrix[] U1DU2 = MatrixFactory.svdk(P12, k);
		SimpleMatrix U1T = U1DU2[0].transpose();
		SimpleMatrix U2 = U1DU2[2];
		SimpleMatrix[] U1DU3 = MatrixFactory.svdk(P13, k);
		SimpleMatrix U3 = U1DU3[2];
		
		// Try to project onto theta 
		for( int i = 0; i < 100; i++ )
		{
			try{
				// Project onto an orthogonal basis set
				SimpleMatrix Theta = MatrixFactory.randomOrthogonal( k );
				SimpleMatrix L = attemptRecovery(k, U1T, U2, U3, P12, P123, Theta);
				
				// Reconstruct
				return U3.mult( Theta.transpose().invert() ).mult( L );
			}
			catch( NumericalException e )
			{
				continue;
			}
		}
			
		return null;
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
	 */
	public SimpleMatrix exactRecovery( int k, SimpleMatrix w, SimpleMatrix M1, SimpleMatrix M2, SimpleMatrix M3 ) {
		assert( M1.svd().rank() >= k );
		assert( M2.svd().rank() >= k );
		assert( M3.svd().rank() >= k );
		
		int d = M1.numRows();
		
		SimpleMatrix P12 = M1.mult( MatrixFactory.diag( w ) ).mult( M2.transpose() );
		SimpleMatrix P13 = M1.mult( MatrixFactory.diag( w ) ).mult( M3.transpose() );
		SimpleMatrix Z = MatrixFactory.vectorStack(d, w);
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
	 */
	public SimpleMatrix sampleRecovery( int k, SimpleMatrix X1, SimpleMatrix X2, SimpleMatrix X3 ) {
		
		SimpleMatrix P12 = MatrixFactory.Pairs(X1, X2);
		SimpleMatrix P13 = MatrixFactory.Pairs(X1, X3);
		SimpleTensor P123 = new SimpleTensor( X1, X2, X3 );
		
		SimpleMatrix M3_ = recoverM3( k, P12, P13, P123 );
		
		return M3_;
	}

}