/**
 * learning.spectral
 * Arun Chaganty (chaganty@stanford.edu)
 *
 */

package learning.spectral;

import learning.linalg.MatrixOps;
import learning.linalg.MatrixFactory;
import learning.linalg.RandomFactory;
import learning.linalg.Tensor;
import learning.linalg.SimpleTensor;
import learning.exceptions.RecoveryFailure;
import learning.exceptions.NumericalException;

import org.ejml.simple.SimpleMatrix;
import fig.basic.LogInfo;

public class MultiViewMixture {

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
  public SimpleMatrix attemptEigendecomposition( int k, SimpleMatrix U1T, SimpleMatrix U2, 
			SimpleMatrix U3, SimpleMatrix P12, Tensor P123, 
			SimpleMatrix Theta ) throws NumericalException {

    SimpleMatrix L = new SimpleMatrix( k, k );
    SimpleMatrix R = new SimpleMatrix( k, k );
    SimpleMatrix R_inv;

    // Project P123 onto the random orthogonal matrix
		Theta = U3.mult( Theta );
		
		SimpleMatrix theta = MatrixOps.col( Theta, 0 );
		SimpleMatrix P123T = P123.project(2, theta);
		assert( MatrixOps.rank( P123T ) >= k );

		// Compute B123 = (U1' P123 U2) (U1' P12 U2)^-1
		SimpleMatrix U1P12U2 = (U1T.mult(P12).mult(U2));
		SimpleMatrix U1P12U2_1 = U1P12U2.invert();
		SimpleMatrix B123 = U1T.mult( P123T )
				.mult( U2 ).mult( U1P12U2_1 );
		
		LogInfo.begin_track("simultaneously diagonalization");
    try { 
      // Try to compute the eigen vectors of B123
      SimpleMatrix[] LR = MatrixOps.eig( B123 );
			MatrixOps.setRow( L, 0, LR[0] );
      R = LR[1];
      R_inv = R.invert();
		} catch( RuntimeException e ) {
			LogInfo.error( e.getMessage() );
			LogInfo.end_track("simultaneously diagonalization");
			throw new NumericalException( e.getMessage() );
		} catch( NumericalException e ) {
			LogInfo.error( e.getMessage() );
			LogInfo.end_track("simultaneously diagonalization");
			throw e;
		}

    // Diagonalise the remaining matrices
		for( int i = 1; i<k; i++ )
		{
			SimpleMatrix theta_ = MatrixOps.col( Theta, i );
			SimpleMatrix B123_ = U1T.mult( P123.project(2, theta_ ) )
					.mult( U2 ).mult( U1P12U2_1 );
			MatrixOps.setRow( L, i, MatrixFactory.diag( R_inv.mult(B123_).mult(R) ) );
		}
		LogInfo.end_track("simultaneously diagonalization");
		
		return L;
  }

  /**
   * Recover M3 from the sample moments using Algorithm B
   */
  public SimpleMatrix algorithmB( int k, SimpleMatrix P12, SimpleMatrix P13, SimpleTensor P123 ) 
    throws RecoveryFailure {
		LogInfo.begin_track("algorithmB");
    // Get the U_i
		SimpleMatrix[] U1DU2 = MatrixOps.svdk(P12, k);
		SimpleMatrix U1T = U1DU2[0].transpose();
		SimpleMatrix U2 = U1DU2[2];
		SimpleMatrix[] U1DU3 = MatrixOps.svdk(P13, k);
		SimpleMatrix U3 = U1DU3[2];
		LogInfo.logsForce( "Subspace computation done." );

		// Try to project onto theta 
		for( int i = 0; i < 100; i++ )
		{
			try{
				LogInfo.logs( "Attempt %d", i );
				// Project onto an orthogonal basis set
				SimpleMatrix Theta = RandomFactory.orthogonal( k );
				SimpleMatrix L = attemptEigendecomposition(k, U1T, U2, U3, P12, P123, Theta);
				
				// Reconstruct
				LogInfo.end_track("algorithmB");
				return U3.mult( Theta.transpose().invert() ).mult( L );
			}
			catch( NumericalException e )
			{
				continue;
			}
		}
		
		LogInfo.end_track("algorithmB");
		throw new RecoveryFailure();
  }


  /** 
   * Use Algorithm B to recover the the means of the third view (M3)
	 * @param k - number of components
	 * @param X1 - Data from first view. Each data point is a row
	 * @param X2 - Data from second view. Each data point is a row
	 * @param X3 - Data from third view. Each data point is a row
	 * @return
	 * @throws RecoveryFailure 
   */
  public SimpleMatrix recoverM3( int k, SimpleMatrix X1, SimpleMatrix X2, SimpleMatrix X3 ) throws RecoveryFailure, NumericalException {

    // Compute the moments
    SimpleMatrix P12 = MatrixOps.Pairs( X1, X2 );
    SimpleMatrix P13 = MatrixOps.Pairs( X1, X2 );
    SimpleTensor P123 = MatrixOps.Triples( X1, X2, X3 );

    // Check input conditions
    if( MatrixOps.rank( P12 ) < k || MatrixOps.rank( P13 ) < k ) {
      throw new NumericalException( "Given input data is of insufficient rank" );
    }

    // Recover from moments
    SimpleMatrix M3 = algorithmB( k, P12, P13, P123 );

    return M3;
  }

}

