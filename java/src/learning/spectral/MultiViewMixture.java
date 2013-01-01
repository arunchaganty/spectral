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
import learning.linalg.ExactTensor;
import learning.exceptions.RecoveryFailure;
import learning.exceptions.NumericalException;

import org.javatuples.*;

import org.ejml.simple.SimpleMatrix;
import fig.basic.LogInfo;

public class MultiViewMixture {

  /**
   * Recover M3 from the sample moments using Algorithm B
   */
  public SimpleMatrix algorithmB( int k, SimpleMatrix P12, SimpleMatrix P13, Tensor P123 ) 
    throws RecoveryFailure {
    LogInfo.begin_track("algorithmB");
    //LogInfo.logsForce("P12: " + P12);
    //LogInfo.logsForce("P13: " + P13);

    // Get the U_i
    SimpleMatrix[] U1DU2 = MatrixOps.svdk(P12, k);
    SimpleMatrix U1T = U1DU2[0].transpose();
    SimpleMatrix U2 = U1DU2[2];
    SimpleMatrix[] U1DU3 = MatrixOps.svdk(P13, k);
    SimpleMatrix U3 = U1DU3[2];

    //LogInfo.logsForce("U1: " + U1T.transpose());
    //LogInfo.logsForce("U2: " + U2);
    //LogInfo.logsForce("U3: " + U3);

    LogInfo.logsForce( "Subspace computation done." );

    SimpleMatrix U1P12U2 = (U1T.mult(P12).mult(U2));
    SimpleMatrix U1P12U2_1 = U1P12U2.invert();

    SimpleMatrix L = new SimpleMatrix( k, k );
    SimpleMatrix R = null;
    SimpleMatrix R_inv = null;

    // Try to project onto theta 
    for( int i = 0; i < 100; i++ )
    {
      try{
        LogInfo.logs( "Attempt %d", i );
        // Project onto an orthogonal basis set
        //SimpleMatrix Theta = MatrixFactory.eye( k );
        SimpleMatrix Theta = RandomFactory.orthogonal( k );
        //LogInfo.logsForce("Theta: " + Theta);
        // Project P123 onto the random orthogonal matrix
        SimpleMatrix Theta_ = U3.mult( Theta );
        SimpleMatrix theta = MatrixOps.col( Theta_, 0 );
        SimpleMatrix P123T = P123.project(2, theta);
        //LogInfo.logsForce("P123T: " + P123T);
        assert( MatrixOps.rank( P123T ) >= k );

        // Compute B123 = (U1' P123 U2) (U1' P12 U2)^-1
        SimpleMatrix B123 = U1T.mult( P123T )
          .mult( U2 ).mult( U1P12U2_1 );
        //LogInfo.logsForce("B123: " + B123);


        // Try to compute the eigen vectors of B123
        SimpleMatrix[] LR = MatrixOps.eig( B123 );
        MatrixOps.setRow( L, 0, LR[0] );
        R = LR[1];
        R_inv = R.invert();

        // Diagonalise the remaining matrices
        LogInfo.begin_track("simultaneously diagonalization");
        for( int j = 1; j<k; j++ )
        {
          SimpleMatrix theta_ = MatrixOps.col( Theta_, j );
          SimpleMatrix B123_ = U1T.mult( P123.project(2, theta_ ) )
            .mult( U2 ).mult( U1P12U2_1 );
          SimpleMatrix Li = R_inv.mult(B123_).mult(R);
          //LogInfo.logsForce("L" + j + " " + Li);
          MatrixOps.setRow( L, j, MatrixFactory.diag( Li  ) );
        }
        LogInfo.end_track("simultaneously diagonalization");


        // Reconstruct
        LogInfo.end_track("algorithmB");

        // We've been storing everything in it's transposed form.
        SimpleMatrix M3T = U3.mult( Theta.transpose().invert() ).mult( L );
        return M3T.transpose();
      } catch( NumericalException e ) {
        LogInfo.error( e.getMessage() );
        continue;
      }
    }

    LogInfo.end_track("algorithmB");
    throw new RecoveryFailure();
  }

  /**
   * Computes the exact moments of the model whose means are given by
   * the columns of M[v].
   */
  public Triplet<SimpleMatrix, SimpleMatrix, Tensor> computeExactMoments( int D, int K, int V, SimpleMatrix weights, SimpleMatrix[] M ) {
    SimpleMatrix M1 = M[0].transpose();
    SimpleMatrix M2 = M[1].transpose();
    SimpleMatrix M3 = M[2].transpose();

    // Compute the moments
    SimpleMatrix P12 = M1.mult( MatrixFactory.diag( weights ) ).mult( M2.transpose() );
    SimpleMatrix P13 = M1.mult( MatrixFactory.diag( weights ) ).mult( M3.transpose() );
    ExactTensor P123 = new ExactTensor( weights, M1, M2, M3 );

    return new Triplet<SimpleMatrix, SimpleMatrix, Tensor>( P12, P13, P123 );
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

