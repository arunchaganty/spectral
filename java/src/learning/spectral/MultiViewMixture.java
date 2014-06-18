/**
 * learning.spectral
 * Arun Chaganty (chaganty@stanford.edu)
 *
 */

package learning.spectral;

import learning.linalg.*;
import learning.exceptions.RecoveryFailure;
import learning.exceptions.NumericalException;

import org.javatuples.*;

import org.ejml.simple.SimpleMatrix;
import fig.basic.LogInfo;

public class MultiViewMixture {

  public Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> computeSubspace( int K, SimpleMatrix P12, SimpleMatrix P13 ) {
    // Get the U_i
    LogInfo.begin_track( "subspace-computation" );
    Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> U1DU2 = MatrixOps.svdk(P12, K);
    SimpleMatrix U1 = U1DU2.getValue0();
    SimpleMatrix U2 = U1DU2.getValue2();
    Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> U1DU3 = MatrixOps.svdk(P13, K);
    SimpleMatrix U3 = U1DU3.getValue2();
    LogInfo.end_track( "subspace-computation" );

    return new Triplet<>( U1, U2, U3 );
  } 


  public SimpleMatrix algorithmB( int K, 
      SimpleMatrix U1T, SimpleMatrix U2, SimpleMatrix U3, 
      SimpleMatrix U1P12U2_1, SimpleMatrix[] P123T, SimpleMatrix Theta ) 
    throws NumericalException {
    assert( MatrixOps.rank( P123T[0] ) >= K );

    // Compute B123 = (U1' P123 U2) (U1' P12 U2)^-1
    SimpleMatrix B123 = U1T.mult( P123T[0] )
      .mult( U2 ).mult( U1P12U2_1 );

    SimpleMatrix L = new SimpleMatrix( K, K );
    // Try to compute the eigen vectors of B123
    SimpleMatrix[] LR = MatrixOps.eig( B123 );
    MatrixOps.setRow( L, 0, LR[0] );
    SimpleMatrix R = LR[1];
    SimpleMatrix R_inv = R.invert();

    // Diagonalise the remaining matrices
    LogInfo.begin_track("simultaneously diagonalization");
    for( int j = 1; j < K; j++ )
    {
      SimpleMatrix B123_ = U1T.mult( P123T[j] ).mult( U2 ).mult( U1P12U2_1 );
      SimpleMatrix Li = R_inv.mult(B123_).mult(R);
      //LogInfo.logsForce("L" + j + " " + Li);
      MatrixOps.setRow( L, j, MatrixFactory.diag( Li ) );
    }
    LogInfo.end_track("simultaneously diagonalization");

    SimpleMatrix M3 = U3.mult( Theta.transpose().invert() ).mult( L );
    LogInfo.end_track("algorithmB");

    return M3;
  }

  public SimpleMatrix algorithmB( int K, SimpleMatrix P12, SimpleMatrix P13, SimpleMatrix[] P123T, SimpleMatrix Theta ) 
    throws NumericalException {
    LogInfo.begin_track("algorithmB");

    Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> subspace = computeSubspace( K, P12, P13 ); 
    SimpleMatrix U1T = subspace.getValue0().transpose();
    SimpleMatrix U2 = subspace.getValue1();
    SimpleMatrix U3 = subspace.getValue2();

    SimpleMatrix U1P12U2 = (U1T.mult(P12).mult(U2));
    SimpleMatrix U1P12U2_1 = U1P12U2.invert();

    return algorithmB( K, U1T, U2, U3, 
        U1P12U2_1, P123T, Theta );
  }

  /**
   * Recover M3 from the sample moments using Algorithm B
   */
  public SimpleMatrix algorithmB( int K, SimpleMatrix P12, SimpleMatrix P13, Tensor P123 ) 
    throws RecoveryFailure {
    LogInfo.begin_track("algorithmB");

    // Get the U_i
    Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> subspace = computeSubspace( K, P12, P13 ); 
    SimpleMatrix U1T = subspace.getValue0().transpose();
    SimpleMatrix U2 = subspace.getValue1();
    SimpleMatrix U3 = subspace.getValue2();

    SimpleMatrix U1P12U2 = (U1T.mult(P12).mult(U2));
    SimpleMatrix U1P12U2_1 = U1P12U2.invert();

    // Try to project onto theta 
    for( int i = 0; i < 100; i++ ) {
      try{
        LogInfo.logsForce( "Attempt %d", i );
        // Project onto an orthogonal basis set
        //SimpleMatrix Theta = MatrixFactory.eye( K );
        SimpleMatrix Theta = RandomFactory.orthogonal( K );
        // Project P123 onto the random orthogonal matrix
        SimpleMatrix Theta_ = U3.mult( Theta );

        SimpleMatrix[] P123T = new SimpleMatrix[K];
        for( int k = 0; k < K; k++ ) {
          P123T[k] = P123.project(2, MatrixOps.col( Theta_, k ) );
        }

        return algorithmB( K, U1T, U2, U3, 
            U1P12U2_1, P123T, Theta );
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
  public static Triplet<SimpleMatrix, SimpleMatrix, Tensor>
    computeExactMoments( SimpleMatrix weights, SimpleMatrix M1,
        SimpleMatrix M2, SimpleMatrix M3 ) {

    // Compute the moments
    SimpleMatrix P12 = M1.mult( MatrixFactory.diag( weights ) ).mult( M2.transpose() );
    SimpleMatrix P13 = M1.mult( MatrixFactory.diag( weights ) ).mult( M3.transpose() );
    FullTensor P123 = FullTensor.fromDecomposition( weights, M1, M2, M3 );

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
    Tensor P123 = MatrixOps.Triples( X1, X2, X3 );

    // Check input conditions
    if( MatrixOps.rank( P12 ) < k || MatrixOps.rank( P13 ) < k ) {
      throw new NumericalException( "Given input data is of insufficient rank" );
    }

    // Recover from moments
    SimpleMatrix M3 = algorithmB( k, P12, P13, P123 );

    return M3;
  }

}

