/**
 * learning.model
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.em;


import learning.linalg.MatrixOps;
import learning.linalg.MatrixFactory;
import learning.linalg.RandomFactory;

import org.ejml.simple.SimpleMatrix;
import org.ejml.data.DenseMatrix64F;

import fig.basic.*;
import fig.exec.*;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import java.util.Random;

/**
 * An E-M method for the mixture of experts model
 */
public class MixtureOfExperts implements Runnable {
  
  public static class Parameters {
    public int K;
    public int D;

    public double[] weights;
    // Each beta is stored as a row
    public double[][] betas;
    public double sigma2;

    public Parameters( int K, int D, double[] weights, double[][] betas, double sigma2 ) {
      this.K = K;
      this.D = D;
      this.weights = weights;
      this.betas = betas;
      this.sigma2 = sigma2;
    }

    public Parameters( int K, int D, SimpleMatrix weights, SimpleMatrix betas, double sigma2 ) {
      this.K = K;
      this.D = D;
      this.weights = MatrixFactory.toVector( weights );
      this.betas = MatrixFactory.toArray( betas );
      this.sigma2 = sigma2;
    }

    /** 
     * Default values. 
     * */
    public Parameters( int K, int D ) {
      this.K = K;
      this.D = D;

      weights = new double[K];
      for( int k = 0; k < K; k++ )
        weights[k] =  1.0/K + RandomFactory.randn(0.01); // Very small noise
      MatrixOps.normalize( weights );

      betas = new double[K][D];
      sigma2 = 1.0;
    }

    public static Parameters random( int K, int D ) {
      double[] weights = new double[K];
      for( int k = 0; k < K; k++ )
        weights[k] =  1.0/K;

      double[][] betas = new double[K][D];
      for( int k = 0; k < K; k++ )
        for( int d = 0; d < D; d++ )
          betas[k][d] = RandomFactory.randn(1.0);
      double sigma2 = 1.0;

      return new Parameters( K, D, weights, betas, sigma2 );
    }
  }

  @Option(gloss="Number of experts") 
  public int K = 2;
  @Option(gloss="Number of dimensions") 
  public int D = 3;

  protected MixtureOfExperts() {}

  public MixtureOfExperts( int K, int D ) {
    this.K = K;
    this.D = D;
  }

  /**
   * Compute the "responsibilities"
   * \tau_{nj} \propto \pi_j N( y_n | \beta_j^T x_n, \sigma^2 ) 
   * \forall n, \sum_{k} \tau_{nk} = 1
   */
  public double[][] computeResponsibilities( double[] y, double[][] X, Parameters state ) {
    int N = X.length;

    double[][] R = new double[N][K]; 

    for( int n = 0; n < N; n++ ) {
      for( int k = 0; k < K; k++ ) {
        // $tau_{nk} \propto 
        //    pi_k \exp( -0.5 (y_n - b_k' x_n)**2 / sigma**2 )
        double delta = (y[n] - MatrixOps.dot( state.betas[k], X[n] ));
        R[n][k] = state.weights[k] * Math.exp( - 0.5 * delta*delta / state.sigma2 );
      }
      // Normalize
      MatrixOps.normalize( R[n] );
    }

    return R;
  }

  /**
   * Update parameter according to the following rules:
   *
   *  \beta_j = \sum_{n} \inv{(\tau_{nj} x^{(n)} x^{(n)T})} \sum_{n=1}^{N} \tau_{nj} x^{(n)T} y^{(n)} \\
   *  \pi_j \propto \sum_{n=1}^{N} \tau_{nj} \\
   *  \sigma^2 = \frac{1}{N} \sum_{n=1}^{N} \sum_{j=1}^{K} \tau_{nj} (y^{(n)} - \beta_j^T x^{(n)})^2.
   */
  public Parameters updateParameters( double[] y, double[][] X, Parameters state, double[][] responsibilities ) {
    int N = X.length;

    double[] weights = new double[K];
    double[][] betas = new double[K][D];
    double sigma2 = 1.0;

    double[][] R = responsibilities;

    // Update pi's
    for( int k = 0; k < K; k++ )
      for( int n = 0; n < N; n++ )
        weights[k] += R[n][k];
    MatrixOps.normalize( weights );

    // Update betas
    for( int k = 0; k < K; k++ ) {
      double[][] X2 = new double[D][D];
      double[] Xy = new double[D];

      // Aggregate over data
      for( int n = 0; n < N; n++ ) {
        for( int d = 0; d < D; d++ ) {
          for( int d_ = 0; d_ < D; d_++ ) {
            X2[d][d_] += (R[n][k] / N) * (X[n][d] * X[n][d_]) ;
          }
          Xy[d] += (R[n][k] / N) * X[n][d] * y[n];
        }
      }

      // Invert 
      SimpleMatrix X2_ = new SimpleMatrix( X2 );
      SimpleMatrix Xy_ = MatrixFactory.fromVector( Xy ).transpose();
      SimpleMatrix B_ = X2_.invert().mult(Xy_);
      betas[k] = MatrixFactory.toVector( B_ );
    }

    // Update sigma2
    sigma2 = 0.0;
    for( int k = 0; k < K; k++ ) {
      for( int n = 0; n < N; n++ ) {
        double delta = (y[n] - MatrixOps.dot( state.betas[k], X[n] ));
        sigma2 += R[n][k]/N * delta*delta;
      }
    }
    
    return new Parameters( K, D, weights, betas, sigma2 );
  }

  /**
   * Update parameter according to the following rules:
   *
   */
  public double computeLogLikelihood( double[] y, double[][] X, Parameters state, double[][] responsibilities ) {
    int N = X.length;

    double lhood = 0.0;

    for( int n = 0; n < N; n++ ) {
      double term = 0.0;
      for( int k = 0; k < K; k++ ) {
        // pi_k \exp( -0.5 (y_n - b_k' x_n)**2 / sigma**2 )
        double delta = (y[n] - MatrixOps.dot( state.betas[k], X[n] ));
        term += state.weights[k] * Math.exp( - 0.5 * delta*delta / state.sigma2 );
      }
      assert( term < 1.0 );
      lhood += Math.log( term );
    }

    lhood += - N * 0.5 * Math.log( 2 * Math.PI * state.sigma2 );

    return lhood;
  }

  /**
   * Run EM 
   */
  public Parameters run( double[] y, double[][] X, Parameters state ) {
    LogInfo.begin_track( "MixtureOfExperts-em" );
    double old_lhood = Double.NEGATIVE_INFINITY;
    Parameters old_state = state;

    for( int i = 0; i < iters; i++ ) {
      double[][] responsibilities = computeResponsibilities( y, X, state );
      state = updateParameters( y, X, state, responsibilities );

      double lhood = computeLogLikelihood( y, X, state, responsibilities );
      LogInfo.logs( "Iter %d, lhood = %f, d lhood = %f\n", i, lhood, lhood - old_lhood ); 

      // Stopping condition; difference in likelihood is less than
      // epsilion.
      if( lhood - old_lhood < eps )
        break;

      old_lhood = lhood;
      old_state = state;
    }

    LogInfo.end_track( "MixtureOfExperts-em" );

    return state;
  }

  public Parameters run( double[] y, double[][] X ) {
    Parameters state = Parameters.random(K, D);
    return run( y, X, state );
  }

  public Parameters run( SimpleMatrix y, SimpleMatrix X ) {
    return run( MatrixFactory.toVector( y ), MatrixFactory.toArray( X ) );
  }

  @Option(gloss="Read data points from this file") 
  public String inputPath;

  @Option(gloss="Maximum iterations of EM") 
  public int iters = 100;

  @Option(gloss="Difference between iterations before stopping") 
  public double eps = 1e-3;

  public SimpleMatrix[] readFromFile( String filename ) throws IOException, ClassNotFoundException {
		ObjectInputStream in = new ObjectInputStream( new FileInputStream( filename ) ); 
    SimpleMatrix[] yX = (SimpleMatrix[]) in.readObject();
		in.close();

    return yX;
  }

  @Override
  public void run() {
    // Read data from a file
    try {
      SimpleMatrix[] yX = readFromFile( inputPath );
      SimpleMatrix y = yX[0];
      SimpleMatrix X = yX[1];

      Parameters params = run( y, X );
      MatrixOps.printVector( params.weights );
      MatrixOps.printArray( params.betas );
      System.out.println( params.sigma2 );

    } catch (IOException e) {
      LogInfo.error( e.getMessage() );
      return;
    } catch (ClassNotFoundException e) {
      LogInfo.error( e.getMessage() );
      return;
    }
  }

  public static void main( String[] args ) {
    Execution.run( args, new MixtureOfExperts() );
  }
}


