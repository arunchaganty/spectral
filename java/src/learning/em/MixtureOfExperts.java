/**
 * learning.model
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.em;


import learning.linalg.MatrixOps;
import learning.linalg.MatrixFactory;
import learning.linalg.RandomFactory;
import learning.models.*;

import learning.spectral.applications.SpectralExperts;
import org.ejml.simple.SimpleMatrix;
import org.ejml.data.DenseMatrix64F;

import org.javatuples.*;

import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.OptionsParser;
import fig.exec.Execution;

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

    public double[] weights;
    // Each beta is stored as a row
    public double[][] betas;
    public double sigma2;

    public Parameters( int K, double[] weights, double[][] betas, double sigma2 ) {
      this.K = K;
      this.weights = weights;
      this.betas = betas;
      this.sigma2 = sigma2;
    }

    /**
     * @param K
     * @param weights
     * @param betas - The regression coefficients in K x D form.
     * @param sigma2
     */
    public Parameters( int K, SimpleMatrix weights, SimpleMatrix betas, double sigma2 ) {
      this.K = K;
      this.weights = MatrixFactory.toVector( weights );
      this.betas = MatrixFactory.toArray( betas );
      this.sigma2 = sigma2;
    }

    /** 
     * Default values. 
     * */
    public Parameters( int K, int D ) {
      this.K = K;

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

      return new Parameters( K, weights, betas, sigma2 );
    }

    public boolean isValid() {
      boolean weightsInSimplex = MatrixOps.equal( MatrixOps.sum( weights ), 1.0 );
      boolean betasFinite = (MatrixOps.sum( betas ) != Double.POSITIVE_INFINITY) && (MatrixOps.sum( betas ) != Double.NEGATIVE_INFINITY);
      boolean sigma2Positive = (sigma2 >= 0.0);
      LogInfo.logsForce( "Valid: " + weightsInSimplex + " " + betasFinite + " " + sigma2Positive );
      if ( !weightsInSimplex ) return false;
      if ( !betasFinite ) return false;
      if ( !sigma2Positive  ) return false;

      return true;
    }
  }

  public int K = 2;

  protected MixtureOfExperts() {}

  public MixtureOfExperts( int K ) {
    this.K = K;
  }

  /**
   * Compute the "responsibilities"
   * \tau_{nj} \propto \pi_j N( y_n | \beta_j^T x_n, \sigma^2 ) 
   * \forall n, \sum_{k} \tau_{nk} = 1
   */
  public double[][] computeResponsibilities( double[] y, double[][] X, Parameters state ) {
    int N = X.length;
    int D = X[0].length;

    double[][] R = new double[N][K]; 

    for( int n = 0; n < N; n++ ) {
      for( int k = 0; k < K; k++ ) {
        // $tau_{nk} \propto 
        //    pi_k \exp( -0.5 (y_n - b_k' x_n)**2 / sigma**2 )
        double delta = (y[n] - MatrixOps.dot( state.betas[k], X[n] ));
        R[n][k] = Math.log( state.weights[k] ) - 0.5 * delta*delta / state.sigma2;
      }
      // Normalize
      double minR = MatrixOps.min( R[n] );
      MatrixOps.minus( R[n], minR );
      double Z = MatrixOps.logsumexp( R[n] );
      if( Z == Double.POSITIVE_INFINITY ) {
        int k = MatrixOps.argmax( R[n] );
        for( int k_ = 0; k_ < K; k_++ )
          R[n][k_] = (k == k_) ? 1.0 : 0.0;
      } else {
        MatrixOps.minus( R[n], Z );
        MatrixOps.exp( R[n] );
      }
      MatrixOps.normalize( R[n] );
      assert( MatrixOps.equal( MatrixOps.sum( R[n] ), 1.0 ) );
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
    int D = X[0].length;

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
    
    return new Parameters( K, weights, betas, sigma2 );
  }

  /**
   * Update parameter according to the following rules:
   *
   */
  public double computeLogLikelihood( double[] y, double[][] X, Parameters state, double[][] responsibilities ) {
    int N = X.length;
    int D = X[0].length;

    double lhood = 0.0;

    for( int n = 0; n < N; n++ ) {
      double term = 0.0;
      for( int k = 0; k < K; k++ ) {
        // pi_k \exp( -0.5 (y_n - b_k' x_n)**2 / sigma**2 )
        double delta = (y[n] - MatrixOps.dot( state.betas[k], X[n] ));
        term += state.weights[k] * Math.exp( - 0.5 * delta*delta / state.sigma2 );
      }
      assert( term <= 1.0 );
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

    assert( state.isValid() );

    for( int i = 0; i < iters; i++ ) {
      double[][] responsibilities = computeResponsibilities( y, X, state );
      state = updateParameters( y, X, state, responsibilities );

      double lhood = computeLogLikelihood( y, X, state, responsibilities );
      LogInfo.logs( "Iter %d, lhood = %f, d lhood = %f\n", i, lhood, lhood - old_lhood ); 

      // Stopping condition; difference in likelihood is less than
      // epsilion.
      if( Math.abs(lhood - old_lhood) < eps )
        break;
      Execution.putOutput( "betas" + i, (new SimpleMatrix( state.betas )).transpose() );

      old_lhood = lhood;
    }

    LogInfo.end_track( "MixtureOfExperts-em" );

    return state;
  }
  public Parameters run( double[] y, double[][] X ) {
    int D = X[0].length;
    Parameters state = Parameters.random(K, D);
    return run( y, X, state );
  }

  public Parameters run( SimpleMatrix y, SimpleMatrix X ) {
    return run( MatrixFactory.toVector( y ), MatrixFactory.toArray( X ) );
  }
  public Parameters run( SimpleMatrix y, SimpleMatrix X, Parameters initState ) {
    return run( MatrixFactory.toVector( y ), MatrixFactory.toArray( X ), initState );
  }

  @Option(gloss="Read data points from this file") 
  public String inputPath;

  @Option(gloss="Maximum iterations of EM") 
  public int iters = 100;

  @Option(gloss="Number of samples to run on (0 for all)")
  public double subsampleN = 0;
	@Option(gloss = "Remove Thirds")
	public boolean removeThirds = false;

  @Option(gloss="Difference between iterations before stopping") 
  public double eps = 1e-3;

  @Override
  public void run() {
    // Read data from a file
    try {
      Pair< Pair<SimpleMatrix, SimpleMatrix>, learning.models.MixtureOfExperts > data =
              learning.models.MixtureOfExperts.readFromFile( inputPath );
      SimpleMatrix y = data.getValue0().getValue0();
      SimpleMatrix X = data.getValue0().getValue1();
      learning.models.MixtureOfExperts model = data.getValue1();
      data = null;

      // Choose a subset of the data
      int N = X.numRows();
      if( removeThirds || subsampleN > N ) {
        // Possibly sample data with thirds removed
        model.removeThirds = removeThirds;
        y = null; X = null; data = null;
        Pair<SimpleMatrix, SimpleMatrix> yX = model.sample( (int) subsampleN );
        y = yX.getValue0();
        X = yX.getValue1();
      } else if( subsampleN > 0 ) {
        y = y.extractMatrix(0, SimpleMatrix.END, 0, (int) subsampleN);
        X = X.extractMatrix(0, (int) subsampleN, 0, SimpleMatrix.END);
      }
      N = X.numRows();

      SimpleMatrix betas = model.getBetas();

      // Recover paramaters
      this.K = model.getK();

      Parameters params = run( y, X );
      SimpleMatrix betas_ = (new SimpleMatrix( params.betas )).transpose();
      betas_ = MatrixOps.alignMatrix( betas_, betas, true );
      Execution.putOutput( "betas", betas );
      Execution.putOutput( "betas_", betas_ );

      double err = MatrixOps.diff(betas, betas_);
      Execution.putOutput( "betasErr", err );

      System.out.printf( "%.4f %.4f\n", err,  SpectralExperts.SpectralExpertsAnalysis.computeLoss(y, X, betas_) );
    } catch( ClassNotFoundException | IOException e ) {
      LogInfo.error( e.getMessage() );
      return;
    }
  }

  public static void main( String[] args ) {
    Execution.run( args, new MixtureOfExperts() );
  }
}


