/**
 * learning.model
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.models;

import learning.Misc;
import learning.linalg.*;

import org.ejml.simple.SimpleMatrix;
import org.ejml.data.DenseMatrix64F;

import fig.basic.*;
import fig.prob.MultGaussian;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.IOException;

import java.util.Random;

import org.javatuples.*;

/**
 * A mixture of experts model
 */
public class MixtureOfGaussians {

  protected int K; // Number of components
  protected int D; // Dimensionality of space
  protected int V; // Number of views

  // Model parameters
  protected SimpleMatrix weights;
  protected SimpleMatrix[] means;
  protected SimpleMatrix[][] covs;

  Random rnd = new Random();

  public MixtureOfGaussians( int K, int D, int V, SimpleMatrix weights, SimpleMatrix[] means, SimpleMatrix[][] covs ) {
    this.K = K;
    this.D = D;
    this.V = V;

    this.weights = weights;
    this.means = means;
    this.covs = covs;
  }

  public SimpleMatrix getWeights() {
    return weights;
  }
  public SimpleMatrix[] getMeans() {
    return means;
  }
  public SimpleMatrix[][] getCovariances() {
    return covs;
  }

  public int getD() { return D; }
  public int getK() { return K; }
  public int getV() { return V; }

  /**
   * Computes the exact moments of the model whose means are given by
   * the columns of M[v].
   */
  public static Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor>
      computeExactMoments( SimpleMatrix weights, 
        SimpleMatrix M1, SimpleMatrix M2, SimpleMatrix M3 ) {
    // Compute the moments
    SimpleMatrix M12 = M1.mult( MatrixFactory.diag( weights ) ).mult( M2.transpose() );
    SimpleMatrix M13 = M1.mult( MatrixFactory.diag( weights ) ).mult( M3.transpose() );
    SimpleMatrix M23 = M2.mult( MatrixFactory.diag( weights ) ).mult( M3.transpose() );
    FullTensor M123 = FullTensor.fromDecomposition( weights, M1, M2, M3 );

    return new Quartet<>( M12, M13, M23, M123 );
  }
  public Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor>
      computeExactMoments() {
    return computeExactMoments( weights, means[0], means[1], means[2] ); 
  }


  /**
   * Sample N points from a particular cluster
   */
  public SimpleMatrix sample( int N, int view, int cluster ) {
    double[] mean = MatrixFactory.toVector( MatrixOps.col( means[view], cluster ) );
    double[][] cov = MatrixFactory.toArray( covs[view][cluster] );
		fig.prob.MultGaussian mgRnd = new MultGaussian( mean, cov );
		double[][] y = new double[N][D];

		for(int n = 0; n < N; n++)
			y[n] = mgRnd.sample(rnd);
		return new SimpleMatrix( y );
  }

  /**
   * Sample n points from each view of the point distribution
   *
   * @return The returned value contains one matrix of points for each
   * view; each row corresponds to the same draw across views (i.e. the
   * cluster they came from is the same).
   */
  public SimpleMatrix[] sample( int N ) {
    // Generate n random points
    SimpleMatrix X[] = new SimpleMatrix[V];
    for( int v = 0; v < V; v++ ) X[v] = new SimpleMatrix( N, D );

    double[] z = RandomFactory.multinomial( MatrixFactory.toVector( weights ), N );
    for( int v = 0; v < V; v++ ) {
      int offset = 0;
      for( int k = 0; k < K; k++ ) {
        // Sample z[k] numbers from the k-th gaussian 
				int n = (int) z[k];
				MatrixOps.setRows( X[v], offset, offset+n, sample(n, v, k) );
        offset += n;
      }
    }

    return X;
  }

  public static enum WeightDistribution {
    Uniform,
      Random
  }

  public static enum MeanDistribution {
		Hypercube,
      Random
  }

  public static enum CovarianceDistribution {
    Eye,
      Spherical,
      Random
  }

  /**
   * Generate a MixtureOfGaussians model
   * @param K - number of components
   * @param D - number of dimensions
   * @param V - number of views
   * @param wDistribution - distribution on the mixture weights
   * @param mDistribution - distribution on the means
   * @param SDistribution - distribution on the variance
   * @param sigma - variance parameter for the points.
   */
  public static MixtureOfGaussians generate( final int K, final int D, final int V, WeightDistribution wDistribution, MeanDistribution mDistribution, CovarianceDistribution SDistribution, double sigma ) {
    double[] w = new double[K];
    double[][][] M = new double[V][K][D];
    double[][][][] S = new double[V][K][D][D];

    switch( wDistribution ) {
      case Uniform:
        for(int i = 0; i < K; i++ ) w[i] = 1.0/K;
        break;
      case Random:
        // Generate random values, and then normalise
        for(int i = 0; i < K; i++ ) w[i] = Math.abs( RandomFactory.randn(1.0) ); 
        MatrixOps.normalize( w );
        break;
    }

    switch( mDistribution ) {
      case Hypercube:
        int bits = Misc.log2( K ) + 1;
        for(int v = 0; v < V; v++) {
          for(int k = 0; k < K; k++) {
            // Generate points by going around the circle of integers
            int pt = ((v + k) % K) + 1;
            for(int d = 0; d < D; d++) {
              double slope = ( ( (d + pt) %D ) % 2 == 0) ? 1.0 : -1.0;
              M[v][k][d] = (d == pt) ? slope : 0.0;
            }
          }
        }
        break;
      case Random:
        for(int v = 0; v < V; v++) {
          for(int k = 0; k < K; k++) {
            for(int d = 0; d < D; d++) {
              M[v][k][d] = RandomFactory.randn(1.0);
            }
          }
        }
    }

    switch( SDistribution ) {
      case Eye:
        for(int v = 0; v < V; v++) {
          for(int k = 0; k < K; k++) {
            for(int i = 0; i < D; i++) {
              for(int j = 0; j < D; j++) {
                S[v][k][i][j] = (i == j) ? sigma : 0.0;
              }
            }
          }
        }
        break;
      case Spherical:
        for(int v = 0; v < V; v++) {
          for(int k = 0; k < K; k++) {
            for(int i = 0; i < D; i++) {
              for(int j = 0; j < D; j++) {
                S[v][k][i][j] = (i == j) ? Math.abs( RandomFactory.randn(sigma) ) : 0.0;
              }
            }
          }
        }
        break;
      case Random:
        throw new NoSuchMethodError();
    }

    SimpleMatrix weights = MatrixFactory.fromVector( w );
    SimpleMatrix[] means  = new SimpleMatrix[V];
    for( int v = 0; v < V; v++ ) means[v] = (new SimpleMatrix( M[v] )).transpose();

    SimpleMatrix[][] covs = new SimpleMatrix[V][K];
    for( int v = 0; v < V; v++ ) 
      for( int k = 0; k < K; k++ ) 
        covs[v][k] = new SimpleMatrix( S[v][k] );

    return new MixtureOfGaussians( K, D, V, weights, means, covs );
  }

  public static MixtureOfGaussians generate( final int K, final int D, final int V, WeightDistribution wDistribution, MeanDistribution mDistribution, CovarianceDistribution SDistribution ) {
    return generate( K, D, V, wDistribution, mDistribution, SDistribution, 1.0 );
  }
  public static MixtureOfGaussians generate( final int K, final int D, final int V ) {
    return generate( K, D, V, WeightDistribution.Uniform, MeanDistribution.Hypercube, CovarianceDistribution.Eye, 1.0 );
  }

  public static MixtureOfGaussians generate( GenerationOptions options ) {
    WeightDistribution wDistribution;
    MeanDistribution mDistribution;
    CovarianceDistribution SDistribution;

    switch( options.weights.toLowerCase() ) {
      case "uniform":
        wDistribution = WeightDistribution.Uniform; break;
      case "random":
        wDistribution = WeightDistribution.Random; break;
      default:
        throw new NoSuchMethodError();
    }

    switch( options.means.toLowerCase() ) {
      case "hypercube":
        mDistribution = MeanDistribution.Hypercube; break;
      case "random":
        mDistribution = MeanDistribution.Random; break;
      default:
        throw new NoSuchMethodError();
    }

    switch( options.covs.toLowerCase() ) {
      case "eye":
        SDistribution = CovarianceDistribution.Eye; break;
      case "spherical":
        SDistribution = CovarianceDistribution.Spherical; break;
      case "random":
        SDistribution = CovarianceDistribution.Random; break;
      default:
        throw new NoSuchMethodError();
    }

    return generate( options.K, options.D, options.V, wDistribution, mDistribution, SDistribution, options.sigma );
  }

  public static class GenerationOptions {
    @Option(gloss="Number of components") 
    public int K = 2;
    @Option(gloss="Number of dimensions") 
    public int D = 3;
    @Option(gloss="Number of views") 
    public int V = 3;
    
    @Option(gloss="Weight distribution = uniform|random") 
    public String weights = "uniform";
    @Option(gloss="Mean distribution = hypercube|random") 
    public String means = "hypercube";
    @Option(gloss="Covariance distribution = eye|spherical|random") 
    public String covs = "eye";
    @Option(gloss="variance parameter") 
    public double sigma = 0.01;
  }
  public static class OutputOptions {
    @Option(gloss="Output file: '-' for STDOUT") 
    public String outputPath = "-";

    @Option(gloss="Number of points") 
    public double N = 1e3;
  }

  /**
   * Generates data with given specifications to stdout. 
   */
  public static void main( String[] args ) {
    GenerationOptions genOptions = new GenerationOptions();
    OutputOptions outOptions = new OutputOptions();
    OptionsParser parser = new OptionsParser( genOptions, outOptions );
    if( !parser.parse( args ) ) {
      return;
    }
    MixtureOfGaussians model = MixtureOfGaussians.generate( genOptions );

    int N = (int) outOptions.N;
    int D = (int) genOptions.D;
    int V = (int) genOptions.V;
    SimpleMatrix[] X = model.sample( N );

    if( outOptions.outputPath == "-" ) {
      // Print data
      for( int v = 0; v < V; v++ ) {
        System.out.printf( "View %d\n", v );
        for( int n = 0; n < N; n++ ) {
          for( int d = 0; d < D; d++ )
            System.out.printf( "%f ", X[v].get(n,d) );
          System.out.printf( "\n" );
        }
      }
    } else {
      try {
        ObjectOutputStream out = new ObjectOutputStream( new FileOutputStream( outOptions.outputPath ) ); 
        out.writeObject(X);
        out.close();
      } catch (IOException e){
        LogInfo.error( e );
      }
    }
  }


}

