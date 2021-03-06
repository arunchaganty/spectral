/**
 * learning.model
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.models;

import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.OptionsParser;
import fig.prob.MultGaussian;
import learning.Misc;
import learning.data.ComputableMoments;
import learning.data.HasExactMoments;
import learning.data.HasSampleMoments;
import learning.linalg.FullTensor;
import learning.linalg.MatrixFactory;
import learning.linalg.MatrixOps;
import learning.linalg.RandomFactory;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.javatuples.Triplet;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Random;

/**
 * A mixture of experts model
 *
 * TODO (arun): I hate this class
 */
@Deprecated
public class MixtureOfGaussians implements HasExactMoments, HasSampleMoments {

  protected int K; // Number of components
  protected int D; // Dimensionality of space
  protected int V; // Number of views

  // Model parameters
  protected SimpleMatrix weights;
  protected SimpleMatrix[] means;
  protected SimpleMatrix[][] covs;
  protected SimpleMatrix[][] invcovs;

  Random rnd = new Random();

  public MixtureOfGaussians(int K, int D, int V, SimpleMatrix weights, SimpleMatrix[] means, SimpleMatrix[][] covs) {
    this.K = K;
    this.D = D;
    this.V = V;

    this.weights = weights;
    this.means = means;
    // The covs are unit? So be it.
    if(covs == null) {
      covs = new SimpleMatrix[V][K];
      for(int view = 0; view < V; view++)
        for(int k = 0; k < K; k++)
          covs[view][k] = MatrixFactory.eye(D);
    }
    this.covs = covs;
    this.invcovs = new SimpleMatrix[V][K];
    for(int view = 0; view < V; view++)
      for(int k = 0; k < K; k++)
        invcovs[view][k] = covs[view][k].invert();
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
    SimpleMatrix M13 = M1.mult( MatrixFactory.diag( weights ) ).mult( M3.transpose() );
    SimpleMatrix M12 = M1.mult( MatrixFactory.diag( weights ) ).mult( M2.transpose() );
    SimpleMatrix M32 = M3.mult( MatrixFactory.diag( weights ) ).mult( M2.transpose() );
    FullTensor M123 = FullTensor.fromDecomposition( weights, M1, M2, M3 );

    Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> x = MatrixOps.svd(M12);

    return new Quartet<>( M13, M12, M32, M123 );
  }
  public Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor>
      computeExactMoments() {
    return computeExactMoments( weights, means[0], means[1], means[2] ); 
  }

  @Deprecated
  public ComputableMoments computeExactMoments_() {
    final SimpleMatrix M1 = means[0];
    final SimpleMatrix M2 = means[1];
    final SimpleMatrix M3 = means[2];
    return new ComputableMoments() {
      @Override
      public MatrixOps.Matrixable computeP13() {
        return MatrixOps.matrixable(M1.mult( MatrixFactory.diag( weights ) ).mult( M3.transpose() ));
      }

      @Override
      public MatrixOps.Matrixable computeP12() {
        return MatrixOps.matrixable(M1.mult( MatrixFactory.diag( weights ) ).mult( M2.transpose() ));
      }

      @Override
      public MatrixOps.Matrixable computeP32() {
        return MatrixOps.matrixable(M3.mult( MatrixFactory.diag( weights ) ).mult( M2.transpose() ));
      }

      @Override
      public MatrixOps.Tensorable computeP123() {
        return MatrixOps.tensorable(FullTensor.fromDecomposition( weights, M1, M2, M3 ));
      }
    };
  }


  public static Triplet<
        Pair<SimpleMatrix, FullTensor>,
        Pair<SimpleMatrix, FullTensor>,
        Pair<SimpleMatrix, FullTensor>>
      computeSymmetricMoments( SimpleMatrix weights, 
        SimpleMatrix M1, SimpleMatrix M2, SimpleMatrix M3 ) {
    // Compute the moments
    SimpleMatrix M11 = M1.mult( MatrixFactory.diag( weights ) ).mult( M1.transpose() );
    FullTensor M111 = FullTensor.fromDecomposition( weights, M1, M1, M1 );
    SimpleMatrix M22 = M2.mult( MatrixFactory.diag( weights ) ).mult( M2.transpose() );
    FullTensor M222 = FullTensor.fromDecomposition( weights, M2, M2, M2 );
    SimpleMatrix M33 = M3.mult( MatrixFactory.diag( weights ) ).mult( M3.transpose() );
    FullTensor M333 = FullTensor.fromDecomposition( weights, M3, M3, M3 );

    return new Triplet<>( 
        new Pair<>(M11,M111),
        new Pair<>(M22,M222),
        new Pair<>(M33,M333)
        );
  }
  public Triplet<
        Pair<SimpleMatrix, FullTensor>,
        Pair<SimpleMatrix, FullTensor>,
        Pair<SimpleMatrix, FullTensor>>
      computeSymmetricMoments() {
    return computeSymmetricMoments( weights, means[0], means[1], means[2] ); 
  }

  /**
   * Sample N points from a particular cluster
   */
  public SimpleMatrix sample( int N, int view, int cluster ) {
    double[] mean = MatrixFactory.toVector( MatrixOps.col( means[view], cluster ) );
    double[][] cov = MatrixFactory.toArray( covs[view][cluster] );
		MultGaussian mgRnd = new MultGaussian( mean, cov );
		double[][] y = new double[N][D];

		for(int n = 0; n < N; n++)
			y[n] = mgRnd.sample(rnd);
		return new SimpleMatrix( y );
  }

  public Pair<SimpleMatrix,int[]> sampleWithCluster( int N, int view, int cluster ) {
    double[] mean = MatrixFactory.toVector( MatrixOps.col( means[view], cluster ) );
    double[][] cov = MatrixFactory.toArray( covs[view][cluster] );
		MultGaussian mgRnd = new MultGaussian( mean, cov );
		double[][] y = new double[N][D];
		int[] h = new int[N];

		for(int n = 0; n < N; n++) {
			y[n] = mgRnd.sample(rnd);
      h[n] = cluster;
    }
		return new Pair<>(new SimpleMatrix( y ), h);
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
  public Pair<SimpleMatrix[],int[]> sampleWithCluster( int N ) {
    // Generate n random points
    SimpleMatrix X[] = new SimpleMatrix[V];
    for( int v = 0; v < V; v++ ) X[v] = new SimpleMatrix( N, D );

    int[] h = new int[N];

    double[] z = RandomFactory.multinomial( MatrixFactory.toVector( weights ), N );
    for( int v = 0; v < V; v++ ) {
      int offset = 0;
      for( int k = 0; k < K; k++ ) {
        // Sample z[k] numbers from the k-th gaussian 
				int n = (int) z[k];
				MatrixOps.setRows( X[v], offset, offset+n, sample(n, v, k) );
        for(int i = offset; i < offset + n; i++ ) h[i] = k;
        offset += n;
      }
    }

    return new Pair<>(X, h);
  }

  public double computeLikelihood(SimpleMatrix[] data) {
    double lhood = 0.;
    for(int view = 0; view < V; view++) {
      SimpleMatrix data_ = data[view];
      SimpleMatrix means_ = means[view];
      for(int row = 0; row < data_.numRows(); row++) {
        // Add the Gaussian of each component.
        SimpleMatrix datum = MatrixOps.row(data_, row);
        for(int k = 0; k < K; k++ ) {
          SimpleMatrix mean = MatrixOps.col(means_,k).transpose();
          lhood += 0.5 * (datum.minus(mean).normF() * datum.minus(mean).normF()) - Math.log(K);
//                  MatrixOps.xMy(datum.minus(mean), invcovs[view][k], datum.minus(mean));
        }
      }
    }

    return lhood;
  }

  @Override
  public Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> computeSampleMoments(int N) {
    SimpleMatrix[] X = sample(N);
    int K = getK();

    // Compute the moments
    SimpleMatrix P13 = MatrixOps.Pairs( X[0], X[2] );
    SimpleMatrix P12 = MatrixOps.Pairs( X[0], X[1] );
    SimpleMatrix P32 = MatrixOps.Pairs( X[2], X[1] );
    FullTensor P123 = MatrixOps.Triples( X[0], X[1], X[2] );

    return Quartet.with(P13, P12, P32, P123);
  }

  public static enum WeightDistribution {
    Uniform,
      Random
  }

  public static enum MeanDistribution {
		Identical,
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
      case Identical:
        // Generate each gaussian randomly of unit norm
        for(int k = 0; k < K; k++) {
          for(int d = 0; d < D; d++) {
            M[0][k][d] = RandomFactory.randn(1.0);
          }
          MatrixOps.normalize(M[0][k]);
          // Copy to the remaining views
          for(int v = 1; v < V; v++) {
            for(int d = 0; d < D; d++) {
              M[v][k][d] = M[0][k][d];
            }
          }
        }
        break;

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
      case "identical":
        mDistribution = MeanDistribution.Identical; break;
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
    @Option(gloss="Mean distribution = identical|hypercube|random") 
    public String means = "hypercube";
    @Option(gloss="Covariance distribution = eye|spherical|random") 
    public String covs = "eye";
    @Option(gloss="variance parameter") 
    public double sigma = 0.01;

    @Option(gloss="genRandom")
    Random genRandom = new Random(42);
  }
  public static class OutputOptions {
    @Option(gloss="Output file: '-' for STDOUT") 
    public String outputPath = "-";

    @Option(gloss="with cluster") 
    public boolean withCluster = false;

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
    MixtureOfGaussians model = MixtureOfGaussians.generate(genOptions);

    int N = (int) outOptions.N;
    int D = (int) genOptions.D;
    int V = (int) genOptions.V;
    if( outOptions.outputPath == "-" && outOptions.withCluster ) {
      Pair<SimpleMatrix[],int[]> Xh = model.sampleWithCluster( N );
      SimpleMatrix[] X = Xh.getValue0();
      int[] h = Xh.getValue1();

      if( outOptions.outputPath == "-" ) {
        // Print data
        for( int v = 0; v < V; v++ ) {
          System.out.printf( "# View %d\n", v );
          for( int n = 0; n < N; n++ ) {
            for( int d = 0; d < D; d++ )
              System.out.printf( "%f ", X[v].get(n,d) );
            System.out.printf( "# %d ", h[n] );
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
    else {
      SimpleMatrix[] X = model.sample( N );

      if( outOptions.outputPath == "-" ) {
        // Print data
        for( int v = 0; v < V; v++ ) {
          System.out.printf( "# View %d\n", v );
          System.out.printf( "# M: " );
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

}

