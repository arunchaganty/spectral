/**
 * learning.model
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.models;

import learning.linalg.MatrixOps;
import learning.linalg.MatrixFactory;
import learning.linalg.RandomFactory;
import static learning.Misc.*;

import learning.models.transforms.*;
import org.ejml.simple.SimpleMatrix;
import org.ejml.data.DenseMatrix64F;

import org.javatuples.*;

import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.OptionsParser;
import fig.exec.Execution;

import java.io.*;

import java.util.ArrayList;
import java.util.Random;

/**
 * A mixture of experts model
 */
public class MixtureOfExperts implements Serializable {
  private static final long serialVersionUID = 2L;

  protected int K; // Number of experts
  protected int D; // Dimensionality of space

  // Model parameters
  protected SimpleMatrix weights;
  protected SimpleMatrix betas;
  protected double sigma2;

  // Point generator
  protected SimpleMatrix mean;
  protected SimpleMatrix cov;

  protected boolean bias;
  protected NonLinearity nl;

  public boolean removeThirds = false;

  Random rnd = new Random();

  public MixtureOfExperts( int K, int D, SimpleMatrix weights,
      SimpleMatrix betas, double sigma2, SimpleMatrix mean, SimpleMatrix
      cov, boolean bias, NonLinearity nl ) {
    this.K = K;
    this.D = D;

    this.weights = weights;
    this.betas = betas;
    this.sigma2 = sigma2;

    this.mean = mean;
    this.cov = cov;

    this.bias = bias;
    this.nl = nl;
  }

  public int getK() {
    return K;
  }
  public int getD() {
    return D;
  }
  public int getDataDimension() {
    return nl.getLinearDimension( D );
  }

  public SimpleMatrix getWeights() {
    return weights;
  }
  public SimpleMatrix getBetas() {
    return betas;
  }
  public double getSigma2() {
    return sigma2;
  }
  public NonLinearity getNonLinearity() {
    return nl;
  }

  /**
   * Sample n points from the point distribution and generate y's
   * according to the mixture of experts model
   *
   * @return The returned value is a tuple of 2 elements, the first
   * containing $y$ and the second $X$.
   */
  public Pair<SimpleMatrix, SimpleMatrix> sample( int N ) {
    // Generate n random points
    SimpleMatrix X = RandomFactory.rand( N, D ); X = X.scale( 1.0 );

    // Add a bias term
    double[][] X_ = MatrixFactory.toArray( X );
    // Remove thirds
    if(removeThirds) {
      X_ = MatrixOps.removeInRange(X_, -0.5, -0.25);
      X_ = MatrixOps.removeInRange(X_, 0.25, 0.5);
      N = X_.length;
    }


    if( bias ) {
        for( int n = 0; n < N; n++ ) {
          // If x in a range, just remove it
          double[] x = X_[n];
          X_[n] = new double[ D + 1 ];
          X_[n][0] = 1.0;
          for( int d = 0; d < D; d++ )
            X_[n][d+1] = x[d];
      }
    }

    // Projected X onto the "feature space"
    X_ = nl.getLinearEmbedding( X_ );

    // Get the betas in a row form to make it easier to generate data
    double[][] betas_ = MatrixFactory.toArray( betas.transpose() );

    double[] y = new double[ N ];
    for( int n = 0; n < N; n++ ) {
      // Sample the coordinates $z$
      int z = (int) RandomFactory.multinomial( weights );

      // Generate the y's
      y[n] = MatrixOps.dot( betas_[z], X_[n] );
      if( sigma2 > 0 )
        y[n] += RandomFactory.randn( sigma2 );
    }

    return new Pair<>( MatrixFactory.fromVector( y ), new SimpleMatrix( X_ ) );
  }

  public static enum WeightDistribution {
    Uniform,
      Random
  }

  public static enum BetaDistribution {
    Eye,
      Random
  }

  public static enum MeanDistribution {
    Zero,
      Random
  }

  public static enum CovarianceDistribution {
    Eye,
      Spherical,
      Random
  }

  /**
   * Generate a MixtureOfExperts model
   * @param K - number of experts
   * @param D - number of dimensions
   * @param sigma2 - noise in y
   * @param wDistribution - distribution on the mixture weights
   * @param bDistribution - distribution on the betas
   * @param mDistribution - distribution on the point means
   * @param SDistribution - distribution on the point variance
   * @param pointSigma - variance parameter for the points.
   * @param nl - Non-Linearity.
   */
  public static MixtureOfExperts generate( final int K, final int D,
      double sigma2, WeightDistribution wDistribution, BetaDistribution
      bDistribution, MeanDistribution mDistribution,
      CovarianceDistribution SDistribution, double pointSigma,
      boolean bias, NonLinearity nl ) {
    int D_ = nl.getLinearDimension( D + (bias ? 1 : 0) ); // The dimension of the linear embedding of the data.

    double[] w = new double[K];
    double[][] B = new double[D_][K];
    double[] M = new double[D_];
    double[][] S = new double[D_][D_];

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

    switch( bDistribution ) {
      case Eye:
        for(int i = 0; i < D_; i++) {
          for(int j = 0; j < K; j++) {
            double slope = ( i % 2 == 0 ) ? 1.0 : -1.0;
            B[i][j] = (i == j) ? slope : 0.0;
          }
        }
        break;
      case Random:
        for(int i = 0; i < D_; i++) {
          for(int j = 0; j < K; j++) {
            B[i][j] = RandomFactory.randn(1.0);
          }
        }
    }

    switch( mDistribution ) {
      case Zero:
        break;
      case Random:
        for( int i = 0; i < D; i++ )
          M[i] = RandomFactory.randn(1.0);
    }

    switch( SDistribution ) {
      case Eye:
        for(int i = 0; i < D; i++) {
          for(int j = 0; j < D; j++) {
            S[i][j] = (i == j) ? pointSigma : 0.0;
          }
        }
        break;
      case Spherical:
        for(int i = 0; i < D; i++) {
          for(int j = 0; j < D; j++) {
            S[i][j] = (i == j) ? Math.abs( RandomFactory.randn(pointSigma) ) : 0.0;
          }
        }
        break;
      case Random:
        throw new NoSuchMethodError();
    }

    SimpleMatrix weights = MatrixFactory.fromVector( w );
    SimpleMatrix betas = new SimpleMatrix( B );
    SimpleMatrix mean = MatrixFactory.fromVector( M );
    SimpleMatrix cov = new SimpleMatrix( S );

    return new MixtureOfExperts( K, D, weights, betas, sigma2, mean, cov, bias, nl );
  }

  public static MixtureOfExperts generate( final int K, final int D,
      double sigma2, WeightDistribution wDistribution, BetaDistribution
      bDistribution, MeanDistribution mDistribution,
      CovarianceDistribution SDistribution ) {
    return generate( K, D, sigma2, wDistribution, bDistribution, mDistribution, SDistribution, 10.0, true, new PolynomialNonLinearity() );
  }

  public static MixtureOfExperts generate( GenerationOptions options ) {
    WeightDistribution wDistribution;
    BetaDistribution bDistribution;
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

    switch( options.betas.toLowerCase() ) {
      case "eye":
        bDistribution = BetaDistribution.Eye; break;
      case "random":
        bDistribution = BetaDistribution.Random; break;
      default:
        throw new NoSuchMethodError();
    }

    switch( options.mean.toLowerCase() ) {
      case "zero":
        mDistribution = MeanDistribution.Zero; break;
      case "random":
        mDistribution = MeanDistribution.Random; break;
      default:
        throw new NoSuchMethodError();
    }

    switch( options.cov.toLowerCase() ) {
      case "eye":
        SDistribution = CovarianceDistribution.Eye; break;
      case "spherical":
        SDistribution = CovarianceDistribution.Spherical; break;
      case "random":
        SDistribution = CovarianceDistribution.Random; break;
      default:
        throw new NoSuchMethodError();
    }

    NonLinearity nl;
    switch(options.nlType) {
      case "poly":
        nl = new PolynomialNonLinearity( options.nlDegree ); break;
      case "fourier":
        nl = new FourierNonLinearity( options.nlDegree ); break;
      case "poly-independent":
        nl = new IndependentPolynomial( options.nlDegree, options.D ); break;
      case "random-fractional":
        nl = new FractionalPolynomial( options.nlDegree, options.D ); break;
      default:
        throw new NoSuchMethodError();
    }

    MixtureOfExperts model =  generate( options.K, options.D, options.sigma2, wDistribution, bDistribution, mDistribution, SDistribution, options.pointSigma, options.bias, nl );
    model.removeThirds = options.removeThirds;

    return model;
  }

  public static class GenerationOptions {
    @Option(gloss="Number of experts") 
    public int K = 2;
    @Option(gloss="Number of dimensions") 
    public int D = 3;
    @Option(gloss="Noise") 
    public double sigma2 = 0.0;

    @Option(gloss="Include a bias term of '1'") 
    public boolean bias = true;
    
    @Option(gloss="Weight distribution = uniform|random") 
    public String weights = "uniform";
    @Option(gloss="Beta distribution = eye|random") 
    public String betas = "eye";

    // TODO: Get rid of these
    @Option(gloss="Mean distribution = zero|random") 
    public String mean = "zero";
    @Option(gloss="Covariance distribution = eye|spherical|random") 
    public String cov = "eye";
    @Option(gloss="Point variance parameter") 
    public double pointSigma = 10.0;

    @Option(gloss="Non-linearity degree") 
    public int nlDegree = 1;
    @Option(gloss="Non-linearity type")
    public String nlType = "poly";
    @Option(gloss="RemoveThirds")
    public boolean removeThirds = false;
  }
  public static class OutputOptions {
    @Option(gloss="Print binary file in plain text and exit")
    public String inputPath = null;

    @Option(gloss="Output file: '-' for STDOUT")
    public String outputPath = "-";

    @Option(gloss="Number of points") 
    public double N = 1e3;
  }

  @SuppressWarnings("unchecked")
  public static Pair< Pair< SimpleMatrix, SimpleMatrix >, MixtureOfExperts >
  readFromFile( String filename ) throws IOException, ClassNotFoundException {
    // Using  DecompressibleInputStream that basically ignores the
    // version number check. Dangerous!
    // DecompressibleInputStream in = new DecompressibleInputStream( new FileInputStream( filename ) );
    
    ObjectInputStream in = new ObjectInputStream( new FileInputStream( filename ) );

    Pair<SimpleMatrix,SimpleMatrix> yX = (Pair<SimpleMatrix,SimpleMatrix>) in.readObject();
    MixtureOfExperts model = (MixtureOfExperts) in.readObject();
    in.close();

    return new Pair<>( yX, model );
  }

  public static void printData( SimpleMatrix y, SimpleMatrix X, MixtureOfExperts model ) {
    double[][] exponents = model.getNonLinearity().getExponents();

    // Pretty print the exponents
    System.out.printf( "# " );
    for( double[] exp : exponents ) {
      for( int d = 0; d < exp.length; d++ )
        System.out.printf( "x_" + d + "^" + exp[d] );
      System.out.printf( " " );
    }
    System.out.printf( "\n" );

    int N = X.numRows();
    for( int i = 0; i < N; i++ ) {
      for( int d = 0; d < X.numCols(); d++ )
        System.out.printf( "%f ", X.get(i,d) );
      System.out.printf( "%f\n", y.get(i) );
    }

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
    // If this has been enabled, just grok the file and exit
    if( outOptions.inputPath != null ) {
      try{
        Pair<Pair<SimpleMatrix, SimpleMatrix>, MixtureOfExperts> data = readFromFile(outOptions.inputPath);
        SimpleMatrix y = data.getValue0().getValue0();
        SimpleMatrix X = data.getValue0().getValue1();
        MixtureOfExperts model = data.getValue1();
        printData( y,  X, model );
      } catch(IOException | ClassNotFoundException e) {
        System.err.println("Corrupt or unreadable input file");
      }
      System.exit(0);
    }
    MixtureOfExperts model = MixtureOfExperts.generate( genOptions );

    int N = (int) outOptions.N;
    Pair<SimpleMatrix, SimpleMatrix> yX = model.sample( N );

    if( outOptions.outputPath.equals( "-" ) ) {
      SimpleMatrix y = yX.getValue0();
      SimpleMatrix X = yX.getValue1();
      printData( y,  X, model );
    } else {
      try {
        ObjectOutputStream out = new ObjectOutputStream( new FileOutputStream( outOptions.outputPath ) ); 
        out.writeObject(yX);
        out.writeObject(model);
        out.close();
      } catch (IOException e){
        LogInfo.error( e );
      }
    }

  }


}

