/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.spectral.applications;

import learning.exceptions.RecoveryFailure;
import learning.exceptions.NumericalException;

import learning.models.MixtureOfExperts;

import learning.spectral.MultiViewMixture;
import learning.linalg.MatrixOps;
import learning.linalg.MatrixFactory;
import learning.linalg.RandomFactory;
import learning.linalg.Tensor;
import learning.linalg.FullTensor;

import learning.data.MomentComputer;
import learning.data.RealSequence;

import org.ejml.alg.dense.mult.GeneratorMatrixMatrixMult;
import org.ejml.alg.dense.mult.MatrixMatrixMult;
import org.ejml.data.DenseMatrix64F;
import org.ejml.simple.SimpleBase;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.*;

import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.OptionsParser;
import fig.exec.Execution;

import java.util.Date;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.ClassNotFoundException;

/**
 * Spectral Experts.
 */
public class SpectralExperts implements Runnable {
	@Option(gloss = "Number of classes")
	public int K = 0;
	@Option(gloss = "Input filename", required=true)
	public String inputPath;
  @Option(gloss = "Regularization")
  public double reg = 0.1;
	@Option(gloss = "Number of Threads to use")
	public int nThreads = 1;
	@Option(gloss = "Random seed")
	public int seed = (int)(new Date()).getTime();

  public SpectralExperts() {}
  public SpectralExperts( int K ) {
    this.K = K;
  }

  /*
    Regularize the matrices A and b which form the solution of the linear equation
    Ax = b; i.e. (A'A + \lambda I) x = A' b;
   */
  Pair<SimpleMatrix, SimpleMatrix> regularize( SimpleMatrix A, SimpleMatrix b, double reg ) {
    int N = A.numRows();
    int D = A.numCols();
    SimpleMatrix At = A.transpose();
    A = At.mult( A ).plus(reg, SimpleMatrix.identity(D));
    b = At.mult(b.transpose());

    return new Pair<>(A, b);
  }

  /**
   * Recover the second-order tensor \beta \otimes \beta by linear regression.
   * @param y
   * @param X
   * @return
   */
  SimpleMatrix recoverPairs(SimpleMatrix y, SimpleMatrix X, double reg) {
    int N = X.numRows();
    int D = X.numCols();
    int D_ = D * (D+1) / 2;

    // Consider only the upper triangular half of x x' (because it is symmetric) and construct a vector.
    double[][] A_ = new double[N][D_];

    for( int n = 0; n < N; n++ ) {
      for( int d = 0; d < D; d++ ) {
        for( int d_ = 0; d_ <= d; d_++ ) {
          int idx = d * (d-1) / 2 + d_;
          A_[n][ idx ] = X.get(n, d) * X.get(n,d_);
          if( d != d_ ) A_[n][idx] /= 2;
        }
      }
    }

    // Solve for the matrix
    SimpleMatrix A = new SimpleMatrix( A_ );
    SimpleMatrix b = y.elementMult(y);
    // Regularize the matrix
    if( reg > 0.0 ) {
      Pair<SimpleMatrix, SimpleMatrix> Ab = regularize(A, b, reg);
      A = Ab.getValue0(); b = Ab.getValue1();
    }
    SimpleMatrix x = A.solve(b);

    // Reconstruct $B$ from $x$
    SimpleMatrix B = new SimpleMatrix(D, D);
    for( int d = 0; d < D; d++ ) {
      for( int d_ = 0; d_ <= d; d_++ ) {
        int idx = d * (d-1) / 2 + d_;
        B.set(d, d_, x.get(idx));
        B.set(d_, d, x.get(idx));
      }
    }

    return B;
  }

  SimpleMatrix recoverPairs(SimpleMatrix y, SimpleMatrix X) {
    return recoverPairs( y, X, 0.0 );
  }


  /**
   * Recover the second-order tensor \beta \otimes \beta by linear regression.
   * @param y
   * @param X
   * @return
   */
  Tensor recoverTriples(SimpleMatrix y, SimpleMatrix X, double reg) {
    int N = X.numRows();
    int D = X.numCols();
    int D_ = D * (D+1) * (D+2) / 3;

    // Consider only the upper triangular half of x x' (because it is symmetric) and construct a vector.
    double[][] A_ = new double[N][D_];

    for( int n = 0; n < N; n++ ) {
      for( int d = 0; d < D; d++ ) {
        for( int d_ = 0; d_ <= d; d_++ ) {
          for( int d__ = 0; d__ <= d_; d__++ ) {
            int idx = (d-1) * d * (d+1) / 6 + (d_-1) * d_ / 2 + d__;
            A_[n][idx] = X.get(n, d) * X.get(n,d_) * X.get(n,d__);
            if( d == d_ && d_ == d__ ) {
            } else if( d == d_ || d_ == d__ || d == d__) {
              A_[n][idx] /= 2;
            } else {
              A_[n][idx] /= 3;
            }
          }
        }
      }
    }

    // Solve for the matrix
    SimpleMatrix A = new SimpleMatrix( A_ );
    SimpleMatrix b = y.elementMult(y).elementMult(y);;
    // Regularize the matrix
    if( reg > 0.0 ) {
      Pair<SimpleMatrix, SimpleMatrix> Ab = regularize(A, b, reg);
      A = Ab.getValue0(); b = Ab.getValue1();
    }
    SimpleMatrix x = A.solve(b);

    // Reconstruct $B$ from $x$
    double[][][] B = new double[D][D][D];
    for( int d = 0; d < D; d++ ) {
      for( int d_ = 0; d_ <= d; d_++ ) {
        for( int d__ = 0; d__ <= d_; d__++ ) {
          int idx = (d-1) * d * (d+1) / 6 + (d_-1) * d_ / 2 + d__;
          B[d][d_][d__] = B[d][d__][d_] =
             B[d_][d][d__] = B[d_][d_][d_] =
             B[d__][d][d_] = B[d__][d_][d] =  x.get(idx);
        }
      }
    }

    return new FullTensor(B);
  }

  Tensor recoverTriples(SimpleMatrix y, SimpleMatrix X) {
    return recoverTriples( y, X, 0.0 );
  }



  /**
   * Run the SpectralExperts algorithm on data $y$, $X$.
   * @param y
   * @param X
   * @return
   * @throws NumericalException
   */
  public SimpleMatrix run(int K, SimpleMatrix y, SimpleMatrix X) throws NumericalException, RecoveryFailure {
    // Set the seed
    RandomFactory.setSeed( seed );

    // Recover Pairs and Triples moments by linear regression
    SimpleMatrix Pairs = recoverPairs( y, X, reg );
    System.out.println( Pairs );
    Tensor Triples = recoverTriples(y, X, reg );

    // Use Algorithm B symmetric to recover the $\beta$
    MultiViewMixture algo = new MultiViewMixture();
    return algo.algorithmB( K, Pairs,  Pairs, Triples );
  }

  public Pair< Pair< SimpleMatrix, SimpleMatrix >, learning.models.MixtureOfExperts >
  readFromFile( String filename ) throws IOException, ClassNotFoundException {
    ObjectInputStream in = new ObjectInputStream( new FileInputStream( filename ) );

    Pair<SimpleMatrix,SimpleMatrix> yX = (Pair<SimpleMatrix,SimpleMatrix>) in.readObject();
    learning.models.MixtureOfExperts model = (learning.models.MixtureOfExperts) in.readObject();
    in.close();

    return new Pair<>( yX, model );
  }

	@Override
	public void run() {
    try {
      // Read data from a file
      Pair< Pair<SimpleMatrix, SimpleMatrix>, learning.models.MixtureOfExperts > data = readFromFile( inputPath );
      SimpleMatrix y = data.getValue0().getValue0();
      SimpleMatrix X = data.getValue0().getValue1();
      learning.models.MixtureOfExperts model = data.getValue1();
      SimpleMatrix betas = model.getBetas();

      // Set K from the model if it hasn't been provided
      if( this.K < 1 )
        this.K = model.getK();

      SimpleMatrix betas_ = run( this.K, y, X );
      betas_ = MatrixOps.alignMatrix( betas_, betas, true );
      Execution.putOutput( "betas", betas );
      Execution.putOutput( "betas_", betas_ );
      double err = MatrixOps.norm( betas.minus( betas_ ) );
      System.out.printf( "%.4f\n", err );
    } catch( ClassNotFoundException | IOException |  NumericalException | RecoveryFailure e ) {
      System.err.println( e.getMessage() );
      return;
    }
	}

	/**
	 * Mixture of Linear Regressions
	 * @param args
	 * @throws IOException 
	 * @throws RecoveryFailure 
	 */
	public static void main(String[] args) throws IOException, RecoveryFailure {
		Execution.run( args, new SpectralExperts() );
	}
}

