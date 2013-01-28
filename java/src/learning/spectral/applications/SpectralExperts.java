/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.spectral.applications;

import learning.exceptions.RecoveryFailure;
import learning.exceptions.NumericalException;

import learning.linalg.*;
import learning.models.MixtureOfExperts;

import learning.spectral.MultiViewMixture;

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

  public static class SpectralExpertsAnalysis {
    public boolean saveToExecution = false;

    public MixtureOfExperts model;
    public SimpleMatrix Pairs;
    public Tensor Triples;
    public SimpleMatrix betas;


    public double PairsErr;
    public double TriplesErr;
    public double betasErr;

    public SpectralExpertsAnalysis(MixtureOfExperts model) {
      this.model = model;

      // Store the exact moments
      Pair<SimpleMatrix, Tensor> moments = SpectralExperts.computeExactMoments( model );
      Pairs = moments.getValue0();
      Triples = moments.getValue1();
      betas = model.getBetas();
    }

    public void reportPairs( SimpleMatrix Pairs_ ) {
      PairsErr = MatrixOps.norm(Pairs.minus(Pairs_));
      if( saveToExecution ) {
        Execution.putOutput( "Pairs", Pairs );
        Execution.putOutput( "Pairs_", Pairs_ );
        Execution.putOutput( "PairsErr", PairsErr );
      }
      LogInfo.logsForce("Pairs: " + PairsErr);
    }

    public void reportTriples( Tensor Triples_ ) {
      int D = Triples_.getDim(0);
      SimpleMatrix eta = MatrixFactory.ones(D); //RandomFactory.rand(D + 1, 1);
      SimpleMatrix TriplesT = Triples.project(2, eta);
      SimpleMatrix TriplesT_ = Triples_.project(2, eta );
      TriplesErr = MatrixOps.norm(TriplesT.minus(TriplesT_));
      if( saveToExecution ) {
        Execution.putOutput( "eta", eta );
        Execution.putOutput( "TriplesT", TriplesT );
        Execution.putOutput( "TriplesT_", TriplesT_ );
        Execution.putOutput("TriplesTErr", TriplesErr);
      }
      LogInfo.logsForce("TriplesT: " + TriplesErr);
      LogInfo.logsForce("TriplesT_: " + TriplesT);
      LogInfo.logsForce("TriplesT_: " + TriplesT_);
    }

    public void reportBetas( SimpleMatrix betas_ ) {
      betas_ = MatrixOps.alignMatrix( betas_, betas, true );
      betasErr = MatrixOps.norm( betas.minus( betas_ ) );

      if( saveToExecution ) {
        Execution.putOutput( "betas", betas );
        Execution.putOutput( "betas_", betas_ );
        Execution.putOutput( "betasErr", betasErr);
      }
      LogInfo.logsForce("betas: " + betasErr);
    }

    public static double checkDataSanity( SimpleMatrix y, SimpleMatrix X, MixtureOfExperts model ) {
      // Check that the empirical moments match
      SimpleMatrix betas = model.getBetas();
      SimpleMatrix weights = model.getWeights();

      int N = y.numCols();

      double Ey = y.elementSum() / N;
      double Ey_ = X.mult( betas.mult(weights.transpose()) ).elementSum() / N;
      double err = Math.abs(Ey - Ey_);

      LogInfo.logsForce("Ey: " + err);

      return err;
    }
    public double checkDataSanity( SimpleMatrix y, SimpleMatrix X ) {
      return checkDataSanity(y, X, model);
    }

    /**
     * Compute the loss \sum min_k( y - X betas[k])^2.
     * @param y
     * @param X
     * @param betas_
     */
    public static double computeLoss( SimpleMatrix y, SimpleMatrix X, SimpleMatrix betas ) {
      int N = y.numRows();
      int K = betas.numCols();
      double[][] betas_ = MatrixFactory.toArray( betas.transpose() );
      double[][] X_ = MatrixFactory.toArray( X );

      double err = 0.0;
      for( int n = 0; n < N; n++ ) {
        double yn = y.get(n);
        double errn = Double.POSITIVE_INFINITY;

        for( int k = 0; k < K; k++) {
          double errny = (yn - MatrixOps.dot( betas_[k],  X_[n] ));
          errny *= errny;
          errn = ( errny < errn ) ? errny : errn;
        }
        err += (errn - err)/(n+1);
      }

      return err;
    }

  }

  public SpectralExpertsAnalysis analysis = null;

  public SpectralExperts() {}
  public SpectralExperts( int K ) {
    this.K = K;
  }

  public static Pair<SimpleMatrix, Tensor> computeExactMoments( SimpleMatrix weights, SimpleMatrix betas ) {
    SimpleMatrix Pairs = betas.mult( MatrixFactory.diag( weights ) ).mult( betas.transpose() );
    ExactTensor Triples = new ExactTensor( weights, betas, betas, betas );

    return new Pair<SimpleMatrix, Tensor>( Pairs, Triples );
  }

  public static Pair<SimpleMatrix, Tensor> computeExactMoments( MixtureOfExperts model ) {
    return computeExactMoments(model.getWeights(), model.getBetas());
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
    b = At.mult(b);

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

    int idx;
    for( int n = 0; n < N; n++ ) {
      idx = 0;
      for( int d = 0; d < D; d++ ) {
        for( int d_ = 0; d_ <= d; d_++ ) {
          A_[n][ idx++ ] = X.get(n, d) * X.get(n,d_);
        }
      }
    }


    // Solve for the matrix
    SimpleMatrix A = new SimpleMatrix( A_ );
    SimpleMatrix b = y.elementMult(y).transpose();
    // Regularize the matrix
    if( reg > 0.0 ) {
      Pair<SimpleMatrix, SimpleMatrix> Ab = regularize(A, b, reg);
      A = Ab.getValue0(); b = Ab.getValue1();
    }
    SimpleMatrix x = A.solve(b);

    // Reconstruct $B$ from $x$
    SimpleMatrix B = new SimpleMatrix(D, D);
    idx = 0;
    for( int d = 0; d < D; d++ ) {
      for( int d_ = 0; d_ <= d; d_++ ) {
        double multiplicity = (d == d_) ? 1 : 2;
        double x_ = x.get(idx++) / multiplicity;
        B.set(d, d_, x_);
        B.set(d_, d, x_);
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

    int idx;
    for( int n = 0; n < N; n++ ) {
      idx = 0;
      for( int d = 0; d < D; d++ ) {
        for( int d_ = 0; d_ <= d; d_++ ) {
          for( int d__ = 0; d__ <= d_; d__++ ) {
            A_[n][idx++] = X.get(n, d) * X.get(n,d_) * X.get(n,d__);
          }
        }
      }
    }

    // Solve for the matrix
    SimpleMatrix A = new SimpleMatrix( A_ );
    SimpleMatrix b = y.elementMult(y).elementMult(y).transpose();;
    // Regularize the matrix
    if( reg > 0.0 ) {
      Pair<SimpleMatrix, SimpleMatrix> Ab = regularize(A, b, reg);
      A = Ab.getValue0(); b = Ab.getValue1();
    }
    SimpleMatrix x = A.solve(b);

    // Reconstruct $B$ from $x$
    double[][][] B = new double[D][D][D];
    idx = 0;
    for( int d = 0; d < D; d++ ) {
      for( int d_ = 0; d_ <= d; d_++ ) {
        for( int d__ = 0; d__ <= d_; d__++ ) {
          double multiplicity =
                  ( d == d_ && d_ == d__ ) ? 1 :
                  ( d == d_ || d_ == d__ || d == d__) ? 2 : 3;
          B[d][d_][d__] = B[d][d__][d_] =
             B[d_][d][d__] = B[d_][d_][d] =
             B[d__][d][d_] = B[d__][d_][d] =  x.get(idx++) / multiplicity;
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
    if( analysis != null ) analysis.reportPairs(Pairs);
    Tensor Triples = recoverTriples(y, X, reg );
    if( analysis != null ) analysis.reportTriples(Triples);

    // Use Algorithm B symmetric to recover the $\beta$
    MultiViewMixture algo = new MultiViewMixture();
    return algo.algorithmB( K, Pairs,  Pairs, Triples );
  }

  @SuppressWarnings("unchecked")
  public Pair< Pair< SimpleMatrix, SimpleMatrix >, learning.models.MixtureOfExperts >
  readFromFile( String filename ) throws IOException, ClassNotFoundException {
    ObjectInputStream in = new ObjectInputStream( new FileInputStream( filename ) );

    Pair<SimpleMatrix,SimpleMatrix> yX = (Pair<SimpleMatrix,SimpleMatrix>) in.readObject();
    learning.models.MixtureOfExperts model = (learning.models.MixtureOfExperts) in.readObject();
    in.close();

    return new Pair<>( yX, model );
  }

  public void enableAnalysis(MixtureOfExperts model, boolean saveToExecution) {
    analysis = new SpectralExpertsAnalysis(model);
    analysis.saveToExecution = saveToExecution;
  }
  public void enableAnalysis(MixtureOfExperts model) {
    enableAnalysis(model, false);
  }

	@Override
	public void run() {
    try {
      // Read data from a file
      Pair< Pair<SimpleMatrix, SimpleMatrix>, learning.models.MixtureOfExperts > data = readFromFile( inputPath );
      SimpleMatrix y = data.getValue0().getValue0();
      SimpleMatrix X = data.getValue0().getValue1();
      learning.models.MixtureOfExperts model = data.getValue1();
      enableAnalysis(model, true);
      analysis.checkDataSanity(y, X);

      // Set K from the model if it hasn't been provided
      if( this.K < 1 )
        this.K = model.getK();

      SimpleMatrix betas_ = run( this.K, y, X );
      analysis.reportBetas(betas_);

      System.out.printf( "%.4f %.4f %.4f\n", analysis.betasErr,
              SpectralExpertsAnalysis.computeLoss(y, X, model.getBetas()),
              SpectralExpertsAnalysis.computeLoss(y, X, betas_) );
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

