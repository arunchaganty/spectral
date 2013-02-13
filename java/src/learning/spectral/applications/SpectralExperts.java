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

import learning.optimization.PhaseRecovery;
import learning.optimization.ProximalGradientSolver;
import learning.optimization.TensorRecovery;
import learning.spectral.MultiViewMixture;

import learning.data.MomentComputer;
import learning.data.RealSequence;

import learning.spectral.TensorMethod;
import org.ejml.alg.dense.mult.GeneratorMatrixMatrixMult;
import org.ejml.alg.dense.mult.MatrixMatrixMult;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
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
  @Option(gloss = "Ridge Regression Regularization")
  public double ridgeReg = 1e-5;
  @Option(gloss = "Trace Norm Regularization")
  public double traceReg = 1e-4;
  @Option(gloss = "Scale Data?")
  public boolean scaleData = false;
  @Option(gloss = "Run EM on spectral output?")
  public boolean runEM = false;
  @Option(gloss = "Use tensor power method")
  public boolean useTensorPowerMethod = true;
  @Option(gloss = "Use low rank recovery")
  public boolean useLowRankRecovery = true;
  @Option(gloss = "Use low rank recovery")
  public int lowRankIters = 20;
	@Option(gloss = "Number of Threads to use")
	public int nThreads = 1;
	@Option(gloss = "Random seed")
	public int seed = (int)(new Date()).getTime();

  public static class SpectralExpertsAnalysis {
    public boolean saveToExecution = false;

    public MixtureOfExperts model;
    public SimpleMatrix Pairs;
    public FullTensor Triples;
    public SimpleMatrix betas;
    public SimpleMatrix weights;

    public double PairsErr;
    public double TriplesErr;
    public double betasErr;
    public double betasEMErr;
    public double weightsErr;
    public double weightsEMErr;

    public SpectralExpertsAnalysis(MixtureOfExperts model) {
      this.model = model;

      // Store the exact moments
      Pair<SimpleMatrix, FullTensor> moments = SpectralExperts.computeExactMoments( model );
      Pairs = moments.getValue0();
      Triples = moments.getValue1();
      betas = model.getBetas();
      weights = model.getWeights();
    }

    public void reportPairs( final SimpleMatrix Pairs_ ) {
      PairsErr = MatrixOps.diff(Pairs, Pairs_);
      if( saveToExecution ) {
        Execution.putOutput( "Pairs", Pairs );
        Execution.putOutput( "Pairs_", Pairs_ );
        Execution.putOutput( "PairsErr", PairsErr );
      }
      LogInfo.logsForce("Pairs: " + PairsErr);
    }

    public void reportTriples( final Tensor Triples_ ) {
      int D = Triples_.getDim(0);
      SimpleMatrix eta = MatrixFactory.ones(D); //RandomFactory.rand(D + 1, 1);
      SimpleMatrix TriplesT = Triples.project(2, eta);
      SimpleMatrix TriplesT_ = Triples_.project(2, eta );
      TriplesErr = MatrixOps.diff(TriplesT, TriplesT_);
      if( saveToExecution ) {
        Execution.putOutput( "eta", eta );
        Execution.putOutput( "TriplesT", TriplesT );
        Execution.putOutput( "TriplesT_", TriplesT_ );
        Execution.putOutput("TriplesTErr", TriplesErr);
      }
      LogInfo.logsForce("TriplesT: " + TriplesErr);
    }

    public void reportTriples0( final FullTensor Triples_ ) {
      TriplesErr = MatrixOps.diff(Triples, Triples_ );
      if( saveToExecution ) {
        Execution.putOutput("Triples0Err", TriplesErr);
      }
      LogInfo.logsForce("Triples0: " + TriplesErr);
    }

    public void reportTriples( final FullTensor Triples_ ) {
      TriplesErr = MatrixOps.diff(Triples, Triples_ );
      if( saveToExecution ) {
        Execution.putOutput("TriplesErr", TriplesErr);
      }
      LogInfo.logsForce("Triples: " + TriplesErr);
    }

    public void reportWeights( SimpleMatrix weights_ ) {
      weights_ = MatrixOps.alignMatrix( weights_, weights, true );
      weightsErr = MatrixOps.norm( weights.minus( weights_ ) );

      if( saveToExecution ) {
        Execution.putOutput( "weights", weights );
        Execution.putOutput( "weights_", weights_ );
        Execution.putOutput( "weightsErr", weightsErr);
      }
      LogInfo.logsForce("weights: " + weights_ );
      LogInfo.logsForce("weights: " + weightsErr);
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
    public void reportBetasEM( SimpleMatrix betas_ ) {
      betas_ = MatrixOps.alignMatrix( betas_, betas, true );
      betasEMErr = MatrixOps.norm( betas.minus( betas_ ) );

      if( saveToExecution ) {
        Execution.putOutput( "betasEM", betas_ );
        Execution.putOutput( "betasEMErr", betasEMErr);
      }
      LogInfo.logsForce("betasEM: " + betasEMErr);
    }
    public void reportWeightsEM( SimpleMatrix weights_ ) {
      weights_ = MatrixOps.alignMatrix( weights_, weights, true );
      weightsEMErr = MatrixOps.norm( weights.minus( weights_ ) );

      if( saveToExecution ) {
        Execution.putOutput( "weightsEM", weights_ );
        Execution.putOutput( "weightsErr", weightsEMErr);
      }
      LogInfo.logsForce("weightsEM: " + weightsEMErr);
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
     * @param betas
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

  public static Pair<SimpleMatrix, FullTensor> computeExactMoments( SimpleMatrix weights, SimpleMatrix betas ) {
    SimpleMatrix Pairs = betas.mult( MatrixFactory.diag( weights ) ).mult( betas.transpose() );
    FullTensor Triples = FullTensor.fromDecomposition( weights, betas );

    return new Pair<>( Pairs, Triples );
  }

  public static Pair<SimpleMatrix, FullTensor> computeExactMoments( MixtureOfExperts model ) {
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
  SimpleMatrix recoverPairs(SimpleMatrix y, SimpleMatrix X, double reg, boolean doScale) {
    int N = X.numRows();
    int D = X.numCols();
    int D_ = D * (D+1) / 2;

    // Normalize the data
    SimpleMatrix xScaling = MatrixFactory.ones(D);
    double yScaling = 1.0;
    if(doScale) {
      Pair<SimpleMatrix, SimpleMatrix> scaleInfo = MatrixOps.columnScale(X);
      X = scaleInfo.getValue0(); xScaling = scaleInfo.getValue1();
      scaleInfo = MatrixOps.rowScale(y);
      y = scaleInfo.getValue0(); yScaling = scaleInfo.getValue1().get(0);
    }

    // Consider only the upper triangular half of x x' (because it is symmetric) and construct a vector.
    double[][] A_ = new double[N][D_];

    int idx;
    for( int n = 0; n < N; n++ ) {
      idx = 0;
      for( int d = 0; d < D; d++ ) {
        for( int d_ = 0; d_ <= d; d_++ ) {
          double multiplicity = (d == d_) ? 1 : 2;
          A_[n][ idx++ ] = X.get(n, d) * X.get(n,d_) * multiplicity;
        }
      }
    }

    // Solve for the matrix
    SimpleMatrix A = new SimpleMatrix( A_ );
    SimpleMatrix b = y.elementMult(y).transpose();
    SimpleMatrix bEntries;
    // Regularize the matrix
    if( reg > 0.0 ) {
      Pair<SimpleMatrix, SimpleMatrix> Ab = regularize(A, b, reg);
      A = Ab.getValue0(); b = Ab.getValue1();
      bEntries = A.solve(b);
    }
    else
      bEntries = A.solve(b);
    LogInfo.logsForce( "Condition Number for Pairs: " + A.conditionP2());

    // Reconstruct $B$ from $x$
    SimpleMatrix B = new SimpleMatrix(D, D);
    idx = 0;
    for( int d = 0; d < D; d++ ) {
      for( int d_ = 0; d_ <= d; d_++ ) {
        double b_ = bEntries.get(idx++);
        B.set(d, d_, b_);
        B.set(d_, d, b_);
      }
    }
    // Rescale
    if(doScale) { // Unscale B
      for( int d = 0; d < D; d++ ) {
        for( int d_ = 0; d_ < D; d_++ ) {
          double scaling = (yScaling * yScaling) / (xScaling.get(d) * xScaling.get(d_));
          B.set( d, d_, B.get(d, d_) * scaling);
        }
      }
    }

    if( useLowRankRecovery ) {
      // Use low rank recovery to improve estimate.
      LogInfo.begin_track("low-rank-pairs");
      // Report the performance before low-rankification
      if( analysis != null ) analysis.reportPairs(B);

      ProximalGradientSolver solver = new ProximalGradientSolver();
      // Tweak the response variables to give an unbiased estimate
      DenseMatrix64F y_ = y.elementMult(y).getMatrix();
      if(analysis != null)
        CommonOps.add(y_, -analysis.model.getSigma2());
      y = SimpleMatrix.wrap(y_);
      PhaseRecovery problem = new PhaseRecovery(y, X, traceReg);
      DenseMatrix64F Pairs_ = solver.optimize(problem, B.getMatrix(), lowRankIters);
      B = SimpleMatrix.wrap(Pairs_);

      LogInfo.end_track("low-rank-pairs");
    }

    return B;
  }

  SimpleMatrix recoverPairs(SimpleMatrix y, SimpleMatrix X) {
    return recoverPairs( y, X, 0.0, true );
  }

  /**
   * Recover the second-order tensor \beta \otimes \beta by linear regression.
   * @param y
   * @param X
   * @return
   */
  FullTensor recoverTriples(SimpleMatrix y, SimpleMatrix X, double reg, boolean  doScale) {

    int N = X.numRows();
    int D = X.numCols();
    int D_ = D * (D+1) * (D+2) / 6;

    SimpleMatrix xScaling = MatrixFactory.ones(D);
    double yScaling = 1.0;
    if(doScale) {
      Pair<SimpleMatrix, SimpleMatrix> scaleInfo = MatrixOps.columnScale(X);
      X = scaleInfo.getValue0(); xScaling = scaleInfo.getValue1();
      scaleInfo = MatrixOps.rowScale(y);
      y = scaleInfo.getValue0(); yScaling = scaleInfo.getValue1().get(0);
    }

    // Consider only the upper triangular half of x x' (because it is symmetric) and construct a vector.
    double[][] A_ = new double[N][D_];

    int idx;
    for( int n = 0; n < N; n++ ) {
      idx = 0;
      for( int d1 = 0; d1 < D; d1++ ) {
        for( int d2 = 0; d2 <= d1; d2++ ) {
          for( int d3 = 0; d3 <= d2; d3++ ) {
            double multiplicity =
                    ( d1 == d2 && d2 == d3 ) ? 1 :
                            ( d1 == d2 || d2 == d3 || d1 == d3) ? 3 : 6;
            A_[n][idx++] = X.get(n, d1) * X.get(n,d2) * X.get(n,d3) * multiplicity;
          }
        }
      }
      assert( idx == D * (D+1) * (D+2) / 6 );
    }

    // Solve for the matrix
    SimpleMatrix A = new SimpleMatrix( A_ );
    //System.out.println(A);
    SimpleMatrix b = y.elementMult(y).elementMult(y).transpose();;
    // Regularize the matrix
    if( reg > 0.0 ) {
      Pair<SimpleMatrix, SimpleMatrix> Ab = regularize(A, b, reg);
      A = Ab.getValue0(); b = Ab.getValue1();
    }
    LogInfo.logsForce( "Condition Number for Triples: " + A.conditionP2());
    SimpleMatrix bEntries = A.solve(b);

    // Reconstruct $B$ from $x$
    double[][][] B = new double[D][D][D];
    idx = 0;
    for( int d1 = 0; d1 < D; d1++ ) {
      for( int d2 = 0; d2 <= d1; d2++ ) {
        for( int d3 = 0; d3 <= d2; d3++ ) {
          double value = bEntries.get(idx++);
          if(doScale)
            value = value * (yScaling * yScaling * yScaling) /
                    (xScaling.get(d1) * xScaling.get(d2) * xScaling.get(d2));
          B[d1][d2][d3] = B[d1][d3][d2] =
             B[d2][d1][d3] = B[d3][d2][d1] =
             B[d3][d1][d2] = B[d3][d2][d1] = value;
        }
      }
    }

    FullTensor Triples = new FullTensor(B);

    if( useLowRankRecovery ) {
      // Use low rank recovery to improve estimate.
      LogInfo.begin_track("low-rank-triples");
      if( analysis != null ) analysis.reportTriples0(Triples);

      ProximalGradientSolver solver = new ProximalGradientSolver();
      TensorRecovery problem = new TensorRecovery(y.elementMult(y).elementMult(y), X, 10 * traceReg);
      //DenseMatrix64F Triples_ = solver.optimize(problem, new DenseMatrix64F( D, D*D ), lowRankIters);
      DenseMatrix64F Triples_ = solver.optimize(problem, Triples.unfold(0).getMatrix(), lowRankIters);
      FullTensor.fold(0, Triples_, Triples);

      LogInfo.end_track("low-rank-triples");
    }

    return Triples;
  }

  FullTensor recoverTriples(SimpleMatrix y, SimpleMatrix X) {
    return recoverTriples( y, X, 0.0, false );
  }


  /**
   * Run the SpectralExperts algorithm on data $y$, $X$.
   * @param y
   * @param X
   * @return
   * @throws NumericalException
   */
  public Pair<SimpleMatrix, SimpleMatrix> run(int K, SimpleMatrix y, SimpleMatrix X) throws NumericalException, RecoveryFailure {
    int D = X.numCols();

    // Set the seed
    RandomFactory.setSeed( seed );

    // Recover Pairs and Triples moments by linear regression
    SimpleMatrix Pairs = recoverPairs( y, X, ridgeReg, scaleData );
    if( analysis != null ) analysis.reportPairs(Pairs);

    FullTensor Triples = recoverTriples(y, X, ridgeReg, scaleData );
    if( analysis != null ) analysis.reportTriples(Triples);

    if( useTensorPowerMethod ) {
      // Use the tensor power method to recover $\betas$.
      TensorMethod algo = new TensorMethod();
      Pair<SimpleMatrix, SimpleMatrix> pair = algo.recoverParameters( K, Pairs, Triples );
      // Somewhat of a "hack" to try and rescale the weights to sum to 1
      SimpleMatrix weights = pair.getValue0();
      SimpleMatrix betas = pair.getValue1();
      analysis.reportWeights(weights);
      analysis.reportBetas(betas);

      double sum = weights.elementSum();
      // Weird
      //betas = betas.scale( sum );
      weights = weights.scale( 1/sum );

      return new Pair<>(weights, betas);
    } else {
      // Use Algorithm B symmetric to recover the $\beta$
      MultiViewMixture algo = new MultiViewMixture();
      SimpleMatrix betas_ = algo.algorithmB( K, Pairs,  Pairs, Triples );
      // TODO: At some point we should compute this from betas_
      SimpleMatrix weights_ = MatrixFactory.zeros(K);
      return new Pair<>( weights_, betas_ );
    }
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
      Pair< Pair<SimpleMatrix, SimpleMatrix>, learning.models.MixtureOfExperts > data =
              MixtureOfExperts.readFromFile( inputPath );
      SimpleMatrix y = data.getValue0().getValue0();
      SimpleMatrix X = data.getValue0().getValue1();
      learning.models.MixtureOfExperts model = data.getValue1();
      enableAnalysis(model, true);
      analysis.checkDataSanity(y, X);

      LogInfo.logs("basis", MatrixOps.arrayToString(model.getNonLinearity().getExponents()));

      // Set K from the model if it hasn't been provided
      if( K < 1 )
        K = model.getK();
      int D = X.numCols();

      Pair<SimpleMatrix, SimpleMatrix> pi_betas_ = run( K, y, X );
      SimpleMatrix weights_ = pi_betas_.getValue0();
      SimpleMatrix betas_ = pi_betas_.getValue1();
      analysis.reportBetas(betas_);
      analysis.reportWeights(weights_);
      if( runEM ) {
        learning.em.MixtureOfExperts.Parameters initState = new learning.em.MixtureOfExperts.Parameters(
                this.K, MatrixFactory.ones(K).scale(1.0/K), betas_.transpose(), 0.1);
        learning.em.MixtureOfExperts emAlgo = new learning.em.MixtureOfExperts(K);
        learning.em.MixtureOfExperts.Parameters params = emAlgo.run( y, X, initState);
        SimpleMatrix betasEM = (new SimpleMatrix( params.betas )).transpose();
        SimpleMatrix weightsEM = MatrixFactory.fromVector(params.weights);
        analysis.reportBetasEM(betasEM);
        analysis.reportWeightsEM(weightsEM);
        System.out.printf( "%.4f %.4f %.4f %.4f %.4f\n",
                analysis.betasErr, analysis.betasEMErr,
                SpectralExpertsAnalysis.computeLoss(y, X, model.getBetas()),
                SpectralExpertsAnalysis.computeLoss(y, X, betas_),
                SpectralExpertsAnalysis.computeLoss(y, X, betasEM) );
      } else {
        System.out.printf( "%.4f %.4f %.4f\n", analysis.betasErr,
                SpectralExpertsAnalysis.computeLoss(y, X, model.getBetas()),
                SpectralExpertsAnalysis.computeLoss(y, X, betas_) );
      }
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

