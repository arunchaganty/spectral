/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.spectral.applications;

import learning.exceptions.RecoveryFailure;
import learning.exceptions.NumericalException;

import learning.Misc;
import learning.linalg.*;
import learning.models.MixtureOfExperts;

import learning.optimization.PhaseRecovery;
import learning.optimization.ProximalGradientSolver;
import learning.optimization.ProximalGradientSolver.LearningRate;
import learning.optimization.TensorRecovery;
import learning.optimization.MatlabProxy;
import learning.spectral.MultiViewMixture;

import learning.spectral.TensorMethod;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.*;

import fig.basic.LogInfo;
import fig.basic.Option;
import fig.exec.Execution;

import java.util.Date;
import java.lang.ClassNotFoundException;
import java.io.*;

/**
 * Spectral Experts.
 */
public class SpectralExperts implements Runnable {
	@Option(gloss = "Number of classes")
	public int K = 0;
	@Option(gloss = "Input filename", required=true)
	public String inputPath;
  @Option(gloss = "Number of samples to use (0 for all)")
  public double subsampleN = 0;
  @Option(gloss = "Forced Resample")
  public boolean forceResample = false;

  @Option(gloss = "Adjust the regularizer to be 1/sqrt(N) of the given number")
  public boolean adjustReg = true;
  @Option(gloss = "Ridge Regression Regularization (Pairs)")
  public double ridgeReg2 = 5e1;
  @Option(gloss = "Ridge Regression Regularization (Triples) (<0 => ridgeReg2*10)")
  public double ridgeReg3 = -1;
  @Option(gloss = "Trace Norm Regularization (Pairs)")
  public double traceReg2 = 1e-4;
  @Option(gloss = "Trace Norm Regularization (Triples)")
  public double traceReg3 = 0;
  @Option(gloss = "Scale Data?")
  public boolean scaleData = false;
  @Option(gloss = "Run spectral? (if false, just compute the moments)")
  public boolean runSpectral = true;

  @Option(gloss = "Run EM on spectral output?")
  public boolean runEM = false;
  @Option(gloss = "Run EM on spectral output?")
  public int emIters = 100;

  @Option(gloss = "Use tensor power method")
  public boolean useTensorPowerMethod = true;
  @Option(gloss = "Iterations for the tensor method")
  public int tensorMethodIters = 1000;
  @Option(gloss = "Attempts for the tensor method")
  public int tensorMethodAttempts = 10;

  @Option(gloss = "Use ridge regression")
  public boolean useRidgeRegression = false;
  @Option(gloss = "Use low rank recovery")
  public boolean useLowRankRecovery = true;
  @Option(gloss = "Use low rank recovery")
  public int lowRankIters = 1000;

  @Option(gloss = "Use Matlab for low rank recovery")
  public boolean useMatlab = true;
  @Option(gloss = "Path for matlab")
  public String matlabPath = System.getenv().get("HOME") + "/scr/spectral/matlab/";

  @Option(gloss = "Adjust the bias factor of <M_1, x>")
  public boolean adjustBias = true;

	@Option(gloss = "Remove Thirds")
	public boolean removeThirds = false;
	@Option(gloss = "Adjust noise")
	public double adjustNoise = -1;

	@Option(gloss = "Number of Threads to use")
	public int nThreads = 1;
	@Option(gloss = "Random seed")
	public int seed = (int)(new Date()).getTime();
	@Option(gloss = "Random seed")
	public int dataSeed = (int)(new Date()).getTime();

  public static class SpectralExpertsAnalysis {
    public boolean saveToExecution = false;

    public MixtureOfExperts model;
    public SimpleMatrix avgBetas;
    public SimpleMatrix Pairs;
    public FullTensor Triples;
    public SimpleMatrix betas;
    public SimpleMatrix weights;

    public double avgBetasErr;
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

      avgBetas = model.getWeights().mult( model.getBetas().transpose() ).transpose();
      Pairs = moments.getValue0();
      Triples = moments.getValue1();
      betas = model.getBetas();
      weights = model.getWeights();
    }

    public void reportCnd() {
      double cnd = MatrixOps.conditionNumber(Pairs, model.getK());
      double sk = MatrixOps.sigmak(Pairs, model.getK());
      Execution.putOutput( "cnd(M_2)",  cnd);
      Execution.putOutput( "sigma_k(M_2)", sk);

      LogInfo.logs( "cnd(M_2) " + cnd);
      LogInfo.logs( "sigma_k(M_2) " + sk);
    }

    public void reportAvg( final SimpleMatrix avgBetas_ ) {
      avgBetasErr = MatrixOps.diff(avgBetas, avgBetas_);
      if( saveToExecution ) {
        Execution.putOutput( "avgBetas", avgBetas );
        Execution.putOutput( "avgBetas_", avgBetas_ );
        Execution.putOutput( "avgBetasErr", avgBetasErr );
      } else {
        LogInfo.logsForce( "avgBetas: " + avgBetas );
        LogInfo.logsForce( "avgBetas_: " + avgBetas_ );
        LogInfo.logsForce( "avgBetasErr: " + avgBetasErr );
      }
      LogInfo.logsForce("avgBetas: " + avgBetasErr);
    }

    public void reportPairs0( final SimpleMatrix Pairs_ ) {
      PairsErr = MatrixOps.diff(Pairs, Pairs_ );
      if( saveToExecution ) {
        Execution.putOutput("Pairs0Err", PairsErr);
        Execution.putOutput( "Pairs0", Pairs_ );
      }
      LogInfo.logsForce("Pairs0: " + PairsErr);
    }

    public void reportPairs( final SimpleMatrix Pairs_ ) {
      PairsErr = MatrixOps.diff(Pairs, Pairs_);
      if( saveToExecution ) {
        Execution.putOutput( "Pairs", Pairs );
        Execution.putOutput( "Pairs_", Pairs_ );
        Execution.putOutput( "PairsErr", PairsErr );
      } else {
        LogInfo.logsForce( "Pairs: " + Pairs );
        LogInfo.logsForce( "Pairs_: " + Pairs_ );
        LogInfo.logsForce( "PairsErr: " + PairsErr );
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
        Execution.putOutput( "Triples0", Triples_ );
      }
      LogInfo.logsForce("Triples0: " + TriplesErr);
    }

    public void reportTriples( final FullTensor Triples_ ) {
      TriplesErr = MatrixOps.diff(Triples, Triples_ );
      if( saveToExecution ) {
        Execution.putOutput("TriplesErr", TriplesErr);
        Execution.putOutput( "Triples", Triples );
        Execution.putOutput( "Triples_", Triples_ );
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

  SimpleMatrix recoverMeans(SimpleMatrix y, SimpleMatrix X) {
    return X.solve(y.transpose());
  }

  /**
   *
   * @param y is expect to have already been squared
   * @param X
   * @return
   */
  SimpleMatrix recoverPairsByRidgeRegression(SimpleMatrix y, SimpleMatrix X) {
    int N = X.numRows();
    // We can't handle extracting whole of X, so let's just take a smaller subsdet, since this is for initialization anyways.
    if( N > (int)1e6 ) {
      LogInfo.logsForce("Ridge regression can handle only 1e6, so truncating and using that much data" );
      N = (int) 1e6;
      y = y.extractMatrix(0, SimpleMatrix.END, 0, N);
    }

    int D = X.numCols();
    int D_ = D * (D+1) / 2;
    LogInfo.begin_track("ridge-regression-pairs");

    // Normalize the data
    SimpleMatrix xScaling = MatrixFactory.ones(D);
    double yScaling = 1.0;
    //if(scaleData) {
    //  Pair<SimpleMatrix, SimpleMatrix> scaleInfo = MatrixOps.columnScale(X);
    //  X = scaleInfo.getValue0(); xScaling = scaleInfo.getValue1();
    //  scaleInfo = MatrixOps.rowScale(y);
    //  y = scaleInfo.getValue0(); yScaling = scaleInfo.getValue1().get(0);
    //}

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
    SimpleMatrix b = y.transpose();
    SimpleMatrix bEntries;
    // Regularize the matrix
    if( ridgeReg2 > 0.0 ) {
      Pair<SimpleMatrix, SimpleMatrix> Ab = regularize(A, b, ridgeReg2);
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
    //if(scaleData) { // Unscale B
    //  for( int d = 0; d < D; d++ ) {
    //    for( int d_ = 0; d_ < D; d_++ ) {
    //      double scaling = (yScaling * yScaling) / (xScaling.get(d) * xScaling.get(d_));
    //      B.set( d, d_, B.get(d, d_) * scaling);
    //    }
    //  }
    // }
    LogInfo.end_track("ridge-regression-pairs");

    return B;
  }
  /**
   * Recover the second-order tensor \beta \otimes \beta by linear regression.
   * @param y
   * @param X
   * @return
   */
  SimpleMatrix recoverPairs(SimpleMatrix y, SimpleMatrix X) {
    int N = X.numRows();
    int D = X.numCols();

    SimpleMatrix Pairs;
    {
      y = y.elementMult(y);
      double sigma2 = (analysis != null) ? analysis.model.getSigma2() : 0.0;
      if( sigma2 != 0 ) {
        DenseMatrix64F y_ = y.getMatrix();
        CommonOps.add(y_, -sigma2);
        y = SimpleMatrix.wrap(y_);
      }
    }

    if( useRidgeRegression ) {
      Pairs = recoverPairsByRidgeRegression(y, X);
    } else {
      //Pairs = RandomFactory.symmetric( D ).scale(0.01);
      Pairs = analysis.Pairs; //RandomFactory.symmetric( D ).scale(0.01);
    }
    if( useLowRankRecovery ) {
      // Use low rank recovery to improve estimate.
      LogInfo.begin_track("low-rank-pairs");
      // Report the performance before low-rankification
      if( analysis != null ) analysis.reportPairs0(Pairs);

      ProximalGradientSolver solver = new ProximalGradientSolver();
      // Tweak the response variables to give an unbiased estimate
      PhaseRecovery problem = new PhaseRecovery(y, X, traceReg2);
      DenseMatrix64F Pairs_ = solver.optimize(problem, Pairs.getMatrix(),
          new LearningRate(LearningRate.Type.CONSTANT, 0.1),
          (int) (lowRankIters * 0.1), 1e0);
      Pairs_ = solver.optimize(problem, Pairs_,
          new LearningRate(LearningRate.Type.BY_SQRT_T, 1.0),
          lowRankIters);
      Pairs = SimpleMatrix.wrap(Pairs_);

      LogInfo.end_track("low-rank-pairs");
    }
    //Pairs = MatrixOps.approxk(Pairs, K );

    return Pairs;
  }

  /**
   * Recover the second-order tensor \beta \otimes \beta by linear regression.
   * @param y - assume you have y^3
   * @param X
   * @return
   */
  FullTensor recoverTriplesByRegression(SimpleMatrix y, SimpleMatrix X) {
    int N = X.numRows();
    // We can't handle extracting whole of X, so let's just take a smaller subsdet, since this is for initialization anyways.
    if( N > (int)1e6 ) {
      LogInfo.logsForce("Ridge regression can handle only 1e6, so truncating and using that much data" );
      N = (int) 1e6;
      y = y.extractMatrix(0, SimpleMatrix.END, 0, N);
    }
    int D = X.numCols();
    int D_ = D * (D+1) * (D+2) / 6;

    LogInfo.begin_track("ridge-regression-triples");

    SimpleMatrix xScaling = MatrixFactory.ones(D);
    double yScaling = 1.0;
    //if(scaleData) {
    //  Pair<SimpleMatrix, SimpleMatrix> scaleInfo = MatrixOps.columnScale(X);
    //  X = scaleInfo.getValue0(); xScaling = scaleInfo.getValue1();
    //  scaleInfo = MatrixOps.rowScale(y);
    //  y = scaleInfo.getValue0(); yScaling = scaleInfo.getValue1().get(0);
    //}

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
    SimpleMatrix b = y.transpose();
    // Regularize the matrix
    if( ridgeReg3 > 0.0 ) {
      Pair<SimpleMatrix, SimpleMatrix> Ab = regularize(A, b, ridgeReg3);
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
          //if(scaleData)
          //  value = value * (yScaling * yScaling * yScaling) /
          //          (xScaling.get(d1) * xScaling.get(d2) * xScaling.get(d2));
          B[d1][d2][d3] = value;
          B[d1][d3][d2] = value;
          B[d2][d1][d3] = value;
          B[d2][d3][d1] = value;
          B[d3][d1][d2] = value;
          B[d3][d2][d1] = value;
        }
      }
    }
    FullTensor B_ = new FullTensor(B);
    LogInfo.end_track("ridge-regression-triples");

    return B_;
  }

  FullTensor recoverTriples(SimpleMatrix y, SimpleMatrix X, SimpleMatrix avgBetas) {
    int N = X.numRows();
    int D = X.numCols();

    FullTensor Triples;
    // Adjust y
    {
      y = y.elementMult(y).elementMult(y);
      double sigma2 = (adjustBias && analysis != null) ? analysis.model.getSigma2() : 0.0;
      SimpleMatrix bias = X.mult(avgBetas).transpose();
      y = y.plus( -3*sigma2, bias );
    }
    if( useRidgeRegression ) {
      Triples = recoverTriplesByRegression(y, X);
    } else {
      //Triples = RandomFactory.symmetricTensor( 1, D ).scale(0.01);
      Triples = analysis.Triples;
    }

    if( useLowRankRecovery ) {
      // Use low rank recovery to improve estimate.
      LogInfo.begin_track("low-rank-triples");
      if( analysis != null ) analysis.reportTriples0(Triples);

      ProximalGradientSolver solver = new ProximalGradientSolver();
      TensorRecovery problem = new TensorRecovery(y, X, traceReg3);
      DenseMatrix64F Triples_ = solver.optimize(problem, Triples.unfold(0).getMatrix(), 
          new LearningRate(LearningRate.Type.CONSTANT, 0.1),
          (int) (lowRankIters * 0.1), 1e0);
      Triples_ = solver.optimize(problem, Triples_, 
          new LearningRate(LearningRate.Type.BY_SQRT_T, 1.0),
          lowRankIters);
      FullTensor.fold(0, Triples_, Triples);
      LogInfo.end_track("low-rank-triples");
    }
    //MatrixOps.approxk(Triples,K);

    return Triples;
  }

  /**
   * Recover Pairs and Triples via a system call to Matlab
   */
  public Pair<SimpleMatrix, FullTensor> recoverMomentsByMatlab( SimpleMatrix y, SimpleMatrix X ) {
    int N = X.numRows();
    int D = X.numCols();
    double sigma2 = (analysis != null) ? analysis.model.getSigma2() : 0.0;

    try {
    // Create a temporary directory for all this stuff
    File dir = Misc.createTemporaryDirectory( "spectral-experts" );

    // Save matrices
    MatlabProxy.save( new File( dir, "y.txt" ), y.transpose() );
    MatlabProxy.save( new File( dir, "X.txt" ), X );
    MatlabProxy.save( new File( dir, "lambda2.txt" ), traceReg2 );
    MatlabProxy.save( new File( dir, "lambda3.txt" ), traceReg3 );
    MatlabProxy.save( new File( dir, "sigma2.txt" ), sigma2 );

    // Run matlab
    MatlabProxy.run( matlabPath, String.format("sdpB2('%s')", dir) );
    MatlabProxy.run( matlabPath, String.format("sdpB3('%s')", dir) );

    // Get output
    SimpleMatrix B2 = MatlabProxy.load( new File( dir, "B2.txt" ) );
    SimpleMatrix B3_ = MatlabProxy.load( new File( dir, "B3.txt" ) );
    FullTensor B3 = FullTensor.reshape( B3_, new int[]{ D, D, D } );

    dir.delete();
   
    return new Pair<>(B2, B3); 
    } catch( IOException e ) {
      throw new RuntimeException();
    } catch( InterruptedException e ) {
      throw new RuntimeException();
    }
  }

  public Pair<SimpleMatrix, FullTensor> recoverMomentsByMatlabSpecial( SimpleMatrix y2, SimpleMatrix y3, SimpleMatrix X ) {
    int N = X.numRows();
    int D = X.numCols();
    double sigma2 = (analysis != null) ? analysis.model.getSigma2() : 0.0;

    try {
    // Create a temporary directory for all this stuff
    File dir = Misc.createTemporaryDirectory( "spectral-experts" );

    // Save matrices
    MatlabProxy.save( new File( dir, "X.txt" ), X );
    MatlabProxy.save( new File( dir, "lambda2.txt" ), traceReg2 );
    MatlabProxy.save( new File( dir, "lambda3.txt" ), traceReg3 );
    MatlabProxy.save( new File( dir, "sigma2.txt" ), 0 ); //sigma2 );

    MatlabProxy.save( new File( dir, "y.txt" ), y2.transpose() );
    MatlabProxy.run( matlabPath, String.format("sdpB2('%s')", dir) );

    // Run matlab
    MatlabProxy.save( new File( dir, "y.txt" ), y3.transpose() );
    MatlabProxy.run( matlabPath, String.format("sdpB3('%s')", dir) );

    // Get output
    SimpleMatrix B2 = MatlabProxy.load( new File( dir, "B2.txt" ) );
    SimpleMatrix B3_ = MatlabProxy.load( new File( dir, "B3.txt" ) );
    System.out.println(B3_);
    FullTensor B3 = FullTensor.reshape( B3_, new int[]{ D, D, D } );

    dir.delete();
   
    return new Pair<>(B2, B3); 
    } catch( IOException e ) {
      throw new RuntimeException();
    } catch( InterruptedException e ) {
      throw new RuntimeException();
    }
  }


  /**
   * Run the SpectralExperts algorithm on data $y$, $X$.
   * @param y
   * @param X
   * @return
   * @throws NumericalException
   */
  public Pair<SimpleMatrix, SimpleMatrix> run(int K, SimpleMatrix y, SimpleMatrix X) throws NumericalException, RecoveryFailure {
    int N = X.numRows();
    int D = X.numCols();

    if( ridgeReg3 < 0 ) ridgeReg3 = ridgeReg2 * 10;
    if( traceReg3 < 0 ) traceReg3 = traceReg2 * 10;
    // Adjust the regularizer for N
    if(adjustReg) {
      // Ridge regression is auto-scaled because we add lambda to XX^T
      traceReg2 /= Math.sqrt(N); //N; 
      traceReg3 /= Math.sqrt(N); //N; 
    }
    LogInfo.logs("Ridge Reg: " + ridgeReg2 + ", " + ridgeReg3 );
    LogInfo.logs("Trace Reg: " + traceReg2 + ", " + traceReg3 );

    // Set the seed
    RandomFactory.setSeed( seed );

    SimpleMatrix Pairs;
    FullTensor Triples;
    if( useMatlab ) {
      Pair<SimpleMatrix,FullTensor> moments = recoverMomentsByMatlab( y, X );
      Pairs = moments.getValue0();
      if( analysis != null ) analysis.reportPairs(Pairs);

      Triples = moments.getValue1();
      if( analysis != null ) analysis.reportTriples(Triples);
    } else {
      // Recover the first moment
      SimpleMatrix avgBetas = recoverMeans(y, X);
      if( analysis != null ) analysis.reportAvg(avgBetas);

      // Recover Pairs and Triples moments by linear regression
      Pairs = recoverPairs( y, X );
      if( analysis != null ) analysis.reportPairs(Pairs);

      Triples = recoverTriples(y, X, avgBetas);
      if( analysis != null ) analysis.reportTriples(Triples);
    }

    if( runSpectral ) {
      if( useTensorPowerMethod ) {
        // Use the tensor power method to recover $\betas$.
        TensorMethod algo = new TensorMethod( tensorMethodIters, tensorMethodAttempts );
        Pair<SimpleMatrix, SimpleMatrix> pair = algo.recoverParameters( K, Pairs, Triples );
        // Somewhat of a "hack" to try and rescale the weights to sum to 1
        SimpleMatrix weights = pair.getValue0();
        SimpleMatrix betas = pair.getValue1();
        analysis.reportWeights(weights);
        analysis.reportBetas(betas);

        // Normalize the weights at the very least
        double sum = weights.elementSum();
        weights = weights.scale( 1/sum );

        return new Pair<>(weights, betas);
      } else {
        // Use Algorithm B symmetric to recover the $\beta$
        MultiViewMixture algo = new MultiViewMixture();
        SimpleMatrix betas_ = algo.algorithmB(K, Pairs, Pairs, Triples);
        // TODO: At some point we should compute this from betas_
        SimpleMatrix weights_ = MatrixFactory.zeros(1, K);
        return new Pair<>( weights_, betas_ );
      }
    }
    else return null;
  }

  public void enableAnalysis(MixtureOfExperts model, boolean saveToExecution) {
    analysis = new SpectralExpertsAnalysis(model);
    analysis.saveToExecution = saveToExecution;
  }
  public void enableAnalysis(MixtureOfExperts model) {
    enableAnalysis(model, false);
  }

//  // Save data to a file
//  private void setData(SimpleMatrix y, SimpleMatrix X) {
//
//  }
//
//  private Pair<SimpleMatrix, SimpleMatrix> getData() {
//
//
//  }

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
      data = null;

      // Choose a subset of the data
      int N = X.numRows();
      if( removeThirds || adjustNoise > 0 || forceResample || subsampleN > N ) {
        // Set the seed
        RandomFactory.setSeed( dataSeed );
        LogInfo.logsForce("Resampling data");
        // Possibly sample data with thirds removed
        model.removeThirds = removeThirds;
        model.sigma2 = ( adjustNoise > 0 ) ?  adjustNoise : (model.getSigma2());
        y = null; X = null; data = null;
        Pair<SimpleMatrix, SimpleMatrix> yX = model.sample( (int) subsampleN );
        y = yX.getValue0();
        X = yX.getValue1();
      } else if( subsampleN > 0 ) {
        y = y.extractMatrix(0, SimpleMatrix.END, 0, (int) subsampleN);
        X = X.extractMatrix(0, (int) subsampleN, 0, SimpleMatrix.END);
      }
      N = X.numRows();
      analysis.checkDataSanity(y, X);

      LogInfo.logs("basis", MatrixOps.arrayToString(model.getNonLinearity().getExponents()));

      analysis.reportCnd();
      // Set K from the model if it hasn't been provided
      if( K < 1 )
        K = model.getK();
      int D = X.numCols();

      Pair<SimpleMatrix, SimpleMatrix> pi_betas_ = run( K, y, X );
      if( runSpectral ) {
        SimpleMatrix weights_ = pi_betas_.getValue0();
        SimpleMatrix betas_ = pi_betas_.getValue1();
        analysis.reportBetas(betas_);
        analysis.reportWeights(weights_);
        if( runEM ) {
          learning.em.MixtureOfExperts.Parameters initState = new learning.em.MixtureOfExperts.Parameters(
                  this.K, MatrixFactory.ones(K).scale(1.0/K), betas_.transpose(), 0.1);
          learning.em.MixtureOfExperts emAlgo = new learning.em.MixtureOfExperts(K); emAlgo.iters = emIters;
          learning.em.MixtureOfExperts.Parameters params = emAlgo.run( y, X, initState);
          SimpleMatrix betasEM = (new SimpleMatrix( params.betas )).transpose();
          SimpleMatrix weightsEM = MatrixFactory.fromVector(params.weights);
          analysis.reportBetasEM(betasEM);
          analysis.reportWeightsEM(weightsEM);
          System.out.printf( "%.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                  analysis.betasErr, analysis.betasEMErr,
                  SpectralExpertsAnalysis.computeLoss(y, X, model.getBetas()),
                  SpectralExpertsAnalysis.computeLoss(y, X, betas_),
                  SpectralExpertsAnalysis.computeLoss(y, X, betasEM),
                  analysis.PairsErr,
                  analysis.TriplesErr
              );
        } else {
          System.out.printf( "%.4f %.4f %.4f %.4f %.4f\n", analysis.betasErr,
                  SpectralExpertsAnalysis.computeLoss(y, X, model.getBetas()),
                  SpectralExpertsAnalysis.computeLoss(y, X, betas_),
                  analysis.PairsErr,
                  analysis.TriplesErr
              );
        }
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

