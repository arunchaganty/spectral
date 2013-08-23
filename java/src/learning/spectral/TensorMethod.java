package learning.spectral;

import learning.linalg.*;
import learning.data.*;

import org.javatuples.*;
import java.util.*;
import java.io.*;

import fig.basic.Option;
import fig.basic.OptionSet;
import fig.basic.LogInfo;
import fig.exec.Execution;

import org.ejml.simple.SimpleMatrix;
import org.apache.commons.io.FilenameUtils;

/**
 * Recover parameters from symmetric multi-views using the Tensor powerup method
 */
public class TensorMethod implements Runnable {
  @Option(gloss="Number of iterations to run the power method to convergence")
  public int iters = 1000;
  @Option(gloss="Number of attempts to find good eigen-vectors")
  public int attempts = 10;
  @Option(gloss="Random number generator for tensor method and random projections")
  Random rnd = new Random(1);

  public TensorMethod() {}
  public TensorMethod(int iters, int attempts) {
    this.iters = iters;
    this.attempts = attempts;
  }

  /**
   * The tensor factorization method is just finding
   * the eigenvalues/eigenvectors of the tensor Triples.
   * @param K
   * @param Pairs
   * @param Triples
   * @return
   */
  public Pair<SimpleMatrix,SimpleMatrix> recoverParameters( int K, SimpleMatrix Pairs, FullTensor Triples ) {
    Pair<SimpleMatrix, SimpleMatrix> pair = TensorFactorization.eigendecompose(Triples, Pairs, K, attempts, iters);
    return pair;
  }

  /**
   * The tensor factorization method is just finding
   * the eigenvalues/eigenvectors of the tensor Triples.
   * @param K
   * @param Pairs
   * @param Triples
   * @return
   */
  public Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix> 
      recoverParameters( int K, SimpleMatrix M12, 
          SimpleMatrix M13, SimpleMatrix M23, 
          FullTensor M123 ) {
    LogInfo.begin_track("recovery-asymmetric");
    // Symmetrize views to get M33, M333
    Pair<SimpleMatrix,FullTensor> symmetricMoments = symmetrizeViews( K, M12, M13, M23, M123 );
    SimpleMatrix Pairs = symmetricMoments.getValue0();
    FullTensor Triples = symmetricMoments.getValue1();

    // Tensor Factorize to get w, M3
    Pair<SimpleMatrix, SimpleMatrix> pair = recoverParameters( K, Pairs, Triples );
    SimpleMatrix pi = pair.getValue0();
    SimpleMatrix M3 = pair.getValue1();

    // Invert M3 to get M1 and M2.

    SimpleMatrix inversion = (M3.transpose()).pseudoInverse().mult( MatrixFactory.diag( MatrixOps.reciprocal(pi) ) );

    SimpleMatrix M1 = M13.mult( inversion );
    SimpleMatrix M2 = M23.mult( inversion );
    LogInfo.end_track("recovery-asymmetric");

    return new Quartet<>( pi, M1, M2, M3 );
  }
  public Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix> 
      recoverParameters( int K, 
        Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,FullTensor> moments ) {
        return recoverParameters( K,
            moments.getValue0(),
            moments.getValue1(),
            moments.getValue2(),
            moments.getValue3() );
    }


  /**
   * Reduce the 3-view mixture model to 1 symmetric view.
   */
  public static Pair<SimpleMatrix,FullTensor> symmetrizeViews( int K, 
        SimpleMatrix M12, 
        SimpleMatrix M13, 
        SimpleMatrix M23, 
        FullTensor M123 ) {
    LogInfo.begin_track("symmetrize-views");

    int D = M12.numRows();

    Triplet<SimpleMatrix,SimpleMatrix,SimpleMatrix> U1WU2 = MatrixOps.svdk( M12, K );
    Triplet<SimpleMatrix,SimpleMatrix,SimpleMatrix> U2WU3 = MatrixOps.svdk( M23, K );
    SimpleMatrix U1 = U1WU2.getValue0(); // d x k
    SimpleMatrix U2 = U1WU2.getValue2();
    SimpleMatrix U3 = U2WU3.getValue2();

    MatrixOps.printSize( U1 );

    assert( U1.numRows() == D );
    assert( U1.numCols() == K );

    // \tilde M_{12} = U_1^T M_{12} U_2
    SimpleMatrix M12_ = // k x k
      U1.transpose().mult // k x d
      (M12).mult(U2); // d x k
    SimpleMatrix M12_i = M12_.invert();

    // P = M_{31} U_1^T (\tilde M_{21})^{-1} U_2 M_{23}
    SimpleMatrix Pairs = // d x d 
        M13.transpose().mult // d x d 
        (U1).mult // d x k 
         (M12_i.transpose()).mult // k x k
         (U2.transpose()).mult // k x d
         (M23);  // d x d 

    // T = M_{123}( M_{32} U_2^T (\tilde M_{12})^{-1} U_1, M_{31} U_1^T (\tilde M_{21})^{-1} U_2,  I )
    FullTensor Triples = 
      M123.rotate(
        (M23.transpose().mult // d x d
          (U2).mult // d x k
          (M12_i).mult // k x k
          (U1.transpose())).transpose(), // k x d
        (M13.transpose().mult
          (U1).mult
          (M12_i.transpose()).mult // k x k
          (U2.transpose())).transpose(), // k x d
        MatrixFactory.eye(D)
        );

    LogInfo.end_track("symmetrize-views");
    return new Pair<>(Pairs, Triples);
  }

  /**
   * Symmetrizes a (large) 3-view mixture model and whitens it.
   * @param K - number of components
   * @param obj - object with computable moments
   * @return - a whitened tensor Triples_3
   */
  public Pair<SimpleMatrix,FullTensor> randomizedTensorRecovery( int K, ComputableMoments obj ) {
    int p = 2*K;
    // First, compute the range
    SimpleMatrix Q2 = MatrixOps.randomizedRangeFinder(obj.computeP12(), p, rnd);
    // Now do the SVD to compute the top-k range of both left and right spaces; truncates everything to $K$
    Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> triplet = MatrixOps.randomizedSvd(obj.computeP12(), Q2, K);
    SimpleMatrix Q1 = triplet.getValue0(); Q2 = triplet.getValue2(); // K x D
    // Compute the range for view 3
    SimpleMatrix Q3 = MatrixOps.randomizedRangeFinder(obj.computeP13(), p, rnd); // p x p

    // Next, project P12, P32 and P13 to their ranges
    SimpleMatrix P12 = obj.computeP12().doubleMultiply(Q1, Q2); // K x K
    SimpleMatrix P32 = obj.computeP32().doubleMultiply(Q3, Q2); // p x K
    SimpleMatrix P13 = obj.computeP13().doubleMultiply(Q1, Q3); // K x p
    // Compute P12inv
    SimpleMatrix P12inv = P12.invert();

    // Finally, compute Pairs3, and whiten.
    // P = U_3 P_{31} U_1^T (\tilde P_{21})^{-1} U_2 P_{23} U_3
    SimpleMatrix Pairs = P32.mult(P12inv).mult(P13);
    SimpleMatrix W = MatrixOps.whitener(Pairs, K);

    // Use this to compute the whitened symmetrized Triples.
    // T( W (P32 (P12)^-1), ((P12^-1)P13)^T, I )
    SimpleMatrix first = W.transpose().mult(P32.mult(P12inv).mult(Q1.transpose()));
    SimpleMatrix second = W.transpose().mult((Q2.mult(P12inv).mult(P13)).transpose());
    SimpleMatrix third = W.transpose().mult(Q3.transpose());

    FullTensor tensor = obj.computeP123().multiply123(first.transpose(), second.transpose(), third.transpose());

    return Pair.with(W,tensor);
  }

  /**
   * Compute the whitened full tensor
   * @param K - Number of components to keep in the whitener
   * @param obj - object with computable moments
   * @return - Pair of whitener and whitened tensor
   */
  public Pair<SimpleMatrix, FullTensor> randomizedSymmetricTensorRecovery( int K, ComputableMoments obj ) {
    int p = 2*K;
    // First, compute the ranges
    SimpleMatrix Q = MatrixOps.randomizedRangeFinder(obj.computeP12(), p, rnd);

    // Compute "projected" Pairs, and whiten.
    SimpleMatrix Pairs = obj.computeP12().doubleMultiply(Q, Q);
    SimpleMatrix W = Q.mult(MatrixOps.whitener(Pairs, K));

    FullTensor tensor = obj.computeP123().multiply123(W, W, W);
    return Pair.with(W, tensor);
  }

  /**
   * Extract parameters right from a data sequence.
   */
  public Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix> 
      recoverParameters( int K, int D, Iterator<double[][]> dataSeq ) {

      // Compute moments 
      MomentAggregator agg = new MomentAggregator(D, dataSeq);
      agg.run();

      // In our case, the moments are symmetric.
      //Pair<SimpleMatrix,SimpleMatrix> params = recoverParameters( K, agg.getMoments().getValue0(),
      //  agg.getMoments().getValue3() );
      Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix> params = recoverParameters( K, agg.getMoments() );

      return new Quartet<>( params.getValue0(), 
          params.getValue1(),
          params.getValue2(),
          params.getValue3() );

      //return recoverParameters( K, agg.getMoments() );
    }

  public class ComputationOptions {
    @OptionSet(name="moments")
      public MomentComputer.Options momentOpts = new MomentComputer.Options();
    @Option(gloss="Path to file containing moments information", required=true)
      public String momentsPath;
    @Option(gloss="Number of clusters to use for the factorization", required=true)
      public int K;
  }
  public ComputationOptions opts = new ComputationOptions();

  @SuppressWarnings("unchecked")
  public void run() {
    try { 
      int K = opts.K;
      MomentComputer.Options momentOpts = opts.momentOpts;
      // Read moments from path. 
      LogInfo.begin_track("file-input");
      ObjectInputStream in = new ObjectInputStream( new FileInputStream( opts.momentsPath ) ); 
      momentOpts.randomProjSeed = (int) in.readObject();
      Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,FullTensor> moments = 
        (Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,FullTensor>)
        in.readObject();
      momentOpts.randomProjDim = moments.getValue0().numRows();
      in.close();
      assert( moments.getValue0().numCols() == momentOpts.randomProjDim );

      // Read corpus from path. Use seed from projection.
      // This is mainly to "unproject"
      ProjectedCorpus PC = new ProjectedCorpus( 
          Corpus.parseText( momentOpts.dataPath, momentOpts.mapPath ),
          momentOpts.randomProjDim, momentOpts.randomProjSeed );
      LogInfo.end_track("file-input");

      // Run the TensorFactorization algorithm
      Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix> 
        parameters = recoverParameters( K, moments );
      SimpleMatrix pi = parameters.getValue0();
      SimpleMatrix M1 = parameters.getValue1();
      SimpleMatrix M2 = parameters.getValue2();
      SimpleMatrix M3 = parameters.getValue3();

      // Unproject.
      LogInfo.begin_track("unfeaturization");
      MatrixOps.printSize( pi );
      MatrixOps.printSize( M1 );
      MatrixOps.printSize( PC.Pinv );
      M1 = PC.unfeaturize( M1 );
      M2 = PC.unfeaturize( M2 );
      M3 = PC.unfeaturize( M3 );
      LogInfo.end_track();

      // Write.
      LogInfo.begin_track("saving");
      parameters = new Quartet<>(pi, M1, M2, M3);
      String outFilename = Execution.getFile( FilenameUtils.getBaseName(momentOpts.dataPath) + 
          "-" + momentOpts.randomProjDim + "-" + momentOpts.randomProjSeed + ".parameters" );
      ObjectOutputStream out = new ObjectOutputStream( new FileOutputStream( outFilename ) ); 
      out.writeObject(parameters);
      out.close();
      LogInfo.end_track();
    } catch( ClassNotFoundException | IOException e ) {
      LogInfo.logsForce( e );
    }
  }

  /**
   * Main routine reads moments from a file and runs the tensor factorization
   * algorithm. 
   * - It can also "unproject" if needed.
   */
  public static void main(String[] args) {
    TensorMethod method = new TensorMethod(); 
    Execution.run(args, method, method.opts);
  }
}
