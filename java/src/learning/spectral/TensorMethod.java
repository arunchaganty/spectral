package learning.spectral;

import fig.basic.IOUtils;
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
public class TensorMethod {
  @Option(gloss="Number of iterations to run the power method to convergence")
  public int iters = 100;
  @Option(gloss="Number of attempts to find good eigen-vectors")
  public int attempts = 300;
  @Option(gloss="Random number generator for tensor method and random projections")
  public Random rnd = new Random();
  @Option(gloss="Oversampling factor for the random projection")
  public double oversamplingFactor = 2.0;

  public enum Method {
    PowerMethod,
    GradientPowerMethod
  };
  @Option(gloss="Method for tensor factorization")
  public Method method = Method.PowerMethod;

  public TensorMethod() {
  }
  public TensorMethod(int iters, int attempts) {
    this();
    this.iters = iters;
    this.attempts = attempts;
  }
  TensorFactorizationAlgorithm getMethod() {
    TensorFactorizationAlgorithm tf;
    switch(method) {
      case PowerMethod:
        tf = new TensorPowerMethod();
        break;
      case GradientPowerMethod:
        tf = new TensorGradientDescent();
        break;
      default:
        throw new RuntimeException("Invalid method");
    }
    return tf;
  }

  /**
   * Whiten the tensor Triples using Pairs.
   * @return - (unwhitening matrix, whitened tensor).
   */
  protected Pair<SimpleMatrix, FullTensor> whiten(int K, SimpleMatrix Pairs, FullTensor Triples) {
    // Whiten
    LogInfo.begin_track("whitening");
    LogInfo.log( "k(P): " + MatrixOps.conditionNumber(Pairs,K) );
    SimpleMatrix W = MatrixOps.whitener(Pairs, K);
    SimpleMatrix Winv = MatrixOps.colorer(Pairs, K);
    FullTensor Tw = Triples.rotate(W,W,W);
    LogInfo.end_track("whitening");

    return Pair.with(Winv, Tw);
  }

  /**
   * Unwhiten the eigenvectors/eigenvalues.
   */
  protected Pair<SimpleMatrix, SimpleMatrix> unwhiten(SimpleMatrix eigenvalues, SimpleMatrix eigenvectors, SimpleMatrix Winv) {
    int K = eigenvectors.numCols();
    int D = eigenvectors.numRows();
    // Color them in again
    LogInfo.begin_track("un-whitening");

    // Scale the vectors by 1/sqrt(eigenvalues);
    {
      for( int d = 0; d < D; d++ )
          for( int k = 0; k < K; k++ )
            eigenvectors.set(d,k, eigenvectors.get(d,k) * eigenvalues.get(k) ) ;
    }
    eigenvectors = Winv.mult( eigenvectors );

    // Eigenvalues are w^{-1/2}; w is what we want.
    for(int i = 0; i < K; i++)
      eigenvalues.set( i, Math.pow(eigenvalues.get(i), -2) );
    LogInfo.end_track("un-whitening");

    return Pair.with(eigenvalues, eigenvectors);
  }

  /**
   * The tensor factorization method is just finding
   * the eigenvalues/eigenvectors of the tensor Triples.
   * Whiten Triples before the eigendecomposition
   * @param K - Rank of the model
   * @param Pairs - 2nd order moments for factorization
   * @param Triples - 3rd order moments.
   * @return - (Eigenvalues, Eigenvectors)
   * // TODO: Make private
   */
  public Pair<SimpleMatrix,SimpleMatrix> recoverParameters( int K, SimpleMatrix Pairs, FullTensor Triples ) {
    // This is actually true!
//    SimpleMatrix Pairs_ = MatrixOps.condenseMoment(Triples);
//    LogInfo.logsForce( "%f", MatrixOps.diff(Pairs, Pairs_));
//    assert  MatrixOps.allclose(Pairs, Pairs_);

    // print error notes
    Execution.putOutput("sigmak", MatrixOps.sigmak(Pairs, K));
    Execution.putOutput("condition-number", MatrixOps.conditionNumber(Pairs, K));

    Pair<SimpleMatrix,FullTensor> whitened = whiten(K, Pairs, Triples);
    SimpleMatrix Winv = whitened.getValue0();
    FullTensor Tw = whitened.getValue1();


    Pair<SimpleMatrix, SimpleMatrix> pair = getMethod().symmetricFactorize(Tw, K);
    SimpleMatrix eigenvalues = pair.getValue0(); SimpleMatrix eigenvectors = pair.getValue1();

    pair = unwhiten(eigenvalues, eigenvectors, Winv);
    eigenvalues = pair.getValue0(); eigenvectors = pair.getValue1();

    FullTensor T_ = FullTensor.fromDecomposition( eigenvalues, eigenvectors );
    Execution.putOutput("tensor-recovery-error", MatrixOps.diff(Triples, T_));

    return Pair.with(eigenvalues, eigenvectors);
  }
  public Pair<SimpleMatrix,SimpleMatrix> recoverParameters( int K, FullTensor Triples ) {
    SimpleMatrix Pairs = MatrixOps.condenseMoment(Triples);
    return recoverParameters(K, Pairs, Triples);
  }

  /**
   * With un-symmetric views, this method will first symmetrize the views to recover the parameters M_3,
   * and then do some post-processing to recover the remaining parameters M_1 and M_2.
   * @param K - Rank of model
   * @param M13 - Pairwise moments
   * @param M12 - Pairwise moments
   * @param M32 - Pairwise moments
   * @param M123 - Tensor
   * @return - (weights, M1, M2, M3).
   */
  public Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix>
      recoverParameters( int K, SimpleMatrix M13,
          SimpleMatrix M12, SimpleMatrix M32,
          FullTensor M123 ) {
    LogInfo.begin_track("recovery-asymmetric");
    LogInfo.dbg(M13);
    LogInfo.dbg(M12);
    LogInfo.dbg(M32);
    // Symmetrize views to get M33, M333
    Pair<SimpleMatrix,FullTensor> symmetricMoments = symmetrizeViews( K, M13, M12, M32, M123 );
    SimpleMatrix Pairs = symmetricMoments.getValue0();
    FullTensor Triples = symmetricMoments.getValue1();

    // Tensor Factorize to get w, M3
    Pair<SimpleMatrix, SimpleMatrix> pair = recoverParameters( K, Pairs, Triples );
    SimpleMatrix pi = pair.getValue0();
    SimpleMatrix M3 = pair.getValue1();

    // Invert M3 to get M1 and M2.
    SimpleMatrix M3i = (M3.transpose()).pseudoInverse().mult( MatrixFactory.diag( MatrixOps.reciprocal(pi) ) );

    SimpleMatrix M1 = M13.mult( M3i );
    SimpleMatrix M2 = M3i.transpose().mult(M32).transpose();

    Execution.putOutput("tensor-reconstruction-error", MatrixOps.diff(M123, FullTensor.fromDecomposition(pi, M1, M2, M3)));

    try {
      if( Execution.getActualExecDir() != null ){
        IOUtils.writeLines( Execution.getFile("tensor-method.results"),
                Arrays.asList(pi.toString(),
                M1.toString(),
                M2.toString(),
                M3.toString()));
      }
    } catch (IOException ignored) {
    }

    LogInfo.end_track("recovery-asymmetric");

    return new Quartet<>( pi, M1, M2, M3 );
  }

  public Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix> recoverParametersAsymmetric( int K, FullTensor M123 ) {
    SimpleMatrix M13 = MatrixOps.condenseMoment(M123, 0, 2, 1);
    SimpleMatrix M12 = MatrixOps.condenseMoment(M123, 0, 1, 2);
    SimpleMatrix M32 = MatrixOps.condenseMoment(M123, 2, 1, 0);
    return recoverParameters(K, M13, M12, M32, M123);
  }

  // TODO: Allow for some specification of permutation of moments?
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
        SimpleMatrix M13,
        SimpleMatrix M12,
        SimpleMatrix M32,
        FullTensor M123 ) {
    LogInfo.begin_track("symmetrize-views");

    int D = M12.numRows();

    LogInfo.log("M13 condition:" + MatrixOps.conditionNumber(M13, K)) ;
    LogInfo.log("M12 condition:" + MatrixOps.conditionNumber(M12, K)) ;
    LogInfo.log("M32 condition:" + MatrixOps.conditionNumber(M32, K)) ;

    Triplet<SimpleMatrix,SimpleMatrix,SimpleMatrix> U1WU2 = MatrixOps.svdk( M12, K );
    SimpleMatrix U1 = U1WU2.getValue0(); // d x k
    SimpleMatrix U2 = U1WU2.getValue2();

    MatrixOps.printSize( U1 );

    assert( U1.numRows() == D );
    assert( U1.numCols() == K );

    // \tilde M_{12} = U_1^T M_{12} U_2
    M12 = U1.transpose().mult(M12).mult(U2); // d x k
    SimpleMatrix M12_i = M12.invert();
    M32 = M32.mult(U2); // M32 U2
    M13 = U1.transpose().mult(M13); // U1^T M13

    // P = M_{32} U_1^T (\tilde M_{12})^{-1} U_2 M_{13}
    SimpleMatrix Pairs =  M32.mult(M12_i).mult(M13);
    double skew = MatrixOps.symmetricSkewMeasure(Pairs);
    LogInfo.log( "Pairs Skew: " + skew );
    Pairs = MatrixOps.symmetrize(Pairs);
    assert( MatrixOps.isSymmetric(Pairs) );

    // T = M_{123}( M_{32} U_2^T (\tilde M_{12})^{-1} U_1, M_{31} U_1^T (\tilde M_{21})^{-1} U_2,  I )
    // T( W (P32 (P12)^-1), ((P12^-1)P13)^T, I )
    FullTensor Triples =
      M123.rotate(
          (M32.mult(M12_i).mult(U1.transpose())).transpose(),
          (M13.transpose().mult(M12_i.transpose()).mult(U2.transpose())).transpose(), // k x d
          MatrixFactory.eye(D)
      );
    skew = MatrixOps.symmetricSkewMeasure(Triples);
    LogInfo.log( "Triples Skew: " + skew );
    Triples = MatrixOps.symmetrize(Triples);
    assert( MatrixOps.isSymmetric(Triples) );

    LogInfo.end_track("symmetrize-views");
    return new Pair<>(Pairs, Triples);
  }


  /**
   * Extract parameters right from a data sequence.
   */
//  @Deprecated
//  public Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix>
//      recoverParameters( int K, int D, Iterator<double[][]> dataSeq ) {
//      // Compute moments
//      MomentAggregator agg = new MomentAggregator(D, dataSeq);
//      agg.run();
//
//      // In our case, the moments are symmetric.
//      //Pair<SimpleMatrix,SimpleMatrix> params = recoverParameters( K, agg.getMoments().getValue0(),
//      //  agg.getMoments().getValue3() );
//      Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix> params = recoverParameters( K, agg.getMoments() );
//
//      return new Quartet<>( params.getValue0(),
//          params.getValue1(),
//          params.getValue2(),
//          params.getValue3() );
//
//      //return recoverParameters( K, agg.getMoments() );
//    }

  /**
   * Compute the parameters of a tensor using random subspace estimation tricks.
   * @param K - rank of model
   * @param obj - object with computable moments
   * @return - (eigenvalue, eigenvector)
   */
  public Pair<SimpleMatrix, SimpleMatrix> randomizedSymmetricRecoverParameters( int K, ComputableMoments obj ) {
    int p = (int) oversamplingFactor*K;
    // First, compute the ranges
    SimpleMatrix Q = MatrixOps.randomizedRangeFinder(obj.computeP12(), p, rnd);

    // Compute "projected" Pairs, and whiten.
    SimpleMatrix Pairs = obj.computeP12().doubleMultiply(Q, Q);
    SimpleMatrix W = Q.mult(MatrixOps.whitener(Pairs, K));
    SimpleMatrix Winv = Q.mult(MatrixOps.colorer(Pairs, K));
    FullTensor Tw = obj.computeP123().multiply123(W, W, W);

    // Eigendecompose to find stuff.
    Pair<SimpleMatrix, SimpleMatrix> pair = getMethod().symmetricFactorize(Tw, K);
    SimpleMatrix eigenvalues = pair.getValue0(); SimpleMatrix eigenvectors = pair.getValue1();

    // Unwhiten
    pair = unwhiten(eigenvalues, eigenvectors, Winv);
    eigenvalues = pair.getValue0(); eigenvectors = pair.getValue1();

    return Pair.with(eigenvalues, eigenvectors);
  }

  /**
   * Symmetrizes a (large) 3-view mixture model and whitens it.
   * @param K - number of components
   * @param obj - object with computable moments
   * @return - a whitened tensor Triples_3
   */
  public Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix> randomizedRecoverParameters( int K, ComputableMoments obj ) {
    int p = 2*K;

    LogInfo.begin_track("find-ranges");
    // First, compute the range
    SimpleMatrix Q2 = MatrixOps.randomizedRangeFinder(obj.computeP12(), p, rnd);
    // Now do the SVD to compute the top-k range of both left and right spaces; truncates everything to $K$
    Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> triplet = MatrixOps.randomizedSvd(obj.computeP12(), Q2, K);
    SimpleMatrix Q1 = triplet.getValue0(); Q2 = triplet.getValue2(); // K x D
    // Compute the range for view 3
    SimpleMatrix Q3 = MatrixOps.randomizedRangeFinder(obj.computeP13(), p, rnd); // p x p
    LogInfo.end_track("find-ranges");

    LogInfo.begin_track("symmetrize-views+whiten");
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
    SimpleMatrix Winv = MatrixOps.colorer(Pairs, K);

    // Use this to compute the whitened symmetrized Triples.
    // T( W (P32 (P12)^-1), ((P12^-1)P13)^T, I )
    SimpleMatrix first = W.transpose().mult(P32.mult(P12inv).mult(Q1.transpose()));
    SimpleMatrix second = W.transpose().mult((Q2.mult(P12inv).mult(P13)).transpose());
    SimpleMatrix third = W.transpose().mult(Q3.transpose());

    FullTensor Tw = obj.computeP123().multiply123(first.transpose(), second.transpose(), third.transpose());
    LogInfo.end_track("symmetrize-views+whiten");

    // Recover parameters
    Pair<SimpleMatrix, SimpleMatrix> pair = getMethod().symmetricFactorize(Tw, K);
    SimpleMatrix eigenvalues = pair.getValue0(); SimpleMatrix eigenvectors = pair.getValue1();

    // Unwhiten
    pair = unwhiten(eigenvalues, eigenvectors, Winv);
    SimpleMatrix weights = pair.getValue0(); SimpleMatrix M3_ = pair.getValue1();

    LogInfo.begin_track("unsymmetrize");
    // Invert M3 to get M1 and M2.
    SimpleMatrix M3i = (M3_.transpose()).pseudoInverse().mult( MatrixFactory.diag( MatrixOps.reciprocal(weights) ) );
    SimpleMatrix M1_ = P13.mult( M3i );
    SimpleMatrix M2_ = M3i.transpose().mult(P32).transpose();
    LogInfo.end_track("unsymmetrize");

    // Now project everything back up using Q*
    return Quartet.with( weights,
        Q1.mult(M1_),
        Q2.mult(M2_),
        Q3.mult(M3_) );
  }
}
