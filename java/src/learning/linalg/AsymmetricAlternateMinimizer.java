package learning.linalg;

import fig.basic.Option;
import learning.common.Utils;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.javatuples.Triplet;

import java.util.*;

/**
 * Implements an alternating minimization with asymmetric rank-1 updates
 *
 * - TODO: include reference
 */
public class AsymmetricAlternateMinimizer implements TensorFactorizationAlgorithm {

  @Option(gloss = "inititalization randomness")
  public Random initRandom = new Random(24);
  @Option(gloss = "Number of iteration to run algorithm")
  public int iters = 100;
  @Option(gloss = "Number of attempts")
  public int attempts = 1000;
  @Option(gloss = "Clustering resolution")
  public double clusterEps = 1e-2;

  @Override
  public Pair<SimpleMatrix, SimpleMatrix> symmetricOrthogonalFactorize(FullTensor T, int K) {
    return null;
  }

  @Override
  public Pair<SimpleMatrix, SimpleMatrix> symmetricFactorize(FullTensor T, int K) {
    return null;
  }

  Quartet<Double, SimpleMatrix, SimpleMatrix, SimpleMatrix> factorizationStep(FullTensor T, SimpleMatrix a, SimpleMatrix b, SimpleMatrix c ) {
    // a' = T(I, b, c)
    SimpleMatrix a_ = T.project2(1, 2, b, c);
    SimpleMatrix b_ = T.project2(0, 2, a, c);
    SimpleMatrix c_ = T.project2(0, 1, a, b);

    double w = Math.pow(MatrixOps.norm(a_) * MatrixOps.norm(b_) * MatrixOps.norm(c_), 1./3);

    a_ = a_.scale(1./MatrixOps.norm(a_));
    b_ = b_.scale(1./MatrixOps.norm(b_));
    c_ = c_.scale(1./MatrixOps.norm(c_));

    return Quartet.with(w, a_, b_, c_);
  }
  Quartet<Double, SimpleMatrix, SimpleMatrix, SimpleMatrix> factorizationStep(FullTensor T, Quartet<Double, SimpleMatrix, SimpleMatrix, SimpleMatrix> iterate) {
    return factorizationStep(T, iterate.getValue1(), iterate.getValue2(), iterate.getValue3());
  }

  Quartet<Double, SimpleMatrix, SimpleMatrix, SimpleMatrix> initRandom(FullTensor T) {
    // Initialize randomly
    SimpleMatrix a = RandomFactory.randn(initRandom, 1, T.D1);
    SimpleMatrix b = RandomFactory.randn(initRandom, 1, T.D2);
    a = a.scale(1./MatrixOps.norm(a));
    b = b.scale(1./MatrixOps.norm(b));

    SimpleMatrix c = T.project2(0,1,a,b);
    c = c.scale(1./MatrixOps.norm(c));

    double w = Math.pow(MatrixOps.norm(T.project2(1,2,b,c)) * MatrixOps.norm(T.project2(0,2,a,c)) * MatrixOps.norm(T.project2(0,1,a,b)), 1./3);

    return Quartet.with(w,a,b,c);
  }

  @Override
  public Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, SimpleMatrix> asymmetricFactorize(FullTensor T, int K) {

    // - Collect fixed points
    List<Quartet<Double, SimpleMatrix, SimpleMatrix, SimpleMatrix>> fixedPoints = new ArrayList<>();
    for( int attempt = 0; attempt < attempts; attempt++ ) {
      // - Initialize
      Quartet<Double,SimpleMatrix,SimpleMatrix,SimpleMatrix> iterate = initRandom(T);
      for(int iter = 0; iter < iters; iter++) {
        iterate = factorizationStep(T, iterate);
      }
      // - Save fixed point
      fixedPoints.add(iterate);
    }
    // - Cluster fixed points
    fixedPoints = clusterFixedPoints(T, K, fixedPoints);

    // Finally save results
    SimpleMatrix w = new SimpleMatrix(1,K);
    SimpleMatrix A = new SimpleMatrix(T.D1,K);
    SimpleMatrix B = new SimpleMatrix(T.D2,K);
    SimpleMatrix C = new SimpleMatrix(T.D3,K);
    for(int k = 0; k < K; k++) {
      Quartet<Double, SimpleMatrix, SimpleMatrix, SimpleMatrix> fixedPoint = fixedPoints.get(k);
      w.set(0, k, fixedPoint.getValue0());
      for( int row = 0; row < T.D1; row++ ) {
        A.set(row, k, fixedPoint.getValue1().get(row));
      }
      for( int row = 0; row < T.D2; row++ ) {
        B.set(row, k, fixedPoint.getValue2().get(row));
      }
      for( int row = 0; row < T.D3; row++ ) {
        C.set(row, k, fixedPoint.getValue3().get(row));
      }
    }

    return Quartet.with(w, A, B, C);
  }

  /**
   * Do simple agglomerative clustering to cluster T
   * @param T
   * @param fixedPoints
   * @return
   */
  private List<Quartet<Double, SimpleMatrix, SimpleMatrix, SimpleMatrix>> clusterFixedPoints(FullTensor T, int K,
                 List<Quartet<Double, SimpleMatrix, SimpleMatrix, SimpleMatrix>> fixedPoints) {

    // Sort the list by value of T(a,b,c)
    List<Double> scores = new ArrayList<>();
    for(Quartet<Double, SimpleMatrix, SimpleMatrix, SimpleMatrix> fixedPoint : fixedPoints ) {
      scores.add(T.project3(fixedPoint.getValue1(), fixedPoint.getValue2(), fixedPoint.getValue3()));
    }
    List<Pair<Double,Integer>> ordering = Utils.argsort(scores);
    Collections.reverse(ordering);
    List<Quartet<Double, SimpleMatrix, SimpleMatrix, SimpleMatrix>> candidates = new LinkedList<>();
    for( Pair<Double, Integer> elem : ordering ) {
      candidates.add( fixedPoints.get(elem.getValue1()) );
    }

    List<Quartet<Double, SimpleMatrix, SimpleMatrix, SimpleMatrix>> centers = new ArrayList<>();
    while(centers.size() < K && candidates.size() > 0) {
      // Choose the largest value and run for some iterations
      Quartet<Double, SimpleMatrix, SimpleMatrix, SimpleMatrix> center = candidates.remove(0);
      for(int iter = 0; iter < iters; iter++) {
        center = factorizationStep(T, center);
      }
      centers.add(center);

      // and remove anything near by
      ListIterator<Quartet<Double, SimpleMatrix, SimpleMatrix, SimpleMatrix>> it = candidates.listIterator();
      while(it.hasNext()) {
        Quartet<Double, SimpleMatrix, SimpleMatrix, SimpleMatrix> candidate = it.next();
        if( clusterDistance(center, candidate) > clusterEps)
          it.remove();
      }
    }

    return centers;
  }

  private double clusterDistance(Quartet<Double,SimpleMatrix,SimpleMatrix,SimpleMatrix> center,
                                 Quartet<Double, SimpleMatrix, SimpleMatrix, SimpleMatrix> candidate) {
    double a = Math.abs(center.getValue1().dot(candidate.getValue1()));
    double b = Math.abs(center.getValue2().dot(candidate.getValue2()));
    double c = Math.abs(center.getValue3().dot(candidate.getValue3()));
    return Math.max(Math.max( a, b ), c);
  }
}
