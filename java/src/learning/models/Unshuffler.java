package learning.models;

import java.io.*;
import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;
import Jama.*;

/**
 * Given K^S D-dimensional input vectors, find S sources, each taking on K
 * values, and each associated with a vector, such that the products
 * (exp-normalize-log'ed if !knownNormalization) are exactly those input
 * vectors.
 * Used for learning factorial models.
 */
public class Unshuffler {
  int D;  // Dimensionality of vectors
  List<double[]> vectors;  // Our input data
  List<String> names;  // Just for debugging
  static final double tolerance = 1e-10;
  boolean knownNormalization;

  public Unshuffler(List<double[]> vectors) {
    this.vectors = vectors;
    this.D = vectors.get(0).length;
  }

  // Return if vectors a and b differ by a constant
  static boolean differByConstant(double[] a, double[] b) {
    double d = a[0] - b[0];
    for (int i = 1; i < a.length; i++)
      if (Math.abs(a[i] - b[i]) > tolerance)
        return false;
    return true;
  }

  class Pair {
    int id1, id2;  // Index into vectors
    double[] diff;

    @Override public String toString() {
      // 1,1,0  and 1,0,0  => *,1-0,*
      String[] n1 = names.get(id1).split(",");
      String[] n2 = names.get(id2).split(",");
      StringBuilder sb = new StringBuilder();
      for (int i = 0; i < n1.length; i++) {
        if (i > 0) sb.append(',');
        if (n1[i].equals(n2[i]))
          sb.append('*');
        else
          sb.append(n1[i] + '-' + n2[i]);
      }
      return sb.toString();
    }
  }

  // Represents the set of pairs with diff's up to a constant. (b2-b1+?)
  // (PairSet = Bin)
  class PairSet {
    List<Pair> pairs = new ArrayList<Pair>();
    @Override public String toString() {
      return pairs.toString();
    }
  }

  // Represents a collection of PairSets (b2-b1, b3-b1, ...)
  // (Factor = Source)
  class Factor {
    List<PairSet> sets = new ArrayList<PairSet>();
  }

  boolean triangle(PairSet set1, PairSet set2, PairSet set3) {
    double[] d1 = set1.pairs.get(0).diff;
    double[] d2 = set2.pairs.get(0).diff;
    double[] d3 = set3.pairs.get(0).diff;
    return differByConstant(ListUtils.sub(d2, d1), d3);
  }

  boolean isZero(double[] v) {
    for (double x : v)
      if (Math.abs(x) > tolerance) return false;
    return true;
  }

  boolean isNegative(PairSet set1, PairSet set2) {
    double[] d1 = set1.pairs.get(0).diff;
    double[] d2 = set2.pairs.get(0).diff;
    return isZero(ListUtils.add(d1, d2));
  }

  public void solve() {
    // Construct pairs b2-b1 of vectors
    List<Pair> pairs = new ArrayList<Pair>();
    for (int i = 0; i < vectors.size(); i++) {
      for (int j = 0; j < vectors.size(); j++) {
        if (i == j) continue;
        Pair pair = new Pair();
        pair.id1 = i;
        pair.id2 = j;
        pair.diff = ListUtils.sub(vectors.get(i), vectors.get(j));
        pairs.add(pair);
      }
    }

    // Group pairs b2-b1 with the same difference (up to a constant) into pair sets.
    Set<Pair> hit = new HashSet<Pair>();
    List<PairSet> sets = new ArrayList<PairSet>();
    int maxPairSetSize = 0;
    for (Pair pair : pairs) {
      if (hit.contains(pair)) continue;
      PairSet set = new PairSet();
      for (Pair pair2 : pairs) {
        if (differByConstant(pair.diff, pair2.diff)) {
          hit.add(pair2);
          set.pairs.add(pair2);
        }
      }
      maxPairSetSize = Math.max(maxPairSetSize, set.pairs.size());
      sets.add(set);
      //logs("Set %s", set);
    }

    // Keep only pair sets of largest cardinality.
    // If each of the F factors have the same number of possible values K,
    // then each of the surviving sets contain only differences on one factor.
    List<PairSet> newSets = new ArrayList<PairSet>();
    for (PairSet set : sets) {
      if (set.pairs.size() == maxPairSetSize)
        newSets.add(set);
    }
    sets = newSets;

    // Try to group the pair sets into factors:
    // {a2-a1, a3-a1, a4-a1, ...}, {b2-b1, b3-b1, b4-b1, ...}.
    // Keep only factors that involve the difference with one pivot (e.g., a1)
    // The key idea is that two pair sets connected if their difference is also a valid pair set
    // (e.g., b2-b1 is connected to b3-b1 because b3-b2 exists).
    List<Factor> factors = new ArrayList<Factor>();
    Set<PairSet> hitSet = new HashSet<PairSet>();
    for (PairSet set : sets) {  // Take set = b2-b1, the first element of this factor
      if (hitSet.contains(set)) continue;

      hitSet.add(set);
      Factor factor = new Factor();
      factor.sets.add(set);
      for (PairSet set3 : sets) if (isNegative(set, set3)) hitSet.add(set3);
      for (PairSet set2 : sets) {  // For each candidate set2...
        if (hitSet.contains(set2)) continue;

        // Check to see if set2 = bi-b1 by seeing if set3 = bi-b2 exists
        boolean found = false;
        for (PairSet set3 : sets) {
          if (triangle(set, set2, set3)) {
            found = true;
            break;
          }
        }

        if (found) {
          // Will add set2, get rid of set1 - set2 for all set1 in factor.sets
          for (PairSet set1 : factor.sets) {
            for (PairSet set3 : sets) {
              if (triangle(set1, set2, set3) || triangle(set2, set1, set3))
                hitSet.add(set3);
            }
          }
          hitSet.add(set2);
          factor.sets.add(set2);
          for (PairSet set3 : sets) if (isNegative(set2, set3)) hitSet.add(set3);
        }
      }

      LogInfo.logs("Factor %s", factor.sets);
      factors.add(factor);
    }

    int F = factors.size();  // Number of factors
    int[] K = new int[F];  // Number of possible values per cluster
    for (int f = 0; f < F; f++)
      K[f] = factors.get(f).sets.size() + 1;
    logs("K: %s", Fmt.D(K));

    // A constraint for each vector and dimension
    Constraint[][] constraints = new Constraint[vectors.size()][D];
    for (int i = 0; i < vectors.size(); i++) {
      for (int j = 0; j < D; j++) {
        constraints[i][j] = new Constraint();
        constraints[i][j].name = names.get(i)+":"+j;
        constraints[i][j].target = vectors.get(i)[j];  // Target value
        if (!knownNormalization)
          constraints[i][j].add("Z"+i, 1);  // Normalization
      }
    }
    for (int f = 0; f < F; f++) {  // For each factor...
      Factor factor = factors.get(f);
      for (int i = 0; i < factor.sets.size(); i++) { // For each possible difference that the factor can take on...
        for (Pair pair : factor.sets.get(i).pairs) {
          for (int j = 0; j < D; j++) {
            String v1 = "F"+f+":"+(i+1)+":"+j;
            String v0 = "F"+f+":"+0+":"+j;
            constraints[pair.id1][j].add(v1, 1);
            constraints[pair.id2][j].add(v0, 1);
          }
        }
      }
    }

    LinearSystem system = new LinearSystem();
    for (int f = 0; f < F; f++)
      for (int i = 0; i < K[f]; i++)
        for (int j = 0; j < D; j++)
          system.varIndexer.getIndex("F"+f+":"+i+":"+j);  // Not necessary, but puts variables in order
    for (int i = 0; i < vectors.size(); i++)
      for (int j = 0; j < D; j++)
        system.add(constraints[i][j]);
    system.solve();
  }

  public static class UnshufflerTest implements Runnable {
    @Option(gloss="Number of sources (h_1, ..., h_F)") public int F = 3;
    @Option(gloss="Number of possible values for each source h_j") public int K = 2;
    @Option(gloss="Dimensionality of x") public int D = 10;
    @Option public Random random = new Random(1);
    @Option boolean knownNormalization = true;

    // For each value for each source, multiply the vectors together
    void createVectors(double[][][] theta, int i, String name, double[] vec, List<double[]> vectors, List<String> names) {
      if (i == theta.length) {
        names.add(name);
        if (!knownNormalization) {
          double Z = 1;
          for (double v : vec) Z *= Math.exp(v);
          double logZ = Math.log(Z);
          for (int x = 0; x < D; x++) vec[x] -= logZ;
        }
        vectors.add(vec);
        return;
      }
      
      for (int j = 0; j < theta[i].length; j++)
        createVectors(theta, i+1, name == null ? ""+j : name+","+j, ListUtils.add(vec, theta[i][j]), vectors, names);
    }

    public void run() {
      // Generate true parameters
      double[][][] theta = new double[F][][];
      for (int f = 0; f < F; f++) {
        theta[f] = new double[K][D];
        for (int j = 0; j < K; j++)
          for (int x = 0; x < D; x++)
            theta[f][j][x] = 2*random.nextDouble() - 1;
      }
      LogInfo.begin_track("True parameters");
      for (int f = 0; f < F; f++)
        for (int j = 0; j < K; j++)
          LogInfo.logs("F%d:%d = %s", f, j, Fmt.D(theta[f][j]));
      LogInfo.end_track();

      List<double[]> vectors = new ArrayList<double[]>();
      List<String> names = new ArrayList<String>();
      createVectors(theta, 0, null, new double[D], vectors, names);
      Unshuffler unshuffler = new Unshuffler(vectors);
      unshuffler.knownNormalization = knownNormalization;
      unshuffler.names = names;
      unshuffler.solve();
    }
  }

  public static void main(String[] args) {
    Execution.run(args, new UnshufflerTest());
  }
}
