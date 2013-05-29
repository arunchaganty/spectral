package learning.models.loglinear;

import fig.basic.Hypergraph;
import fig.basic.Fmt;

public class Example {
  public Hypergraph Hq;  // For inference conditioned on the observations (represents q(h|x)).
  public int[] x;  // Values of observed nodes
  public int[] h;  // Values of labelled (hidden) nodes
  public Example() {}
  public Example(int[] x) {
    this.x = new int[x.length];
    System.arraycopy(x, 0, this.x, 0, x.length);
  }
  public Example(int[] x, int[] h) {
    this(x);
    this.h = new int[h.length];
    System.arraycopy(h, 0, this.h, 0, h.length);
  }

  public String toString() {
    return "x=" + Fmt.D( x ) + ",h=" + Fmt.D(h);
  }

  /**
   * Creates a copy of this example with only the observed data.
   * Useful to predict the hidden data and compare.
   */
  public Example copyData() {
    Example ex = new Example( x );
    ex.h = new int[h.length];
    return ex;
  }

}


