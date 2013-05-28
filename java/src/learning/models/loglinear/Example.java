package learning.models.loglinear;

import fig.basic.Hypergraph;

public class Example {
  public Hypergraph Hq;  // For inference conditioned on the observations (represents q(h|x)).
  public int[] x;  // Values of observed nodes
  public int[] h;  // Values of labelled (hidden) nodes
  public Example() {}
  public Example(int[] x) {this.x = x;}
  public Example(int[] x, int[] h) {this.x = x; this.h = h;}
}


