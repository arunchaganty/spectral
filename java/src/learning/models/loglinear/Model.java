package learning.models.loglinear;

import java.io.*;
import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;

public abstract class Model {
  int K;  // Number of latent states
  Indexer<Feature> featureIndexer = new Indexer<Feature>();
  int numFeatures() { return featureIndexer.size(); }

  ParamsVec newParamsVec() { return new ParamsVec(K, featureIndexer); }

  abstract Example newExample();
  abstract Example newExample(int[] x);
  abstract Hypergraph<Example> createHypergraph(Example ex, double[] params, double[] counts, double increment);

  Hypergraph.LogHyperedgeInfo<Example> nullInfo = new Hypergraph.LogHyperedgeInfo<Example>() {
    public double getLogWeight() { return 0; }
    public void setPosterior(double prob) { }
    public Example choose(Example ex) { return ex; }
  };

  Hypergraph.LogHyperedgeInfo<Example> edgeInfo(double[] params, double[] counts, int f, double increment) {
    return new Hypergraph.MultinomialLogHyperedgeInfo<Example>(params, counts, f, increment);
  }
  Hypergraph.LogHyperedgeInfo<Example> edgeInfo(double[] params, double[] counts, int f, double increment, final int j, final int v) {
    return new Hypergraph.MultinomialLogHyperedgeInfo<Example>(params, counts, f, increment) {
      public Example choose(Example ex) {
        ex.x[j] = v;
        return ex;
      }
    };
  }
}

class MixtureModel extends Model {
  int L, D; // L = number of views, D = choices per view;

  public Example newExample() {
    Example ex = new Example();
    ex.x = new int[L];
    return ex;
  }

  // Used to set observed data
  public Example newExample(int[] x) {
    assert( x.length == L );
    Example ex = new Example();
    ex.x = new int[x.length];
    System.arraycopy(x, 0, ex.x, 0, x.length);
    return ex;
  }

  public Hypergraph<Example> createHypergraph(Example ex, double[] params, double[] counts, double increment) {
    Hypergraph<Example> H = new Hypergraph<Example>();
    //H.debug = true;
    
    // The root node disjuncts over the possible hidden state values
    Object rootNode = H.sumStartNode();
    for (int h = 0; h < K; h++) {  // For each value of hidden states...
      String hNode = "h="+h;
      H.addProdNode(hNode); // this is product node over each view

      // probability of choosing this value for h.
      int hf = featureIndexer.getIndex(new UnaryFeature(h, "h="+h));
      H.addEdge(rootNode, hNode, edgeInfo(params, counts, hf, increment));

      for (int j = 0; j < L; j++) {  // For each view j...
        String xNode = "h="+h+",x"+j;
        if (ex != null) {  // Numerator: generate x[j]
          int f = featureIndexer.getIndex(new UnaryFeature(h, "x="+ex.x[j]));
          if (params != null)
            H.addEdge(hNode, H.endNode, edgeInfo(params, counts, f, increment));
        } else {  // Denominator: generate each possible assignment x[j] = a
          H.addSumNode(xNode);
          H.addEdge(hNode, xNode);
          for (int a = 0; a < D; a++) {
            int f = featureIndexer.getIndex(new UnaryFeature(h, "x="+a));
            if (params != null)
              H.addEdge(xNode, H.endNode, edgeInfo(params, counts, f, increment, j, a));
          }
        }
      }
    }
    return H;
  }
}

class HiddenMarkovModel extends Model {
  int L, D; // L = number of views, D = choices per view;

  public Example newExample() {
    Example ex = new Example();
    ex.x = new int[L];
    return ex;
  }

  // Used to set observed data
  public Example newExample(int[] x) {
    Example ex = new Example();
    ex.x = new int[x.length];
    System.arraycopy(x, 0, ex.x, 0, x.length);
    return ex;
  }

  public void addInit(Hypergraph<Example> H, Example ex, double[] params, double[] counts, double increment) {
    // Transitions
    Object rootNode = H.sumStartNode();
    for (int h = 0; h < K; h++) {  // For each value of start state...
      String hNode = String.format("h_%d=%d", 0, h); 
      H.addProdNode(hNode); // this is a product node over emissions and transistions.

      // probability of choosing this value for h.
      int pi_h = featureIndexer.getIndex(new UnaryFeature(h, "pi="+h));
      H.addEdge(rootNode, hNode, edgeInfo(params, counts, pi_h, increment));
    }
  }

  public void addTransition(Hypergraph<Example> H, int i, int h, Example ex, double[] params, double[] counts, double increment) {
    // Transitions
    // Create a link to the sum node that will enumerate over next states
    String hNode = String.format("h_%d=%d", i, h );
    String h_Node = String.format("h_%d", i+1 );
    H.addSumNode(h_Node); // this is a sum node over values of x_i.
    H.addEdge(hNode, h_Node); // Again, no potential required; hNode is a product node

    for (int h_ = 0; h_ < K; h_++) {  // For each value of start state...
      // Allocate the next node and assign transistion probabilities
      String hNextNode = String.format("h_%d=%d", i+1, h_); 
      H.addProdNode(hNextNode); // this is a product node over emissions and transistions.

      // probability of choosing this value for h.
      int trans_h_h_ = featureIndexer.getIndex(new UnaryFeature(h, "h'="+h_));
      H.addEdge(h_Node, hNextNode, edgeInfo(params, counts, trans_h_h_, increment));
    }
  }

  public void addEmission(Hypergraph<Example> H, int i, int h, Example ex, double[] params, double[] counts, double increment) {
    String hNode = String.format("h_%d=%d", i, h );
    // We need to index with $h_i=h$ because we average over that value.
    String xNode = String.format("h_%d=%d,x_%d", i, h, i ); 

    if (ex != null) {  // Numerator: generate x[j]
      int f_xj = featureIndexer.getIndex(new UnaryFeature(h, "x="+ex.x[i]));
      if (params != null)
        H.addEdge(hNode, H.endNode, edgeInfo(params, counts, f_xj, increment));
    } else {  // Denominator: generate each possible assignment x[j] = a
      H.addSumNode(xNode);
      H.addEdge(hNode, xNode); // No potential required; hNode is a product node
      for (int a = 0; a < D; a++) {
        int f_xa = featureIndexer.getIndex(new UnaryFeature(h, "x="+a));
        if (params != null)
          H.addEdge(xNode, H.endNode, edgeInfo(params, counts, f_xa, increment, i, a));
      }
    }
  }

  public Hypergraph<Example> createHypergraph(Example ex, double[] params, double[] counts, double increment) {
    // Set length to be length of example data.
    int L = (ex != null) ? ex.x.length : this.L;

    Hypergraph<Example> H = new Hypergraph<Example>();
    //H.debug = true;
    
    // The root node disjuncts over the possible hidden state values for
    // the start state
    addInit(H, ex, params, counts, increment);

    // For the remaining nodes
    for( int i = 0; i < L-1; i++ ) { // For each state in the sequence
      for (int h = 0; h < K; h++) {  // For each value of state...
        addEmission( H, i, h, ex, params, counts, increment );
        addTransition( H, i, h, ex, params, counts, increment );
      }
    }

    // The last node doesn't have a transition
    {
      int i = L-1;
      for (int h = 0; h < K; h++) {  // And finally an end state.
        addEmission( H, i, h, ex, params, counts, increment );
      }
    }
    return H;
  }
}

