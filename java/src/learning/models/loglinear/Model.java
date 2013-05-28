package learning.models.loglinear;

import java.io.*;
import java.util.*;

import learning.linalg.MatrixOps;

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
      //int hf = featureIndexer.getIndex(new UnaryFeature(h, "pi"));
      //H.addEdge(rootNode, hNode, edgeInfo(params, counts, hf, increment));
      H.addEdge(rootNode, hNode);

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
      //int pi_h = featureIndexer.getIndex(new UnaryFeature(h, "pi="+h));
      //H.addEdge(rootNode, hNode, edgeInfo(params, counts, pi_h, increment));
      // TODO: Handle non-uniform start probabilities.
      H.addEdge(rootNode, hNode);
    }
  }

  public void addTransition(Hypergraph<Example> H, int i, int h, Example ex, double[] params, double[] counts, double increment) {
    // Transitions
    // Create a link to the sum node that will enumerate over next states
    String hNode = String.format("h_%d=%d", i, h );
    String h_Node = String.format("h_%d=%d,h_%d", i, h, i+1 );
    H.addSumNode(h_Node); // this is a sum node over values of x_i.
    H.addEdge(hNode, h_Node); // Again, no potential required; hNode is a product node

    for (int h_ = 0; h_ < K; h_++) {  // For each value of start state...
      // Allocate the next node and assign transistion probabilities
      String hNextNode = String.format("h_%d=%d", i+1, h_); 
      H.addProdNode(hNextNode); // this is a product node over emissions and transistions.

      // probability of choosing this value for h.
      int trans_h_h_ = featureIndexer.getIndex(new BinaryFeature(h, h_));
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

class TallMixture extends Model {
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

  public void addInit(Hypergraph<Example> H, Example ex, double[] params, double[] counts, double increment) {
    Object rootNode = H.sumStartNode();
    for (int h = 0; h < K; h++) {  // For each value of start state...
      String hNode = String.format("h_%d=%d", 0, h); 
      H.addProdNode(hNode); // Each sub node is a product over views

      // probability of choosing this value for h.
      int pi_h = featureIndexer.getIndex(new UnaryFeature(h, "pi"));
      H.addEdge(rootNode, hNode, edgeInfo(params, counts, pi_h, increment));
    }
  }

  public void addHiddenNode(Hypergraph<Example> H, int i, int h, Example ex, double[] params, double[] counts, double increment) {
    String hNode = String.format("h_%d=%d", 0, h); 
    String h_Node = String.format("h_%d", i+1); 
    H.addSumNode(h_Node); // Sum over hidden states
    H.addEdge( hNode, h_Node ); // No potential

    for (int h_ = 0; h_ < K; h_++) {  // For each value of start state...
      String hNextNode = String.format("h_%d=%d", i+1, h_); 
      H.addSumNode(hNextNode); // Sum over observations

      // probability of choosing this value for h.
      int pi_h = featureIndexer.getIndex(new BinaryFeature(h, h_));
      H.addEdge(h_Node, hNextNode, edgeInfo(params, counts, pi_h, increment));
    }
  }

  public void addObserved(Hypergraph<Example> H, int i, int h, Example ex, double[] params, double[] counts, double increment) {
    String hNode = String.format("h_%d=%d", i+1, h); 
    String xNode = String.format("h_%d=%d,x_%d", i+1, h, i+1); 

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
    for( int i = 0; i < L; i++ ) { // For each view
      for( int h = 0; h < K; h++ ) { // For each choice of the hidden start state
        addHiddenNode( H, i, h, ex, params, counts, increment );
      }
      for( int h = 0; h < K; h++ ) { // For each choice of the hidden state for the view
        addObserved( H, i, h, ex, params, counts, increment );
      }
    }
    return H;
  }
}

class Grid extends Model {
  int L, D; // L = number of grid cells, D = choices per X;
  int width, height;


  public Example newExample() {
    Example ex = new Example();
    assert( L == width * height );
    ex.x = new int[2 * L]; // Each grid cell has two units
    return ex;
  }

  // Used to set observed data
  public Example newExample(int[] x) {
    assert( L == width * height );
    assert( x.length == 2 * L );
    Example ex = new Example();
    ex.x = new int[x.length];
    System.arraycopy(x, 0, ex.x, 0, x.length);
    return ex;
  }

  public void addObserved(Hypergraph<Example> H, int i, int j, Example ex, double[] params, double[] counts, double increment) {
    LogInfo.begin_track("grid-add-observed");
    // Iterate over hidden states of the cell at (i,j)
    for( int h = 0; h < K; h++ ) {
      String hNode = String.format("h_{%d,%d}=%d", i, j, h); // Defined to be a product node.
      String xaNode = String.format("h_{%d,%d}=%d,x_a", i, j, h); // Register x nodes
      String xbNode = String.format("h_{%d,%d}=%d,x_b", i, j, h); 

      if (ex != null) {  // Numerator: generate x[j]
        int idx = 2*(width*i + j);
        int f_xa = featureIndexer.getIndex(new UnaryFeature(h, String.format("x^a=%d", ex.x[idx]))); // Features are shared across cells
        int f_xb = featureIndexer.getIndex(new UnaryFeature(h, String.format("x^b=%d", ex.x[idx+1])));
        if (params != null) {
          H.addEdge(hNode, H.endNode, edgeInfo(params, counts, f_xa, increment));
          H.addEdge(hNode, H.endNode, edgeInfo(params, counts, f_xb, increment));
        }
      } else {  // Denominator: generate each possible assignment x[j] = a
        H.addSumNode(xaNode);
        H.addSumNode(xbNode);
        H.addEdge(hNode, xaNode); // No potential required; hNode is a product node
        H.addEdge(hNode, xbNode); // No potential required; hNode is a product node

        for (int a = 0; a < D; a++) {
          int idx = 2*(height*i + j);
          int f_xa = featureIndexer.getIndex(new UnaryFeature(h, String.format("x^a=%d", a) ) );
          int f_xb = featureIndexer.getIndex(new UnaryFeature(h, String.format("x^b=%d", a) ) );
          if (params != null) {
            H.addEdge(xaNode, H.endNode, edgeInfo(params, counts, f_xa, increment, idx, a));
            H.addEdge(xbNode, H.endNode, edgeInfo(params, counts, f_xb, increment, idx+1, a)); // Remember that this is where we store the variable
          }
        }
      }
    }
    LogInfo.end_track("grid-add-observed");
  }

  public void addEdges(Hypergraph<Example> H, int i, int j, Example ex, double[] params, double[] counts, double increment) {
    LogInfo.begin_track("grid-add-edges");
    for( int h = 0; h < K; h++ ) {
      String hNode = String.format("h_{%d,%d}=%d", i, j, h); 
      // Add links to adjacent nodes; note that nodes earlier in the lex.
      // ordering have already added links.
      
      if( i + 1 < height ) {
        // Temporary factor that will enumerate over the sum
        String h_Node = String.format("h_{%d,%d}=%d,h_{%d,%d}", i, j, h, i+1, j); 
        H.addSumNode( h_Node );
        H.addEdge( hNode, h_Node );
        for (int h_ = 0; h_ < K; h_++) {  // For each value of start state...
          String hRightNode = String.format("h_{%d,%d}=%d", i+1, j, h_); 
          H.addProdNode(hRightNode); // Sum over observations
          // The h->h_ potential
          int T_h_h_ = featureIndexer.getIndex(new BinaryFeature(h, h_));
          H.addEdge(h_Node, hRightNode, edgeInfo(params, counts, T_h_h_, increment));
        }
      }
      if( j + 1 < width ) {
        String h_Node = String.format("h_{%d,%d}=%d,h_{%d,%d}", i, j, h, i, j+1); 
        H.addSumNode( h_Node );
        H.addEdge( hNode, h_Node );
        for (int h_ = 0; h_ < K; h_++) {  // For each value of start state...
          String hLowerNode = String.format("h_{%d,%d}=%d", i, j+1, h_); 
          H.addProdNode(hLowerNode); // Sum over observations
          // The h->h_ potential
          int T_h_h_ = featureIndexer.getIndex(new BinaryFeature(h, h_));
          H.addEdge(h_Node, hLowerNode, edgeInfo(params, counts, T_h_h_, increment));
        }
      }
    }
    LogInfo.end_track("grid-add-edges");
  }

  public Hypergraph<Example> createHypergraph(Example ex, double[] params, double[] counts, double increment) {
    LogInfo.begin_track("create-grid");
    // Set length to be length of example data.

    Hypergraph<Example> H = new Hypergraph<Example>();
    H.debug = true;
    
    // Add the first hidden node first.
    Object rootNode = H.sumStartNode(); // Think of it as h_{0,0}
    for (int h = 0; h < K; h++) {  // For each value of start state...
      String hNode = String.format("h_{%d,%d}=%d", 0, 0, h); 
      H.addProdNode(hNode); // this is a product node over emissions and transistions.
      // probability of choosing this value for h.
      int pi_h = featureIndexer.getIndex(new UnaryFeature(h, "pi="+h));
      H.addEdge(rootNode, hNode, edgeInfo(params, counts, pi_h, increment));
    }
    // Traverse the nodes in cantor diagonal fashion
    for( int w = 0; w < width+height; w++ ) {
      // Now add all the hidden nodes and their observeds.
      for(int i = 0; i < MatrixOps.min(w+1,height); i++ ) {
        if( w - i >= width ) continue;
        int j = w - i;
        // LogInfo.logs( w + ": " + i + ", " + j );
        addEdges(H, i, j, ex, params, counts, increment);
        addObserved(H, i, j, ex, params, counts, increment);
      }
    }
    LogInfo.end_track("create-grid");
    
    return H;
  }
}


