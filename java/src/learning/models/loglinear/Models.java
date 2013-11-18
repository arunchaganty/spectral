package learning.models.loglinear;

import java.io.*;
import java.util.*;

import learning.linalg.MatrixOps;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;

public class Models {
  public static class MixtureModel extends Model {

    public Example newExample() {
      Example ex = new Example();
      ex.x = new int[L];
      ex.h = new int[1];
      return ex;
    }
    // Used to set observed data
    public Example newExample(int[] x) {
      assert( x.length == L );
      return new Example( x, new int[1] );
    }

    public Hypergraph<Example> createHypergraph(int L, Example ex, double[] params, double[] counts, double increment) {
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
        H.addEdge(rootNode, hNode, hiddenEdgeInfo(0,h) ); 

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

  public static class HiddenMarkovModel extends Model {

    public HiddenMarkovModel(int K, int D, int L) {
      this.K = K;
      this.D = D;
      this.L = L;
    }
    HiddenMarkovModel(){}

    public Example newExample() {
      Example ex = new Example();
      ex.x = new int[L];
      ex.h = new int[L];
      return ex;
    }

    // Used to set observed data
    public Example newExample(int[] x) {
      return new Example( x, new int[x.length] );
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
        H.addEdge(rootNode, hNode, hiddenEdgeInfo(0,h));
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
        H.addEdge(h_Node, hNextNode, hiddenEdgeInfo(params, counts, trans_h_h_, increment, i+1, h_));
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

    public Hypergraph<Example> createHypergraph(int L, Example ex, double[] params, double[] counts, double increment) {
      // Set length to be length of example data.
      assert( (ex == null) || ex.x.length == L );

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
          // Add an end-transition state
          // Create a link to the sum node that will enumerate over next states
          String hNode = String.format("h_%d=%d", i, h );
          H.addEdge(hNode, H.endNode); // Again, no potential required; hNode is a product node
        }
      }
      return H;
    }
  }

  public static class TallMixture extends Model {
    int D; // L = number of views, D = choices per view;

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


    public Hypergraph<Example> createHypergraph(int L, Example ex, double[] params, double[] counts, double increment) {
      // Set length to be length of example data.
      assert( (ex == null) || ex.x.length == L );

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

  // 2 x width grid.  Each node has two observations, 
  // 1     L
  // o-o-o-o
  // | | | |
  // o-o-o-o
  public static class GridModel extends Model {
    final int height = 2;
    int width;

    public GridModel(int L, int D) {
      this.L = L;
      this.D = D;
      this.width = L / height;
      assert L % height == 0 : L;
    }

    public Example newExample() {
      Example ex = new Example();
      ex.h = new int[L];
      ex.x = new int[L * 2];
      return ex;
    }

    // row r, column c, direction d (a or b) => index into x
    public int observedNodeIndex(int r, int c, int d) { return (r * width + c) * 2 + d; }

    // row r, column c => index into h
    public int hiddenNodeIndex(int r, int c) { return r * width + c; }

    // Used to set observed data
    public Example newExample(int[] x) {
      assert x.length == L;
      Example ex = new Example();
      ex.x = new int[L];
      ListUtils.set(ex.x, x);
      return ex;
    }

    private Object genNode(Hypergraph<Example> H, Example ex, double[] params, double[] counts, double increment, int c, int ha, int hb) {
      String node = "G"+c+"="+ha+","+hb;
      if (H.addProdNode(node)) {
        // 4 emissions
        H.addEdge(node, emitNode(H, ex, params, counts, increment, 0, c, 0, ha));  // Top row, left observation
        H.addEdge(node, emitNode(H, ex, params, counts, increment, 0, c, 1, hb));  // Top row, right observation
        H.addEdge(node, emitNode(H, ex, params, counts, increment, 1, c, 0, ha));  // Bottom row, left observation
        H.addEdge(node, emitNode(H, ex, params, counts, increment, 1, c, 1, hb));  // Bottom row, right observation
        // 1 transition
        H.addEdge(node, transNode(H, ex, params, counts, increment, c+1, ha, hb));
      }
      return node;
    }

    // Generate the d-th observation at cell (r, c), given that the hidden node is h.
    private Object emitNode(Hypergraph<Example> H, Example ex, double[] params, double[] counts, double increment, int r, int c, int d, int h) {
      String node = "E("+r+","+c+","+d+")="+h;
      if (H.addSumNode(node)) {
        int j = observedNodeIndex(r, c, d);
        if (ex != null) {  // Numerator: generate x[j]
          int f = featureIndexer.getIndex(new UnaryFeature(h, "x="+ex.x[j]));
          if (params != null)
            H.addEdge(node, H.endNode, edgeInfo(params, counts, f, increment));
        } else {  // Denominator: generate each possible assignment x[j] = a
          for (int a = 0; a < D; a++) {
            int f = featureIndexer.getIndex(new UnaryFeature(h, "x="+a));
            if (params != null)
              H.addEdge(node, H.endNode, edgeInfo(params, counts, f, increment, j, a));
          }
        }
      }
      return node;
    }

    // Generate the grid from column c on, given that hidden nodes at c-1 are (prev_ha, prev_hb).
    // Transition into the nodes at column c.
    private Object transNode(Hypergraph<Example> H, Example ex, double[] params, double[] counts, double increment, int c, int prev_ha, int prev_hb) {
      if (c == width) return H.endNode;

      String node = "T"+c+"="+prev_ha+","+prev_hb;
      if (H.addSumNode(node)) {
        // For each possible setting of hidden nodes at column c (ha, hb)...
        for (int ha = 0; ha < K; ha++) {
          for (int hb = 0; hb < K; hb++) {
            if (c == 0) {
              // No prev_ha, prev_hb...
              H.addEdge(node, genNode(H, ex, params, counts, increment, c, ha, hb));
            } else {
              // Intermediate node, allows us to add two features.
              String node2 = "T2"+c+"="+prev_ha+","+prev_hb;
              H.addSumNode(node2);
              int f1 = featureIndexer.getIndex(new BinaryFeature(prev_ha, ha));
              if (params != null)
                H.addEdge(node, node2, hiddenEdgeInfo(params, counts, f1, increment, hiddenNodeIndex(0, c), ha));
              int f2 = featureIndexer.getIndex(new BinaryFeature(prev_hb, hb));
              if (params != null)
                H.addEdge(node2, genNode(H, ex, params, counts, increment, c, ha, hb), hiddenEdgeInfo(params, counts, f2, increment, hiddenNodeIndex(1, c), hb));
            }
          }
        }
      }
      return node;
    }

    public Hypergraph<Example> createHypergraph(int L, Example ex, double[] params, double[] counts, double increment) {
      Hypergraph<Example> H = new Hypergraph<Example>();
      //H.debug = true;
      H.addEdge(H.sumStartNode(), transNode(H, ex, params, counts, increment, 0, -1, -1));
      return H;
    }
  }
}
