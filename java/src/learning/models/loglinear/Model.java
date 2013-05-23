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
  int L, D;

  public Example newExample() {
    Example ex = new Example();
    ex.x = new int[L];
    return ex;
  }

  public Hypergraph<Example> createHypergraph(Example ex, double[] params, double[] counts, double increment) {
    Hypergraph<Example> H = new Hypergraph<Example>();
    //H.debug = true;
    Object rootNode = H.sumStartNode();
    for (int h = 0; h < K; h++) {  // For each value of hidden states...
      String hNode = "h="+h;
      H.addProdNode(hNode);
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
