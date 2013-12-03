package learning.experiments;

import fig.basic.*;
import fig.exec.Execution;
import learning.models.loglinear.Example;
import learning.models.loglinear.ParamsVec;
import learning.models.loglinear.UndirectedHiddenMarkovModel;
import learning.utils.Counter;

import java.util.ArrayList;
import java.util.List;

/**
 * Compute marginal counts
 */
public class MarginalCounts implements  Runnable {
  @Option(gloss="length of sequence") public int L = 3;

  /**
   * Choose idx
   */
  public void generateExamples(Example current, int idx, List<Example> examples) {
    if( idx == current.x.length ) {
      examples.add(new Example(current.x));
    } else {
      // Make a choice for this index
      current.x[idx] = 0;
      generateExamples(current, idx+1, examples);
      current.x[idx] = 1;
      generateExamples(current, idx+1, examples);
    }
  }
  public List<Example> generateExamples(int L) {
    List<Example> examples = new ArrayList<>((int)Math.pow(2,L));
    Example ex = new Example(new int[L]);
    generateExamples(ex, 0, examples);
    return examples;
  }

  public void run() {
    // Initialize the Hidden Markov model
    int K = 2; int D = 2;
    UndirectedHiddenMarkovModel model;
    model = new UndirectedHiddenMarkovModel(K, D, L);

    ParamsVec params1 = model.newParamsVec();
    params1.set(UndirectedHiddenMarkovModel.o(0, 0), Math.log(1.));
    params1.set(UndirectedHiddenMarkovModel.o(0, 1), Math.log(1.));
    params1.set(UndirectedHiddenMarkovModel.o(1, 0), Math.log(1.));
    params1.set(UndirectedHiddenMarkovModel.o(1, 1), Math.log(1.));

    params1.set(UndirectedHiddenMarkovModel.t(0, 0), Math.log(1));
    params1.set(UndirectedHiddenMarkovModel.t(0, 1), Math.log(4.));
    params1.set(UndirectedHiddenMarkovModel.t(1, 0), Math.log(4.));
    params1.set(UndirectedHiddenMarkovModel.t(1, 1), Math.log(1.));

//    params1.set(UndirectedHiddenMarkovModel.o(0, 0), 0.4549482263236486);
//    params1.set(UndirectedHiddenMarkovModel.o(0, 1), 0.7055357649634446);
//    params1.set(UndirectedHiddenMarkovModel.o(1, 0), -0.9284808331570149);
//    params1.set(UndirectedHiddenMarkovModel.o(1, 1), 0.15254212472490214);
//
//    params1.set(UndirectedHiddenMarkovModel.t(0, 0), 0.7580337425845372);
//    params1.set(UndirectedHiddenMarkovModel.t(0, 1), 0.894597081189912);
//    params1.set(UndirectedHiddenMarkovModel.t(1, 0), -0.8812829777726168);
//    params1.set(UndirectedHiddenMarkovModel.t(1, 1), -0.5384561639523937);

    {
      LogInfo.begin_track("Params 1");
      LogInfo.logs("Params: " + params1);
      ParamsVec marginals = model.getMarginals(params1);
      LogInfo.logs("Marginals: " + marginals);

      List<Example> hiddenStates = generateExamples(L);

      for( Example ex : generateExamples(L) ) {
        for( Example hx : hiddenStates ) {
          ex.h = hx.x;
          LogInfo.logs(ex + ": " + model.getFullProbability(params1, ex));
        }
      }
      LogInfo.end_track("Params 1");
    }

    ParamsVec params2 = model.newParamsVec();
    params2.set(UndirectedHiddenMarkovModel.o(0, 0), Math.log(4.));
    params2.set(UndirectedHiddenMarkovModel.o(0, 1), Math.log(1.));
    params2.set(UndirectedHiddenMarkovModel.o(1, 0), Math.log(1.));
    params2.set(UndirectedHiddenMarkovModel.o(1, 1), Math.log(4.));

    params2.set(UndirectedHiddenMarkovModel.t(0, 0), Math.log(1));
    params2.set(UndirectedHiddenMarkovModel.t(0, 1), Math.log(1.));
    params2.set(UndirectedHiddenMarkovModel.t(1, 0), Math.log(1.));
    params2.set(UndirectedHiddenMarkovModel.t(1, 1), Math.log(1.));
//
//    params2.set(UndirectedHiddenMarkovModel.o(0, 0), 0.6939332743963367);
//    params2.set(UndirectedHiddenMarkovModel.o(0, 1), 0.6391140014026073);
//    params2.set(UndirectedHiddenMarkovModel.o(1, 0), -0.665006807009526);
//    params2.set(UndirectedHiddenMarkovModel.o(1, 1), 0.7482242232940524);
//
//    params2.set(UndirectedHiddenMarkovModel.t(0, 0),  0.0736020599531431);
//    params2.set(UndirectedHiddenMarkovModel.t(0, 1),  0.2341375127766598);
//    params2.set(UndirectedHiddenMarkovModel.t(1, 0), -0.434180932249893);
//    params2.set(UndirectedHiddenMarkovModel.t(1, 1), -0.3969206003267549);

    {
      LogInfo.begin_track("Params 2");
      LogInfo.logs("Params: " + params2);
      ParamsVec marginals = model.getMarginals(params2);
      LogInfo.logs("Marginals: " + marginals);

//      for( Example ex : generateExamples(L) )
//        LogInfo.logs(ex + ": " + model.getProbability(params2, ex));
      List<Example> hiddenStates = generateExamples(L);

      for( Example ex : generateExamples(L) ) {
        for( Example hx : hiddenStates ) {
          ex.h = hx.x;
          LogInfo.logs(ex + ": " + model.getFullProbability(params2, ex));
        }
      }
      LogInfo.end_track("Params 2");
    }

  }

  public static void main(String[] args) {
    Execution.run(args, new MarginalCounts());
  }
}
