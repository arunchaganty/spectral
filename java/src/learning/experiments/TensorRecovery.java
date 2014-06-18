package learning.experiments;

import fig.basic.*;
import fig.exec.Execution;
import learning.data.ComputableMoments;
import learning.data.HasSampleMoments;
import learning.linalg.FullTensor;
import learning.linalg.MatrixOps;
import learning.models.ExponentialFamilyModel;
import learning.models.loglinear.Example;
import learning.models.loglinear.Models;
import learning.models.loglinear.ParamsVec;
import learning.models.loglinear.UndirectedHiddenMarkovModel;
import learning.spectral.TensorMethod;
import learning.common.Counter;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Quartet;

import java.util.Random;

/**
 * Tests methods for tensor recovery
 */
public class TensorRecovery implements  Runnable {

  @OptionSet(name="TensorMethod") public TensorMethod tf = new TensorMethod();

  public enum ModelType { mixture, hmm, tallMixture, grid, factMixture };
  public static class ModelOptions {
    @Option(gloss="Type of modelA") public ModelType modelType = ModelType.mixture;
    @Option(gloss="Number of values of the hidden variable") public int K = 2;
    @Option(gloss="Number of possible values of output") public int D = 2;
    @Option(gloss="Length of observation sequence") public int L = 3;
  }
  @OptionSet(name="modelA") public ModelOptions modelOpts = new ModelOptions();

  public static class GenerationOptions {
    @Option(gloss="Random seed for generating artificial data") public Random genRandom = new Random(42);
    @Option(gloss="Random seed for the true modelA") public Random trueParamsRandom = new Random(43);
    @Option(gloss="Number of examples to generate") public int genNumExamples = 100;
    @Option(gloss="How much variation in true parameters") public double trueParamsNoise = 1.0;
  }
  @OptionSet(name="gen") public GenerationOptions genOpts = new GenerationOptions();;

  ExponentialFamilyModel<Example> modelA;

  /**
   * Generates random data from the modelA.
   *  - Uses genRand as a seed.
   */
  ParamsVec generateParameters( ExponentialFamilyModel<Example> model, GenerationOptions opts ) {
    ParamsVec trueParams = (ParamsVec) model.newParams();
    trueParams.initRandom(opts.trueParamsRandom, opts.trueParamsNoise);
//    for(int i = 0; i < trueParams.weights.length; i++)
//      trueParams.weights[i] = Math.sin(i);
    return trueParams;
  }

  /*
  * Generates a modelA of a particular type
  */
  void initializeModels(ModelOptions opts) {
    switch (opts.modelType) {
      case mixture: {
        modelA = new Models.MixtureModel(opts.K, opts.D, opts.L);
        break;
      }
      case hmm: {
        modelA = new UndirectedHiddenMarkovModel(opts.K, opts.D, opts.L);
        break;
      }
      case tallMixture: {
        throw new RuntimeException("Tall mixture not implemented");
        //break;
      }
      case grid: {
        modelA = new Models.GridModel(opts.K, opts.D, opts.L);
        break;
      }
      default:
        throw new RuntimeException("Unhandled modelA type: " + opts.modelType);
    }
  }

  /**
   * Computes moments from the sequences in an Example.
   */
  class ExampleMoments implements ComputableMoments, HasSampleMoments {
    //List<Integer> indices;
    SimpleMatrix P13;
    SimpleMatrix P12;
    SimpleMatrix P32;
    FullTensor P123;
    double preconditioning = 0.0;

    public ExampleMoments() {
    }
    public ExampleMoments(double preconditioning) {
      this.preconditioning = preconditioning;
    }

    <T> ExampleMoments(ExponentialFamilyModel<T> model, final Counter<T> data) {
      int D = model.getD();
      // Create P13
      P13 = new SimpleMatrix(D, D);
      P12 = new SimpleMatrix(D, D);
      P32 = new SimpleMatrix(D, D);
      P123 = new FullTensor(D,D,D);
      double count = 0;
      for( T ex : data ) {
        count += model.updateMoments(ex, data.getCount(ex), P12, P13, P32, P123);
      }
      // Scale down everything
      P13 = P13.scale(1./count);
      P12 = P12.scale(1./count);
      P32 = P32.scale(1./count);
      P123.scale(1./count);

      // Add some to the diagonal term
      if( preconditioning > 0. ) {
        for( int d = 0; d < D; d++ ) {
          P123.set(d,d,d, P123.get(d,d,d) + preconditioning);
        }
      }
      P123.scale(1./P123.elementSum());
    }

    @Override
    public MatrixOps.Matrixable computeP13() {
      return MatrixOps.matrixable(P13);
    }

    @Override
    public MatrixOps.Matrixable computeP12() {
      return MatrixOps.matrixable(P12);
    }

    @Override
    public MatrixOps.Matrixable computeP32() {
      return MatrixOps.matrixable(P32);
    }

    @Override
    public MatrixOps.Tensorable computeP123() {
      return MatrixOps.tensorable(P123);
    }

    @Override
    public Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> computeSampleMoments(int N) {
      return Quartet.with(P13, P12, P32, P123);
    }
  }

  void computeSpectralMoments( final Counter<Example> data ) {
    LogInfo.begin_track("solveBottleneck");
    ParamsVec measurements;
    // Construct triples of three observed variables around the hidden
    // node.
    int K = modelA.getK(); int D = modelA.getD();

    Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> moments_ = new ExampleMoments(modelA, data).computeSampleMoments(0);
    tf.recoverParametersAsymmetric(K, moments_.getValue3());
  }


  public void run() {
    // Setup modelA, modelB
    initializeModels( modelOpts );

    // Generate parameters
    ParamsVec trueParams = generateParameters( modelA, genOpts );

    // Get true parameters
//    Counter<Example> data = modelA.drawSamples(trueParams, genOpts.genRandom, genOpts.genNumExamples);
    Counter<Example> data = modelA.getDistribution(trueParams);

    computeSpectralMoments(data);
  }

  public static void main(String[] args) {
    Execution.run(args, new TensorRecovery());
  }

}
