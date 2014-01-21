/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
  package learning.applications;

import fig.basic.*;
import fig.basic.IOUtils;
import fig.exec.Execution;
import learning.data.ComputableMoments;
import learning.data.Corpus;
import learning.data.MomentComputationWorkers;
import learning.data.ParsedCorpus;
import learning.linalg.MatrixOps;
import learning.models.BasicParams;
import learning.models.Params;
import learning.models.loglinear.Example;
import learning.unsupervised.ExpectationMaximization;
import learning.models.loglinear.UndirectedHiddenMarkovModel;
import learning.spectral.TensorMethod;
import learning.common.*;
import learning.common.Utils;
import learning.unsupervised.MeasurementsEM;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Quartet;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static fig.basic.LogInfo.*;
import static learning.models.loglinear.UndirectedHiddenMarkovModel.*;
import static learning.common.Utils.outputList;

/**
 * Perform POS induction, aka HMM learning.
 */
public class POSInduction implements Runnable {

  @OptionSet(name="em")
  public ExpectationMaximization emSolver = new ExpectationMaximization();
  @OptionSet(name="mem")
  public MeasurementsEM mEMSolver = new MeasurementsEM();
  @Option(gloss="em iterations")
  public int iters = 200;
  @Option(gloss="em eps")
  public double eps = 1e-4;

  @Option(gloss="Seed to generate initial point")
  public Random initRandom = new Random();
  @Option(gloss="Noise in parameters")
  public double initParamsNoise = 1.0;

  @OptionSet(name="TensorMethod")
  public TensorMethod tensorMethod = new TensorMethod();

  @Option(gloss="File containing text in word_TAG format")
  public File trainData;
  @Option(gloss="File containing text in word_TAG format")
  public File testData;
  @Option(gloss="File containing trained model parameters")
  public File modelPath;
  @Option(gloss="Threshold before grouping words into classes")
  public int THRESHOLD = 0;

  @Option(gloss="Smooth the spectral measurements")
  public double smoothMeasurements = 1e-2;

  @Option(gloss="Threads")
  public int nThreads = 1;

  @Option(gloss="Verbosity")
  public int VERBOSE = 0;

  public static enum TrainMode {
    EM,
    TrueMeasurements,
    SpectralMeasurements
  }
  @Option(gloss="Training mode")
  public TrainMode mode = TrainMode.EM;

  @Option(gloss="Possibly truncate the number of sentences read")
    public int maxSentences = Integer.MAX_VALUE;

  public List<Example> corpusToExamples( Corpus C, int maxN ) {
     List<Example> examples = new ArrayList<>();

     for( int i = 0; i < C.getInstanceCount() && i < maxN; i++ ) {
       examples.add( new Example( C.C[i] ) );
     }

     return examples;
  }
  public List<Example> corpusToExamples( ParsedCorpus C, int maxN ) {
     List<Example> examples = new ArrayList<>();

     for( int i = 0; i < C.getInstanceCount() && i < maxN; i++ ) {
       Example ex = new Example( C.C[i], C.L[i] );
       examples.add( ex );
       // LogInfo.logs( Fmt.D( ex.x ) );
     }
     return examples;
  }
  public void truncate( ParsedCorpus C, int maxN ) {
    if( maxN > C.getInstanceCount() ) return;

    int[][] C_ = new int[ maxN ][];
    int[][] L_ = new int[ maxN ][];
    for( int i = 0; i < maxN; i++ ) {
      C_[i] = new int[ C.C[i].length ];
      L_[i] = new int[ C.L[i].length ];
      System.arraycopy( C.C[i], 0, C_[i], 0, C.C[i].length );
      System.arraycopy( C.L[i], 0, L_[i], 0, C.L[i].length );
    }
    C.C = C_;
    C.L = L_;
  }

  public void printTopK(UndirectedHiddenMarkovModel model, Params params, ParsedCorpus C, int[] perm) {
    int K = model.getK();
    // (b) Print top 10 words
    LogInfo.begin_track("Top-k words");
    int TOP_K = 20;
    for( int k = 0; k < K; k++ ) {
      StringBuilder topk = new StringBuilder();
      int k_ = perm[k];

      topk.append(C.tagDict[k_]).append(": ");

      Parameters parameters = (Parameters) params;
      double[] options = new double[model.D];
      for(int d = 0; d < model.D; d++)
        options[d] = parameters.weights[parameters.o(k, d)];
      Integer[] sortedWords = MatrixOps.argsort(options);
      for(int i = 0; i < TOP_K; i++ ) {
        topk.append(C.dict[sortedWords[i]]).append(", ");
      }
      logsForce(topk);
    }
    LogInfo.end_track("Top-k words");
  }

  /**
   * Print the viterbi decoding of the first @nExamples of @C
   */
  public void printExamples(UndirectedHiddenMarkovModel model, Params params, ParsedCorpus C, int nExamples, int[] perm) {
    begin_track("Examples");
    for(int i = 0; i < nExamples; i++) {
      begin_track("Example " + i);
      Example ex = new Example( C.C[i], C.L[i]);
      int[] tags = model.viterbi(params, ex);
      // Push through the dictionary
      for(int tag = 0; tag < tags.length; tag++) tags[tag] = perm[tags[tag]];

      // Print example and sentence.
      log(C.translateSentence(C.C[i]));
      log(C.translateTags(tags));
      end_track("Example " + i);
    }

    end_track("Examples");
  }

  public Pair<Double, Double> reportAccuracies(UndirectedHiddenMarkovModel model, Params params, ParsedCorpus C, int verbose) {
    begin_track("accuracy");
    params.cache(); // Useful!
    // Create a confusion matrix
    int K = C.getTagDimension();
    double[][] confusion = new double[K][K];
    double[][] confusionVsAll = new double[K][K]; // Align differently
    for( int n = 0; n < C.getInstanceCount(); n++ ) {
      int[] l = C.L[n];
      int[] l_ = model.viterbi( (Parameters)params, new Example(C.C[n]) );

      for( int i = 0; i < l.length; i++ ) {
        confusion[l[i]][l_[i]] += 1;
        confusionVsAll[l_[i]][l[i]] += 1;
      }
    }

    begin_track("best-match accuracy");
    double acc; int[] perm;
    {
      // Find best alignment
      BipartiteMatcher matcher = new BipartiteMatcher();
      perm = matcher.findMaxWeightAssignment(confusion);
      // Compute hamming score
      long correct = 0;
      long total = 0;
      for( int k = 0; k < K; k++ ) {
        for( int k_ = 0; k_ < K; k_++ ) {
          total += confusion[k][k_];
        }
        correct += confusion[k][perm[k]];
      }
      acc = (double) correct/ (double) total;
    }
    end_track("best-match accuracy");

    begin_track("vs-all accuracy");
    double accVsAll;
    {
      // Find greedy alignment
      int[] permVsAll = new int[K];
      for(int k = 0; k < K; k++) permVsAll[k] = MatrixOps.argmax(confusionVsAll[k]);

      // Compute hamming score
      long correct = 0;
      long total = 0;
      for( int k = 0; k < K; k++ ) {
        for( int k_ = 0; k_ < K; k_++ ) {
          total += confusionVsAll[k][k_];
        }
        correct += confusionVsAll[k][permVsAll[k]];
      }
      accVsAll = (double) correct/ (double) total;
    }
    end_track("vs-all accuracy");




    // Now (a) print a confusion matrix
    if(verbose > 2) {
      LogInfo.begin_track("Confusion matrix");
      StringBuilder table = new StringBuilder();
      table.append( "\t" );
      for( int k = 0; k < K; k++ ) {
        table.append(C.tagDict[k]).append("\t");
      }
      table.append("\n");
      for( int k = 0; k < K; k++ ) {
        table.append(C.tagDict[k]).append("\t");
        for( int k_ = 0; k_ < K; k_++ ) {
          table.append(confusion[k][perm[k_]]).append("\t");
        }
        table.append( "\n" );
      }
      logsForce(table);
      LogInfo.end_track("Confusion matrix");
    }
    if(verbose > 1)
      // (b) Print top 10 words
      printTopK(model, params, C, perm);
    if(verbose > 0)
      // (c) print top 10 examples
      printExamples(model, params, C, 10, perm);

    end_track("accuracy");

    return Pair.newPair(acc, accVsAll);
  }

  String logStat(String key, Object value) {
    LogInfo.logs("%s = %s", key, value);
    Execution.putOutput(key, value);
    return key+"="+value;
  }

  double getAverageLength(final Corpus C) {
    double avgLength = 0.;
    int count = 0;
    for(int[] sentence : C.C) {
      avgLength += (sentence.length - avgLength)/(++count);
    }

    return avgLength;
  }

  Params computeFromSpectral(final ParsedCorpus C) {
    int D = C.getDimension();
    int K = C.getTagDimension();

    LogInfo.begin_track("Spectral recovery");

    Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, SimpleMatrix> result =
            tensorMethod.randomizedRecoverParameters(K,
                    new ComputableMoments() {
                      @Override
                      public MatrixOps.Matrixable computeP13() {
                        // NOTE: We want the middle element ("x2") to be 'x3' for stable recovery
                        return MomentComputationWorkers.matrixable(C, 0, 1, nThreads);
                      }

                      @Override
                      public MatrixOps.Matrixable computeP12() {
                        return MomentComputationWorkers.matrixable(C, 0, 2, nThreads);
                      }

                      @Override
                      public MatrixOps.Matrixable computeP32() {
                        return MomentComputationWorkers.matrixable(C, 1, 2, nThreads);
                      }

                      @Override
                      public MatrixOps.Tensorable computeP123() {
                        return MomentComputationWorkers.tensorable(C, 0, 2, 1, nThreads);
                      }
                    });
    // Create a fake indexer for all of this
    SimpleMatrix pi = result.getValue0();
    SimpleMatrix O = result.getValue3(); // Because of above magic
    pi = MatrixOps.projectOntoSimplex( pi.transpose(), smoothMeasurements ).transpose();
    O = MatrixOps.projectOntoSimplex( O, smoothMeasurements );
    LogInfo.end_track("Spectral recovery");

    LogInfo.begin_track("Converting to measurements");
    Indexer<String> measuredFeatureIndexer = new Indexer<>();
    for( int h = 0; h < K; h++ ) {
      for( int x = 0; x < D; x++ ) {
        // O
        measuredFeatureIndexer.add( UndirectedHiddenMarkovModel.oString(h, x) );
      }
    }

    Params measurements = new BasicParams(measuredFeatureIndexer);
    double L = getAverageLength(C);
    for( int h = 0; h < K; h++ ) {
      for( int x = 0; x < D; x++ ) {
        // Assuming identical distribution.
        // multiplying by pi to go from E[x|h] -> E[x,h]
        // multiplying by L because true.counts aggregates
        // over x1, x2 and x3.
        measurements.set(
                UndirectedHiddenMarkovModel.oString(h, x),
                L * O.get( x, h ) * pi.get(h));
      }
    }
    LogInfo.end_track("Converting to measurements");

    return measurements;
  }




  /**
   * Reads a data file in the format word_TAG,
   * Example:
   *    Pierre_NNP Vinken_NNP ,_, 61_CD years_NNS old_JJ ,_, will_MD join_VB the_DT board_NN as_IN a_DT nonexecutive_JJ director_NN Nov._NNP 29_CD ._.
   * @param input
   * @return
   */
  public ParsedCorpus readData(File input) {
    Indexer<String> wordIndex = new Indexer<>();
    Indexer<String> tagIndex = new Indexer<>();
    List<int[]> sentences = new ArrayList<>();
    List<int[]> answers = new ArrayList<>();
    Counter<String> counter = new Counter<>();

    wordIndex.add(Corpus.UPPER_CLASS);
    wordIndex.add(Corpus.LOWER_CLASS);
    wordIndex.add(Corpus.DIGIT_CLASS);
    wordIndex.add(Corpus.MISC_CLASS);
    // Anything that appears less than THRESHOLD times is set to rare
    try(BufferedReader stream = Utils.openReader(input)) {
      // Compute the words and statistics
      int lineCount = 0;
      for(String line : IOUtils.readLines(stream)) {
        for(String token : line.split(" ")) {
          String[] word_tag = token.split("_");
          assert word_tag.length == 2;
          String word = word_tag[0];
          counter.add(word);
        }
        if(++lineCount > maxSentences) break;
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    try(BufferedReader stream = learning.common.Utils.openReader(input)) {
      // Go back and actually add the words to the corpus
      int lineCount = 0;
      for(String line : IOUtils.readLines(stream)) {
        String[] tokens = line.split(" ");
        if(tokens.length == 0) continue;

        int[] words = new int[tokens.length];
        int[] tags = new int[tokens.length];
        for(int i = 0; i < tokens.length; i++) {
          String token = tokens[i];
          String[] word_tag = token.split("_");
          assert word_tag.length == 2;
          String word = word_tag[0];

          // Handle rare words
          if( counter.getCount(word) < THRESHOLD ) {
            if(Character.isUpperCase(word.charAt(0)))
              word = Corpus.UPPER_CLASS;
            else if(Character.isDigit(word.charAt(0)))
              word = Corpus.DIGIT_CLASS;
            else if(Character.isLowerCase(word.charAt(0)))
              word = Corpus.LOWER_CLASS;
            else
              word = Corpus.MISC_CLASS;
          }

          String tag = word_tag[1];
          words[i] = wordIndex.getIndex(word);
          tags[i] = tagIndex.getIndex(tag);
        }
        sentences.add(words);
        answers.add(tags);

        if(++lineCount > maxSentences) break;
      }
      // TODO: handle close with a finally
    } catch (IOException e) {
        throw new RuntimeException(e);
    }

    return new ParsedCorpus(
      wordIndex.toArray(new String[wordIndex.size()]),
            sentences.toArray(new int[sentences.size()][]),
      tagIndex.toArray(new String[tagIndex.size()]),
              answers.toArray(new int[answers.size()][])
      );
  }


  public void train(File dataFile) {
    try {
      LogInfo.begin_track("file-input");
      // Read data as word-index sequences
      ParsedCorpus C = readData(dataFile);
      // Convert to Example sequences
      LogInfo.logs( "Corpus has %d instances, with %d words and %d tags",
              C.getInstanceCount(), C.getDimension(), C.getTagDimension() );
      truncate( C, maxSentences );
      LogInfo.end_track();

      Counter<Example> data = new Counter<>(corpusToExamples(C, C.getInstanceCount()));

      // TODO: Optionally featurize the data.
      // Train an (undirected) HMM
      int K = C.getTagDimension();
      int D = C.getDimension();
      int L = 3;
      UndirectedHiddenMarkovModel hmm = new UndirectedHiddenMarkovModel(K, D, L);
      Params params = hmm.newParams();
      params.initRandom(initRandom, initParamsNoise);

      switch(mode) {
        case EM: {
          ExpectationMaximization solver = emSolver;
          solver.backtrack.tolerance = eps;

          ExpectationMaximization.EMState state = solver.newState(hmm, data, params);

          for(int i = 0; i < iters; i++) {
            // Report likelihood.
            Pair<Double, Double> acc = reportAccuracies(hmm, params, C, VERBOSE);
            LogInfo.log(outputList(
                    "iter", i,
                    "objective", state.objective.value(),
                    "accuracy", acc.getFirst(),
                    "vs-all-accuracy", acc.getSecond())
                    );
            // Print information.
            solver.takeStep(state);
          }

          // Report likelihood.
          Pair<Double, Double> acc = reportAccuracies(hmm, params, C, VERBOSE);
          LogInfo.log(outputList(
                  "objective", state.objective.value(),
                  "accuracy", acc.getFirst(),
                  "vs-all-accuracy", acc.getSecond())
          );
        } break;
        case SpectralMeasurements: {
          MeasurementsEM solver = mEMSolver;
          solver.backtrack.tolerance = eps;

          // Recover from spectral
          Params measurements = computeFromSpectral(C);

          Params beta = measurements.newParams();

          MeasurementsEM.State state = solver.newState(hmm, hmm, data, measurements, params, beta);

          for(int i = 0; i < iters; i++) {
            // Report likelihood.
            Pair<Double, Double> acc = reportAccuracies(hmm, params, C, VERBOSE);
            LogInfo.log(outputList(
                    "iter", i,
                    "m-objective", state.mObjective.value(),
                    "e-objective", state.eObjective.value(),
                    "accuracy", acc.getFirst(),
                    "vs-all-accuracy", acc.getSecond())
            );
            // Print information.
            solver.takeStep(state);
          }

          // Report likelihood.
          Pair<Double, Double> acc = reportAccuracies(hmm, params, C, VERBOSE);
          LogInfo.log(outputList(
                  "m-objective", state.mObjective.value(),
                  "e-objective", state.eObjective.value(),
                  "accuracy", acc.getFirst(),
                  "vs-all-accuracy", acc.getSecond())
          );
        } break;
        default:
          throw new RuntimeException("Method not supported.");
      }

      String output = Execution.getFile(String.format("model-%s.ser", mode));
      IOUtils.writeObjFile(output, params);

      test(dataFile, new File(output));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public void test(File dataFile, File modelPath) {
    try {
      LogInfo.begin_track("file-input");
      // Read data as word-index sequences
      ParsedCorpus C = readData(dataFile);
      // Convert to Example sequences
      LogInfo.logs( "Corpus has %d instances, with %d words and %d tags",
              C.getInstanceCount(), C.getDimension(), C.getTagDimension() );
//      truncate( C, maxSentences );
      LogInfo.end_track();

      Counter<Example> data = new Counter<>(corpusToExamples(C, C.getInstanceCount()));

      // TODO: Optionally featurize the data.
      // Train an (undirected) HMM
      int K = C.getTagDimension();
      int D = C.getDimension();
      int L = 3;
      UndirectedHiddenMarkovModel hmm = new UndirectedHiddenMarkovModel(K, D, L);
      Parameters params = (Parameters) IOUtils.readObjFile(modelPath);

      // Report likelihood.
      Pair<Double, Double> acc = reportAccuracies(hmm, params, C, VERBOSE);
      LogInfo.log(outputList(
              "accuracy", acc.getFirst(),
              "vs-all-accuracy", acc.getSecond())
      );

    } catch (IOException | ClassNotFoundException e) {
      throw new RuntimeException(e);
    }
  }

  public void run() {
    // Train an (undirected) HMM
    train(trainData);
    // Evaluate a (undirected) HMM
//    test(testData, modelPath);

  }
  
  public static void main(String[] args) {
    Execution.run(args, new POSInduction() );
  }
}


