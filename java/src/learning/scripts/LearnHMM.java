package learning.scripts;

import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.OptionSet;
import fig.exec.Execution;
import learning.data.ParsedCorpus;
import learning.models.HiddenMarkovModel;

import java.io.*;

/**
 * Learn a fully supervised HMM
 */
public class LearnHMM implements Runnable {

  @OptionSet(name="corpus")
  public ParsedCorpus.Options corpusOptions = new ParsedCorpus.Options();

  @Option(gloss="File to output to", required=true)
  public String outputPath;

  public void run() {
    try {
    LogInfo.begin_track("file-input");
    // Read data as word-index sequences
    ParsedCorpus C = null;
      C = ParsedCorpus.parseText(
              corpusOptions.dataPath, corpusOptions.mapPath,
              corpusOptions.labelledDataPath, corpusOptions.labelledMapPath);
      LogInfo.end_track("file-input");
      LogInfo.begin_track("learn-model");
      HiddenMarkovModel model = HiddenMarkovModel.learnFullyObserved(
              C.getTagDimension(),
              C.getDimension(),
              C.C,
              C.L,
              true);
      LogInfo.end_track("learn-model");

      // Write out to a file.
      ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(outputPath));
      out.writeObject(model);
      out.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) {
    Execution.run(args, new LearnHMM());
  }
}
