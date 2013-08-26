package learning.scripts;

import fig.basic.LogInfo;
import fig.basic.Option;
import fig.exec.Execution;
import learning.data.ParsedCorpus;
import learning.models.HiddenMarkovModel;
import org.apache.commons.lang3.StringUtils;
import org.javatuples.Pair;

import java.io.*;
import java.util.Random;

/**
 * Learn a fully supervised HMM
 */
public class HMMGenerateData implements Runnable {

  @Option(gloss="HMM File to read from", required=true)
  public String inputPath;
  @Option(gloss="File to output words", required=true)
  public String wordOutputPath;
  @Option(gloss="File to output labels", required=true)
  public String labelOutputPath;

  @Option(gloss="Number of sentences to produce")
  public int numSentences = 100;
  @Option(gloss="Sentence length")
  public int numWords = 10;

  @Option(gloss="Generator for words")
  public Random rnd = new Random();


  public void run() {
    try {
      LogInfo.begin_track("file-input");
      // Write out to a file.
      ObjectInputStream in = new ObjectInputStream(new FileInputStream(inputPath));
      HiddenMarkovModel model = (HiddenMarkovModel) in.readObject();
      LogInfo.end_track("file-input");
      LogInfo.begin_track("generation");

      OutputStreamWriter wordOut = new OutputStreamWriter(new FileOutputStream(wordOutputPath));
      OutputStreamWriter labelOut = new OutputStreamWriter(new FileOutputStream(labelOutputPath));
      for(int sent = 0; sent < numSentences; sent++) {
        Pair<int[],int[]> sent_ = model.generate(rnd, numWords);
        wordOut.write( StringUtils.join(sent_.getValue0(), " "));
        wordOut.write("\n");

        labelOut.write( StringUtils.join(sent_.getValue1(), " "));
        labelOut.write("\n");
      }
      wordOut.close();
      labelOut.close();
      LogInfo.end_track("generation");


    } catch (IOException | ClassNotFoundException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) {
    Execution.run(args, new LearnHMM());
  }
}
