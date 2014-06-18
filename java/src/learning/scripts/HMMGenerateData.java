package learning.scripts;

import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.OptionSet;
import fig.exec.Execution;
import learning.data.ParsedCorpus;
import learning.models.HiddenMarkovModelOld;
import org.apache.commons.lang3.StringUtils;
import org.javatuples.Pair;

import java.io.*;
import java.util.Random;

/**
 * Learn a fully supervised HMM
 */
public class HMMGenerateData implements Runnable {

  @OptionSet(name="Corpus")
  public ParsedCorpus.Options corpusOptions = new ParsedCorpus.Options();

  @Option(gloss="HMM File to read from", required=true)
  public String inputPath;
  @Option(gloss="Output prefix", required=true)
  public String outputPrefix;

  @Option(gloss="Number of sentences to produce")
  public int numSentences = 100;
  @Option(gloss="Sentence length")
  public int numWords = 10;

  @Option(gloss="Generator for words")
  public Random rnd = new Random();


  public void run() {
    try {
      LogInfo.begin_track("file-input");
      // Just parse the corpus for stupid fun.
      String[] dict = ParsedCorpus.readDict(corpusOptions.mapPath);
      String[] tagDict = ParsedCorpus.readDict(corpusOptions.labelledMapPath);

      // Write out to a file.
      ObjectInputStream in = new ObjectInputStream(new FileInputStream(inputPath));
      HiddenMarkovModelOld model = (HiddenMarkovModelOld) in.readObject();
      LogInfo.end_track("file-input");
      LogInfo.begin_track("generation");

      OutputStreamWriter wordOut = new OutputStreamWriter(new FileOutputStream(outputPrefix + ".words"));
      OutputStreamWriter labelOut = new OutputStreamWriter(new FileOutputStream(outputPrefix + ".tags"));
      OutputStreamWriter rawOut = new OutputStreamWriter(new FileOutputStream(outputPrefix + ".raw"));
      for(int sent = 0; sent < numSentences; sent++) {
        Pair<int[],int[]> sent_ = model.generate(rnd, numWords);
        int[] words = sent_.getValue0();
        int[] tags = sent_.getValue1();

        for(int i = 0; i < words.length; i++ ){
          wordOut.write( words[i] + " " );
          labelOut.write( tags[i] + " " );
          rawOut.write( tagDict[tags[i]] + " " + dict[words[i]] + " ");
        }
        wordOut.write("\n");
        labelOut.write("\n");
        rawOut.write("\n");
      }
      wordOut.close();
      labelOut.close();
      rawOut.close();
      LogInfo.end_track("generation");


    } catch (IOException | ClassNotFoundException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) {
    Execution.run(args, new HMMGenerateData());
  }
}
