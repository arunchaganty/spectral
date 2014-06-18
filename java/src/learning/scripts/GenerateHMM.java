package learning.scripts;

import fig.basic.LogInfo;
import fig.basic.Option;
import fig.exec.Execution;

import java.io.IOException;

/**
 * Script to generate a random HMM
 */
public class GenerateHMM {

  @Option(gloss="Number of states")
  public int stateCount = 2;
  @Option(gloss="Number of emmissions")
  public int emissionCount = 2;

  public void run() {
  }

  public static void main(String[] args) {
    Execution.run(args, new GenerateHMM());
  }
}
