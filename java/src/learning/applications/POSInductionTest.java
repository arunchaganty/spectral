package learning.applications;

import learning.data.ParsedCorpus;
import org.junit.Test;
import org.junit.Assert;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.List;

/**
 * Tests for the POSInduction class
 */
public class POSInductionTest {

  @Test
  public void testReadData() throws IOException {
    InputStream in = this.getClass().getClassLoader()
                                .getResourceAsStream("learning/applications/POSInduction_data.txt");
    in.mark(0);
    List<String> input = fig.basic.IOUtils.readLines(new BufferedReader(new InputStreamReader(in)));
    in.reset();
    ParsedCorpus out = POSInduction.readData(new InputStreamReader(in));
    List<String> output = out.toLines();
    Assert.assertEquals(input, output);
  }
}
