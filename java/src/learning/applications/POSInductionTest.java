package learning.applications;

import learning.data.ParsedCorpus;
import org.junit.Test;
import org.junit.Assert;

import java.io.*;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.List;

/**
 * Tests for the POSInduction class
 */
public class POSInductionTest {

  @Test
  public void testReadData() throws IOException, URISyntaxException {
    URL url = this.getClass().getClassLoader()
            .getResource("learning/applications/POSInduction_data.txt");
    File in = null;
    in = new File(url.toURI());

    List<String> input = fig.basic.IOUtils.readLines(Files.newBufferedReader(in.toPath(), Charset.forName("UTF-8")));
    POSInduction program = new POSInduction();
    ParsedCorpus out = program.readData(in);
    List<String> output = out.toLines();
    Assert.assertEquals(input, output);
  }
}
