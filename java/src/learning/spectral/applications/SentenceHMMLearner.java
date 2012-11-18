/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.spectral.applications;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import fig.basic.Option;
import fig.basic.OptionsParser;

import learning.models.SentenceHMM;
import learning.utils.ParsedCorpus;

/**
 * 
 */
public class SentenceHMMLearner {
	
	@Option(gloss="Whether to do a +1 smoothing")
	public boolean shouldSmooth = false;
	@Option(gloss="Corpus location", required = true)
	public String inputPath;
	@Option(gloss="Output location for hmm", required = true)
	public String outputPath;
	@Option(gloss="Number of words to truncate in each hidden state")
	public int wordLimit = 1000;
	@Option(gloss="Number of components")
	public int componentCount = -1;
	@Option(gloss="Is fully observed?")
	public boolean isObserved = true;
	
	public void fit() throws IOException {
		//TODO: Add a control for the number of classes
		ParsedCorpus C = ParsedCorpus.parseText(Paths.get(inputPath));
		
		if( componentCount != -1 ) 
			C.shrink(componentCount);
		
		// Use the Z and C to learn the HMM parameters
		SentenceHMM hmm;
		if( isObserved )
			hmm = SentenceHMM.learnFullyObserved( C, wordLimit, shouldSmooth );
		else
			throw new NotImplementedException();
		
		ObjectOutputStream out = new ObjectOutputStream( new FileOutputStream( outputPath ) ); 
		out.writeObject(hmm);
		out.close();
	}
	
	/**
	 * Word Clustering using a HMM kind of model
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		SentenceHMMLearner prog = new SentenceHMMLearner();
		OptionsParser parser = new OptionsParser(prog);
		parser.doParse(args);
		prog.fit();
	}
}
