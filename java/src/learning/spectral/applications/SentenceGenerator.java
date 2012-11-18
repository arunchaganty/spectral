/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.spectral.applications;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;


import fig.basic.Option;
import fig.basic.OptionsParser;

import learning.models.SentenceHMM;

/**
 * 
 */
public class SentenceGenerator {
	@Option(gloss="HMM location", required = true)
	public String inputPath;
	@Option(gloss="Output location for text", required = true)
	public String outputPath;
	@Option(gloss="Number of output classes")
	public int classes = -1;
	@Option(gloss="Number of words in a document")
	public int documentLength = 1000;
	@Option(gloss="Number of documents")
	public int documents = 1000;
	
	/**
	 * Generate n sentences with m words each from the hmm stored in fname
	 * @param fname
	 * @param n
	 * @param m
	 * @param ofname
	 * @throws IOException
	 * @throws ClassNotFoundException 
	 */
	public void generate() throws IOException, ClassNotFoundException {
		ObjectInputStream in = new ObjectInputStream( new FileInputStream( inputPath ) ); 
		SentenceHMM hmm = (SentenceHMM) in.readObject();
		in.close();
		
		BufferedWriter writer = Files.newBufferedWriter( Paths.get(outputPath), Charset.defaultCharset() );
		for( int i = 0; i < documents; i++ )
			writer.write( hmm.generateString(documentLength) + "\n");
		writer.close();
	}
	
	/**
	 * Word Clustering using a HMM kind of model
	 * @param args
	 * @throws IOException 
	 * @throws ClassNotFoundException 
	 */
	public static void main(String[] args) throws IOException, ClassNotFoundException {
		SentenceGenerator prog = new SentenceGenerator();
		OptionsParser parser = new OptionsParser(prog);
		if( parser.doParse(args) );
			prog.generate();
	}
}