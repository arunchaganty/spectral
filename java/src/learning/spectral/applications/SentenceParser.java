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

import learning.models.SentenceHMM;
import learning.utils.ParsedCorpus;

/**
 * 
 */
public class SentenceParser {
	
	public static void fit(Path fname, int wordsPerState, Path ofname) throws IOException {
		ParsedCorpus C = ParsedCorpus.parseText(fname);
		// Use the Z and C to learn the HMM parameters
		
		SentenceHMM hmm = SentenceHMM.learnFullyObserved(C.dict.length, C.Zdict.length, wordsPerState, C.C, C.Z, C.dict);
		
		ObjectOutputStream out = new ObjectOutputStream( new FileOutputStream( ofname.toFile() ) ); 
		out.writeObject(hmm);
		out.close();
	}
	
	/**
	 * Generate n sentences with m words each from the hmm stored in fname
	 * @param fname
	 * @param n
	 * @param m
	 * @param ofname
	 * @throws IOException
	 * @throws ClassNotFoundException 
	 */
	public static void generate(Path fname, int n, int m, Path ofname) throws IOException, ClassNotFoundException {
		ObjectInputStream in = new ObjectInputStream( new FileInputStream( fname.toFile() ) ); 
		SentenceHMM hmm = (SentenceHMM) in.readObject();
		in.close();
		
		BufferedWriter writer = Files.newBufferedWriter( ofname, Charset.defaultCharset() );
		for( int i = 0; i<n; i++ )
			writer.write( hmm.generateString(m) + "\n");
		writer.close();
	}
	
	public static void printUsage( String[] args ) {
		System.out.println("Usage: -f <fname> <n> <ofname>\t\t Fit a HMM to the data in <fname>. Save in <ofname>");
		System.out.println("Usage: -g <fname> <n> <m> <ofname>\t Generate <n> sentences of length <m> from HMM in <fname>. Save in <ofname>");
		System.out.println("Usage: -h \t\t\t\t This message");
	}

	/**
	 * Word Clustering using a HMM kind of model
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Get a better argument parsing mechanism
		
		if( args.length > 0 && args[0].equals("-h") ) {
			printUsage(args);
			System.exit(0);
		}
		
		if( args.length == 4 && args[0].equals("-f") ) {
			Path fname = Paths.get(args[1]);
			int n = Integer.parseInt(args[2]);
			Path ofname = Paths.get(args[3]);
			
			try {
				fit( fname, n, ofname );
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		else if (args.length == 5 && args[0].equals("-g")) {
			Path fname = Paths.get(args[1]);
			int n = Integer.parseInt(args[2]);
			int m = Integer.parseInt(args[3]);
			Path ofname = Paths.get(args[4]);
			
			try {
				generate(fname, n, m, ofname);
			} catch (ClassNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} else {
			printUsage(args);
			System.exit(1);
		}
	}
}
