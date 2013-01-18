/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning;

import learning.Misc;
import learning.linalg.MatrixOps;

import org.junit.Assert;
import org.junit.Test;
import org.junit.Before;

/**
 * 
 */
public class MiscTest {

  @Test 
  public void traverseMultiCombination() {
    Misc.traverseMultiCombination( 3, 1, new Misc.TraversalFunction() {
      int i = 0;
      int[][] values = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},};
      @Override
      public void run( int[] state ) {
        Assert.assertTrue( MatrixOps.equal( values[i], state ) );
        i++;
      }
    } );

    Misc.traverseMultiCombination( 3, 2, new Misc.TraversalFunction() {
      int i = 0;
      int[][] values = {
        { 2, 0, 0 }, 
        { 1, 1, 0 },
        { 1, 0, 1 },
        { 0, 2, 0 },
        { 0, 1, 1 },
        { 0, 0, 2 }, };
      @Override
      public void run( int[] state ) {
        Assert.assertTrue( MatrixOps.equal( values[i], state ) );
        i++;
      }
    } );

  }

}
