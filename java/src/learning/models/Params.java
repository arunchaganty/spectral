package learning.models;

import java.io.Serializable;
import java.util.Random;

import static learning.utils.UtilsJ.writeStringHard;

/**
 * Uniform interface for (vectorizable)-parameters
 */
public abstract class Params implements Serializable {

  /**
   * Create another params with the same configuration
   * @return new empty params
   */
  public abstract Params newParams();
  public abstract void initRandom(Random rnd, double variance);
  /**
   * Replace with other
   */
  public abstract void copyOver(Params other);
  /**
   * Replace with other
   */
  public abstract Params merge(Params other);
  /**
   * To double array
   */
  public abstract double[] toArray();
  public abstract int size();

  public abstract void clear();

  // Algebraic operations
  /**
   * Update by adding other with scale
   */
  public abstract void plusEquals(double scale, Params other);
  /**
   * Update by scaling each entry
   */
  public abstract void scaleEquals(double scale);
  /**
   * Take the dot product of two params
   */
  public abstract double dot(Params other);

  // TODO: Support matching

  /**
   * Create another params with the same configuration
   * @return new params with same entries as other
   */
  public Params copy() {
    Params other = newParams();
    other.copyOver(this);
    return other;
  }
  public void plusEquals(Params other) {
    plusEquals(1.0, other);
  }
  /**
   * Update by creating a new object and adding
   */
  public Params plus(double scale, Params other) {
    Params ret = newParams();
    ret.plusEquals(scale, other);
    return ret;
  }
  public Params plus(Params other) {
    return plus(1.0, other);
  }
  /**
   * Update by scaling each entry
   */
  public Params scale(double scale) {
    Params ret = newParams();
    ret.scaleEquals(scale);
    return ret;
  }

  public Params restrict(Params other) {
    Params ret = newParams();
    ret.copyOver(other);
    return ret;
  }

  public void write(String path) {
    writeStringHard(path, toString());
  }


}
