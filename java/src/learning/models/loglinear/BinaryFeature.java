package learning.models.loglinear;

import java.io.*;
import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;

// Conjunction of two latent states (for edge potentials in HMM).
public class BinaryFeature implements Feature {
  public final int h1;
  public final int h2;
  public BinaryFeature(int h1, int h2) { this.h1 = h1; this.h2 = h2; }
  @Override public String toString() { return "h1="+h1+",h2="+h2; }
  @Override public boolean equals(Object _that) {
    if (!(_that instanceof BinaryFeature)) return false;
    BinaryFeature that = (BinaryFeature)_that;
    return this.h1 == that.h1 && this.h2 == that.h2;
  }
  @Override public int hashCode() { return h1 * 37 + h2; }
}
