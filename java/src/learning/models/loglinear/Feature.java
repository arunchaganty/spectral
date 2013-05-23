package learning.models.loglinear;

import java.io.*;
import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;

public interface Feature {
}

// Conjunction of latent state and some description (for node potentials in HMM).
class UnaryFeature implements Feature {
  final int h;  // Latent state associated with this feature
  final String description;
  UnaryFeature(int h, String description) {
    this.h = h;
    this.description = description;
  }
  @Override public String toString() { return "h="+h+":"+description; }
  @Override public boolean equals(Object _that) {
    if (!(_that instanceof UnaryFeature)) return false;
    UnaryFeature that = (UnaryFeature)_that;
    return this.h == that.h && this.description.equals(that.description);
  }
  @Override public int hashCode() { return h * 37 + description.hashCode(); }
}

// Conjunction of two latent states (for edge potentials in HMM).
class BinaryFeature implements Feature {
  final int h1, h2;
  BinaryFeature(int h1, int h2) { this.h1 = h1; this.h2 = h2; }
  @Override public String toString() { return "h1="+h1+",h2="+h2; }
  @Override public boolean equals(Object _that) {
    if (!(_that instanceof BinaryFeature)) return false;
    BinaryFeature that = (BinaryFeature)_that;
    return this.h1 == that.h1 && this.h2 == that.h2;
  }
  @Override public int hashCode() { return h1 * 37 + h2; }
}
