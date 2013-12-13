package learning.models.loglinear;

import java.io.*;
import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;

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

