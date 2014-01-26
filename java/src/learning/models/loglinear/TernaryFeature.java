package learning.models.loglinear;

// Conjunction of two latent states (for edge potentials in HMM).
public class TernaryFeature implements Feature {
  public final int h1;
  public final int h2;
  public final int h3;
  public TernaryFeature(int h1, int h2, int h3) { this.h1 = h1; this.h2 = h2; this.h3 = h3; }
  @Override public String toString() { return "h1="+h1+",h2="+h2+",h3="+h3; }
  @Override public boolean equals(Object _that) {
    if (!(_that instanceof TernaryFeature)) return false;
    TernaryFeature that = (TernaryFeature)_that;
    return this.h1 == that.h1 && this.h2 == that.h2 && this.h3 == that.h3;
  }
  @Override public int hashCode() { return h1 * 37 + 29 * h2 + h3; }
}
