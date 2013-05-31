package learning.models;

import java.io.*;
import java.util.*;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;
import Jama.*;

/**
Allows one to easily create a linear system (Ax = b) with named variables.
*/
public class LinearSystem {
  Indexer<String> varIndexer = new Indexer<String>();
  List<Constraint> constraints = new ArrayList<Constraint>();
  int rank;

  public void add(Constraint constraint) {
    for (Term term : constraint.terms)
      varIndexer.getIndex(term.var);
    constraints.add(constraint);
  }

  public void solve(boolean display) {
    // Ax = b
    int V = varIndexer.size();
    int C = constraints.size();
    LogInfo.begin_track("LinearSystem.solve(%d variables, %d constraints)", V, C);
    Matrix A = new Matrix(C, V);
    Matrix b = new Matrix(C, 1);
    for (int c = 0; c < C; c++) {
      for (Term term : constraints.get(c).terms)
        A.set(c, varIndexer.indexOf(term.var), term.coeff);
      b.set(c, 0, constraints.get(c).target);
    }

    // Display the system
    if (display) {
      logs("%s", "name\ttarget\t" + StrUtils.join(varIndexer.getObjects(), "\t"));
      for (int c = 0; c < C; c++) {
        List<String> row = new ArrayList<String>();
        row.add(constraints.get(c).name);
        row.add(Fmt.D(b.get(c, 0)));
        for (int v = 0; v < V; v++) {
          row.add(Fmt.D(A.get(c, v)));
        }
        logs("%s", StrUtils.join(row, "\t"));
      }
    }

    Matrix At = A.transpose();
    Matrix S = At.times(A);
    //S = S.plus(Matrix.identity(V, V).times(0.01));  // Regularization
    rank = S.rank();
    logs("S: %dx%d has rank %d", S.getRowDimension(), S.getColumnDimension(), S.rank());
    /*Matrix x = S.inverse().times(At.times(b));
    for (int v = 0; v < V; v++) {
      LogInfo.logs("%s: %s", varIndexer.getObject(v), x.get(v, 0));
    }*/
    LogInfo.end_track();
  }
}

class Term {
  String var;
  double coeff;
}

class Constraint {
  String name;
  List<Term> terms = new ArrayList<Term>();
  double target;
  Constraint add(String var, double coeff) {
    Term t = new Term();
    t.var = var;
    t.coeff = coeff;
    terms.add(t);
    return this;
  }
}
