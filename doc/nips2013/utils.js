var G = sfig.serverSide ? global : this;
sfig.importAllMethods(G);
sfig.latexMacro('cl', 1, '\\textsf{#1}');  // Category labels
sfig.latexMacro('il', 1, '\\texttt{#1}');  // Identifier labels
sfig.latexMacro('E', 0, '\\mathbb{E}');
sfig.latexMacro('phix', 0, 't');
sfig.initialize();

G.prez = sfig.presentation();

G.node = function(x, shaded) { return overlay(circle(25).fillColor(shaded ? 'lightgray' : 'white') , std(x).orphan(true)).center(); }

G.moveLeftOf = function(a, b, offset) { return transform(a).pivot(1, 0).shift(b.left().sub(offset == null ? 5 : offset), b.ymiddle()); }
G.moveRightOf = function(a, b, offset) { return transform(a).pivot(-1, 0).shift(b.right().add(offset == null ? 5 : offset), b.ymiddle()); }
G.moveTopOf = function(a, b, offset) { return transform(a).pivot(0, 1).shift(b.xmiddle(), b.top().up(offset == null ? 5 : offset)); }
G.moveBottomOf = function(a, b, offset) { return transform(a).pivot(0, -1).shift(b.xmiddle(), b.bottom().down(offset == null ? 5 : offset)); }
G.moveCenterOf = function(a, b) { return transform(a).pivot(0, 0).shift(b.xmiddle(), b.ymiddle()); }
G.moveBottomLeftOf = function(a, b, offset) { return transform(a).pivot(1, -1).shift(b.left().sub(offset == null ? 5 : offset), b.bottom().down(offset == null ? 5 : offset)); }
G.moveBottomRightOf = function(a, b, offset) { return transform(a).pivot(-1, -1).shift(b.right().add(offset == null ? 5 : offset), b.bottom().down(offset == null ? 5 : offset)); }

G.red = function(x) { return x.fontcolor('red'); }
G.green = function(x) { return x.fontcolor('green'); }
G.blue = function(x) { return x.fontcolor('blue'); }
G.darkblue = function(x) { return x.fontcolor('darkblue'); }

G.wholeNumbers = function(n) {
  var result = [];
  for (var i = 0; i < n; i++) result[i] = i;
  return result;
}

G.indent = function(x, n) { return frame(x).xpadding(n != null ? n : 20).xpivot(1); }

////////////////////////////////////////////////////////////

G.schema = function() {
  boxed = function(a) { return frame(a).padding(5).bg.strokeWidth(2).strokeColor('gray').end; }
  style = function(x) { return x.strokeWidth(10).color('brown'); }
  return overlay(
    table(
      [boxed('Data $x^{(1)}, \\dots, x^{(n)}$'), nil(), boxed('Parameters $\\theta$')],
      [a1 = style(downArrow(s=70)), nil(), a3 = style(upArrow(s))],
      [boxed('Observed moments $\\E[\\phix(x)]$'), a2 = style(rightArrow(200)), boxed('Latent moments $\\E[t(x) \\mid h]$')],
    _).margin(50, 20).center(),
    moveRightOf('(1) aggregation'.bold(), a1),
    moveBottomOf('(2) factorization'.bold(), a2),
    moveRightOf('(3) optimization'.bold(), a3),
  _);
}

G.mixtureModel = function(directed) {
  var edge = directed ? arrow : line;
  return overlay(
    table(
      [nil(), h = node('$h$'), nil()],
      [x1 = node('$x_1$', true), x2 = node('$x_2$', true), x3 = node('$x_3$', true)],
    _).margin(50, 50),
    edge(h, x1),
    edge(h, x2),
    edge(h, x3),
  _);
}

G.tallMixtureModel = function(directed) {
  var edge = directed ? arrow : line;
  return overlay(
    table(
      [nil(), h0 = node('$h_0$'), nil()],
      [h1 = node('$h_1$', false), h2 = node('$h_2$', false), h3 = node('$h_3$', false)],
      [x1 = node('$x_1$', true), x2 = node('$x_2$', true), x3 = node('$x_3$', true)],
    _).margin(50, 50),
    edge(h0, h1),
    edge(h0, h2),
    edge(h0, h3),
    edge(h1, x1),
    edge(h2, x2),
    edge(h3, x3),
  _);
}

G.hmm = function(opts) {
  var edge = opts.directed ? arrow : line;
  var h = wholeNumbers(opts.len).map(function(i) { return node('$h_{'+(i+1)+'}$'); });
  var x = wholeNumbers(opts.len).map(function(i) { return node('$x_{'+(i+1)+'}$', true); });
  var edges = [];
  for (var i = 0; i < opts.len; i++) {
    if (i+1 < opts.len) edges.push(edge(h[i], h[i+1]));
    edges.push(edge(h[i], x[i]));
  }
  return overlay(
    table(
      h, x,
    _).margin(50, 50),
    new Overlay(edges),
  _);
}

G.gridModel = function(opts) {
  var edge = opts.directed ? arrow : line;
  var h = wholeNumbers(opts.len).map(function(i) { return node('$h_{'+(i+1)+'}$'); });
  var x = wholeNumbers(opts.len).map(function(i) { return node('$x_{'+(i+1)+'}$', true); });
  var nodes = [];
  for (var i = 0; i < opts.numRows; i++) {
    nodes[i] = [];
    for (var j = 0; j < opts.numCols; j++) {
      nodes[i][j] = node('$h_{'+(i+1)+(j+1)+'}$');
    }
  }
  var O = [];
  O.push(new Table(nodes).margin(150, 50));
  for (var i = 0; i < opts.numRows; i++) {
    for (var j = 0; j < opts.numCols; j++) {
      var h = nodes[i][j];
      var x1 = moveBottomLeftOf(node('$x_{'+(i+1)+(j+1)+'}^a$', true), h);
      var x2 = moveBottomRightOf(node('$x_{'+(i+1)+(j+1)+'}^b$', true), h);
      O.push(x1); O.push(edge(h, x1));
      O.push(x2); O.push(edge(h, x2));
      if (j+1 < opts.numCols) O.push(edge(nodes[i][j], nodes[i][j+1]));
      if (i+1 < opts.numRows) O.push(edge(nodes[i][j], nodes[i+1][j]));
    }
  }
  return new Overlay(O);
}

G.factorialMixtureModel = function(directed) {
  var edge = directed ? arrow : line;
  return overlay(
    ytable(
      xtable(h1 = node('$h_1$'), h2 = node('$h_2$')).margin(50),
      xtable(x1 = node('$x_1$', true), x2 = node('$x_2$', true), x3 = node('$x_3$', true)).margin(50),
    _).margin(50).center(),
    edge(h1, x1), edge(h1, x2), edge(h1, x3),
    edge(h2, x1), edge(h2, x2), edge(h2, x3),
  _);
}

G.factorialHMM = function() {
  numChains = 2
  numTimeSteps = 3
  h = wholeNumbers(numChains).map(function(i) {
    return wholeNumbers(numTimeSteps).map(function(j) {
      return node('$h_{'+(j+1)+(i+1)+'}$');
      //return node('$h^{'+(j+1)+'}_{'+(i+1)+'}$');
    });
  });
  x = wholeNumbers(numTimeSteps).map(function(j) {
    return node('$x_{'+(j+1)+'}$', true);
  });
  nodes = h.concat([x.map(function(a) { return indent(a, 40); })]);
  edges = [];
  wholeNumbers(numChains).forEach(function(i) {
    wholeNumbers(numTimeSteps).forEach(function(j) {
      if (j+1 < numTimeSteps) edges.push(arrow(h[i][j], h[i][j+1]));
      edges.push(arrow(h[i][j], x[j]));
    });
  });
  return overlay(
    new Table(nodes).margin(20, 20),
    new Overlay(edges),
  _);
}

G.unshuffle = function() {
  var lhs = [];
  var K = 3;
  for (var i = 0; i < K; i++) {
    for (var j = 0; j < K; j++) {
      var s = text('$f_'+(i+1)+'+g_'+(j+1)+'-Z_{'+(i+1)+(j+1)+'}$');
      lhs.push(s);
    }
  }

  var rhs = [];
  for (var i = 0; i < K*K; i++) {
    rhs.push(text('$\\green{L_{'+(i+1)+'}}$'));
  }

  Math.seedrandom(2);
  var N = K*K;
  var perm = wholeNumbers(N);
  for (var i = 0; i < N; i++) {
    var j = Math.floor(Math.random()*(N-i+1));
    var tmp = perm[i];
    perm[i] = perm[j];
    perm[j] = tmp;
  }
  var arrows = [];
  for (var i = 0; i < N; i++) {
    var s = lhs[i];
    var t = rhs[perm[i]];
    arrows.push(line([s.right(), s.ymiddle()], [t.left(), t.ymiddle()]).dashed());
  }

  var rows = [['unknown', 'known']];
  for (var i = 0; i < N; i++)
    rows.push([lhs[i], rhs[i]]);

  inputOutput = overlay(
    new Table(rows).xmargin(50).center(),
    new Overlay(arrows),
  _);

  bin = function() { return frame(table.apply(null, arguments)).padding(5).bg.strokeWidth(1).dashed().end; }

  group = function() {
    var title = arguments[0];
    var body = Array.prototype.slice.call(arguments, 1);
    var f = frame(ytable.apply(null, body).ymargin(10)).padding(10);
    f.bg.strokeWidth(1).round(15).strokeColor(title ? 'red' : 'gray').end;
    if (title != null) f.title(opaquebg(title));
    return f;
  }

  contents = ytable(
    xtable(
      group('Source 1',
        bin(
          ['$f_1 - f_2 - (Z_{11} - Z_{21})$'],
          ['$f_1 - f_2 - (Z_{12} - Z_{22})$'],
          ['$f_1 - f_2 - (Z_{13} - Z_{23})$'],
        _),
        bin(
          ['$f_1 - f_3 - (Z_{11} - Z_{31})$'],
          ['$f_1 - f_3 - (Z_{12} - Z_{32})$'],
          ['$f_1 - f_3 - (Z_{13} - Z_{33})$'],
        _),
      _),
      group('Source 2',
        bin(
          ['$g_1 - g_2 - (Z_{11} - Z_{12})$'],
          ['$g_1 - g_2 - (Z_{21} - Z_{22})$'],
          ['$g_1 - g_2 - (Z_{31} - Z_{32})$'],
        _),
        bin(
          ['$g_1 - g_3 - (Z_{11} - Z_{13})$'],
          ['$g_1 - g_3 - (Z_{21} - Z_{23})$'],
          ['$g_1 - g_3 - (Z_{31} - Z_{33})$'],
        _),
      _),
    _).margin(20),
    group(null,
      bin(
        ['$f_1 + f_2 - g_1 - g_2 - (Z_{11} - Z_{22})$'],
      _),
    _),
    group(null,
      bin(
        ['$f_1 + g_2 - f_2 - g_1 - (Z_{12} - Z_{21})$'],
      _),
    _),
    '$\\cdots$',
  _).ymargin(20).center();

  return xtable(
    inputOutput,
    '$\\Rightarrow$',
    contents.scale(0.8),
  _).center().margin(30);
}
