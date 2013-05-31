require('./sfig/internal/sfig.js');
require('./sfig/internal/metapost.js');
require('./utils.js');

sfig.Text.defaults.setProperty('font', 'Times New Roman');  // For paper

prez.addSlide(schema().id('schema'));

simpleModels = table(
  [mixtureModel(true), mixtureModel(false)],
  ['(a) Directed mixture model', '(b) Undirected mixture model'],
_).margin(100, 20).center();
prez.addSlide(simpleModels.id('simpleModels'));

generalModels = table(
  [hmm({directed: false, len: 4}), tallMixtureModel(false), gridModel({directed: false, numRows: 3, numCols: 2})],
  ['(a) Hidden Markov model', '(b) Tall mixture model', '(c) Grid model'],
_).margin(100, 20).center();
prez.addSlide(generalModels.id('generalModels'));

factorialModels = table(
  [factorialMixtureModel(), factorialHMM()],
  ['(a) Factorial mixture model', '(b) Factorial HMM'],
_).margin(50, 20).center();
prez.addSlide(factorialModels.id('factorialModels'));

prez.writePdf({outPrefix: 'figures', combine: false});