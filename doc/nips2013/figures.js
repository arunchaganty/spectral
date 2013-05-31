require('./sfig/internal/sfig.js');
require('./sfig/external/seedrandom.js');
require('./sfig/internal/metapost.js');
require('./utils.js');

sfig.Text.defaults.setProperty('font', 'Times New Roman');  // For paper

prez.addSlide(schema().id('schema'));

model = table(
  [mixtureModel(false)],
_).margin(100, 20).center();
prez.addSlide(model.id('mixtureModel'));

/*simpleModels = table(
  [mixtureModel(true), mixtureModel(false)],
  ['(a) Directed mixture model', '(b) Undirected mixture model'],
_).margin(100, 20).center();
prez.addSlide(simpleModels.id('simpleModels'));*/

directedModels = table(
  [mixtureModel(true), hmm({directed: true, len: 4}), factorialMixtureModel(true)],
  ['(a) Mixture model', '(b) Hidden Markov model', '(c) Factorial mixture model'],
_).margin(100, 20).center();
prez.addSlide(directedModels.id('directedModels'));

generalModels = table(
  [
    hmm({directed: false, len: 4}),
    gridModel({directed: false, numRows: 3, numCols: 2}),
    tallMixtureModel(false),
  _],
  [
    '(a) Hidden Markov model',
    '(b) Grid model',
    '(c) Tall mixture model',
  _],
_).margin(100, 20).center();
prez.addSlide(generalModels.id('generalModels'));

/*factorialModels = table(
  [factorialMixtureModel(), factorialHMM()],
  ['(a) Factorial mixture model', '(b) Factorial HMM'],
_).margin(50, 20).center();
prez.addSlide(factorialModels.id('factorialModels'));*/

factorialModels = table(
  [factorialMixtureModel(false), unshuffle()],
  ['(a) Factorial mixture model', '(b) Unshuffling factorization'],
_).margin(50, 20).center();
prez.addSlide(factorialModels.id('factorialModels'));

factorialHMM = table(
  [factorialHMM()],
  ['(a) Factorial HMM'],
_).margin(50, 20).center();
prez.addSlide(factorialHMM.id('factorialHMM'));

prez.writePdf({outPrefix: 'figures', combine: false});
