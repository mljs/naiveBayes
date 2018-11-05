import { GaussianNB } from '..';

import { separateClasses } from '../utils';

import Matrix from 'ml-matrix';

const cases = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]];
const predictions = [0, 0, 0, 1, 1];
describe('Naive bayes', () => {
  test('Basic test', () => {
    var nb = new GaussianNB();
    nb.train(cases, predictions);
    var results = nb.predict(cases);

    expect(results).toEqual(predictions);
  });

  test('Separate classes', () => {
    var matrixCases = new Matrix(cases);
    var sc = separateClasses(matrixCases, predictions);

    var separatedResult = sc.separatedClasses;
    var symbols = sc.actualClassesMap;
    expect(separatedResult).toHaveLength(2);
    expect(separatedResult[0].rows).toEqual(3);
    expect(separatedResult[1].rows).toEqual(2);
    expect(symbols).toEqual([0, 1]);
  });

  test('Separate classes where predictions are non related alphanumerical symbols', () => {
    const cases = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]];
    const predictions = [-1, -1, -1, 'a', 'a'];

    var matrixCases = new Matrix(cases);
    var sc = separateClasses(matrixCases, predictions);
    var separatedResult = sc.separatedClasses;
    var symbols = sc.actualClassesMap;

    expect(Object.keys(separatedResult)).toHaveLength(2);
    expect(separatedResult[0].rows).toEqual(3);
    expect(separatedResult[1].rows).toEqual(2);
    expect(symbols).toEqual([-1, 'a']);
  });

  test('Small test', () => {
    var cases = [
      [6, 148, 72, 35, 0, 33.6, 0.627, 5],
      [1.5, 85, 66.5, 29, 0, 26.6, 0.351, 31],
      [8, 183, 64, 0, 0, 23.3, 0.672, 32],
      [0.5, 89, 65.5, 23, 94, 28.1, 0.167, 21],
      [0, 137, 40, 35, 168, 43.1, 2.288, 33]
    ];
    var predictions = [1, 0, 1, 0, 1];
    var nb = new GaussianNB();
    nb.train(cases, predictions);
    var result = nb.predict(cases);

    expect(result).toEqual(predictions);
  });

  test('Small test where predictions are non related alphanumerical symbols', () => {
    var cases = [
      [6, 148, 72, 35, 0, 33.6, 0.627, 5],
      [1.5, 85, 66.5, 29, 0, 26.6, 0.351, 31],
      [8, 183, 64, 0, 0, 23.3, 0.672, 32],
      [0.5, 89, 65.5, 23, 94, 28.1, 0.167, 21],
      [0, 137, 40, 35, 168, 43.1, 2.288, 33]
    ];
    var predictions = [-1, 'a', -1, 'a', -1];
    var nb = new GaussianNB();
    nb.train(cases, predictions);
    var result = nb.predict(cases);

    expect(result).toEqual(predictions);
  });

  test('Export and import', () => {
    var cases = [
      [6, 148, 72, 35, 0, 33.6, 0.627, 5],
      [1.5, 85, 66.5, 29, 0, 26.6, 0.351, 31],
      [8, 183, 64, 0, 0, 23.3, 0.672, 32],
      [0.5, 89, 65.5, 23, 94, 28.1, 0.167, 21],
      [0, 137, 40, 35, 168, 43.1, 2.288, 33]
    ];
    var predictions = [1, 0, 1, 0, 1];
    var nb = new GaussianNB();
    nb.train(cases, predictions);

    var model = JSON.parse(JSON.stringify(nb));
    nb = GaussianNB.load(model);

    var result = nb.predict(cases);

    expect(result).toEqual(predictions);
  });
});
