import {GaussianNB} from '..';
import {separateClasses} from '../utils';
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

    test('separate classes', () => {
        var matrixCases = new Matrix(cases);
        var separatedResult = separateClasses(matrixCases, predictions);

        expect(separatedResult.length).toEqual(2);
        expect(separatedResult[0].rows).toEqual(3);
        expect(separatedResult[1].rows).toEqual(2);
    });

    test('Small test', () => {
        var cases = [[6, 148, 72, 35, 0, 33.6, 0.627, 5],
            [1.50, 85, 66.5, 29, 0, 26.6, 0.351, 31],
            [8, 183, 64, 0, 0, 23.3, 0.672, 32],
            [0.5, 89, 65.5, 23, 94, 28.1, 0.167, 21],
            [0, 137, 40, 35, 168, 43.1, 2.288, 33]];
        var predictions = [1, 0, 1, 0, 1];
        var nb = new GaussianNB();
        nb.train(cases, predictions);
        var result = nb.predict(cases);

        expect(result).toEqual(predictions);
    });

    test('single column clustering', () => {
        var cases = [[1, 3],
            [2, 5],
            [3, 8],
            [4, 10],
            [5, 6],
            [6, 11]];
        var predictions = [0, 0, 1, 1, 0, 1];
        var nb = new GaussianNB();
        nb.train(cases, predictions);

        var result = nb.predict([[10, 4], [0, 9]]);

        expect(result).toEqual([0, 1]);
    });

    test('low deviation column', () => {
        var cases = [[1.00001, 3],
            [1.00001001, 5],
            [1.00001, 8],
            [1.00001002, 10],
            [1.00001, 6],
            [1.00001, 11]];
        var predictions = [0, 0, 1, 1, 0, 1];
        var nb = new GaussianNB();
        nb.train(cases, predictions);

        var result = nb.predict([[1, 4], [1, 9]]);

        expect(result).toEqual([0, 1]);
    });

    test('zero deviation column', () => {
        var cases = [[0, 3],
            [0, 5],
            [0, 8],
            [0, 10],
            [0, 6],
            [0, 11]];
        var predictions = [0, 0, 1, 1, 0, 1];
        var nb = new GaussianNB();
        nb.train(cases, predictions);

        var result = nb.predict([[0, 4], [0, 9]]);

        expect(result).toEqual([0, 1]);
    });

    test('Export and import', () => {
        var cases = [[6, 148, 72, 35, 0, 33.6, 0.627, 5],
            [1.50, 85, 66.5, 29, 0, 26.6, 0.351, 31],
            [8, 183, 64, 0, 0, 23.3, 0.672, 32],
            [0.5, 89, 65.5, 23, 94, 28.1, 0.167, 21],
            [0, 137, 40, 35, 168, 43.1, 2.288, 33]];
        var predictions = [1, 0, 1, 0, 1];
        var nb = new GaussianNB();
        nb.train(cases, predictions);

        var model = JSON.parse(JSON.stringify(nb));
        nb = GaussianNB.load(model);

        var result = nb.predict(cases);

        expect(result).toEqual(predictions);
    });
});

