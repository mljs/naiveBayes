import {
    MultinomialNB,
    GaussianNB
} from '..';

import irisDataset from 'ml-dataset-iris';
import Random from 'random-js';

var r = new Random(Random.engines.mt19937().seed(42));

describe('Test with iris dataset', () => {
    var X = irisDataset.getNumbers();
    var y = irisDataset.getClasses();
    var classes = irisDataset.getDistinctClasses();

    var transform = {};
    for (var i = 0; i < classes.length; ++i) {
        transform[classes[i]] = i;
    }

    for (i = 0; i < y.length; ++i) {
        y[i] = transform[y[i]];
    }

    shuffle(X, y);
    var Xtrain = X.slice(0, 110);
    var ytrain = y.slice(0, 110);
    var Xtest = X.slice(110);
    var ytest = y.slice(110);

    test('Gaussian naive bayes', () => {
        var gnb = new GaussianNB();
        gnb.train(Xtrain, ytrain);
        var prediction = gnb.predict(Xtest);
        var acc = accuracy(prediction, ytest);

        expect(acc).toBeGreaterThan(0.8);
    });

    test('Multinomial naive bayes', () => {
        var mnb = new MultinomialNB();
        mnb.train(Xtrain, ytrain);
        var prediction = mnb.predict(Xtest);
        var acc = accuracy(prediction, ytest);

        expect(acc).toBeGreaterThan(0.8);
    });
});

function shuffle(X, y) {
    for (let i = X.length; i; i--) {
        let j = Math.floor(r.real(0, 1) * i);
        [X[i - 1], X[j]] = [X[j], X[i - 1]];
        [y[i - 1], y[j]] = [y[j], y[i - 1]];
    }
}

function accuracy(arr1, arr2) {
    var len = arr1.length;
    var total = 0;
    for (var i = 0; i < len; ++i) {
        if (arr1[i] === arr2[i]) {
            total++;
        }
    }

    return total / len;
}
