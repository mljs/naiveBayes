import {MultinomialNB} from '../MultinomialNB';

describe('Multinomial Naive Bayes', function () {

    var cases = [[2, 1, 0, 0, 0, 0],
        [2, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 1]];
    var predictions = [0, 0, 0, 1];

    test('main test', () => {

        var predict = [[3, 0, 0, 0, 1, 1]];

        var mnb = new MultinomialNB();
        mnb.train(cases, predictions);
        var prediction = mnb.predict(predict);

        expect(prediction).toEqual([0]);
    });

    test('save and load', () => {
        var predict = [[3, 0, 0, 0, 1, 1]];

        var mnb = new MultinomialNB();
        mnb.train(cases, predictions);
        mnb = MultinomialNB.load(JSON.parse(JSON.stringify(mnb)));
        var prediction = mnb.predict(predict);

        expect(prediction).toEqual([0]);
    });

});
