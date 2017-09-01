'use strict';

Object.defineProperty(exports, '__esModule', { value: true });

function _interopDefault (ex) { return (ex && (typeof ex === 'object') && 'default' in ex) ? ex['default'] : ex; }

var Matrix = _interopDefault(require('ml-matrix'));
var Stat = _interopDefault(require('ml-stat'));

/**
 * Function that retuns an array of matrices of the cases that belong to each class.
 * @param {Matrix} X - dataset
 * @param {Array} y - predictions
 * @return {Array}
 */
function separateClasses(X, y) {
    var features = X.columns;

    var classes = 0;
    var totalPerClasses = new Array(10000); // max upperbound of classes
    for (var i = 0; i < y.length; i++) {
        if (totalPerClasses[y[i]] === undefined) {
            totalPerClasses[y[i]] = 0;
            classes++;
        }
        totalPerClasses[y[i]]++;
    }
    var separatedClasses = new Array(classes);
    var currentIndex = new Array(classes);
    for (i = 0; i < classes; ++i) {
        separatedClasses[i] = new Matrix(totalPerClasses[i], features);
        currentIndex[i] = 0;
    }
    for (i = 0; i < X.rows; ++i) {
        separatedClasses[y[i]].setRow(currentIndex[y[i]], X.getRow(i));
        currentIndex[y[i]]++;
    }
    return separatedClasses;
}

class GaussianNB {

    /**
     * Constructor for the Gaussian Naive Bayes classifier, the parameters here is just for loading purposes.
     *
     * @param {boolean} reload
     * @param {object} model
     * @constructor
     */
    constructor(reload, model) {
        if (reload) {
            this.means = model.means;
            this.calculateProbabilities = model.calculateProbabilities;
        }
    }

    /**
     * Function that trains the classifier with a matrix that represents the training set and an array that
     * represents the label of each row in the training set. the labels must be numbers between 0 to n-1 where
     * n represents the number of classes.
     *
     * WARNING: in the case that one class, all the cases in one or more features have the same value, the
     * Naive Bayes classifier will not work well.
     * @param {Matrix|Array} trainingSet
     * @param {Matrix|Array} trainingLabels
     */
    train(trainingSet, trainingLabels) {
        var C1 = Math.sqrt(2 * Math.PI); // constant to precalculate the squared root
        trainingSet = Matrix.checkMatrix(trainingSet);//if (!Matrix.isMatrix(trainingSet)) trainingSet = new Matrix(trainingSet);

        if (trainingSet.rows !== trainingLabels.length) {
            throw new RangeError('the size of the training set and the training labels must be the same.');
        }

        var separatedClasses = separateClasses(trainingSet, trainingLabels);
        var calculateProbabilities = new Array(separatedClasses.length);
        this.means = new Array(separatedClasses.length);
        for (var i = 0; i < separatedClasses.length; ++i) {
            var means = Stat.matrix.mean(separatedClasses[i]);
            var std = Stat.matrix.standardDeviation(separatedClasses[i], means);

            var logPriorProbability = Math.log(separatedClasses[i].rows / trainingSet.rows);
            calculateProbabilities[i] = new Array(means.length + 1);

            calculateProbabilities[i][0] = logPriorProbability;
            for (var j = 1; j < means.length + 1; ++j) {
                var currentStd = std[j - 1];
                calculateProbabilities[i][j] = [(1 / (C1 * currentStd)), -2 * currentStd * currentStd];
            }

            this.means[i] = means;
        }

        this.calculateProbabilities = calculateProbabilities;
    };

    /**
     * function that predicts each row of the dataset (must be a matrix).
     *
     * @param {Matrix|Array} dataset
     * @return {Array}
     */
    predict(dataset) {
        if (dataset[0].length === this.calculateProbabilities[0].length) {
            throw new RangeError('the dataset must have the same features as the training set');
        }

        var predictions = new Array(dataset.length);

        for (var i = 0; i < predictions.length; ++i) {
            predictions[i] = getCurrentClass(dataset[i], this.means, this.calculateProbabilities);
        }

        return predictions;
    };

    /**
     * Function that export the NaiveBayes model.
     * @return {object}
     */
    toJSON() {
        return {
            modelName: 'NaiveBayes',
            means: this.means,
            calculateProbabilities: this.calculateProbabilities
        };
    };

    /**
     * Function that create a GaussianNB classifier with the given model.
     * @param {object} model
     * @return {GaussianNB}
     */
    static load(model) {
        if (model.modelName !== 'NaiveBayes') {
            throw new RangeError('The given model is invalid!');
        }

        return new GaussianNB(true, model);
    };
}

/**
 * @private
 * Function the retrieves a prediction with one case.
 *
 * @param {Array} currentCase
 * @param {Array} mean - Precalculated means of each class trained
 * @param {Array} classes - Precalculated value of each class (Prior probability and probability function of each feature)
 * @return {number}
 */
function getCurrentClass(currentCase, mean, classes) {
    var maxProbability = 0;
    var predictedClass = -1;

    // going through all precalculated values for the classes
    for (var i = 0; i < classes.length; ++i) {
        var currentProbability = classes[i][0]; // initialize with the prior probability
        for (var j = 1; j < classes[0][1].length + 1; ++j) {
            currentProbability += calculateLogProbability(currentCase[j - 1], mean[i][j - 1], classes[i][j][0], classes[i][j][1]);
        }

        currentProbability = Math.exp(currentProbability);
        if (currentProbability > maxProbability) {
            maxProbability = currentProbability;
            predictedClass = i;
        }
    }

    return predictedClass;
}

/**
 * @private
 * function that retrieves the probability of the feature given the class.
 * @param {number} value - value of the feature.
 * @param {number} mean - mean of the feature for the given class.
 * @param {number} C1 - precalculated value of (1 / (sqrt(2*pi) * std)).
 * @param {number} C2 - precalculated value of (2 * std^2) for the denominator of the exponential.
 * @return {number}
 */
function calculateLogProbability(value, mean, C1, C2) {
    value = value - mean;
    return Math.log(C1 * Math.exp((value * value) / C2));
}

class MultinomialNB {

    /**
     * Constructor for Multinomial Naive Bayes, the model parameter is for load purposes.
     *
     * @param {object} model - for load purposes.
     * @constructor
     */
    constructor(model) {
        if (model) {
            this.conditionalProbability = Matrix.checkMatrix(model.conditionalProbability);
            this.priorProbability = Matrix.checkMatrix(model.priorProbability);
        }
    }

    /**
     * Train the classifier with the current training set and labels, the labels must be numbers between 0 and n.
     * @param {Matrix|Array} trainingSet
     * @param {Array} trainingLabels
     */
    train(trainingSet, trainingLabels) {
        trainingSet = Matrix.checkMatrix(trainingSet);
        var separateClasses$$1 = separateClasses(trainingSet, trainingLabels);
        this.priorProbability = new Matrix(separateClasses$$1.length, 1);

        for (var i = 0; i < separateClasses$$1.length; ++i) {
            this.priorProbability[i][0] = Math.log(separateClasses$$1[i].length / trainingSet.rows);
        }

        var features = trainingSet.columns;
        this.conditionalProbability = new Matrix(separateClasses$$1.length, features);
        for (i = 0; i < separateClasses$$1.length; ++i) {
            var classValues = Matrix.checkMatrix(separateClasses$$1[i]);
            var total = classValues.sum();
            var divisor = total + features;
            this.conditionalProbability.setRow(i, classValues.sum('column').add(1).div(divisor).apply(matrixLog));
        }
    }

    /**
     * Retrieves the predictions for the dataset with the current model.
     * @param {Matrix|Array} dataset
     * @return {Array} - predictions from the dataset.
     */
    predict(dataset) {
        dataset = Matrix.checkMatrix(dataset);
        var predictions = new Array(dataset.rows);
        for (var i = 0; i < dataset.rows; ++i) {
            var currentElement = dataset.getRowVector(i);
            predictions[i] = this.conditionalProbability.clone().mulRowVector(currentElement).sum('row')
                .add(this.priorProbability).maxIndex()[0];
        }

        return predictions;
    }

    /**
     * Function that saves the current model.
     * @return {object} - model in JSON format.
     */
    toJSON() {
        return {
            name: 'MultinomialNB',
            priorProbability: this.priorProbability,
            conditionalProbability: this.conditionalProbability
        };
    }

    /**
     * Creates a new MultinomialNB from the given model
     * @param {object} model
     * @return {MultinomialNB}
     */
    static load(model) {
        if (model.name !== 'MultinomialNB') {
            throw new RangeError('The current model is not a Multinomial Naive Bayes');
        }

        return new MultinomialNB(model);
    }
}

function matrixLog(i, j) {
    this[i][j] = Math.log(this[i][j]);
}

exports.GaussianNB = GaussianNB;
exports.MultinomialNB = MultinomialNB;
