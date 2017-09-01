# Naive Bayes

  [![NPM version][npm-image]][npm-url]
  [![build status][travis-image]][travis-url]
  [![David deps][david-image]][david-url]
  [![npm download][download-image]][download-url]

Naive bayes classifiers.

## Installation

`$ npm install ml-naivebayes`

## Usage

### [GaussianNB](./src/GaussianNB.js)

```js
// assuming that you created Xtrain, Xtest, Ytrain, Ytest
import {GaussianNB} from 'ml-naivebayes'

var model = new GaussianNB();
model.train(Xtrain, Ytest);

var predictions = model.predict(Xtest);
```

### [MultinomialNB](./src/MultinomialNB.js)

```js
// assuming that you created Xtrain, Xtest, Ytrain, Ytest
import {MultinomialNB} from 'ml-naivebayes'

var model = new MultinomialNB();
model.train(Xtrain, Ytest);

var predictions = model.predict(Xtest);
```

## [API Documentation](http://mljs.github.io/naive-bayes/)

## Authors

- [Jefferson Hernandez](https://github.com/JeffersonH44)

## License

[MIT](./LICENSE)

[npm-image]: https://img.shields.io/npm/v/ml-naivebayes.svg?style=flat-square
[npm-url]: https://npmjs.org/package/ml-naivebayes
[travis-image]: https://img.shields.io/travis/mljs/naive-bayes/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/mljs/naive-bayes
[david-image]: https://img.shields.io/david/mljs/naive-bayes.svg?style=flat-square
[david-url]: https://david-dm.org/mljs/naive-bayes
[download-image]: https://img.shields.io/npm/dm/ml-naivebayes.svg?style=flat-square
[download-url]: https://npmjs.org/package/ml-naivebayes