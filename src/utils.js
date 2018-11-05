import Matrix from 'ml-matrix';

/**
 * @private
 * Function that retuns an array of matrices of the cases that belong to each class.
 * @param {Matrix} X - dataset
 * @param {Array} y - predictions
 * @return {Object} An object containing the separated classes array and another array with the prediction symbols
 */
export function separateClasses(X, y) {
  var features = X.columns;

  var classes = 0;
  var totalPerClasses = new Array(10000); // max upperbound of classes
  var actualClassesMap = [];
  var usedClasses = {};
  var reverseClasses = {};

  for (var i = 0; i < y.length; i++) {
    if (!usedClasses[y[i]]) {
      usedClasses[y[i]] = true;
      totalPerClasses[y[i]] = 0;
      reverseClasses[y[i]] = actualClassesMap.length;
      actualClassesMap.push(y[i]);
      classes++;
    }
    totalPerClasses[y[i]]++;
  }
  var separatedClasses = new Array(classes);
  var currentIndex = new Array(classes);
  for (i = 0; i < classes; ++i) {
    separatedClasses[i] = new Matrix(totalPerClasses[actualClassesMap[i]], features);
    currentIndex[i] = 0;
  }
  for (i = 0; i < X.rows; ++i) {
    var index = reverseClasses[y[i]];
    separatedClasses[index].setRow(currentIndex[index], X.getRow(i));
    currentIndex[index]++;
  }

  return {
    separatedClasses,
    actualClassesMap
  };
}
