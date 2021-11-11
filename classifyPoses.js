import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
import * as data from './data.js';

const cricketShotClasses = {
    "0": "cover-drive",
    "1": "flick",
    "2": "square-cut"
}

export async function buildClassifier() {
    // Create the classifier.
    const classifier = knnClassifier.create();
    classifier.clearAllClasses();

    //Add examples for all classes
    for(const coverDrivePose of data.coverDrivePoses) {
        classifier.addExample(tf.tensor1d(coverDrivePose), 0);
    }

    for(const flickPose of data.flickPoses) {
        classifier.addExample(tf.tensor1d(flickPose), 1);
    }

    for(const squareCutPose of data.squareCutPoses) {
        classifier.addExample(tf.tensor1d(squareCutPose), 2);
    }

    const exampleCountByClass = classifier.getClassExampleCount();
    console.log(exampleCountByClass);

    return classifier;
}

export async function classify(classifier, testDataArr) {
    // Make a prediction.
    let result
    let predictedClassesArr = [];
    for(const testData of testDataArr) {
        console.log(`test data for prediction: ${testData}`);
        result = await classifier.predictClass(tf.tensor1d(testData));
        console.log('Predictions: ' + JSON.stringify(result));
        console.log(`Cricket Shot: ${cricketShotClasses[result.classIndex]}`);

        predictedClassesArr.push(cricketShotClasses[result.classIndex]);
    }

    return predictedClassesArr;
}