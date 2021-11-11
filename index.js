import {getPoses} from './poseDetection.js';
import {buildClassifier, classify} from './classifyPoses.js';
import * as data from './data.js';

let classifier;
async function classifyCricketShots() {
    //Build and load the classifier
    if(classifier == null) {
      classifier = await buildClassifier();
    }

    //Get the poseVectors for test data images
    let testDataArr = await getPoses(data.testDataImgURLArr);
    console.log(`Test Data: ${testDataArr}`);

    //Prediction
    //let predictedClassesArr = await classify(classifier, data.testDataArr);
    let predictedClassesArr = await classify(classifier, testDataArr);
    document.getElementById("output").innerHTML = predictedClassesArr;
}

document.getElementById('predictButton').onclick = () => {
    classifyCricketShots();    
}

/* 
  Uncomment this function to get the poses on a set of images that will be used to train a KNN classifier. 
  Populate the trainDataImgURLArr with the URLs of the images in the data.js file. 
  The sample dataset of images can be found in the dataset folder.   
*/
//getPoses(data.trainDataImgURLArr);