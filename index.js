import {getPoses} from './poseDetection.js';
import {buildClassifier, classify} from './classifyPoses.js';

const chooseFiles = document.getElementById('chooseFiles');
const predictButton = document.getElementById('predictButton');

let selectedImgURL;
let classifier;

async function classifyCricketShots() {
    //Build and load the classifier
    if(classifier == null) {
      classifier = await buildClassifier();
    }

    //Get the poseVectors for test data images
    let imgURLArr = [selectedImgURL];
    let testDataArr = await getPoses(imgURLArr);
    /* If using direct pose data, then use the below statements */
    //let testDataArr = await getPoses(data.testDataImgURLArr);
    //console.log(`Test Data: ${testDataArr}`);

    //Prediction
    //let predictedClassesArr = await classify(classifier, data.testDataArr);
    let predictedClassesArr = await classify(classifier, testDataArr);
    document.getElementById("output").innerHTML = predictedClassesArr;
}

predictButton.onclick = () => {
    classifyCricketShots();    
};

chooseFiles.onchange = () => {
  const [file] = chooseFiles.files
  if (file) {
    selectedImgURL = URL.createObjectURL(file);
  }
};

async function main() {
  //Build and load the KNN Classifier
  classifier = await buildClassifier();
  document.getElementById("info").innerHTML = "KNN classifier loaded successfully!";
}

main();

/* 
  Uncomment this function to get the poses on a set of images that will be used to train a KNN classifier. 
  Populate the trainDataImgURLArr with the URLs of the images in the data.js file. 
  The sample dataset of images can be found in the dataset folder.   
*/
//getPoses(data.trainDataImgURLArr, true);