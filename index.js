import {getPoses} from './poseDetection.js';
import {buildClassifier, classify} from './classifyPoses.js';

/* var trainDataImgURLArr = [
    "http://localhost:8080/cricket-shot/sample-images/thao-le-hoang.jpg",
    "http://localhost:8080/cricket-shot/sample-images/david-hofmann.jpg",
    "http://localhost:8080/cricket-shot/sample-images/tennis-forehand.jpg",
    "http://localhost:8080/cricket-shot/sample-images/woman.jpg"
] */

var trainDataImgURLArr = [
    "http://localhost:8080/cricket-shot/training-set/square-cut/images021.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images022.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images023.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images024.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images025.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images026.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images027.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images028.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images029.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images030.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images031.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images032.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images033.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images034.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images035.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images036.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images037.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images038.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images039.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images041.jpeg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images042.jpeg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images043.jpeg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images045.jpeg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images046.jpeg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images047.jpeg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images048.jpeg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images049.jpeg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images050.jpeg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images052.jpeg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images053.jpeg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images054.jpeg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images056.jpeg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images057.jpeg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images066.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images068.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images077.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images21.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images24.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images27.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images29.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images31.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images32.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images33.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images34.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images35.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images36.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images37.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images38.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images40.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images41.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images42.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images43.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images44.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images49.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images77.jpg"
];

/* const testDataArr = [
    [0.146,0.01,0.144,0.009,0.144,0.001,0.144,0,0.143,0.01,0.142,0.011,0.134,0.004,0.131,0.002,0.12,0.007,0.135,0.025,0.133,0.026,0.095,0.015,0.089,0.035,0.158,0.005,0.098,0.089,0.181,0.012,0.135,0.089,0.182,0.01,0.141,0.087,0.183,0.003,0.136,0.083,0.185,0.008,0.134,0.086,0.08,0.185,0.068,0.188,0.091,0.175,0.08,0.211,0.011,0.223,0.038,0.226,0.003,0.23,0.025,0.223,0,0.247,0.03,0.229],
    [0.057,0.06,0.056,0.052,0.056,0.052,0.057,0.052,0.049,0.054,0.046,0.053,0.04,0.054,0.05,0.053,0.032,0.055,0.061,0.067,0.053,0.068,0.086,0.08,0,0.094,0.137,0.062,0.009,0.147,0.144,0.021,0.03,0.159,0.149,0.007,0.031,0.163,0.144,0.001,0.03,0.159,0.15,0.007,0.034,0.156,0.155,0.166,0.106,0.19,0.127,0.102,0.099,0.257,0.157,0.044,0.101,0.346,0.17,0.041,0.108,0.357,0.123,0,0.077,0.359],
    [0.09,0.012,0.094,0.003,0.102,0.005,0.105,0.004,0.089,0.002,0.087,0.002,0.08,0,0.108,0.005,0.079,0.001,0.101,0.016,0.088,0.014,0.134,0.025,0.075,0.034,0.164,0.085,0.074,0.092,0.128,0.031,0.099,0.1,0.118,0.016,0.105,0.1,0.119,0.007,0.103,0.094,0.11,0.009,0.103,0.097,0.108,0.173,0.063,0.162,0.127,0.248,0.017,0.237,0.104,0.266,0.004,0.258,0.102,0.265,0.002,0.274,0.111,0.296,0,0.286]
]; */

var testDataImgURLArr = [
    "http://localhost:8080/cricket-shot/training-set/square-cut/images43.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images44.jpg",
    "http://localhost:8080/cricket-shot/training-set/square-cut/images49.jpg"
];

async function classifyCricketShots() {
    //Load the classifier
    const classifier = await buildClassifier();

    //Get the poseVectors for test data images
    let testDataArr = await getPoses(testDataImgURLArr);
    console.log(`Test Data: ${testDataArr}`);

    //Prediction
    classify(classifier, testDataArr);
}

classifyCricketShots();

//getPoses(trainDataImgURLArr);