import * as poseDetection from '@tensorflow-models/pose-detection';
import '@tensorflow/tfjs-backend-webgl';

/* var imgURLArr = [
    "http://localhost:8080/cricket-shot/sample-images/thao-le-hoang.jpg",
    "http://localhost:8080/cricket-shot/sample-images/david-hofmann.jpg",
    "http://localhost:8080/cricket-shot/sample-images/tennis-forehand.jpg",
    "http://localhost:8080/cricket-shot/sample-images/woman.jpg"
] */

var imgURLArr = [
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

let image = document.getElementById("imgElement");
let outputCanvas = document.getElementById("outputCanvas");
const canvasCtx = outputCanvas.getContext('2d');
var link = document.getElementById('link');
let detector, model;
const scoreThreshold = 0.6;

async function createDetector() {
    model = poseDetection.SupportedModels.BlazePose;
    const detectorConfig = {
        runtime: "tfjs",
        enableSmoothing: true,
        modelType: "full"
    };
    detector = await poseDetection.createDetector(model, detectorConfig);
}

async function predictPoses(imageName, canvas, ctx) {
    let poses = null;
    let pose = null;
    if (detector != null) {
        try {
            poses = await detector.estimatePoses(canvas); 
        } catch (error) {
            detector.dispose();
            detector = null;
            console.log(error);
        }
    }

    if (poses && poses.length > 0) {
        for (pose of poses) {
            //console.log(`Poses for ${imageName}: `, JSON.stringify(pose));
            if (pose.keypoints != null) {
                drawKeypoints(pose.keypoints, ctx);
                drawSkeleton(pose.keypoints, ctx);
            } else {
                console.log(`No keypoints identified for ${imageName}`);
            }
        }
    } else {
        console.log(`No poses identified for ${imageName}`);
    }

    return pose;
}

function drawKeypoints(keypoints, ctx) {
    ctx.fillStyle = 'Green';
    ctx.strokeStyle = 'White';
    ctx.lineWidth = 2;
    for(let i=0; i<keypoints.length; i++) {
        drawKeypoint(keypoints[i], ctx);    
    }
}

function drawKeypoint(keypoint, ctx) {
    const radius = 4;
    if (keypoint.score >= scoreThreshold) {
      const circle = new Path2D();
      circle.arc(keypoint.x, keypoint.y, radius, 0, 2 * Math.PI);
      ctx.fill(circle);
      ctx.stroke(circle);
    }
}

function drawSkeleton(keypoints, ctx) {
    const color = "#fff";
    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;

    poseDetection.util.getAdjacentPairs(model)
        .forEach(([i, j]) => {
            const kp1 = keypoints[i];
            const kp2 = keypoints[j];
            if (kp1.score >= scoreThreshold && kp2.score >= scoreThreshold) {
                ctx.beginPath();
                ctx.moveTo(kp1.x, kp1.y);
                ctx.lineTo(kp2.x, kp2.y);
                ctx.stroke();
            }
    });
}

function loadImage(imgURL) {
    return new Promise(resolve => {
        image.onload = () => resolve(image);
        image.src = imgURL;
    });
}

function downloadOutputImage(imageName) {
    const outputImageName = "output_" + imageName;
    link.setAttribute('download', outputImageName);
    link.setAttribute('href', outputCanvas.toDataURL("image/jpg").replace("image/jpg", "image/octet-stream"));
    link.click(); 
}

function drawImageInCanvas(image, outputCanvas) {
    outputCanvas.width = image.width;
    outputCanvas.height = image.height;
    outputCanvas.imageSmoothingEnabled = false;
    canvasCtx.drawImage(image, 0, 0, image.width, image.height);
}

function getImageName(imgURL) {
    return imgURL.substring(imgURL.lastIndexOf("/") + 1);
}

function flattenPoseData(pose) {
    let poseVector = [];
    let xMin = Number.POSITIVE_INFINITY;
    let yMin = Number.POSITIVE_INFINITY;
    let scalingFactor = Number.NEGATIVE_INFINITY;

    pose.keypoints.forEach(keypoint => {
        const x = keypoint.x;
        const y = keypoint.y;

        poseVector.push(x, y);

        xMin = Math.min(xMin, x);
        yMin = Math.min(yMin, y);
        scalingFactor = Math.max(scalingFactor, Math.max(x, y));
    });

    return [poseVector, xMin, yMin, scalingFactor];
}

function resizeAndScale(poseVector, xMin, yMin, scalingFactor) {
    return poseVector.map((value, index) => {
        return (index % 2 == 0 ?
            (value - xMin) / scalingFactor :
            (value - yMin) / scalingFactor);
    });
}

function normalize(scaledPoseVector) {
    let poseVectorSquaredSum = 0;
    let poseVectorAbsSum = 0;

    scaledPoseVector.forEach(value => {
        poseVectorSquaredSum += Math.pow(value, 2);
    });

    poseVectorAbsSum = Math.sqrt(poseVectorSquaredSum);

    return scaledPoseVector.map(value => {
        return Math.round((value / poseVectorAbsSum) * 1000) / 1000;
    });
}

const wait = (ms) => new Promise((resolve, reject) => setTimeout(resolve, ms));

async function main() {
    //Create a detector
    await createDetector();

    let imageName;
    for(let i = 0; i < imgURLArr.length; i++) {
        imageName = getImageName(imgURLArr[i]);
       
        //Load the image to predict the poses
        let image = await loadImage(imgURLArr[i]);

        //Draw the image on canvas
        drawImageInCanvas(image, outputCanvas);
        
        //Predict poses for the image
        await wait(5000);
        let pose = await predictPoses(imageName, outputCanvas, canvasCtx);

        //Download the image with keypoints and skeletons drawn on it
        downloadOutputImage(imageName);

        //Flatten and normalize pose data
        if(pose != null && pose.keypoints != null) {
            let [poseVector, xMin, yMin, scalingFactor] = flattenPoseData(pose);
            let scaledPoseVector = resizeAndScale(poseVector, xMin, yMin, scalingFactor);
            let normalizedPoseVector = JSON.stringify(normalize(scaledPoseVector));
            console.log(`${imageName}: ${normalizedPoseVector}`);
        }
    }
}

main();