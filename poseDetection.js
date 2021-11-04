import * as poseDetection from '@tensorflow-models/pose-detection';
import '@tensorflow/tfjs-backend-webgl';

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

export async function getPoses(imgURLArr) {
    let poseVectorArr = [];

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
            //let normalizedPoseVector = JSON.stringify(normalize(scaledPoseVector));
            let normalizedPoseVector = normalize(scaledPoseVector);
            poseVectorArr.push(normalizedPoseVector);
            console.log(`${imageName}: ${normalizedPoseVector}`);
        }
    }
    return poseVectorArr;
}