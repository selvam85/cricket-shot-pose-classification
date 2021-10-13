import * as poseDetection from '@tensorflow-models/pose-detection';
import '@tensorflow/tfjs-backend-webgl';

var imgArr = [
    "http://localhost:8080/cricket-shot/square-cut/david-hofmann.jpg",
    "http://localhost:8080/cricket-shot/square-cut/thao-le-hoang.jpg"
]

var link = document.getElementById('link');
let img = document.getElementById('imgElement');
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let detector, model;
const scoreThreshold = 0.6;

async function loadImage(imgURL) {
    return new Promise((resolve) => {
        img.onload = () => {
            console.log("Image Loaded from URL: ", imgURL);

            canvas.width = img.width;
            canvas.height = img.height;

            ctx.drawImage(img, 0, 0, img.width, img.height);

            link.setAttribute('download', 'MintyPaper.png');
            link.setAttribute('href', canvas.toDataURL("image/png").replace("image/png", "image/octet-stream"));
            link.click(); 

            resolve(img);
        };
        img.src = imgURL;
    });
}

async function app() {
    for(let i = 0; i < imgArr.length; i++) {
        console.log("Img Source URL: ", imgArr[i]);
        await loadImage(imgArr[i]);
    }
}

app();

/* async function createDetector() {
    model = poseDetection.SupportedModels.BlazePose;
    const detectorConfig = {
        runtime: "tfjs",
        enableSmoothing: true,
        modelType: "full"
    };
    detector = await poseDetection.createDetector(model, detectorConfig);
}

async function predictPoses() {
    let poses = null;
    
    canvas.width = img.width;
    canvas.height = img.height;

    if (detector != null) {
        try {
            poses = await detector.estimatePoses(img);
        } catch (error) {
            detector.dispose();
            detector = null;
            alert(error);
        }
    }

    ctx.drawImage(img, 0, 0, img.width, img.height);

    if (poses && poses.length > 0) {
        for (const pose of poses) {
            console.log(pose);
            if (pose.keypoints != null) {
                drawKeypoints(pose.keypoints);
                drawSkeleton(pose.keypoints);
            }
        }
    }
}

function drawKeypoints(keypoints) {
    ctx.fillStyle = 'Green';
    ctx.strokeStyle = 'White';
    ctx.lineWidth = 2;
    for(let i=0; i<keypoints.length; i++) {
        drawKeypoint(keypoints[i]);    
    }
}

function drawKeypoint(keypoint) {
    const radius = 4;
    if (keypoint.score >= scoreThreshold) {
      const circle = new Path2D();
      circle.arc(keypoint.x, keypoint.y, radius, 0, 2 * Math.PI);
      ctx.fill(circle);
      ctx.stroke(circle);
    }
}

function drawSkeleton(keypoints) {
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

async function app() {
    await createDetector();

    imgArr.forEach(processImages);  

    //await predictPoses();

    var link = document.getElementById('link');
    link.setAttribute('download', 'MintyPaper.png');
    link.setAttribute('href', canvas.toDataURL("image/png").replace("image/png", "image/octet-stream"));
    link.click(); 
}

app(); */