const pose = {
    keypoints: [
        { x: 2, y: 0 },
        { x: 3, y: 4 },
        { x: 2, y: 0 }
    ]
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
        return value / poseVectorAbsSum;
    });
}

function main() {
    let [poseVector, xMin, yMin, scalingFactor] = flattenPoseData(pose);
    console.log(`Flattened Pose Vector: ${poseVector}`);
    console.log(`xMin: ${xMin}, yMin: ${yMin}, scalingFactor: ${scalingFactor}`);

    let scaledPoseVector = resizeAndScale(poseVector, xMin, yMin, scalingFactor);
    console.log(`Scaled Pose Vector: ${scaledPoseVector}`);

    let normalizedPoseVector = normalize(scaledPoseVector);
    console.log(`Normalized Pose Vector: ${normalizedPoseVector}`);
}

main();