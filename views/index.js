import { HandLandmarker, FilesetResolver } from "https://cdn.skypack.dev/@mediapipe/tasks-vision@latest";

const LETTERS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", " "];

const video = document.getElementById("video");
const textElem = document.getElementById("transcription_text");
const text = new TypeIt("#transcription_text").go();

let handCoordinates = null;
let lastLetter = null;
let skippedLastLetter = false;
let liveHands, staticHands;
let model;

// Setup webcam camera
const camera = new Camera(video, {
  onFrame: () => {},
  width: 1920,
  height: 1080,
});

// Declare the options for the mediapipe hand landmark detector
const options = (mode) => ({
  baseOptions: {
    modelAssetPath: "assets/hand_landmarker.task",
  },
  numHands: 1,
  min_hand_detection_confidence: 0,
  min_hand_presence_confidence: 0,
  runningMode: mode,
});

async function main() {
  // Load the ASL interpreter model
  model = await tf.loadGraphModel("assets/model/model.json");

  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm");

  // Setup the mediapipe landmark models
  liveHands = await HandLandmarker.createFromOptions(vision, options("LIVE_STREAM")); // For detecting hand location in webcam input
  staticHands = await HandLandmarker.createFromOptions(vision, options("IMAGE")); // For detecting hand landmarks in cropped frame

  camera.start();

  // Start detecting once webcam data is ready
  video.addEventListener("loadeddata", () => {
    resizeVideo();
    renderLoop();
    setInterval(detectSign, 500);
    setTimeout(() => document.getElementById("loading").classList.add("inactive"), 1500);
  });
}

// Main loop to detect hand locations on every frame
function renderLoop() {
  const results = liveHands.detectForVideo(video, Date.now());
  computeHandsLocation(results);
  requestAnimationFrame(renderLoop);
}

// Given hand landmark locations of webcam data, calculate the hands bounding box
function computeHandsLocation(results) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = 0;
  let maxY = 0;

  const h = video.videoHeight;
  const w = video.videoWidth;

  if (results.landmarks.length > 0) {
    results.landmarks[0].forEach((pos) => {
      minX = Math.min(minX, pos.x * w);
      minY = Math.min(minY, pos.y * h);
      maxX = Math.max(maxX, pos.x * w);
      maxY = Math.max(maxY, pos.y * h);
    });

    const epsilon = Math.max((maxX - minX) * 0.5, (maxY - minY) * 0.5); // Add margin to cropped bounding box
    const height = maxY - minY + epsilon;
    const width = maxX - minX + epsilon;
    const top = minY - epsilon / 2;
    const right = minX - epsilon / 2;

    handCoordinates = { width: width, height: height, top: top, right: right };
  } else {
    handCoordinates = null;
  }
}

// Detect sign in the hand bounding box
async function detectSign() {
  if (handCoordinates) {
    // Create a canvas to place cropped hand image
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const maxDimension = Math.max(handCoordinates.width, handCoordinates.height);

    canvas.width = maxDimension;
    canvas.height = maxDimension;

    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, maxDimension, maxDimension);
    ctx.scale(-1, 1);
    ctx.translate(-canvas.width, 0);

    const x = (maxDimension - handCoordinates.width) / 2;
    const y = (maxDimension - handCoordinates.height) / 2;

    // Place cropped video data onto canvas
    ctx.drawImage(video, handCoordinates.right, handCoordinates.top, handCoordinates.width, handCoordinates.height, x, y, handCoordinates.width, handCoordinates.height);

    // Detect hand landmarks from cropped hands data
    const results = staticHands.detect(canvas);

    if (results.landmarks.length == 0) return;

    // Detect corresponding letter from hand landmarks
    let datapoints = [];
    results.landmarks[0].forEach((pos) => {
      datapoints.push(1 - pos.x, pos.y, pos.z);
    });
    datapoints.push(results.handednesses[0][0].categoryName == "Right" ? 0 : 1);

    let img = tf.tensor1d(datapoints).expandDims(0);
    let prediction = model.predict(img).squeeze();
    let highestIndex = prediction.argMax().arraySync();
    let letter = LETTERS[highestIndex];

    tf.dispose(img);

    // If two consecutive detection agree on same letter, confirm and display the letter
    if (lastLetter == letter) {
      displayLetter(letter);
      lastLetter = null;
    } else {
      lastLetter = letter;
    }
  } else {
    lastLetter = null;
  }
}

// Delete previous letter if delete key is pressed
document.addEventListener("keydown", function (event) {
  if (event.key === "Backspace" || event.key === "Delete") {
    text.delete(1).flush();
  }
});

// Display a letter in the UI
function displayLetter(letter) {
  let txt = textElem.innerText.substring(0, textElem.innerText.length - 1);

  // Capitalize first letter
  if (txt.length == 0) {
    letter = letter.toUpperCase();
  }

  // Skip letter once if duplicate, most likely still signing last character
  else if (letter.toUpperCase() == txt.at(-1).toUpperCase() && !skippedLastLetter) {
    skippedLastLetter = true;
    return;
  }

  // Don't add duplicate space characters
  else if ((txt.length == 0 || txt.at(-1) == " ") && letter == " ") {
    return;
  }

  skippedLastLetter = false;
  text.type(letter).flush();
}

// Resize video to be full width or full height depending on aspect ratio
function resizeVideo() {
  if (window.innerWidth / window.innerHeight < video.getBoundingClientRect().width / video.getBoundingClientRect().height) video.classList.add("wide");
  else video.classList.remove("wide");
}

window.onresize = resizeVideo;

main();
