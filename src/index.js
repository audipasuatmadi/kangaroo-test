import React from "react";
import ReactDOM from "react-dom";
import * as tf from "@tensorflow/tfjs";
import "./styles.css";
tf.setBackend("webgl");

const threshold = 0.75;

async function load_model() {
  // It's possible to load the model locally or from a repo
  // You can choose whatever IP and PORT you want in the "http://127.0.0.1:8080/model.json" just set it before in your https server
  // const model = await loadGraphModel("http://127.0.0.1:8080/models/model-3");
  // const model = await loadGraphModel(
  //   "https://raw.githubusercontent.com/audipasuatmadi/kangguru/master/model.json"
  // );
  console.log("loading model");
  const model = await tf.loadGraphModel(`inspix2/model.json`);
  console.log("model loaded");
  return model;
}

let classesDir = {
  1: {
    name: "Masker",
    id: 1,
  },
  2: {
    name: "No mask",
    id: 2,
  },
  3: {
    name: "Incorrect",
    id: 3,
  },
};
let did = 0;
class App extends React.Component {
  videoRef = React.createRef();
  canvasRef = React.createRef();

  /**
   * initialization
   */
  componentDidMount() {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const webCamPromise = navigator.mediaDevices
        .getUserMedia({
          audio: false,
          video: {
            facingMode: "user",
          },
        })
        .then((stream) => {
          // streams the video from webcam to web.
          window.stream = stream;
          this.videoRef.current.srcObject = stream;
          // resolve when video is streamed
          return new Promise((resolve, reject) => {
            this.videoRef.current.onloadedmetadata = () => {
              resolve();
            };
          });
        });

      const modelPromise = load_model();
      // start detecting frame when both modelPromise and webCamPromise resolved.
      Promise.all([modelPromise, webCamPromise])
        .then((values) => {
          // passes videoRef and model to this.detectFrame
          this.detectFrame(this.videoRef.current, values[0]);
        })
        .catch((error) => {
          console.error(error);
        });
    }
  }

  /**
   * runs the model, accepts video and the loaded model
   */

  detectFrame = (video, model) => {
    tf.engine().startScope(); // avoids memory leak, optional
    // calls executeAsync on `process_input` result of video

    model.executeAsync(this.process_input(video)).then((predictions) => {
      // renders the prediction (draws bounding box)
      this.renderPredictions(predictions, video);
      requestAnimationFrame(() => {
        this.detectFrame(video, model);
      });
      tf.engine().endScope();
    });
  };

  /**
   * preprocesses the video_frame, turning it into tfimg (fromPixels)
   * also to int
   * Also transposes it by [0, 1, 2], and expanded its dimension for batch
   */
  process_input(video_frame) {
    // converts <video></video> into a tensor
    const tff = tf.browser.fromPixels(video_frame);
    const tfimg = tff.toInt();
    const expandedimg = tfimg.transpose([0, 1, 2]).expandDims();
    // if (did <= 10) {
    //   const arrTf = expandedimg.arraySync();
    //   axios
    //     .post("http://localhost:8000/masks", {
    //       tensor: JSON.stringify(arrTf),
    //     })
    //     .then((pred) => console.log(pred));
    //   did = did + 1;
    // }
    return expandedimg;
  }

  /**
   * supplies "renderPredictions" with bounding box value,
   * class of the prediction
   * score of the prediction
   * bounding box ()
   *
   * SELECTION whether an object is kangaroo or not is held here!
   * If score above threshold, then it is a kangaroo, return it
   */
  buildDetectedObjects(scores, threshold, boxes, classes, classesDir) {
    const detectionObjects = [];
    var video_frame = document.getElementById("frame");

    const a = {
      scores: scores,
      boxes: boxes,
      classes: classes,
    };
    // setiap objek yang dilihat, draw bounding box disekitarnya
    scores[0].forEach((score, i) => {
      // jika dia mendekati kangguru, draw bounding box
      if (score > 0.5) {
        const bbox = [];
        const minY = boxes[0][i][0] * video_frame.offsetHeight;
        const minX = boxes[0][i][1] * video_frame.offsetWidth;
        const maxY = boxes[0][i][2] * video_frame.offsetHeight;
        const maxX = boxes[0][i][3] * video_frame.offsetWidth;
        bbox[0] = minX;
        bbox[1] = minY;
        bbox[2] = maxX - minX;
        bbox[3] = maxY - minY;
        detectionObjects.push({
          class: classes[i],
          label: classesDir[classes[i]].name,
          score: score.toFixed(4),
          bbox: bbox,
        });
      }
    });
    return detectionObjects;
  }
  /**
   * draws bounding box and its label
   * called by: detectFrame
   */
  renderPredictions = (predictions) => {
    const ctx = this.canvasRef.current.getContext("2d");
    // clears canvas from previous predictions
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // Font options.
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";

    const boxes = predictions[0].arraySync();
    const scores = predictions[1].arraySync();
    const classes = predictions[2].dataSync();
    // find kangaroo(s) object.
    console.log(scores);
    const detections = this.buildDetectedObjects(
      scores,
      threshold,
      boxes,
      classes,
      classesDir
    );
    // foreach kangaroo(s) object, draw a bounding box!
    detections.forEach((item) => {
      const x = item["bbox"][0];
      const y = item["bbox"][1];
      const width = item["bbox"][2];
      const height = item["bbox"][3];

      // Draw the bounding box.
      if (item["label"] == "Masker") {
        ctx.strokeStyle = "#ba03fc";
      }
      ctx.strokeStyle =
        item["label"] === "Masker"
          ? "#0bde00"
          : item["label"] === "No mask"
          ? "#ff1717"
          : "#fff719";
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, width, height);

      // Draw the label background.
      ctx.fillStyle =
        item["label"] === "Masker"
          ? "#0bde00"
          : item["label"] === "No mask"
          ? "#ff1717"
          : "#fff719";
      const textWidth = ctx.measureText(
        item["label"] + " " + (100 * item["score"]).toFixed(2) + "%"
      ).width;
      const textHeight = parseInt(font, 10); // base 10
      ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
    });

    detections.forEach((item) => {
      const x = item["bbox"][0];
      const y = item["bbox"][1];

      // Draw the text last to ensure it's on top.
      ctx.fillStyle = "#000000";
      ctx.fillText(
        item["label"] + " " + (100 * item["score"]).toFixed(2) + "%",
        x,
        y
      );
    });
  };

  render() {
    return (
      <div className="container">
        <video
          // style={{height: "1000px", width: "1000px"}}
          className="size"
          autoPlay
          playsInline
          muted
          ref={this.videoRef}
          // width="1000"
          // height="1000"
          id="frame"
        />
        <canvas
          className="bounding-box"
          ref={this.canvasRef}
          width="1000"
          height="1000"
        />
      </div>
    );
  }
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
