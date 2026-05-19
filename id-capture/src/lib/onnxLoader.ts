/**
 * onnxLoader.ts — ONNX Runtime Web session init + inference wrapper.
 *
 * Loads the MobileNetV3 quality classifier from /models/quality_classifier.onnx
 * Session is created once and reused across frames.
 */

import * as ort from "onnxruntime-web";

let session: ort.InferenceSession | null = null;
let loadingPromise: Promise<ort.InferenceSession> | null = null;
let modelFailed = false;

export interface QualityResult {
  blur_score: number;
  glare_score: number;
  orientation: string;
  brightness: string;
  local_pass: boolean;
}

const MODEL_PATH = "/models/quality_classifier.onnx";

/**
 * Load the ONNX model once. Returns cached session on subsequent calls.
 */
export async function loadQualityModel(): Promise<ort.InferenceSession> {
  if (session) return session;
  if (modelFailed) throw new Error("ONNX model unavailable — using heuristic fallback");
  if (loadingPromise) return loadingPromise;

  loadingPromise = (async () => {
    try {
      // Configure ONNX Runtime Web
      ort.env.wasm.numThreads = 1;

      const s = await ort.InferenceSession.create(MODEL_PATH, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      session = s;
      return s;
    } catch (err) {
      loadingPromise = null;
      modelFailed = true;
      console.warn("ONNX quality model not available, using heuristic fallback");
      throw err;
    }
  })();

  return loadingPromise;
}

/**
 * Run quality inference on a 224×224 ImageData.
 * If the ONNX model is not available, falls back to canvas-based heuristics.
 */
export async function runQualityCheck(
  imageData: ImageData
): Promise<QualityResult> {
  try {
    const model = await loadQualityModel();
    return await runOnnxInference(model, imageData);
  } catch {
    // Fallback: canvas-based heuristic quality checks
    return runHeuristicCheck(imageData);
  }
}

async function runOnnxInference(
  model: ort.InferenceSession,
  imageData: ImageData
): Promise<QualityResult> {
  const { data, width, height } = imageData;

  // Convert RGBA ImageData to float32 CHW tensor [1, 3, 224, 224]
  const floatData = new Float32Array(3 * width * height);
  for (let i = 0; i < width * height; i++) {
    floatData[i] = data[i * 4] / 255.0; // R
    floatData[width * height + i] = data[i * 4 + 1] / 255.0; // G
    floatData[2 * width * height + i] = data[i * 4 + 2] / 255.0; // B
  }

  const inputTensor = new ort.Tensor("float32", floatData, [1, 3, height, width]);

  const inputName = model.inputNames[0] || "input";
  const feeds: Record<string, ort.Tensor> = { [inputName]: inputTensor };
  const results = await model.run(feeds);

  // Parse output — expected shape [1, 4]: [blur, glare, orientation_idx, brightness_idx]
  const outputName = model.outputNames[0] || "output";
  const output = results[outputName].data as Float32Array;

  const blur_score = Math.max(0, Math.min(1, output[0]));
  const glare_score = Math.max(0, Math.min(1, output[1]));
  const orientationIdx = Math.round(output[2]);
  const brightnessIdx = Math.round(output[3]);

  const orientations = ["straight", "tilted_left", "tilted_right", "upside_down"];
  const brightnesses = ["ok", "too_dark", "too_bright"];

  const orientation = orientations[orientationIdx] || "straight";
  const brightness = brightnesses[brightnessIdx] || "ok";

  const local_pass =
    blur_score > 0.5 &&
    glare_score < 0.3 &&
    orientation === "straight" &&
    brightness === "ok";

  return { blur_score, glare_score, orientation, brightness, local_pass };
}

/**
 * Heuristic fallback when ONNX model is not available.
 * Uses pixel statistics from the 224×224 canvas frame.
 */
function runHeuristicCheck(imageData: ImageData): QualityResult {
  const { data, width, height } = imageData;
  const totalPixels = width * height;

  // Compute grayscale statistics
  let sum = 0;
  let sumSq = 0;
  let brightCount = 0;

  for (let i = 0; i < totalPixels; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;
    sum += gray;
    sumSq += gray * gray;
    if (gray > 240) brightCount++;
  }

  const mean = sum / totalPixels;
  const variance = sumSq / totalPixels - mean * mean;

  // Blur estimate: low variance = blurry
  // Laplacian approximation via pixel variance
  const blur_score = Math.min(1, variance / 2500);

  // Glare: ratio of very bright pixels
  const glare_score = brightCount / totalPixels;

  // Brightness classification
  let brightness: string = "ok";
  if (mean < 60) brightness = "too_dark";
  else if (mean > 200) brightness = "too_bright";

  const orientation = "straight"; // Cannot detect from statistics alone

  const local_pass =
    blur_score > 0.3 &&
    glare_score < 0.15 &&
    brightness === "ok";

  return { blur_score, glare_score, orientation, brightness, local_pass };
}
