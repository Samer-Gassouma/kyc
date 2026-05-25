/**
 * Silent-Face Anti-Spoofing — client-side liveness check via ONNX Runtime Web.
 *
 * Expects silent_face.onnx in /models/ (MiniFASNetV2).
 * Input:  1x3x80x80 RGB float32 normalized to [0, 1]
 * Output: 3-class logits [print_attack, replay_attack, live]
 */

import * as ort from "onnxruntime-web";

const MODEL_PATH = "/models/silent_face.onnx";

let _session: ort.InferenceSession | null = null;
let _loadPromise: Promise<ort.InferenceSession> | null = null;

async function getSession(): Promise<ort.InferenceSession> {
  if (_session) return _session;
  if (!_loadPromise) {
    _loadPromise = ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["wasm"],
    });
    _session = await _loadPromise;
  }
  return _loadPromise;
}

/**
 * Crop and resize a face region from a canvas frame for liveness input.
 * Returns RGB pixel data as Float32Array (1x3x80x80).
 */
export function prepareLivenessInput(
  source: HTMLCanvasElement | HTMLVideoElement,
  faceBbox: { x: number; y: number; width: number; height: number }
): Float32Array | null {
  const canvas = document.createElement("canvas");
  canvas.width = 80;
  canvas.height = 80;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  // Expand bbox slightly for context
  const margin = 0.2;
  const sx = Math.max(0, faceBbox.x - faceBbox.width * margin);
  const sy = Math.max(0, faceBbox.y - faceBbox.height * margin);
  const sw = Math.min(
    (source instanceof HTMLVideoElement ? source.videoWidth : source.width) - sx,
    faceBbox.width * (1 + margin * 2)
  );
  const sh = Math.min(
    (source instanceof HTMLVideoElement ? source.videoHeight : source.height) - sy,
    faceBbox.height * (1 + margin * 2)
  );

  ctx.drawImage(source, sx, sy, sw, sh, 0, 0, 80, 80);

  const imageData = ctx.getImageData(0, 0, 80, 80);
  const { data } = imageData;

  // Convert RGBA → CHW float32 (normalize to [-1, 1] or [0, 1] depending on model)
  const chw = new Float32Array(3 * 80 * 80);
  for (let i = 0; i < 80 * 80; i++) {
    chw[i] = data[i * 4] / 255.0;                    // R
    chw[80 * 80 + i] = data[i * 4 + 1] / 255.0;       // G
    chw[2 * 80 * 80 + i] = data[i * 4 + 2] / 255.0;   // B
  }
  return chw;
}

/**
 * Run Silent-Face liveness check on a prepared input tensor.
 * Returns real-face probability in [0, 1].
 *
 * Model outputs 3 logits: [print_attack, replay_attack, live].
 * We softmax and return the "live" probability (index 2).
 */
export async function checkLiveness(input: Float32Array): Promise<number> {
  try {
    const session = await getSession();
    const tensor = new ort.Tensor("float32", input, [1, 3, 80, 80]);
    const outputs = await session.run({ input: tensor });
    const logits = Object.values(outputs)[0].data as Float32Array;
    // 3-class softmax: classes = [print_attack, replay_attack, live]
    const maxLogit = Math.max(logits[0], logits[1], logits[2]);
    const exp0 = Math.exp(logits[0] - maxLogit);
    const exp1 = Math.exp(logits[1] - maxLogit);
    const exp2 = Math.exp(logits[2] - maxLogit);
    const sum = exp0 + exp1 + exp2;
    return exp2 / sum; // P(live)
  } catch {
    return 0.0;
  }
}
