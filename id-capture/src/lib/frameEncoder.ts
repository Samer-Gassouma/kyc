/**
 * frameEncoder.ts — canvas → JPEG binary for WebSocket transport.
 *
 * Converts a canvas element to a JPEG Blob at configurable quality.
 * Never downsamples the source — uses the full canvas resolution.
 */

export async function canvasToJpegBlob(
  canvas: HTMLCanvasElement,
  quality: number = 0.85
): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) resolve(blob);
        else reject(new Error("Failed to create JPEG blob from canvas"));
      },
      "image/jpeg",
      quality
    );
  });
}

export async function canvasToJpegArrayBuffer(
  canvas: HTMLCanvasElement,
  quality: number = 0.85
): Promise<ArrayBuffer> {
  const blob = await canvasToJpegBlob(canvas, quality);
  return blob.arrayBuffer();
}

/**
 * Grab the current video frame onto a canvas at full resolution.
 * Returns the canvas (caller can read pixels or encode).
 */
export function grabFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement
): HTMLCanvasElement {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Cannot get 2d context");
  ctx.drawImage(video, 0, 0);
  return canvas;
}

/**
 * Grab a downscaled frame for ONNX inference (224×224).
 */
export function grabFrameResized(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  width: number = 224,
  height: number = 224
): ImageData {
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Cannot get 2d context");
  ctx.drawImage(video, 0, 0, width, height);
  return ctx.getImageData(0, 0, width, height);
}
