"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

interface Face3DViewerProps {
  points: number[][];  // 68 × [x, y] normalized (0–1)
  colors?: string[];   // 68 CSS color strings sampled from video
  width?: number;
  height?: number;
}

const CONNECTIONS: [number, number][] = [
  // Jaw
  [0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],
  [9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16],
  // Left brow
  [17,18],[18,19],[19,20],[20,21],
  // Right brow
  [22,23],[23,24],[24,25],[25,26],
  // Nose bridge
  [27,28],[28,29],[29,30],
  // Nose bottom
  [30,31],[31,32],[32,33],[33,34],[34,35],[35,30],
  // Left eye
  [36,37],[37,38],[38,39],[39,40],[40,41],[41,36],
  // Right eye
  [42,43],[43,44],[44,45],[45,46],[46,47],[47,42],
  // Outer mouth
  [48,49],[49,50],[50,51],[51,52],[52,53],[53,54],
  [54,55],[55,56],[56,57],[57,58],[58,59],[59,48],
  // Inner mouth
  [60,61],[61,62],[62,63],[63,64],[64,65],[65,66],[66,67],[67,60],
  // Cross-connectors: eyes to nose
  [27,39],[27,42],[30,33],[30,48],[30,54],
  // Brows to nose bridge
  [21,27],[22,27],
  // Eyes to brows
  [19,37],[19,36],[24,44],[24,43],
];

export default function Face3DViewer({ points, colors, width = 400, height = 460 }: Face3DViewerProps) {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    if (!points || points.length < 30) {
      if (!mount.querySelector("p")) {
        mount.innerHTML = "";
        const p = document.createElement("p");
        p.className = "text-xs text-zinc-500 text-center pt-32";
        p.textContent = "Face scan will appear here";
        mount.appendChild(p);
      }
      return;
    }
    mount.innerHTML = "";

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#18181b");

    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 10);
    camera.position.set(0, 0.02, 2.2);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, -0.02, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.7;
    controls.maxDistance = 3.8;
    controls.maxPolarAngle = Math.PI * 0.72;
    controls.update();

    scene.add(new THREE.AmbientLight(0x3388cc, 0.7));

    // ── Build 3D points ──────────────────────────────────────────
    const pts3 = new Float32Array(points.length * 3);
    for (let i = 0; i < points.length; i++) {
      pts3[i * 3] = (points[i][0] - 0.5) * 1.8;
      pts3[i * 3 + 1] = -(points[i][1] - 0.5) * 2.2;
      pts3[i * 3 + 2] = 0;
    }
    // Add fake z-depth for nose, brows, and lips so it looks 3D
    const zOff = [27,28,29,30,31,32,33,34,35, 17,18,19,20,21,22,23,24,25,26, 0,1,2,3,4,12,13,14,15,16];
    for (const i of zOff) {
      if (i < points.length) pts3[i * 3 + 2] = -0.15;
    }
    // Nose tip protrudes
    pts3[30 * 3 + 2] = -0.35;
    // Chin protrudes slightly
    pts3[8 * 3 + 2] = -0.1;

    // Parse colors from video samples
    const hasColors = colors && colors.length >= points.length;
    const dotColors: number[] = [];
    for (let i = 0; i < points.length; i++) {
      if (hasColors) {
        const m = colors![i].match(/[\d.]+/g);
        if (m && m.length >= 3) {
          dotColors.push(+m[0] / 255, +m[1] / 255, +m[2] / 255);
        } else {
          dotColors.push(0.22, 0.74, 0.97); // fallback cyan
        }
      } else {
        dotColors.push(0.22, 0.74, 0.97);
      }
    }

    // ── Wireframe with per-vertex colors ──────────────────────────
    const lineVerts: number[] = [];
    const lineCols: number[] = [];
    for (const [a, b] of CONNECTIONS) {
      if (a >= points.length || b >= points.length) continue;
      lineVerts.push(pts3[a * 3], pts3[a * 3 + 1], pts3[a * 3 + 2]);
      lineVerts.push(pts3[b * 3], pts3[b * 3 + 1], pts3[b * 3 + 2]);
      const ca = dotColors.slice(a * 3, a * 3 + 3);
      const cb = dotColors.slice(b * 3, b * 3 + 3);
      lineCols.push(...ca, ...cb);
    }
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute("position", new THREE.Float32BufferAttribute(lineVerts, 3));
    lineGeo.setAttribute("color", new THREE.Float32BufferAttribute(lineCols, 3));
    scene.add(new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.65,
    })));

    // ── Colored dots at landmarks ─────────────────────────────────
    const dotVerts: number[] = [];
    for (let i = 0; i < points.length; i++) {
      dotVerts.push(pts3[i * 3], pts3[i * 3 + 1], pts3[i * 3 + 2]);
    }
    const dotGeo = new THREE.BufferGeometry();
    dotGeo.setAttribute("position", new THREE.Float32BufferAttribute(dotVerts, 3));
    dotGeo.setAttribute("color", new THREE.Float32BufferAttribute(dotColors, 3));
    scene.add(new THREE.Points(dotGeo, new THREE.PointsMaterial({
      size: 0.012, vertexColors: true, transparent: true, opacity: 0.9,
    })));

    let animId: number;
    function animate() {
      animId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    return () => {
      cancelAnimationFrame(animId);
      controls.dispose();
      renderer.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points]);

  return <div ref={mountRef} className="overflow-hidden rounded-xl" style={{ width, height, maxWidth: "100%" }} />;
}
