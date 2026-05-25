"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

interface Face3DViewerProps {
  points: number[][];
  colors?: string[];
  width?: number;
  height?: number;
}

const CONNECTIONS: [number, number][] = [
  [0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16],
  [17,18],[18,19],[19,20],[20,21],[22,23],[23,24],[24,25],[25,26],
  [27,28],[28,29],[29,30],[30,31],[31,32],[32,33],[33,34],[34,35],[35,30],
  [36,37],[37,38],[38,39],[39,40],[40,41],[41,36],
  [42,43],[43,44],[44,45],[45,46],[46,47],[47,42],
  [48,49],[49,50],[50,51],[51,52],[52,53],[53,54],[54,55],[55,56],[56,57],[57,58],[58,59],[59,48],
  [60,61],[61,62],[62,63],[63,64],[64,65],[65,66],[66,67],[67,60],
  [27,39],[27,42],[30,33],[30,48],[30,54],[21,27],[22,27],[19,37],[19,36],[24,44],[24,43],
];

export default function Face3DViewer({ points, colors, width = 400, height = 460 }: Face3DViewerProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{
    scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer;
    controls: OrbitControls; lines: THREE.LineSegments; dots: THREE.Points; animId: number;
  } | null>(null);

  // Init once
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || sceneRef.current) return;

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
    controls.enableDamping = true; controls.dampingFactor = 0.08;
    controls.minDistance = 0.7; controls.maxDistance = 3.8; controls.maxPolarAngle = Math.PI * 0.72;
    controls.update();
    scene.add(new THREE.AmbientLight(0x3388cc, 0.7));

    // Empty placeholders
    const lines = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.65 }));
    const dots = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ size: 0.012, vertexColors: true, transparent: true, opacity: 0.9 }));
    scene.add(lines); scene.add(dots);

    function animate() { const id = requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); sceneRef.current!.animId = id; }
    const animId = requestAnimationFrame(animate);

    sceneRef.current = { scene, camera, renderer, controls, lines, dots, animId };

    return () => {
      cancelAnimationFrame(sceneRef.current!.animId);
      controls.dispose();
      renderer.dispose();
      sceneRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update geometry when points change (throttled via ref comparison)
  const lastUpdateRef = useRef(0);

  useEffect(() => {
    const s = sceneRef.current;
    if (!s || !points || points.length < 30) return;

    // Throttle to ~15fps to avoid geometry churn
    const now = Date.now();
    if (now - lastUpdateRef.current < 66) return;
    lastUpdateRef.current = now;

    const pts3 = new Float32Array(points.length * 3);
    for (let i = 0; i < points.length; i++) {
      pts3[i * 3] = (points[i][0] - 0.5) * 1.8;
      pts3[i * 3 + 1] = -(points[i][1] - 0.5) * 2.2;
      pts3[i * 3 + 2] = 0;
    }
    const zOff = [27,28,29,30,31,32,33,34,35, 17,18,19,20,21,22,23,24,25,26, 0,1,2,3,4,12,13,14,15,16];
    for (const i of zOff) { if (i < points.length) pts3[i * 3 + 2] = -0.15; }
    pts3[30 * 3 + 2] = -0.35; pts3[8 * 3 + 2] = -0.1;

    const hasColors = colors && colors.length >= points.length;
    const dotColors: number[] = [];
    for (let i = 0; i < points.length; i++) {
      if (hasColors) {
        const m = colors![i].match(/[\d.]+/g);
        dotColors.push(m && m.length >= 3 ? +m[0] / 255 : 0.22, m && m.length >= 3 ? +m[1] / 255 : 0.74, m && m.length >= 3 ? +m[2] / 255 : 0.97);
      } else dotColors.push(0.22, 0.74, 0.97);
    }

    const lineVerts: number[] = [], lineCols: number[] = [];
    for (const [a, b] of CONNECTIONS) {
      if (a >= points.length || b >= points.length) continue;
      lineVerts.push(pts3[a*3], pts3[a*3+1], pts3[a*3+2], pts3[b*3], pts3[b*3+1], pts3[b*3+2]);
      lineCols.push(...dotColors.slice(a*3, a*3+3), ...dotColors.slice(b*3, b*3+3));
    }
    s.lines.geometry.dispose();
    const lg = new THREE.BufferGeometry();
    lg.setAttribute("position", new THREE.Float32BufferAttribute(lineVerts, 3));
    lg.setAttribute("color", new THREE.Float32BufferAttribute(lineCols, 3));
    s.lines.geometry = lg;

    const dotVerts: number[] = [];
    for (let i = 0; i < points.length; i++) dotVerts.push(pts3[i*3], pts3[i*3+1], pts3[i*3+2]);
    s.dots.geometry.dispose();
    const dg = new THREE.BufferGeometry();
    dg.setAttribute("position", new THREE.Float32BufferAttribute(dotVerts, 3));
    dg.setAttribute("color", new THREE.Float32BufferAttribute(dotColors, 3));
    s.dots.geometry = dg;
  }, [points, colors]);

  return <div ref={mountRef} className="overflow-hidden rounded-xl" style={{ width, height, maxWidth: "100%" }} />;
}
