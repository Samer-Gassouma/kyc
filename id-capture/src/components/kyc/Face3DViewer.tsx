"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

interface Face3DViewerProps {
  landmarks: number[][]; // 468 points of [x, y, z] normalized
  tessellation: [number, number][];
  width?: number;
  height?: number;
}

export default function Face3DViewer({
  landmarks,
  tessellation,
  width = 400,
  height = 500,
}: Face3DViewerProps) {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    if (!landmarks || landmarks.length < 100) {
      // Show placeholder while waiting
      if (mount.children.length === 0) {
        const p = document.createElement("p");
        p.className = "text-xs text-zinc-500 text-center pt-20";
        p.textContent = "No face data yet — complete the scan first";
        mount.appendChild(p);
      }
      return;
    }
    // Clear placeholder
    mount.innerHTML = "";

    // ── Scene setup ──────────────────────────────────────────────
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#09090b"); // zinc-950

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 10);
    camera.position.set(0, 0.05, 2.2);
    camera.lookAt(0, -0.05, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, -0.05, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.8;
    controls.maxDistance = 3.5;
    controls.maxPolarAngle = Math.PI * 0.75;
    controls.update();

    // ── Ambient light ────────────────────────────────────────────
    const ambient = new THREE.AmbientLight(0x4488cc, 0.6);
    scene.add(ambient);
    const point = new THREE.PointLight(0x88ccff, 1.5, 5);
    point.position.set(0, 0.3, 1.5);
    scene.add(point);

    // ── Build mesh geometry ──────────────────────────────────────
    // Convert normalized landmarks to 3D points (center at origin)
    // MediaPipe coords: x→right, y→down, z→forward (away from camera)
    // Three.js: x→right, y→up, z→toward viewer
    const points3D: THREE.Vector3[] = [];
    for (const lm of landmarks) {
      const px = (lm[0] - 0.5) * 1.8;   // x: width
      const py = -(lm[1] - 0.5) * 2.2;  // y: height (flip for Three.js)
      const pz = (lm[2] + 0.05) * 2.0;  // z: depth
      points3D.push(new THREE.Vector3(px, py, pz));
    }

    // Wireframe lines from tessellation
    const lineGeo = new THREE.BufferGeometry();
    const lineVerts: number[] = [];
    for (const [a, b] of tessellation) {
      if (a >= points3D.length || b >= points3D.length) continue;
      lineVerts.push(points3D[a].x, points3D[a].y, points3D[a].z);
      lineVerts.push(points3D[b].x, points3D[b].y, points3D[b].z);
    }
    lineGeo.setAttribute("position", new THREE.Float32BufferAttribute(lineVerts, 3));
    const lineMat = new THREE.LineBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.5,
      linewidth: 1,
    });
    const lines = new THREE.LineSegments(lineGeo, lineMat);
    scene.add(lines);

    // Points at landmarks
    const dotGeo = new THREE.BufferGeometry();
    const dotVerts: number[] = [];
    for (let i = 0; i < points3D.length; i += 2) {
      dotVerts.push(points3D[i].x, points3D[i].y, points3D[i].z);
    }
    dotGeo.setAttribute("position", new THREE.Float32BufferAttribute(dotVerts, 3));
    const dotMat = new THREE.PointsMaterial({
      color: 0x38bdf8,
      size: 0.008,
      transparent: true,
      opacity: 0.8,
    });
    const dots = new THREE.Points(dotGeo, dotMat);
    scene.add(dots);

    // ── Animation loop ───────────────────────────────────────────
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
      if (mount.contains(renderer.domElement)) {
        mount.removeChild(renderer.domElement);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [landmarks]);

  return (
    <div
      ref={mountRef}
      className="relative overflow-hidden rounded-2xl"
      style={{ width, height, maxWidth: "100%" }}
    />
  );
}
