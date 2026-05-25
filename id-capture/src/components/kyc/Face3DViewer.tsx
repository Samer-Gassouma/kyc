"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { getFaceTessellation } from "@/hooks/useMediaPipeFace";

interface Face3DViewerProps {
  landmarks: number[][]; // 468 × [x, y, z] from MediaPipe
  width?: number;
  height?: number;
}

export default function Face3DViewer({
  landmarks,
  width = 400,
  height = 460,
}: Face3DViewerProps) {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    if (!landmarks || landmarks.length < 100) {
      if (mount.querySelector("p") === null) {
        mount.innerHTML = "";
        const p = document.createElement("p");
        p.className = "text-xs text-zinc-500 text-center pt-32";
        p.textContent = "Face scan will appear here";
        mount.appendChild(p);
      }
      return;
    }

    // Clear previous renderer
    mount.innerHTML = "";

    // ── Scene ────────────────────────────────────────────────────
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#18181b"); // zinc-900

    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 10);
    camera.position.set(0, 0.02, 2.2);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, -0.02, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enableZoom = true;
    controls.minDistance = 0.7;
    controls.maxDistance = 3.8;
    controls.maxPolarAngle = Math.PI * 0.72;
    controls.autoRotate = false;
    controls.update();

    // ── Lighting ─────────────────────────────────────────────────
    scene.add(new THREE.AmbientLight(0x3388cc, 0.7));
    const p1 = new THREE.PointLight(0x88ccff, 1.2, 4);
    p1.position.set(0.3, 0.5, 1.5);
    scene.add(p1);
    const p2 = new THREE.PointLight(0x4488bb, 0.8, 4);
    p2.position.set(-0.3, -0.2, -1.0);
    scene.add(p2);

    // ── Convert landmarks to 3D points ───────────────────────────
    const pts = new Float32Array(landmarks.length * 3);
    for (let i = 0; i < landmarks.length; i++) {
      const [x, y, z] = landmarks[i];
      pts[i * 3] = (x - 0.5) * 1.8;
      pts[i * 3 + 1] = -(y - 0.5) * 2.2;
      pts[i * 3 + 2] = (z + 0.05) * 3.0;
    }

    // ── Filled translucent skin mesh (triangles from tessellation) ─
    const tess = getFaceTessellation();
    // Build triangle indices from tessellation edges: every 3 edges that share vertices form a tri
    const triIndices: number[] = [];
    const edgeMap = new Map<number, Set<number>>();
    for (const [a, b] of tess) {
      if (!edgeMap.has(a)) edgeMap.set(a, new Set());
      edgeMap.get(a)!.add(b);
      if (!edgeMap.has(b)) edgeMap.set(b, new Set());
      edgeMap.get(b)!.add(a);
    }
    // Find triangle faces: for each edge (a,b), find c where both (a,c) and (b,c) exist
    const added = new Set<string>();
    for (const [a, b] of tess) {
      const aNeighbors = edgeMap.get(a);
      if (!aNeighbors) continue;
      for (const c of aNeighbors) {
        if (c === b) continue;
        const bNeighbors = edgeMap.get(b);
        if (bNeighbors && bNeighbors.has(c)) {
          // Triangle found: (a, b, c)
          const key = [a, b, c].sort((x, y) => x - y).join(",");
          if (!added.has(key)) {
            added.add(key);
            if (a < landmarks.length && b < landmarks.length && c < landmarks.length) {
              triIndices.push(a, b, c);
            }
          }
        }
      }
    }

    // Filled translucent mesh
    const filledGeo = new THREE.BufferGeometry();
    filledGeo.setAttribute("position", new THREE.BufferAttribute(pts, 3));
    filledGeo.setIndex(triIndices);
    filledGeo.computeVertexNormals();
    const filledMat = new THREE.MeshPhongMaterial({
      color: 0x0ea5e9,
      specular: 0x111111,
      shininess: 10,
      transparent: true,
      opacity: 0.12,
      side: THREE.DoubleSide,
    });
    scene.add(new THREE.Mesh(filledGeo, filledMat));

    // ── Wireframe (real tessellation edges) ──────────────────────
    const lineVerts: number[] = [];
    for (const [a, b] of tess) {
      if (a >= landmarks.length || b >= landmarks.length) continue;
      lineVerts.push(pts[a * 3], pts[a * 3 + 1], pts[a * 3 + 2]);
      lineVerts.push(pts[b * 3], pts[b * 3 + 1], pts[b * 3 + 2]);
    }
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute("position", new THREE.Float32BufferAttribute(lineVerts, 3));
    const lineMat = new THREE.LineBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.55,
    });
    scene.add(new THREE.LineSegments(lineGeo, lineMat));

    // ── Dots at landmarks ────────────────────────────────────────
    const dotVerts: number[] = [];
    for (let i = 0; i < landmarks.length; i += 2) {
      dotVerts.push(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]);
    }
    const dotGeo = new THREE.BufferGeometry();
    dotGeo.setAttribute("position", new THREE.Float32BufferAttribute(dotVerts, 3));
    const dotMat = new THREE.PointsMaterial({
      color: 0x38bdf8,
      size: 0.007,
      transparent: true,
      opacity: 0.7,
    });
    scene.add(new THREE.Points(dotGeo, dotMat));

    // ── Render loop ──────────────────────────────────────────────
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
  }, [landmarks]);

  return (
    <div ref={mountRef} className="relative overflow-hidden rounded-xl" style={{ width, height, maxWidth: "100%" }} />
  );
}
