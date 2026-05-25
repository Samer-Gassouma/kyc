"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { buildTriangleIndices } from "@/hooks/useFaceDetection";

interface Props {
  points: number[][];  // 468 × [x, y, z] normalized
  width?: number; height?: number;
}

export default function Face3DViewer({ points, width=400, height=500 }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || !points || points.length < 200) return;
    mount.innerHTML = "";

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#09090b");

    const cam = new THREE.PerspectiveCamera(40, width/height, 0.1, 10);
    cam.position.set(0, 0.02, 2.2);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(cam, renderer.domElement);
    controls.target.set(0, -0.02, 0);
    controls.enableDamping = true; controls.dampingFactor = 0.06;
    controls.minDistance = 0.7; controls.maxDistance = 3.8;
    controls.autoRotate = true; controls.autoRotateSpeed = 0.6;
    controls.update();

    // Lighting
    scene.add(new THREE.AmbientLight(0x8888aa, 0.6));
    const key = new THREE.DirectionalLight(0xffffff, 0.9);
    key.position.set(0, 1, 2); scene.add(key);
    const fill = new THREE.DirectionalLight(0x8899cc, 0.4);
    fill.position.set(-2, 0, 1); scene.add(fill);
    const rim = new THREE.DirectionalLight(0xffffff, 0.25);
    rim.position.set(0, -0.5, -1); scene.add(rim);

    // Geometry
    const positions = new Float32Array(points.length * 3);
    for (let i = 0; i < points.length; i++) {
      const [x, y, z] = points[i];
      positions[i*3]   = (x - 0.5) * 1.8;
      positions[i*3+1] = -(y - 0.5) * 2.2;
      positions[i*3+2] = (z + 0.05) * 3.5;
    }

    const triIndices = buildTriangleIndices();
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setIndex(triIndices);
    geo.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
      color: 0x1e3a5f, roughness: 0.5, metalness: 0.1, side: THREE.DoubleSide,
    });
    scene.add(new THREE.Mesh(geo, material));

    // Subtle wireframe overlay
    const lineVerts: number[] = [];
    for (let i = 0; i < triIndices.length; i += 3) {
      for (const [a,b] of [[0,1],[1,2],[2,0]]) {
        const ai=triIndices[i+a], bi=triIndices[i+b];
        lineVerts.push(positions[ai*3],positions[ai*3+1],positions[ai*3+2], positions[bi*3],positions[bi*3+1],positions[bi*3+2]);
      }
    }
    const lg = new THREE.BufferGeometry();
    lg.setAttribute("position", new THREE.Float32BufferAttribute(lineVerts, 3));
    scene.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({color:0x38bdf8, transparent:true, opacity:0.15})));

    let animId: number;
    function animate() { animId=requestAnimationFrame(animate); controls.update(); renderer.render(scene,cam); }
    animate();
    return () => { cancelAnimationFrame(animId); controls.dispose(); renderer.dispose(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points]);

  return <div ref={mountRef} className="overflow-hidden rounded-xl" style={{width,height,maxWidth:"100%"}}/>;
}
