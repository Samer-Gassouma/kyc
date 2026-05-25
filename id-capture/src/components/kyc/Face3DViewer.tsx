"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { buildTriangleIndices } from "@/hooks/useFaceDetection";

interface Props {
  points: number[][];       // 468 × [x, y, z] normalized
  faceTexture?: string;     // cropped face data URL
  cropMeta?: { sx:number; sy:number; sw:number; sh:number; vw:number; vh:number } | null;
  width?: number; height?: number;
}

export default function Face3DViewer({ points, faceTexture, cropMeta, width=400, height=500 }: Props) {
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

    // ── Lighting ─────────────────────────────────────────────────
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const key = new THREE.DirectionalLight(0xffffff, 0.9);
    key.position.set(0, 1, 2); scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.35);
    fill.position.set(-2, 0, 1); scene.add(fill);
    const rim = new THREE.DirectionalLight(0x8899cc, 0.3);
    rim.position.set(0, -0.5, -1); scene.add(rim);

    // ── Build geometry ───────────────────────────────────────────
    const positions = new Float32Array(points.length * 3);
    const uvs = new Float32Array(points.length * 2);

    for (let i = 0; i < points.length; i++) {
      const [x, y, z] = points[i];
      positions[i*3]   = (x - 0.5) * 1.8;
      positions[i*3+1] = -(y - 0.5) * 2.2;
      positions[i*3+2] = (z + 0.05) * 3.5;

      // UV from cropped region or full frame
      if (cropMeta) {
        const ux = x * cropMeta.vw; // absolute pixel in original frame
        const uy = y * cropMeta.vh;
        uvs[i*2]   = (ux - cropMeta.sx) / cropMeta.sw;
        uvs[i*2+1] = 1.0 - (uy - cropMeta.sy) / cropMeta.sh;
      } else {
        uvs[i*2]   = x;
        uvs[i*2+1] = 1.0 - y;
      }
    }

    const triIndices = buildTriangleIndices();
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
    geo.setIndex(triIndices);
    geo.computeVertexNormals();

    // ── Material with face texture ───────────────────────────────
    let material: THREE.Material;
    if (faceTexture) {
      const tex = new THREE.TextureLoader().load(faceTexture);
      tex.colorSpace = THREE.SRGBColorSpace;
      material = new THREE.MeshStandardMaterial({
        map: tex, roughness: 0.7, metalness: 0.05, side: THREE.DoubleSide,
      });
    } else {
      material = new THREE.MeshStandardMaterial({
        color: 0x334455, roughness: 0.8, metalness: 0.0, side: THREE.DoubleSide,
      });
    }

    scene.add(new THREE.Mesh(geo, material));

    // ── Subtle dot overlay at key points ─────────────────────────
    const dotVerts: number[] = [];
    for (let i = 0; i < points.length; i += 4) {
      dotVerts.push(positions[i*3], positions[i*3+1], positions[i*3+2]);
    }
    const dg = new THREE.BufferGeometry();
    dg.setAttribute("position", new THREE.Float32BufferAttribute(dotVerts, 3));
    scene.add(new THREE.Points(dg, new THREE.PointsMaterial({ color:0xffffff, size:0.004, transparent:true, opacity:0.35 })));

    let animId: number;
    function animate() { animId=requestAnimationFrame(animate); controls.update(); renderer.render(scene,cam); }
    animate();

    return () => { cancelAnimationFrame(animId); controls.dispose(); renderer.dispose(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, faceTexture]);

  return <div ref={mountRef} className="overflow-hidden rounded-xl" style={{width,height,maxWidth:"100%"}}/>;
}
