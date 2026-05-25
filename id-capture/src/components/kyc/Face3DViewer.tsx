"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { buildTriangleIndices } from "@/hooks/useFaceDetection";

interface Props {
  points: number[][];       // 468 × [x, y, z] normalized [0,1]
  faceTexture?: string;     // cropped face dataURL
  width?: number; height?: number;
}

function dataURLToTexture(dataURL: string): THREE.Texture {
  const img = new Image();
  img.src = dataURL;
  const tex = new THREE.Texture(img);
  tex.flipY = false;
  tex.colorSpace = THREE.SRGBColorSpace;
  img.onload = () => { tex.needsUpdate = true; };
  return tex;
}

export default function Face3DViewer({ points, faceTexture, width=400, height=500 }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || !points || points.length < 200) return;
    mount.innerHTML = "";

    const w = mount.clientWidth || width;
    const h = mount.clientHeight || height;
    console.log("[3D] points:", points.length, "hasTex:", !!faceTexture);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#09090b");

    const cam = new THREE.PerspectiveCamera(50, w / h, 0.1, 10);
    cam.position.set(0, 0, 2.5);
    cam.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(cam, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.enableDamping = true; controls.dampingFactor = 0.08;
    controls.minDistance = 0.8; controls.maxDistance = 5;
    controls.autoRotate = true; controls.autoRotateSpeed = 0.5;
    controls.update();

    // Lighting
    scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const k = new THREE.DirectionalLight(0xffffff, 0.7); k.position.set(0, 1, 2); scene.add(k);

    // ── Geometry ─────────────────────────────────────────────────
    const positions = new Float32Array(points.length * 3);
    for (let i = 0; i < points.length; i++) {
      const [x, y, z] = points[i];
      positions[i*3]   = (x - 0.5) * 2.5;
      positions[i*3+1] = -(y - 0.5) * 3.0;
      positions[i*3+2] = z * 4.0;
    }

    const triIndices = buildTriangleIndices();
    console.log("[3D] tris:", triIndices.length / 3);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    if (triIndices.length > 0) {
      geo.setIndex(triIndices);
    }
    geo.computeVertexNormals();

    // ── Material ─────────────────────────────────────────────────
    if (faceTexture) {
      const uvs = new Float32Array(points.length * 2);
      for (let i = 0; i < points.length; i++) {
        uvs[i*2]   = points[i][0];
        uvs[i*2+1] = 1.0 - points[i][1];
      }
      geo.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));

      const tex = dataURLToTexture(faceTexture);
      const mat = new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide });
      console.log("[3D] texture material applied");
      scene.add(new THREE.Mesh(geo, mat));
    } else {
      const mat = new THREE.MeshStandardMaterial({
        color: 0x1e3a5f, roughness: 0.5, metalness: 0.05, side: THREE.DoubleSide,
      });
      scene.add(new THREE.Mesh(geo, mat));
    }

    // ── Debug: white dot cloud ───────────────────────────────────
    const dotGeo = new THREE.BufferGeometry();
    dotGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    scene.add(new THREE.Points(dotGeo, new THREE.PointsMaterial({ color: 0xffffff, size: 0.008 })));

    let animId: number;
    function animate() { animId = requestAnimationFrame(animate); controls.update(); renderer.render(scene, cam); }
    animate();
    return () => { cancelAnimationFrame(animId); controls.dispose(); renderer.dispose(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, faceTexture]);

  return <div ref={mountRef} className="overflow-hidden rounded-xl" style={{ width, height, maxWidth: "100%" }} />;
}
