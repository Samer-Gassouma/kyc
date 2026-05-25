"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { REGION_EDGES } from "@/hooks/useFaceDetection";

interface Face3DViewerProps {
  points: number[][];  // 468 × [x, y, z] normalized
  width?: number; height?: number;
}

export default function Face3DViewer({ points, width=400, height=500 }: Face3DViewerProps) {
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
    controls.maxPolarAngle = Math.PI * 0.72;
    controls.autoRotate = true; controls.autoRotateSpeed = 0.6;
    controls.update();

    scene.add(new THREE.AmbientLight(0x334466, 0.8));
    const pl = new THREE.PointLight(0x88aacc, 1.0, 5);
    pl.position.set(0.2, 0.4, 1.5); scene.add(pl);

    // Convert to 3D
    const toV3 = (p: number[]) => new THREE.Vector3((p[0]-0.5)*1.8, -(p[1]-0.5)*2.2, (p[2]+0.05)*3.0);

    // Draw each region in its own color
    for (const [, region] of Object.entries(REGION_EDGES)) {
      const verts: number[] = [];
      for (const [a,b] of region.edges) {
        if (a>=points.length||b>=points.length) continue;
        const va=toV3(points[a]), vb=toV3(points[b]);
        verts.push(va.x,va.y,va.z, vb.x,vb.y,vb.z);
      }
      if (verts.length===0) continue;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
      scene.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({
        color: new THREE.Color(region.color), transparent: true, opacity: 0.7,
      })));
    }

    // Dots
    const dotVerts: number[] = [];
    for (let i=0; i<points.length; i+=2) {
      const v=toV3(points[i]); dotVerts.push(v.x,v.y,v.z);
    }
    const dg = new THREE.BufferGeometry();
    dg.setAttribute("position", new THREE.Float32BufferAttribute(dotVerts, 3));
    scene.add(new THREE.Points(dg, new THREE.PointsMaterial({ color:0xffffff, size:0.008, transparent:true, opacity:0.6 })));

    let animId: number;
    function animate() { animId=requestAnimationFrame(animate); controls.update(); renderer.render(scene,cam); }
    animate();

    return () => { cancelAnimationFrame(animId); controls.dispose(); renderer.dispose(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points]);

  return <div ref={mountRef} className="overflow-hidden rounded-xl" style={{width,height,maxWidth:"100%"}}/>;
}
