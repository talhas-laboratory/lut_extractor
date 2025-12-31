
import React, { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';

interface LatticeProps {
    originalPoints?: number[][];
    warpedPoints?: number[][];
}

const PointCloud: React.FC<{ points: number[][], color: string, size?: number }> = ({ points, color, size = 0.05 }) => {
    const meshRef = useRef<THREE.InstancedMesh>(null);
    const dummy = useMemo(() => new THREE.Object3D(), []);

    useFrame(() => {
        if (!meshRef.current) return;
        // Optional animation: rotate minimal
        // meshRef.current.rotation.y += 0.001;
    });

    useMemo(() => {
        if (!meshRef.current) return;
        points.forEach((pt, i) => {
            dummy.position.set(pt[0] - 0.5, pt[1] - 0.5, pt[2] - 0.5); // Center the 0-1 cube
            dummy.updateMatrix();
            meshRef.current!.setMatrixAt(i, dummy.matrix);
        });
        meshRef.current.instanceMatrix.needsUpdate = true;
    }, [points, dummy]);

    return (
        <instancedMesh ref={meshRef} args={[undefined, undefined, points.length]}>
            <sphereGeometry args={[size, 8, 8]} />
            <meshBasicMaterial color={color} />
        </instancedMesh>
    );
};

export const LatticeVisualizer: React.FC<LatticeProps> = ({ originalPoints, warpedPoints }) => {
    return (
        <div className="w-full h-full bg-black border-4 border-black relative">
            <div className="absolute top-2 left-2 z-10 font-mono text-xs text-white bg-black p-1">
                LATTICE_ENGINE_V1
            </div>
            <Canvas>
                <PerspectiveCamera makeDefault position={[1.5, 1.5, 2]} />
                <OrbitControls enableZoom={true} enablePan={false} autoRotate autoRotateSpeed={0.5} />

                <gridHelper args={[2, 10, 0x333333, 0x111111]} />
                <axesHelper args={[1]} />

                {/* Draw Bounding Box */}
                <mesh position={[0, 0, 0]}>
                    <boxGeometry args={[1, 1, 1]} />
                    <meshBasicMaterial color="white" wireframe opacity={0.1} transparent />
                </mesh>

                {originalPoints && (
                    <PointCloud points={originalPoints} color="#444444" size={0.015} />
                )}

                {warpedPoints && (
                    <PointCloud points={warpedPoints} color="#ffff00" size={0.025} />
                )}
            </Canvas>
        </div>
    );
};
