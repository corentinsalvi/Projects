// -------------------------------------------------------------------------- //
//          modelisation-3D.py - Importation des modeles 3D et gestion de     //
//                               la fenêtre d'affichage                       //
// -------------------------------------------------------------------------- //
// Auteur : Corentin SALVI                                                    //
// Date : 2024-10-13                                                          //
// -------------------------------------------------------------------------- //

// -------------------------------------------
// Importation des bilbiothèques
// -------------------------------------------
//#region
import * as THREE from 'https://cdn.skypack.dev/three@0.128.0/build/three.module.js';
import { GLTFLoader } from 'https://cdn.skypack.dev/three@0.128.0/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'https://cdn.skypack.dev/three@0.128.0/examples/jsm/controls/OrbitControls.js';
//#endregion

// -------------------------------------------
// Creation de la scène 3D
// -------------------------------------------
//#region
// Canvas 
const canvas = document.querySelector('.webgl');

// Scene
const scene = new THREE.Scene();

// Lumières
const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
directionalLight.position.set(5, 10, 7.5);
scene.add(directionalLight);

const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);

// Camera
const sizes = { width: window.innerWidth, height: window.innerHeight };
const camera = new THREE.PerspectiveCamera(75, sizes.width / sizes.height, 0.5, 50);
scene.add(camera);

// Renderer
const renderer = new THREE.WebGLRenderer({ canvas });
renderer.setSize(sizes.width, sizes.height);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
// Background
const ctx=document.createElement('canvas').getContext('2d');
ctx.canvas.width=ctx.canvas.height=512;
const gradient= ctx.createRadialGradient(256, 256, 0, 256, 256, 256);
gradient.addColorStop(0, '#ffffff');
gradient.addColorStop(1, '#0a0a0a');
ctx.fillStyle=gradient;
ctx.fillRect(0,0,512,512);
const texture= new THREE.CanvasTexture(ctx.canvas);
scene.background=texture;
// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.update();
//#endregion

// -------------------------------------------
// Gestion et configuration des modèles 3D
// -------------------------------------------
//#region

// Loader
const loader = new GLTFLoader();

// Centre le modèle sur son pivot et applique rotation & scale
function centerAndPositionModel(model, obj) {
    const box = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    model.position.set(
        obj.position[0] - center.x,
        obj.position[1] - center.y,
        obj.position[2] - center.z
    );
    if (obj.rotation) {
        model.rotation.set(
            obj.rotation[0] || 0,
            obj.rotation[1] || 0,
            obj.rotation[2] || 0
        );
    }
    model.scale.set(obj.scale, obj.scale, obj.scale);
}

// Configure les meshes : opacité, ombres, matériaux
function configureMeshes(model) {
    model.traverse((child) => {
        if (child.isMesh) {
            child.material.transparent = false;
            child.material.opacity = 1;
            child.material.side = THREE.DoubleSide;
            child.material.metalness = 0;
            child.material.roughness = 0.7;
            child.castShadow = true;
            child.receiveShadow = true;
        }
    });
}

// Charge un modèle et l’ajoute à la scène
function loadModel(obj, onLoad) {
    loader.load(
        obj.file,
        (glb) => {
            const model = glb.scene;
            model.userData = obj.userData || {};
            centerAndPositionModel(model, obj);
            configureMeshes(model);
            scene.add(model);
            onLoad(model);
        },
        undefined,
        (error) => console.error('Erreur de chargement :', error)
    );
}
//#endregion

// -------------------------------------------
// Importation des modèles 3D
// -------------------------------------------
//#region
const models = [
    // Pièces blanches
    { file: 'static/models/Pions/pion1.glb', position: [8.8, 8.7, 6.3], scale: 1, userData: { isPiece: true } },
    { file: 'static/models/Pions/pion2.glb', position: [6.3, 8.7, 6.3], scale: 1 , userData: { isPiece: true } },
    { file: 'static/models/Pions/pion3.glb', position: [3.8, 8.7, 6.3], scale: 1 , userData: { isPiece: true } },
    { file: 'static/models/Pions/pion4.glb', position: [1.3, 8.7, 6.3], scale: 1 , userData: { isPiece: true } },     
    { file: 'static/models/Pions/pion5.glb', position: [-1.3, 8.7, 6.3], scale: 1 , userData: { isPiece: true } },
    { file: 'static/models/Pions/pion6.glb', position: [-3.8, 8.7, 6.3], scale: 1 , userData: { isPiece: true } },
    { file: 'static/models/Pions/pion7.glb', position: [-6.3, 8.7, 6.3], scale: 1, userData: { isPiece: true } }, 
    { file: 'static/models/Pions/pion8.glb', position: [-8.8, 8.7, 6.3], scale: 1 , userData: { isPiece: true } },    
    { file: 'static/models/Tours/tour2.glb', position: [8.8, 9.5, 8.8], scale: 1,  userData: { isPiece: true } },
    { file: 'static/models/Fous/fou3.glb', position: [-3.8, 9.5, 8.8], scale: 1, userData: { isPiece: true } },
    { file: 'static/models/Fous/fou4.glb', position: [3.8, 9.5, 8.8], scale: 1,    userData: { isPiece: true } },
    { file: 'static/models/Tours/tour3.glb', position: [-8.8, 9.5, 8.8], scale: 1, userData: { isPiece: true } },
    { file: 'static/models/Cavaliers/cavalier1.glb', position: [6.2, 9.2, 8.8], scale: 1, userData: { isPiece: true } },
    { file: 'static/models/Cavaliers/cavalier2.glb', position: [-6.2, 9.2, 8.8], scale: 1,  userData: { isPiece: true } },
    { file: 'static/models/Rois/roi2.glb', position: [-1.3, 9.8, 8.8], scale: 1.2, userData: { isPiece: true } },
    { file: 'static/models/Reines/reine2.glb', position: [1.3, 9.8, 8.8], scale: 1.2, rotation: [0, 30, 0], userData: { isPiece: true } },
    //Pièces noires
    { file: 'static/models/Pions/pion9.glb', position: [8.8, 8.7, -6.3], scale: 1, userData: { isPiece: true } },
    { file: 'static/models/Pions/pion10.glb', position: [6.3, 8.7, -6.3], scale: 1, userData: { isPiece: true } },
    { file: 'static/models/Pions/pion11.glb', position: [3.8, 8.7, -6.3], scale: 1, userData: { isPiece: true } },
    { file: 'static/models/Pions/pion12.glb', position: [1.3, 8.7, -6.3], scale: 1, userData: { isPiece: true } },     
    { file: 'static/models/Pions/pion13.glb', position: [-1.3, 8.7,-6.3], scale: 1, userData: { isPiece: true } },
    { file: 'static/models/Pions/pion14.glb', position: [-3.8, 8.7,-6.3], scale: 1, userData: { isPiece: true } },
    { file: 'static/models/Pions/pion15.glb', position: [-6.3, 8.7, -6.3], scale: 1, userData: { isPiece: true }  }, 
    { file: 'static/models/Pions/pion16.glb', position: [-8.8, 8.7, -6.3], scale: 1 , userData: { isPiece: true } },
    { file: 'static/models/Tours/tour1.glb', position: [8.8, 9.5, -8.8], scale: 1, userData: { isPiece: true } },
    { file: 'static/models/Tours/tour4.glb', position: [-8.8, 9.5, -8.8], scale: 1, userData: { isPiece: true } },
    { file: 'static/models/Fous/fou1.glb', position: [-3.8, 9.5, -8.8], scale: 1, userData: { isPiece: true } },
    { file: 'static/models/Fous/fou2.glb', position: [3.8, 9.5, -8.8], scale: 1, userData: { isPiece: true } },
    { file: 'static/models/Cavaliers/cavalier4.glb', position: [6.3, 9.2, -8.8], scale: 1, rotation: [0, 179, 0], userData: { isPiece: true } },
    { file: 'static/models/Cavaliers/cavalier3.glb', position: [-6.3, 9.2, -8.8], scale: 1, rotation: [0, 179, 0], userData: { isPiece: true } },
    { file: 'static/models/Rois/roi1.glb', position: [-1.3, 9.8, -8.8], scale: 1.2, userData: { isPiece: true } },
    { file: 'static/models/Reines/reine1.glb', position: [1.3, 9.8, -8.8], scale: 1.2, rotation: [0, 30, 0], userData: { isPiece: true } },
    //Echiquier
    { file: 'static/models/echiquier-principal.glb', position: [0,0.45,-0.2], scale: 1,userData: { isBoard: true } },
];
//#endregion

// Chargement de tous les modèles sur la scène
let loadedCount = 0;
models.forEach(obj => {
    loadModel(obj, (model) => {
        loadedCount++;
        // Cadre la caméra sur l'échiquier une fois chargé
        if (obj.file.toLowerCase().includes('echiquier-principal')) {
            const box = new THREE.Box3().setFromObject(model);
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const fov = camera.fov * (Math.PI / 180);
            const cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2)) * 1.5;
            camera.position.set(cameraZ, cameraZ, cameraZ);
            camera.lookAt(0, 0, 0);
            controls.update();
        }
        // Stocke la pièce chargée
        if (obj.userData?.isPiece) {
        window.Chess3D.pieces[`piece_${loadedCount}`] = model;
    }
    });
});

// -------------------------------------------
// Animation de la scène
// -------------------------------------------
//#region
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();
//#endregion

// -------------------------------------------
// Redimensionnement de la fenêtre 
// -------------------------------------------
//#region
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});
//#endregion

//-------------------------------------------
// Rendre les objets de la scène accessibles globalement
//-------------------------------------------
//#region
window.Chess3D = {
    scene: scene,
    camera: camera,
    renderer: renderer,
    controls: controls,
    pieces: {}
};
//#endregion

// -------------------------------------------------------------------------- //