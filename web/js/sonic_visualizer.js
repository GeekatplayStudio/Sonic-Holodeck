import { app } from "../../scripts/app.js";

// Load Three.js from CDN
import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";

app.registerExtension({
    name: "Geekatplay.SonicHolodeck",
    async nodeCreated(node, app) {
        if (node.comfyClass === "SonicHoloSynth") {
            const widget = {
                type: "SONIC_VISUALIZER",
                name: "holo_visualizer",
                draw(y, step, ctx) { return y; }, 
                computeSize(width) {
                    return [width, 400]; // 600x400 requested, but width is flexible
                }
            };
            
            node.addCustomWidget(widget);

            // Create Container
            const container = document.createElement("div");
            container.style.position = "absolute";
            container.style.width = "100%";
            container.style.height = "400px";
            container.style.top = "20px"; // approximate offset
            container.style.left = "0";
            container.style.overflow = "hidden";
            container.style.zIndex = "10";
            // We append to body and position manually because Comfy doesn't expose inner DOM easily
            document.body.appendChild(container);

            // Create Canvas
            const canvas = document.createElement("canvas");
            canvas.style.width = "100%";
            canvas.style.height = "100%";
            container.appendChild(canvas);

            // Create UI Overlay (Knobs)
            const overlay = document.createElement("div");
            overlay.style.position = "absolute";
            overlay.style.top = "0";
            overlay.style.left = "0";
            overlay.style.width = "100%";
            overlay.style.height = "100%";
            overlay.style.pointerEvents = "none"; // Let clicks pass through for some parts
            container.appendChild(overlay);

            // Geekatplay Branding
            const branding = document.createElement("div");
            branding.innerText = "Geekatplay Studio";
            branding.style.position = "absolute";
            branding.style.bottom = "5px";
            branding.style.right = "10px";
            branding.style.fontFamily = "monospace";
            branding.style.fontSize = "10px";
            branding.style.color = "rgba(0, 243, 255, 0.5)";
            branding.style.pointerEvents = "none";
            overlay.appendChild(branding);

            // Helper to create knob
            function createKnob(label, initialValue, min, max, onChange, x, y) {
                const knobContainer = document.createElement("div");
                knobContainer.style.position = "absolute";
                knobContainer.style.left = x;
                knobContainer.style.top = y;
                knobContainer.style.width = "60px";
                knobContainer.style.height = "80px";
                knobContainer.style.pointerEvents = "auto";
                knobContainer.style.textAlign = "center";
                knobContainer.style.fontFamily = "monospace";
                knobContainer.style.color = "#00f3ff";
                knobContainer.style.fontSize = "10px";

                const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
                svg.setAttribute("width", "60");
                svg.setAttribute("height", "60");
                
                const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                circle.setAttribute("cx", "30");
                circle.setAttribute("cy", "30");
                circle.setAttribute("r", "25");
                circle.setAttribute("stroke", "#00f3ff");
                circle.setAttribute("stroke-width", "2");
                circle.setAttribute("fill", "rgba(0, 243, 255, 0.1)");
                
                const indicator = document.createElementNS("http://www.w3.org/2000/svg", "line");
                indicator.setAttribute("x1", "30");
                indicator.setAttribute("y1", "30");
                indicator.setAttribute("x2", "30");
                indicator.setAttribute("y2", "5");
                indicator.setAttribute("stroke", "#ff0055");
                indicator.setAttribute("stroke-width", "2");
                
                svg.appendChild(circle);
                svg.appendChild(indicator);
                knobContainer.appendChild(svg);
                
                const labelDiv = document.createElement("div");
                labelDiv.innerText = label;
                knobContainer.appendChild(labelDiv);
                
                overlay.appendChild(knobContainer);

                let isDragging = false;
                let startY = 0;
                let value = initialValue;

                const updateIndicator = () => {
                    const norm = (value - min) / (max - min);
                    const angle = (norm * 270) - 135; // -135 to 135
                    indicator.setAttribute("transform", `rotate(${angle}, 30, 30)`);
                };
                updateIndicator();

                knobContainer.addEventListener("mousedown", (e) => {
                    isDragging = true;
                    startY = e.clientY;
                    document.addEventListener("mousemove", onMouseMove);
                    document.addEventListener("mouseup", onMouseUp);
                });

                const onMouseMove = (e) => {
                    if (!isDragging) return;
                    e.preventDefault();
                    const delta = startY - e.clientY;
                    const range = max - min;
                    value += delta * (range / 200);
                    value = Math.min(Math.max(value, min), max);
                    updateIndicator();
                    startY = e.clientY;
                    onChange(value);
                };

                const onMouseUp = () => {
                    isDragging = false;
                    document.removeEventListener("mousemove", onMouseMove);
                    document.removeEventListener("mouseup", onMouseUp);
                };
            }

            // Sync knobs to widgets
            // Need to find widget indexes. Usually:
            // 0: prompt, 1: bpm, 2: duration, 3: cfg, 4: temperature, 5: seed
            // This order depends on nodes.py INPUT_TYPES order.
            // prompt, bpm, duration, cfg, temperature, seed
            
            const findWidget = (name) => node.widgets.find(w => w.name === name);

            // Wait a tick for widgets to init
            setTimeout(() => {
                const bpmWidget = findWidget("bpm");
                const durationWidget = findWidget("duration");
                const cfgWidget = findWidget("cfg");
                const tempWidget = findWidget("temperature");

                createKnob("BPM", bpmWidget?.value || 120, 60, 200, (v) => { if(bpmWidget) bpmWidget.value = Math.round(v); }, "10px", "10px");
                createKnob("DUR", durationWidget?.value || 10, 5, 30, (v) => { if(durationWidget) durationWidget.value = Math.round(v); }, "80px", "10px");
                createKnob("CFG", cfgWidget?.value || 3.0, 0.1, 10.0, (v) => { if(cfgWidget) cfgWidget.value = v; }, "10px", "100px");
                createKnob("TEMP", tempWidget?.value || 1.0, 0.1, 2.0, (v) => { if(tempWidget) tempWidget.value = v; }, "80px", "100px");
            }, 100);

            // Three.js setup
            let scene, camera, renderer, reactor, grid;
            let analyser, dataArray;
            let audioContext;

            function initThree() {
                scene = new THREE.Scene();
                camera = new THREE.PerspectiveCamera(75, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);
                camera.position.set(0, 5, 10);
                camera.lookAt(0, 0, 0);

                renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
                renderer.setSize(canvas.clientWidth, canvas.clientHeight); // Initial size

                // The Reactor (Torus)
                const geometry = new THREE.TorusGeometry(3, 0.2, 16, 100);
                const material = new THREE.MeshBasicMaterial({ color: 0xff0055, wireframe: true });
                reactor = new THREE.Mesh(geometry, material);
                scene.add(reactor);

                // The Waveform Grid
                // Creating a grid of points or lines
                const gridSize = 20;
                const gridDivisions = 20;
                grid = new THREE.Group();
                
                // Helper grid
                const gridHelper = new THREE.GridHelper(gridSize, gridDivisions, 0xff00ff, 0xff00ff);
                grid.add(gridHelper);
                scene.add(grid);

                // Setup Audio
                try {
                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    analyser = audioContext.createAnalyser();
                    analyser.fftSize = 256;
                    dataArray = new Uint8Array(analyser.frequencyBinCount);
                } catch(e) {
                    console.error("AudioContext error", e);
                }

                // Reactor interaction (Seed)
                let isDraggingReactor = false;
                let lastX = 0;
                
                canvas.addEventListener("mousedown", (e) => {
                     // specific hit testing is complex, assume center area
                     isDraggingReactor = true; // simplified
                     lastX = e.clientX;
                });
                
                 window.addEventListener("mouseup", () => isDraggingReactor = false);
                 
                 window.addEventListener("mousemove", (e) => {
                     if(isDraggingReactor) {
                         const delta = e.clientX - lastX;
                         reactor.rotation.y += delta * 0.01;
                         lastX = e.clientX;
                         // Update seed widget technically here
                         const seedWidget = findWidget("seed");
                         if(seedWidget) seedWidget.value = Math.floor(Math.random() * 1000000);
                     }
                 });
            }

            initThree();

            function animate() {
                requestAnimationFrame(animate);

                if (analyser) {
                    analyser.getByteFrequencyData(dataArray);
                    
                    // Bass is low logical indices
                    const bass = dataArray[2]; 
                    const scale = 1 + (bass / 256);
                    
                    reactor.scale.set(scale, scale, scale);
                    reactor.rotation.x += 0.01;
                    
                    // Simple grid bounce
                    grid.position.y = (bass / 256) * 2 - 2; 
                } else {
                     reactor.rotation.x += 0.01;
                }

                renderer.render(scene, camera);
            }
            animate();
            
            // Positioning Logic
             const updatePosition = () => {
                if (!node || !container) return;
                const visible = app.canvas.isNodeVisible(node);
                container.style.display = visible ? "block" : "none";
                
                if (visible) {
                    const ds = app.canvas.ds; 
                    const offset = app.canvas.convertPosToDOM([node.pos[0], node.pos[1]]);
                    const scale = ds.scale;
                    
                    const w = (node.size[0]) * scale;
                    const h = (node.size[1]) * scale; 

                    container.style.left = `${offset[0]}px`;
                    container.style.top = `${offset[1] + (30*scale)}px`; // Below title
                    container.style.width = `${w}px`;
                    container.style.height = `${h - (30*scale)}px`;
                    
                    // Resize buffers if needed
                    if (canvas.width !== container.clientWidth || canvas.height !== container.clientHeight) {
                        renderer.setSize(container.clientWidth, container.clientHeight);
                        camera.aspect = container.clientWidth / container.clientHeight;
                        camera.updateProjectionMatrix();
                    }
                }
            };
            
            // Hook drawing
            const originalOnDraw = node.onDraw;
            node.onDraw = function(ctx) {
                if (originalOnDraw) originalOnDraw.apply(this, arguments);
                updatePosition();
            }

            // Cleanup
             const originalOnRemoved = node.onRemoved;
            node.onRemoved = function() {
                if (originalOnRemoved) originalOnRemoved.apply(this, arguments);
                if (container.parentNode) container.parentNode.removeChild(container);
            }
            
            // Set Size
            node.size = [600, 440];
        }
    }
});
