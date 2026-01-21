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

        if (node.comfyClass === "SonicMixer") {
             const widget = {
                type: "SONIC_MIXER_UI",
                name: "mixer_visualizer",
                draw(y, step, ctx) { return y; }, 
                computeSize(width) { return [width, 500]; }
            };
            node.addCustomWidget(widget);

            // Container for Mixer
            const container = document.createElement("div");
            container.style.position = "absolute";
            container.style.background = "#111";
            container.style.border = "2px solid #333";
            container.style.borderRadius = "4px";
            container.style.boxShadow = "inset 0 0 20px #000";
            container.style.color = "#eee";
            container.style.fontFamily = "sans-serif";
            container.style.padding = "10px";
            container.style.boxSizing = "border-box";
            container.style.display = "grid";
            container.style.gridTemplateColumns = "1fr 1fr 1fr";
            container.style.gridTemplateRows = "auto auto 1fr";
            container.style.gap = "10px";
            container.style.zIndex = "10";
            document.body.appendChild(container);

            // Header/Branding
            const header = document.createElement("div");
            header.style.gridColumn = "1 / span 3";
            header.innerHTML = "<div style='color:#00f3ff; font-weight:bold; letter-spacing:2px;'>SONIC MIXER <span style='font-size:0.8em; color:#666'>PRO</span></div>";
            container.appendChild(header);

            // Left Panel: Style & Instruments
            const leftPanel = document.createElement("div");
            leftPanel.style.gridColumn = "1";
            leftPanel.style.background = "#222";
            leftPanel.style.padding = "10px";
            leftPanel.style.borderRadius = "4px";
            leftPanel.innerHTML = "<div style='font-size:10px; color:#888; margin-bottom:5px'>STYLE & INSTRUMENTS</div>";
            container.appendChild(leftPanel);

            // Style Select Interface
            const styleDisplay = document.createElement("div");
            styleDisplay.style.background = "#000";
            styleDisplay.style.color = "#0f0";
            styleDisplay.style.padding = "5px";
            styleDisplay.style.marginBottom = "5px";
            styleDisplay.style.fontFamily = "monospace";
            styleDisplay.innerText = "SELECT STYLE";
            leftPanel.appendChild(styleDisplay);
            
            // Instruments Input
            const instInput = document.createElement("textarea");
            instInput.style.width = "100%";
            instInput.style.height = "60px";
            instInput.style.background = "#333";
            instInput.style.border = "none";
            instInput.style.color = "#fff";
            instInput.style.fontSize = "11px";
            instInput.style.padding = "5px";
            instInput.placeholder = "Instruments...";
            leftPanel.appendChild(instInput);

            // Middle Panel: Lyrics & Vocoder
            const midPanel = document.createElement("div");
            midPanel.style.gridColumn = "2";
            midPanel.style.background = "#222";
            midPanel.style.padding = "10px";
            midPanel.style.borderRadius = "4px";
            midPanel.innerHTML = "<div style='font-size:10px; color:#888; margin-bottom:5px'>LYRICS & VOCODER</div>";
            container.appendChild(midPanel);

            const lyricsInput = document.createElement("textarea");
            lyricsInput.style.width = "100%";
            lyricsInput.style.height = "100px";
            lyricsInput.style.background = "#000";
            lyricsInput.style.border = "1px solid #444";
            lyricsInput.style.color = "#0ff";
            lyricsInput.style.fontSize = "10px";
            lyricsInput.style.fontFamily = "monospace";
            lyricsInput.placeholder = "Enter Lyrics...";
            midPanel.appendChild(lyricsInput);

            // Right Panel: Faders (BPM, Duration)
            const rightPanel = document.createElement("div");
            rightPanel.style.gridColumn = "3";
            rightPanel.style.background = "#222";
            rightPanel.style.padding = "10px";
            rightPanel.style.borderRadius = "4px";
            rightPanel.style.display = "flex";
            rightPanel.style.justifyContent = "space-around";
            container.appendChild(rightPanel);

            function createFader(label, color) {
                const wrap = document.createElement("div");
                wrap.style.textAlign = "center";
                wrap.innerHTML = `<div style='font-size:10px; color:${color}; margin-bottom:5px'>${label}</div>`;
                
                const track = document.createElement("div");
                track.style.width = "10px";
                track.style.height = "120px";
                track.style.background = "#111";
                track.style.margin = "0 auto";
                track.style.borderRadius = "5px";
                track.style.position = "relative";
                wrap.appendChild(track);

                const thumb = document.createElement("div");
                thumb.style.width = "20px";
                thumb.style.height = "10px";
                thumb.style.background = color;
                thumb.style.position = "absolute";
                thumb.style.left = "-5px";
                thumb.style.bottom = "50%";
                thumb.style.borderRadius = "2px";
                thumb.style.cursor = "pointer";
                track.appendChild(thumb);
                
                return { wrap, thumb, track };
            }

            const bpmFader = createFader("BPM", "#f00");
            const durFader = createFader("DUR", "#ff0");
            rightPanel.appendChild(bpmFader.wrap);
            rightPanel.appendChild(durFader.wrap);

            // MAKE Button
            const makeBtn = document.createElement("button");
            makeBtn.innerText = "MAKE TRACK";
            makeBtn.style.gridColumn = "1 / span 3";
            makeBtn.style.background = "linear-gradient(90deg, #00f3ff, #ff0055)";
            makeBtn.style.border = "none";
            makeBtn.style.color = "white";
            makeBtn.style.padding = "10px";
            makeBtn.style.fontSize = "14px";
            makeBtn.style.fontWeight = "bold";
            makeBtn.style.cursor = "pointer";
            makeBtn.style.letterSpacing = "2px";
            makeBtn.style.borderRadius = "4px";
            makeBtn.onclick = () => {
                app.queuePrompt(0);
                makeBtn.innerText = "QUEUED...";
                setTimeout(() => makeBtn.innerText = "MAKE TRACK", 2000);
            };
            container.appendChild(makeBtn);

            // Sync Logic
             const findWidget = (name) => node.widgets.find(w => w.name === name);

            setTimeout(() => {
                const w_bpm = findWidget("bpm");
                const w_dur = findWidget("duration");
                const w_lyrics = findWidget("lyrics");
                const w_style = findWidget("style");
                const w_inst = findWidget("instruments");

                // Sync Lyrics
                if(w_lyrics) {
                    lyricsInput.value = w_lyrics.value;
                    lyricsInput.addEventListener("input", () => w_lyrics.value = lyricsInput.value);
                }
                
                // Sync Instruments
                if(w_inst) {
                    instInput.value = w_inst.value;
                    instInput.addEventListener("input", () => w_inst.value = instInput.value);
                }

                // Sync Style (Simplified generic sync)
                if(w_style) {
                    styleDisplay.innerText = w_style.value;
                    styleDisplay.onclick = () => {
                        // Cycle styles simply for this demo
                        const styles = ["Cyberpunk", "Techno", "Lo-Fi", "Orchestral"];
                        let idx = styles.indexOf(w_style.value);
                        if(idx === -1) idx = 0;
                        else idx = (idx + 1) % styles.length;
                        w_style.value = styles[idx];
                        styleDisplay.innerText = styles[idx];
                    }
                }

                // Fader Logic
                const setupFader = (fader, widget, min, max) => {
                    if(!widget) return;
                    let isDragging = false;
                    
                    const updateUI = () => {
                        const pct = (widget.value - min) / (max - min);
                        fader.thumb.style.bottom = `${pct * 100}%`;
                    };
                    updateUI();

                    fader.thumb.addEventListener("mousedown", (e) => {
                        isDragging = true;
                        e.preventDefault(); 
                        document.addEventListener("mousemove", onMM);
                        document.addEventListener("mouseup", onMU);
                    });
                    
                    const onMM = (e) => {
                        if(!isDragging) return;
                        const rect = fader.track.getBoundingClientRect();
                        let y = rect.bottom - e.clientY;
                        y = Math.max(0, Math.min(y, rect.height));
                        const pct = y / rect.height;
                        
                        fader.thumb.style.bottom = `${pct * 100}%`;
                        widget.value = Math.floor(min + (pct * (max - min)));
                    };
                    
                    const onMU = () => {
                        isDragging = false;
                        document.removeEventListener("mousemove", onMM);
                        document.removeEventListener("mouseup", onMU);
                    };
                };

                setupFader(bpmFader, w_bpm, 60, 200);
                setupFader(durFader, w_dur, 5, 30);

            }, 200);

            // Position Logic (shared/similar to above)
            const updatePosition = () => {
                if (!node || !container) return;
                const visible = app.canvas.isNodeVisible(node);
                container.style.display = visible ? "grid" : "none";
                
                if (visible) {
                    const ds = app.canvas.ds;
                    const offset = app.canvas.convertPosToDOM([node.pos[0], node.pos[1]]);
                    const scale = ds.scale;
                    
                    const w = node.size[0] * scale;
                    const h = node.size[1] * scale; 

                    container.style.left = `${offset[0]}px`;
                    container.style.top = `${offset[1] + (30*scale)}px`; // Title offset
                    container.style.width = `${w}px`;
                    container.style.height = `${h - (30*scale)}px`;
                    container.style.fontSize = `${12 * scale}px`;
                }
            };

            const originalOnDraw = node.onDraw;
            node.onDraw = function(ctx) {
                if (originalOnDraw) originalOnDraw.apply(this, arguments);
                updatePosition();
            }
             const originalOnRemoved = node.onRemoved;
            node.onRemoved = function() {
                if (originalOnRemoved) originalOnRemoved.apply(this, arguments);
                if (container.parentNode) container.parentNode.removeChild(container);
            }
            node.size = [500, 350];
        }
    }
});
