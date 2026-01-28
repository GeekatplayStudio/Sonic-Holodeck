import { app } from "/scripts/app.js";

function createAudioWidget(node) {
    // Unique ID for this specific widget instance
    const widgetId = `sonic-audio-${node.id}`;

    // 1. COMPONENT: Create DOM Container
    // We use a persistent container attached to document.body to escape Canvas clipping
    const container = document.createElement("div");
    container.id = widgetId;
    Object.assign(container.style, {
        position: "absolute",
        display: "flex",
        zIndex: "99999", // Extremely High Z-Index
        backgroundColor: "#1a1a1a",
        border: "1px solid #333",
        borderRadius: "0 0 8px 8px", // Rounded bottom
        padding: "4px",
        boxSizing: "border-box",
        alignItems: "center",
        justifyContent: "center",
        pointerEvents: "auto", // Ensure clicks register
        boxShadow: "0 4px 6px rgba(0,0,0,0.3)"
    });

    // 2. COMPONENT: Audio Player
    const audioEl = document.createElement("audio");
    audioEl.controls = true;
    audioEl.preload = "auto";
    Object.assign(audioEl.style, {
        width: "100%",
        height: "32px",
        outline: "none",
        filter: "invert(0.9)" // Make it look dark-themed (standard HTML audio is white)
    });
    
    // Label for empty state
    const label = document.createElement("div");
    label.innerText = "Waiting for audio...";
    Object.assign(label.style, {
        position: "absolute",
        color: "#666",
        fontSize: "10px",
        fontFamily: "monospace",
        pointerEvents: "none",
        display: "none" // Hidden by default
    });

    container.appendChild(label);
    container.appendChild(audioEl);
    document.body.appendChild(container);

    // 3. LOGIC: Widget Integration
    const widget = {
        type: "AUDIO_PREVIEW",
        name: "sonic_player",
        
        // Reserve space in the node connection area
        draw(y, step, ctx) {
             return y + 50; // Reserve 50px
        },
        computeSize(width) {
            return [width, 50];
        },
        onRemove() {
            container.remove();
        }
    };
    node.addCustomWidget(widget);

    // 4. LOGIC: Position Sync
    // This function moves the HTML overlay to match the Canvas Node
    function updatePosition() {
        if (!node || !app.canvas || !app.canvas.ds) return;

        // Visibility Check
        if (node.flags.collapsed || !app.canvas.isNodeVisible(node)) {
            container.style.display = "none";
            return;
        }

        const ds = app.canvas.ds; // Draw State
        const scale = ds.scale;
        
        // Convert Node Logic Coordinates (Workflow space) to DOM Pixel Coordinates (Screen space)
        // node.pos is [X, Y]
        // app.canvas.convertPosToDOM handles Pan and Zoom transforms
        const pos = app.canvas.convertPosToDOM([node.pos[0], node.pos[1]]);
        
        const nodeWidth = node.size[0] * scale;
        const nodeHeight = node.size[1] * scale;
        
        // Dimensions
        const widgetHeight = 45 * scale;
        const fontSize = Math.max(10 * scale, 8); // Minimum font size

        container.style.display = "flex";
        
        // Position: Align to Bottom of Node
        // We calculate the reserved space at the bottom used by our 'draw' method
        // But simply pinning to the visual bottom of the node rect is robust.
        container.style.left = `${pos[0]}px`;
        container.style.top = `${pos[1] + nodeHeight}px`; // Just below the connections
        container.style.width = `${nodeWidth}px`;
        container.style.height = `${widgetHeight}px`;
        
        // Scale internals
        audioEl.style.height = `${30 * scale}px`;
        
        // Show/Hide label based on state
        if (!audioEl.src) {
            label.style.display = "block";
            label.style.fontSize = `${fontSize}px`;
            audioEl.style.opacity = "0.2"; // Dim player when empty
        } else {
            label.style.display = "none";
            audioEl.style.opacity = "1.0";
        }
    }

    // Hook into Node Draw loop for smooth 60fps tracking
    const onDrawOriginal = node.onDraw;
    node.onDraw = function(ctx) {
        if (onDrawOriginal) onDrawOriginal.apply(this, arguments);
        updatePosition();
    };
    
    // Resize handling
    const onResizeOriginal = node.onResize;
    node.onResize = function(size) {
        if (onResizeOriginal) onResizeOriginal.apply(this, arguments);
        updatePosition();
    }
    
    // Cleanup
    const onRemovedOriginal = node.onRemoved;
    node.onRemoved = function() {
        if (onRemovedOriginal) onRemovedOriginal.apply(this, arguments);
        container.remove();
    };

    // 5. LOGIC: Handle Execution Output
    const onExecutedOriginal = node.onExecuted;
    node.onExecuted = function(message) {
        if (onExecutedOriginal) onExecutedOriginal.apply(this, arguments);
        
        // Check for 'audio' output in the UI payload
        if (message && message.audio && message.audio.length > 0) {
            const item = message.audio[0];
            
            // Build URL
            const params = new URLSearchParams({
                filename: item.filename,
                subfolder: item.subfolder,
                type: item.type
            });
            
            // Add timestamp to prevent browser caching of old audio
            const src = `/view?${params.toString()}&t=${Date.now()}`;
            
            console.log(`[SonicHolodeck] Loading Audio: ${src}`);
            audioEl.src = src;
            
            // Auto-play
            audioEl.play().then(() => {
                console.log("[SonicHolodeck] Auto-playing...");
            }).catch(e => {
                console.warn("[SonicHolodeck] Auto-play blocked by policy or error:", e);
            });
        }
    };

    return widget;
}

// REGISTER EXTENSION
app.registerExtension({
    name: "Geekatplay.SonicHolodeck.UI",
    async nodeCreated(node, app) {
        if (node.comfyClass === "SonicSaver") {
            try {
                createAudioWidget(node);
                // Force a minimal size to accommodate the player
                if (node.size[0] < 200) node.setSize([200, node.size[1]]);
            } catch (e) {
                console.error("[SonicHolodeck] Failed to create audio widget:", e);
            }
        }
    }
});
