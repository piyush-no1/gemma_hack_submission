document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropZone = document.getElementById("drop-zone");
    const videoFileInput = document.getElementById("video-file");
    const videoPreviewContainer = document.getElementById("video-preview-container");
    const videoPlayer = document.getElementById("video-player");
    const videoMetaName = document.getElementById("video-meta-name");
    const videoMetaDuration = document.getElementById("video-meta-duration");
    const languageSelect = document.getElementById("language-select");
    const processBtn = document.getElementById("process-btn");
    const stopBtn = document.getElementById("stop-btn");

    // Stream elements
    const streamIdle = document.getElementById("stream-idle");
    const streamMessages = document.getElementById("stream-messages");
    const streamStatusBadge = document.getElementById("stream-status-badge");
    const streamStatusText = document.getElementById("stream-status-text");

    // Progress bar
    const progressBarContainer = document.getElementById("progress-bar-container");
    const progressStatusText = document.getElementById("progress-status-text");
    const progressPercent = document.getElementById("progress-percent");
    const progressFill = document.getElementById("progress-fill");

    // Result cards
    const broadcastCard = document.getElementById("broadcast-card");
    const reportSummary = document.getElementById("report-summary");
    const reportExplanation = document.getElementById("report-explanation");
    const reportRecommendations = document.getElementById("report-recommendations");
    const severityBadge = document.getElementById("severity-badge");

    // Telemetry & Rollouts (hidden until done)
    const telemetryCard = document.getElementById("telemetry-card");
    const overlayGalleryCard = document.getElementById("overlay-gallery-card");
    const telYolo = document.getElementById("tel-yolo");
    const telCrowd = document.getElementById("tel-crowd");
    const telMotion = document.getElementById("tel-motion");
    const telOcr = document.getElementById("tel-ocr");
    const overlaysPlaceholder = document.getElementById("overlays-placeholder");
    const overlaysGrid = document.getElementById("overlays-grid");

    // State
    let selectedFile = null;
    let ws = null;
    let activeJobId = null;
    let isProcessing = false;

    // Buffered telemetry during processing — applied when done
    let pendingTelemetry = {};
    let pendingRollouts = [];

    // -----------------------------------------
    // Node → human-readable stream messages
    // -----------------------------------------
    const NODE_MESSAGES = {
        "run_vjepa":                    { icon: "fa-wave-square",         text: "V-JEPA scanning video for temporal anomalies using self-supervised prediction error..." },
        "extract_and_upscale_frames":   { icon: "fa-image",               text: "Extracting keyframes with scene detection and upscaling to 4× resolution..." },
        "abrupt_motion_detection":      { icon: "fa-bolt",                text: "Analyzing frame differences for abrupt motion events via optical flow contours..." },
        "yolo_world":                   { icon: "fa-crosshairs",          text: "YOLO-World detecting public safety objects across all extracted frames..." },
        "gemma_transcription":          { icon: "fa-comment-dots",        text: "Gemma Vision generating natural-language descriptions for each scene frame..." },
        "orchestrator_judge":           { icon: "fa-scale-balanced",      text: "Orchestrator Judge (Gemma 4B) fusing all evidence and forming a decision..." },
        "tool_caller":                  { icon: "fa-toolbox",             text: "Dispatching auxiliary tools (RAG · Dynamic YOLO · OCR · Crowd Analytics)..." },
        "reasoning_agent":              { icon: "fa-brain",               text: "Deep Reasoning Agent (Gemma 26B) performing forensic analysis of the incident..." },
        "gemma_output_module":          { icon: "fa-file-shield",         text: "Gemma 2B writing the final emergency report from the orchestrator verdict..." },
        "reasoning_to_speech":          { icon: "fa-circle-check",        text: "Pipeline complete. Compiling results..." },
    };

    const NODE_PROGRESS = {
        "extract_and_upscale_frames": 15,
        "run_vjepa":                  35,
        "abrupt_motion_detection":    40,
        "yolo_world":                 50,
        "gemma_transcription":        62,
        "orchestrator_judge":         78,
        "tool_caller":                84,
        "reasoning_agent":            91,
        "gemma_output_module":        97,
        "reasoning_to_speech":        100,
    };

    // -----------------------------------------
    // Drag and Drop
    // -----------------------------------------
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault(); e.stopPropagation();
            dropZone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault(); e.stopPropagation();
            dropZone.classList.remove("dragover");
        }, false);
    });

    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files.length > 0) handleVideoSelected(dt.files[0]);
    });

    videoFileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) handleVideoSelected(e.target.files[0]);
    });

    dropZone.addEventListener("click", () => videoFileInput.click());

    function handleVideoSelected(file) {
        selectedFile = file;
        videoMetaName.innerHTML = `<i class="fa-solid fa-file"></i> ${file.name}`;
        const fileUrl = URL.createObjectURL(file);
        videoPlayer.src = fileUrl;
        videoPlayer.onloadedmetadata = () => {
            videoMetaDuration.innerHTML = `<i class="fa-solid fa-clock"></i> ${videoPlayer.duration.toFixed(1)}s`;
            processBtn.disabled = false;
        };
        videoPreviewContainer.classList.remove("hidden");
        pushStreamMessage("system", "fa-upload", `Video selected: <strong>${file.name}</strong>. Ready to analyze.`);
    }

    // -----------------------------------------
    // Stream message helpers
    // -----------------------------------------
    function showStreamPanel() {
        streamIdle.classList.add("hidden");
        streamMessages.classList.remove("hidden");
        streamStatusBadge.classList.remove("hidden");
        progressBarContainer.classList.remove("hidden");
    }

    function pushStreamMessage(type, icon, html, typewrite = false) {
        const msg = document.createElement("div");
        msg.className = `stream-msg stream-msg--${type}`;
        const ts = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

        if (typewrite) {
            // Render icon + timestamp, then typewrite the text
            const plain = html.replace(/<[^>]+>/g, ""); // strip tags for typewriting
            msg.innerHTML = `
                <span class="sm-icon"><i class="fa-solid ${icon}"></i></span>
                <span class="sm-body">
                    <span class="sm-time">${ts}</span>
                    <span class="sm-text"></span>
                    <span class="sm-cursor">▋</span>
                </span>`;
            streamMessages.appendChild(msg);
            scrollStream();
            typewriteInto(msg.querySelector(".sm-text"), msg.querySelector(".sm-cursor"), plain);
        } else {
            msg.innerHTML = `
                <span class="sm-icon"><i class="fa-solid ${icon}"></i></span>
                <span class="sm-body">
                    <span class="sm-time">${ts}</span>
                    <span class="sm-text">${html}</span>
                </span>`;
            streamMessages.appendChild(msg);
            scrollStream();
        }
        return msg;
    }

    function scrollStream() {
        streamMessages.scrollTop = streamMessages.scrollHeight;
    }

    function typewriteInto(el, cursor, text, speed = 18) {
        let i = 0;
        const iv = setInterval(() => {
            el.textContent += text[i];
            i++;
            scrollStream();
            if (i >= text.length) {
                clearInterval(iv);
                if (cursor) cursor.remove();
            }
        }, speed);
    }

    function setStreamStatus(text, active = true) {
        streamStatusText.textContent = text;
        streamStatusBadge.classList.toggle("active", active);
        streamStatusBadge.classList.toggle("done", !active);
    }

    // -----------------------------------------
    // Process button
    // -----------------------------------------
    processBtn.addEventListener("click", async () => {
        if (!selectedFile || isProcessing) return;
        isProcessing = true;

        // Reset state
        broadcastCard.classList.add("hidden");
        telemetryCard.classList.add("hidden");
        overlayGalleryCard.classList.add("hidden");
        streamMessages.innerHTML = "";
        pendingTelemetry = {};
        pendingRollouts = [];
        resetTelemetry();

        showStreamPanel();
        setStreamStatus("Uploading video...", true);
        updateProgressBar("Uploading video...", 2);

        processBtn.classList.add("hidden");
        stopBtn.classList.remove("hidden");
        stopBtn.disabled = false;
        stopBtn.innerHTML = '<i class="fa-solid fa-circle-stop"></i> Stop Scan';
        
        pushStreamMessage("system", "fa-cloud-arrow-up", "Uploading video to the surveillance API...");

        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("output_language", languageSelect.value);

        try {
            const res = await fetch("/api/jobs", { method: "POST", body: formData });
            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            activeJobId = data.job_id;
            pushStreamMessage("system", "fa-circle-play", `Analysis job started. ID: <code>${activeJobId}</code>`);
            updateProgressBar("Pipeline initializing...", 5);
            setStreamStatus("Pipeline running...", true);
            connectWebSocket(activeJobId);
        } catch (e) {
            pushStreamMessage("error", "fa-triangle-exclamation", `Failed to start scan: ${e.message}`);
            setStreamStatus("Upload failed", false);
            endProcessing();
        }
    });

    function endProcessing() {
        isProcessing = false;
        processBtn.classList.remove("hidden");
        stopBtn.classList.add("hidden");
        processBtn.disabled = false;
    }

    // -----------------------------------------
    // WebSocket
    // -----------------------------------------
    function connectWebSocket(jobId) {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/api/ws/jobs/${jobId}`;
        ws = new WebSocket(wsUrl);

        let pingInterval = null;

        ws.onopen = () => {
            // Start heartbeats to prevent proxy/browser idle timeouts
            pingInterval = setInterval(() => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send("ping");
                }
            }, 30000);
        };

        ws.onmessage = (event) => {
            if (event.data === "pong") return;
            const data = JSON.parse(event.data);
            handlePipelineUpdate(data);
        };

        ws.onclose = () => {
            if (pingInterval) clearInterval(pingInterval);
            if (isProcessing) {
                pushStreamMessage("system", "fa-plug-circle-xmark", "Socket connection closed.");
            }
        };

        ws.onerror = () => {
            if (pingInterval) clearInterval(pingInterval);
            pushStreamMessage("error", "fa-triangle-exclamation", "WebSocket communication error.");
        };
    }

    function updateProgressBar(statusText, percent) {
        progressStatusText.textContent = statusText;
        progressPercent.textContent = `${percent}%`;
        progressFill.style.width = `${percent}%`;
    }

    // -----------------------------------------
    // Pipeline update handler
    // -----------------------------------------
    function handlePipelineUpdate(data) {
        const node = data.node;
        const status = data.status;

        if (status === "running") {
            // Entering a node — show a quick "starting" note (not typewritten)
            const info = NODE_MESSAGES[node];
            if (info) {
                pushStreamMessage("node-start", "fa-arrow-right", `<em>Starting:</em> ${node.replace(/_/g, " ")}`);
            }
        }

        if (status === "completed" && node !== "END") {
            const info = NODE_MESSAGES[node];
            if (info) {
                // Typewrite the main action description
                pushStreamMessage("node", info.icon, info.text, true);
            }

            // Progress bar
            const pct = NODE_PROGRESS[node];
            if (pct !== undefined) {
                const label = info ? info.text.split("...")[0] : node;
                updateProgressBar(label, pct);
            }

            // Buffer telemetry data — will be shown after completion
            if (data.update) {
                bufferTelemetry(node, data.update);
            }

            // Update live status badge with current node
            if (info) {
                const shortLabel = node.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
                setStreamStatus(shortLabel, true);
            }
        }

        if (status === "failed") {
            if (data.error && (data.error.includes("stopped by user request") || data.error.includes("cancelled"))) {
                pushStreamMessage("system", "fa-circle-stop", "Pipeline execution stopped successfully.");
                setStreamStatus("Stopped ✓", false);
            } else {
                pushStreamMessage("error", "fa-skull", `Pipeline error in node <strong>'${node}'</strong>: ${data.error}`);
                setStreamStatus("Pipeline failed", false);
            }
            endProcessing();
        }

        // Fully done
        if (node === "END" && status === "completed") {
            updateProgressBar("Complete!", 100);
            setStreamStatus("Analysis complete ✓", false);
            pushStreamMessage("done", "fa-circle-check", "<strong>All pipeline stages completed.</strong> Loading final report...");
            fetchJobResults(activeJobId);
        }
    }

    // -----------------------------------------
    // Buffer telemetry during processing
    // -----------------------------------------
    function bufferTelemetry(node, update) {
        if (node === "yolo_world" && update.yolo_detections) {
            pendingTelemetry.yolo = update.yolo_detections;
        }
        if (node === "abrupt_motion_detection" && update.motion_evidence) {
            pendingTelemetry.motion = update.motion_evidence;
        }
        if (node === "tool_caller" && update.tool_results) {
            const tr = update.tool_results;
            if (tr.ocr && tr.ocr.text) pendingTelemetry.ocr = tr.ocr.text;
            if (tr.crowd_analytics) pendingTelemetry.crowd = tr.crowd_analytics;
            if (tr.attention_rollout && tr.attention_rollout.length) {
                pendingRollouts.push(...tr.attention_rollout);
            }
        }
        if (node === "run_vjepa" && update.vjepa_result && update.vjepa_result.anomalies) {
            const rolls = update.vjepa_result.anomalies
                .filter(a => a.attention_overlay_path)
                .map(a => ({ timestamp_sec: a.timestamp_sec, overlay_path: a.attention_overlay_path }));
            pendingRollouts.push(...rolls);
        }
    }

    function safeToFixed(value, decimals = 2, fallback = "—") {
        if (value === undefined || value === null || isNaN(value)) {
            return fallback;
        }
        const parsed = parseFloat(value);
        if (isNaN(parsed)) {
            return fallback;
        }
        return parsed.toFixed(decimals);
    }

    // -----------------------------------------
    // Reveal all buffered data after completion
    // -----------------------------------------
    function revealPostProcessingData() {
        // Telemetry
        if (pendingTelemetry.yolo) {
            const labels = pendingTelemetry.yolo.map(d => `${d.label} (${safeToFixed(d.confidence, 2)})`);
            telYolo.innerHTML = labels.length ? labels.join("<br>") : "None";
        }
        if (pendingTelemetry.motion) {
            const lines = pendingTelemetry.motion.map(m =>
                `Time: ${safeToFixed(m.timestamp_sec, 1)}s | Score: ${safeToFixed(m.motion_score, 2)} | Z: ${safeToFixed(m.z_score, 2)} ${m.is_abrupt ? "🔥" : ""}`
            );
            telMotion.innerHTML = lines.length ? lines.join("<br>") : "None";
        }
        if (pendingTelemetry.ocr) {
            telOcr.innerHTML = pendingTelemetry.ocr.join("<br>") || "None";
        }
        if (pendingTelemetry.crowd) {
            const ca = pendingTelemetry.crowd;
            telCrowd.innerHTML = `State: <strong>${ca.state ? ca.state.toUpperCase() : "—"}</strong><br>Flow: ${ca.flow_direction || "—"}<br>Density: ${ca.density || "—"}<br>Count: ${safeToFixed(ca.people_count, 0)}`;
        }

        // Rollouts
        if (pendingRollouts.length > 0) {
            overlaysPlaceholder.classList.add("hidden");
            overlaysGrid.classList.remove("hidden");
            pendingRollouts.forEach(r => {
                const cardId = `rollout-card-${safeToFixed(r.timestamp_sec, 2)}`;
                if (document.getElementById(cardId)) return;
                const card = document.createElement("div");
                card.className = "overlay-card";
                card.id = cardId;
                const parts = r.overlay_path.split(/[\\\/]/);
                const filename = parts[parts.length - 1];
                const srcUrl = `/api/jobs/${activeJobId}/artifacts/${filename}`;
                card.innerHTML = `
                    <img src="${srcUrl}" class="overlay-img" alt="Attention Rollout at ${safeToFixed(r.timestamp_sec, 2)}s">
                    <div class="overlay-label">Rollout @ ${safeToFixed(r.timestamp_sec, 2)}s</div>
                `;
                overlaysGrid.appendChild(card);
            });
        }

        // Slide in cards with staggered animation
        setTimeout(() => { telemetryCard.classList.remove("hidden"); telemetryCard.classList.add("reveal"); }, 200);
        setTimeout(() => { overlayGalleryCard.classList.remove("hidden"); overlayGalleryCard.classList.add("reveal"); }, 450);
    }

    function resetTelemetry() {
        telYolo.textContent = "-";
        telCrowd.textContent = "-";
        telMotion.textContent = "-";
        telOcr.textContent = "-";
        overlaysPlaceholder.classList.remove("hidden");
        overlaysGrid.classList.add("hidden");
        overlaysGrid.innerHTML = "";
    }

    // -----------------------------------------
    // Fetch final report
    // -----------------------------------------
    async function fetchJobResults(jobId) {
        try {
            const res = await fetch(`/api/jobs/${jobId}`);
            const job = await res.json();

            if (job.result) {
                const report = job.result;

                reportSummary.textContent = report.incident_summary;
                reportExplanation.textContent = report.explanation;
                reportRecommendations.textContent = report.recommended_action;

                severityBadge.textContent = `${report.severity.toUpperCase()} SEVERITY`;
                severityBadge.className = `severity-badge ${report.severity}`;

                broadcastCard.classList.remove("hidden");
                broadcastCard.classList.add("reveal");

                pushStreamMessage("done", "fa-shield-halved",
                    `Verdict: <strong>${report.incident_summary}</strong> — Severity: <strong>${report.severity.toUpperCase()}</strong>`
                );
            }

            // Now reveal telemetry + rollouts
            revealPostProcessingData();

            endProcessing();
        } catch (e) {
            pushStreamMessage("error", "fa-triangle-exclamation", `Failed to load final report: ${e.message}`);
            endProcessing();
        }
    }

    // Stop button event listener
    stopBtn.addEventListener("click", async () => {
        if (!activeJobId || !isProcessing) return;

        stopBtn.disabled = true;
        stopBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Stopping...';
        pushStreamMessage("system", "fa-circle-stop", "Requesting pipeline cancellation from server...");

        try {
            const res = await fetch(`/api/jobs/${activeJobId}/stop`, { method: "POST" });
            if (!res.ok) throw new Error(await res.text());
            pushStreamMessage("system", "fa-circle-stop", "Pipeline cancellation request accepted by server.");
        } catch (e) {
            pushStreamMessage("error", "fa-triangle-exclamation", `Failed to stop pipeline: ${e.message}`);
            stopBtn.disabled = false;
            stopBtn.innerHTML = '<i class="fa-solid fa-circle-stop"></i> Stop Scan';
        }
    });
});
