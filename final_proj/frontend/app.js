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
    const consoleOutput = document.getElementById("console-output");
    
    const progressBarContainer = document.getElementById("progress-bar-container");
    const progressStatusText = document.getElementById("progress-status-text");
    const progressPercent = document.getElementById("progress-percent");
    const progressFill = document.getElementById("progress-fill");
    
    const broadcastCard = document.getElementById("broadcast-card");
    const reportSummary = document.getElementById("report-summary");
    const reportExplanation = document.getElementById("report-explanation");
    const reportAudio = document.getElementById("report-audio");
    const reportRecommendations = document.getElementById("report-recommendations");
    const severityBadge = document.getElementById("severity-badge");
    
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

    // --- Drag and Drop File Handlers ---
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("dragover");
        }, false);
    });

    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files.length > 0) {
            handleVideoSelected(dt.files[0]);
        }
    });

    videoFileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleVideoSelected(e.target.files[0]);
        }
    });

    dropZone.addEventListener("click", () => {
        videoFileInput.click();
    });

    function handleVideoSelected(file) {
        selectedFile = file;
        
        // Show metadata
        videoMetaName.textContent = file.name;
        
        const fileUrl = URL.createObjectURL(file);
        videoPlayer.src = fileUrl;
        
        videoPlayer.onloadedmetadata = () => {
            videoMetaDuration.textContent = `${videoPlayer.duration.toFixed(1)}s`;
            processBtn.disabled = false;
        };
        
        videoPreviewContainer.classList.remove("hidden");
        writeLog("system", `Selected video: ${file.name}. Ready to analyze.`);
    }

    // --- Logger helper ---
    function writeLog(type, message) {
        const entry = document.createElement("div");
        entry.className = `log-entry ${type}`;
        
        // Icon mapping
        let icon = "fa-info";
        if (type === "node") icon = "fa-microchip";
        if (type === "finished") icon = "fa-circle-check";
        if (type === "error") icon = "fa-triangle-exclamation";
        
        entry.innerHTML = `<i class="fa-solid ${icon}"></i> [${new Date().toLocaleTimeString()}] ${message}`;
        consoleOutput.appendChild(entry);
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
    }

    // --- Action triggering ---
    processBtn.addEventListener("click", async () => {
        if (!selectedFile) return;

        processBtn.disabled = true;
        broadcastCard.classList.add("hidden");
        resetFlowChartNodes();
        clearTelemetry();
        
        progressBarContainer.classList.remove("hidden");
        updateProgressBar("Ingesting video package", 5);

        writeLog("system", "Uploading video to API service...");
        
        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("output_language", languageSelect.value);

        try {
            const res = await fetch("/api/jobs", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                throw new Error(await res.text());
            }

            const data = await res.json();
            activeJobId = data.job_id;
            
            writeLog("system", `Job successfully queued. Job ID: ${activeJobId}`);
            connectWebSocket(activeJobId);
        } catch (e) {
            writeLog("error", `Failed to initialize scan: ${e.message}`);
            processBtn.disabled = false;
            updateProgressBar("Scan failed", 0);
        }
    });

    // --- WebSocket Sync logic ---
    function connectWebSocket(jobId) {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/api/ws/jobs/${jobId}`;
        
        ws = new WebSocket(wsUrl);
        writeLog("system", "Opening socket link to monitor pipeline progress...");

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handlePipelineUpdate(data);
        };

        ws.onclose = () => {
            writeLog("system", "Socket connection closed.");
        };

        ws.onerror = (err) => {
            writeLog("error", "Socket communication error occurred.");
        };
    }

    function updateProgressBar(statusText, percent) {
        progressStatusText.textContent = statusText;
        progressPercent.textContent = `${percent}%`;
        progressFill.style.width = `${percent}%`;
    }

    function resetFlowChartNodes() {
        const steps = document.querySelectorAll(".flow-step");
        steps.forEach(s => {
            s.className = "flow-step";
        });
    }

    function clearTelemetry() {
        telYolo.textContent = "-";
        telCrowd.textContent = "-";
        telMotion.textContent = "-";
        telOcr.textContent = "-";
        overlaysPlaceholder.classList.remove("hidden");
        overlaysGrid.classList.add("hidden");
        overlaysGrid.innerHTML = "";
    }

    // --- State update router ---
    function handlePipelineUpdate(data) {
        const node = data.node;
        const status = data.status;

        if (status === "running") {
            const nodeEl = document.getElementById(`node-${node}`);
            if (nodeEl) {
                nodeEl.classList.add("running");
            }
            writeLog("node", `Pipeline entering node execution step: '${node}'`);
        } else if (status === "completed") {
            const nodeEl = document.getElementById(`node-${node}`);
            if (nodeEl) {
                nodeEl.classList.remove("running");
                nodeEl.classList.add("completed");
            }
            writeLog("finished", `Step completed successfully: '${node}'`);
            
            // Increment progress bar based on node completed
            if (node === "extract_and_upscale_frames") updateProgressBar("Scene detection complete", 20);
            if (node === "run_vjepa") updateProgressBar("V-JEPA sliding window anomaly scan finished", 40);
            if (node === "yolo_world") updateProgressBar("Static object tags extracted", 50);
            if (node === "gemma_transcription") updateProgressBar("Multimodal frame descriptions generated", 65);
            if (node === "orchestrator_judge") updateProgressBar("Orchestrator evaluation finalized", 80);
            if (node === "tool_caller") updateProgressBar("Auxiliary evidence tools dispatched", 85);
            if (node === "reasoning_agent") updateProgressBar("Deep reasoning logic analyzed", 90);
            if (node === "gemma_output_module") updateProgressBar("Human-readable emergency report created", 95);
            if (node === "reasoning_to_speech") updateProgressBar("Audio broadcast files synthesized", 100);
            
            // Extract and update inline telemetry data
            if (data.update) {
                updateTelemetryView(node, data.update);
            }
        } else if (status === "failed") {
            const steps = document.querySelectorAll(".flow-step.running");
            steps.forEach(s => {
                s.classList.remove("running");
                s.classList.add("failed");
            });
            writeLog("error", `Pipeline crashed inside loop. Error: ${data.error}`);
            processBtn.disabled = false;
        }

        // Job fully completed
        if (node === "END" && status === "completed") {
            writeLog("finished", "Pipeline completed all processing cycles. Fetching safety reports...");
            fetchJobResults(activeJobId);
        }
    }

    // --- Dynamic Telemetry Viewer ---
    function updateTelemetryView(node, update) {
        if (node === "yolo_world" && update.yolo_detections) {
            const labels = update.yolo_detections.map(d => `${d.label} (${d.confidence.toFixed(2)})`);
            telYolo.innerHTML = labels.length ? labels.join("<br>") : "None";
        }
        
        if (node === "abrupt_motion_detection" && update.motion_evidence) {
            const lines = update.motion_evidence.map(m => `Time: ${m.timestamp_sec.toFixed(1)}s | Area Score: ${m.motion_score.toFixed(2)} | Z: ${m.z_score.toFixed(2)} ${m.is_abrupt ? '🔥' : ''}`);
            telMotion.innerHTML = lines.length ? lines.join("<br>") : "None";
        }
        
        if (node === "tool_caller" && update.tool_results) {
            const tr = update.tool_results;
            if (tr.ocr && tr.ocr.text) {
                telOcr.innerHTML = tr.ocr.text.join("<br>") || "None";
            }
            if (tr.crowd_analytics) {
                const ca = tr.crowd_analytics;
                telCrowd.innerHTML = `State: <strong>${ca.state.toUpperCase()}</strong><br>Flow Vector: ${ca.flow_direction}<br>Density: ${ca.density}<br>Count: ${ca.people_count.toFixed(1)}`;
            }
            if (tr.attention_rollout && tr.attention_rollout.length) {
                renderAttentionGallery(tr.attention_rollout);
            }
        }
        
        if (node === "run_vjepa" && update.vjepa_result) {
            // Render any inline overlays generated automatically during anomaly confirmations
            const vjepa_res = update.vjepa_result;
            if (vjepa_res.anomalies) {
                const auto_rollouts = vjepa_res.anomalies
                    .filter(a => a.attention_overlay_path)
                    .map(a => ({
                        "timestamp_sec": a.timestamp_sec,
                        "overlay_path": a.attention_overlay_path
                    }));
                if (auto_rollouts.length) {
                    renderAttentionGallery(auto_rollouts);
                }
            }
        }
    }

    function renderAttentionGallery(rollouts) {
        overlaysPlaceholder.classList.add("hidden");
        overlaysGrid.classList.remove("hidden");
        
        rollouts.forEach(r => {
            // Check if card is already rendered to avoid duplicate renders
            const cardId = `rollout-card-${r.timestamp_sec.toFixed(2)}`;
            if (document.getElementById(cardId)) return;
            
            const card = document.createElement("div");
            card.className = "overlay-card";
            card.id = cardId;
            
            // Extract just the filename to request artifact securely
            const parts = r.overlay_path.split(/[\\/]/);
            const filename = parts[parts.length - 1];
            const srcUrl = `/api/jobs/${activeJobId}/artifacts/${filename}`;
            
            card.innerHTML = `
                <img src="${srcUrl}" class="overlay-img" alt="Attention Rollout at ${r.timestamp_sec}s">
                <div class="overlay-label">Rollout @ ${r.timestamp_sec.toFixed(2)}s</div>
            `;
            overlaysGrid.appendChild(card);
        });
    }

    // --- Result loading details ---
    async function fetchJobResults(jobId) {
        try {
            const res = await fetch(`/api/jobs/${jobId}`);
            const job = await res.json();
            
            // Render Report Card
            if (job.result) {
                const report = job.result;
                
                reportSummary.textContent = report.incident_summary;
                reportExplanation.textContent = report.explanation;
                reportRecommendations.textContent = report.recommended_action;
                
                severityBadge.textContent = `${report.severity.toUpperCase()} SEVERITY`;
                severityBadge.className = `severity-badge ${report.severity}`;
                
                // Mount audio endpoint
                if (job.speech_audio) {
                    reportAudio.src = job.speech_audio;
                    reportAudio.load();
                }
                
                broadcastCard.classList.remove("hidden");
            }
            
            processBtn.disabled = false;
        } catch (e) {
            writeLog("error", `Failed to load final report outcomes: ${e.message}`);
            processBtn.disabled = false;
        }
    }
});
