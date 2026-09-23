/**
 * LiDAR Environment Perception & Adaptive Mapping Playback — Frontend Application
 *
 * Architecture:
 *   - Dynamic WebSocket URL derived from window.location (ws:// or wss://)
 *   - Automatic exponential backoff reconnection (500ms → 1s → 2s → 4s → 5s cap)
 *   - 5 map layers: Adaptive Map, LiDAR Points, Terrain, Candidate Obstacles, Changes
 *   - Ego vehicle always drawn at canvas origin (0,0)
 *   - Allocation stats read from snapshot.tiers (accurate cell counts from AdaptiveMap25D)
 *   - Diagnostics banner shows errors from server or WS
 *   - All replay controls go through /api/replay/* endpoints
 */

(function () {
  'use strict';

  // ── State Container ──────────────────────────────────────────────────
  const state = {
    isStreaming: false,
    playbackState: 'STOPPED',   // STOPPED, PLAYING, PAUSED
    playbackMode: 'replay',     // replay, synthetic
    playbackSpeed: 0.25,
    currentFrameIdx: 0,
    totalFrames: 5,

    // Map Snapshot Data
    currentSnapshot: null,
    colorMode: 'tier',          // 'tier' or 'elevation'
    autoFitDone: false,

    // Camera Viewport
    camera: {
      x: 0,       // Center world X (meters)
      y: 0,       // Center world Y (meters)
      zoom: 16.0, // Scale factor: pixels per meter
    },

    // Map Layers Toggle State
    layers: {
      adaptiveMap: true,
      points: true,
      terrainRisk: true,
      candidateObstacles: true,
      changes: true,
    },

    // Canvas Interaction
    isDragging: false,
    dragStart: { x: 0, y: 0 },
    cameraStart: { x: 0, y: 0 },

    // WebSocket & Polling
    ws: null,
    wsConnected: false,
    reconnectTimer: null,
    reconnectDelay: 500,        // Exponential backoff start
    pingInterval: null,
    pollInterval: null,
  };

  // ── Color Schemes ───────────────────────────────────────────────────
  const tierColors = {
    coarse:    { fill: 'rgba(37, 99, 235, 0.65)',  stroke: 'rgba(59, 130, 246, 0.45)' },
    medium:    { fill: 'rgba(16, 185, 129, 0.70)', stroke: 'rgba(52, 211, 153, 0.50)' },
    fine:      { fill: 'rgba(245, 158, 11, 0.75)', stroke: 'rgba(251, 191, 36, 0.55)' },
    ultrafine: { fill: 'rgba(239, 68, 68, 0.80)',  stroke: 'rgba(248, 113, 113, 0.60)' },
    ultra_fine:{ fill: 'rgba(239, 68, 68, 0.80)',  stroke: 'rgba(248, 113, 113, 0.60)' },
  };

  // ── DOM References ──────────────────────────────────────────────────
  const elements = {
    // Playback Controls
    btnPlay:          document.getElementById('btn-play') || document.getElementById('btn-play-pause'),
    btnPause:         document.getElementById('btn-pause'),
    btnPlayPause:     document.getElementById('btn-play-pause'),
    btnReset:         document.getElementById('btn-reset'),
    btnPrev:          document.getElementById('btn-prev') || document.getElementById('btn-step-prev'),
    btnNext:          document.getElementById('btn-next') || document.getElementById('btn-step-next'),
    btnSpeed01:       document.getElementById('btn-speed-01'),
    btnSpeed025:      document.getElementById('btn-speed-025'),
    btnSpeed05:       document.getElementById('btn-speed-05'),
    btnSpeed1:        document.getElementById('btn-speed-1'),
    frameScrubber:    document.getElementById('frame-scrubber') || document.getElementById('frame-slider'),

    // Legacy Compat Buttons
    btnStart:         document.getElementById('btn-start'),
    btnStop:          document.getElementById('btn-stop'),

    // Map Toolbar
    btnRefreshMap:    document.getElementById('btn-refresh-map'),
    btnFitMap:        document.getElementById('btn-fit-map'),
    btnZoomIn:        document.getElementById('btn-zoom-in'),
    btnZoomOut:       document.getElementById('btn-zoom-out'),
    btnToggleColor:   document.getElementById('btn-toggle-color'),
    colorModeLabel:   document.getElementById('color-mode-label'),

    // Layer Toggles
    layerAdaptiveMap: document.getElementById('layer-adaptive-map'),
    layerPoints:      document.getElementById('layer-points'),
    layerTerrainRisk: document.getElementById('layer-terrain-risk'),
    layerCandidateObs:document.getElementById('layer-candidate-obs'),
    layerChanges:     document.getElementById('layer-changes'),

    // Canvas & Overlay
    mapCanvas:        document.getElementById('map-canvas'),
    mapEmptyOverlay:  document.getElementById('map-empty-overlay'),
    overviewCanvas:   document.getElementById('overview-canvas'),
    overviewEmptyOverlay: document.getElementById('overview-empty-overlay'),
    btnFitOverview:   document.getElementById('btn-fit-overview'),
    mapTooltip:       document.getElementById('map-hover-tooltip'),
    mapFrameBadge:    document.getElementById('map-frame-badge'),

    // Badges
    wsBadge:          document.getElementById('ws-badge') || document.getElementById('ws-status-badge'),
    wsStatusText:     document.getElementById('ws-status-text') || (document.getElementById('ws-status-badge') ? document.getElementById('ws-status-badge').querySelector('.status-label') : null),
    streamBadge:      document.getElementById('stream-badge') || document.getElementById('sim-status-badge'),
    streamStatusText: document.getElementById('stream-status-text') || (document.getElementById('sim-status-badge') ? document.getElementById('sim-status-badge').querySelector('.status-label') : null),

    // Diagnostics
    diagnosticsBanner:  document.getElementById('diagnostics-banner'),
    diagnosticsText:    document.getElementById('diagnostics-text') || document.getElementById('diag-message'),
    btnDismissDiag:     document.getElementById('btn-dismiss-diag'),

    // Top Summary Strip
    mFrameId:         document.getElementById('m-frame-id'),
    mFrameCurrent:    document.getElementById('m-frame-current') || document.getElementById('curr-frame-display'),
    mFrameTotal:      document.getElementById('m-frame-total') || document.getElementById('total-frames-display'),
    sumCellCount:     document.getElementById('sum-cell-count'),
    mLatency:         document.getElementById('m-latency'),
    liveFpsBadge:     document.getElementById('live-fps-badge'),
    mapStatusPill:    document.getElementById('map-status-pill'),

    // Environment Information (B. What LiDAR Sees)
    mEnvRawPoints:    document.getElementById('m-env-raw-points'),
    mEnvRepCells:     document.getElementById('m-env-rep-cells'),
    mEnvCandObs:      document.getElementById('m-env-cand-obs'),
    mEnvTerrainRisk:  document.getElementById('m-env-terrain-risk'),
    mEnvElevRange:    document.getElementById('m-env-elev-range'),

    // Vehicle / Sensor Panel (A.)
    vmSensorPos:      document.getElementById('vm-sensor-pos'),
    vmHeading:        document.getElementById('vm-heading'),
    vmPlaybackMode:   document.getElementById('vm-playback-mode'),

    // Simulation Status Panel (C. Adaptive Mapping)
    simFrameIdx:      document.getElementById('sim-frame-idx'),
    simFrameTotal:    document.getElementById('sim-frame-total'),
    simPlaybackState: document.getElementById('sim-playback-state'),
    simPlaybackSpeed: document.getElementById('sim-playback-speed'),
    simLidarPts:      document.getElementById('sim-lidar-pts'),
    simMapCells:      document.getElementById('sim-map-cells'),
    simCandObs:       document.getElementById('sim-cand-obs'),
    simTerrainRisk:   document.getElementById('sim-terrain-risk'),
    simTierC:         document.getElementById('sim-tier-c'),
    simTierM:         document.getElementById('sim-tier-m'),
    simTierF:         document.getElementById('sim-tier-f'),
    simTierUF:        document.getElementById('sim-tier-uf'),
    simLatency:       document.getElementById('sim-latency'),
    simFps:           document.getElementById('sim-fps'),

    // Allocation Cards
    cntCoarse:        document.getElementById('cnt-coarse'),
    pctCoarse:        document.getElementById('pct-coarse'),
    barCoarse:        document.getElementById('bar-tier-coarse'),

    cntMedium:        document.getElementById('cnt-medium'),
    pctMedium:        document.getElementById('pct-medium'),
    barMedium:        document.getElementById('bar-tier-medium'),

    cntFine:          document.getElementById('cnt-fine'),
    pctFine:          document.getElementById('pct-fine'),
    barFine:          document.getElementById('bar-tier-fine'),

    cntUltrafine:     document.getElementById('cnt-ultrafine'),
    pctUltrafine:     document.getElementById('pct-ultrafine'),
    barUltrafine:     document.getElementById('bar-tier-ultrafine'),

    allocTotalCells:  document.getElementById('alloc-total-cells'),
    allocNoData:      document.getElementById('alloc-no-data'),
    allocContent:     document.getElementById('alloc-content'),

    // Temporal
    mTempNew:         document.getElementById('m-temp-new'),
    mTempChanged:     document.getElementById('m-temp-changed'),
    mTempUnchanged:   document.getElementById('m-temp-unchanged'),
    mTempRemoved:     document.getElementById('m-temp-removed'),

    // Timings
    tPreproc:         document.getElementById('t-preproc'),
    tBasemap:         document.getElementById('t-basemap'),
    tTerrain:         document.getElementById('t-terrain'),
    tResselect:       document.getElementById('t-resselect'),
    tMapupdate:       document.getElementById('t-mapupdate'),

    // Query
    inputQX:          document.getElementById('qx'),
    inputQY:          document.getElementById('qy'),
    btnQuery:         document.getElementById('btn-query'),
    queryResultBox:   document.getElementById('query-result-box'),

    // Map Legend
    mapCellCount:     document.getElementById('map-cell-count'),
    legendMinZ:       document.getElementById('legend-min-z'),
    legendMaxZ:       document.getElementById('legend-max-z'),
    egoPositionLabel: document.getElementById('ego-position-label'),
  };

  // ── Initialization ──────────────────────────────────────────────────
  function init() {
    setupCanvas();
    setupEventListeners();

    // Draw initial frame immediately (grid + ego marker) — no black screen on load
    renderMap();

    connectWebSocket();
    fetchStreamStatus();
    fetchMapSnapshot();

    // Periodic map snapshot polling when streaming
    state.pollInterval = setInterval(() => {
      if (state.isStreaming || state.playbackState === 'PLAYING') {
        fetchMapSnapshot();
      }
    }, 400);
  }

  // ── Canvas Setup ────────────────────────────────────────────────────
  let ctx = null;
  let overviewCtx = null;

  function setupCanvas() {
    if (!elements.mapCanvas) return;
    ctx = elements.mapCanvas.getContext('2d');
    resizeCanvasToDisplaySize();

    // Initialize overview canvas context
    if (elements.overviewCanvas) {
      overviewCtx = elements.overviewCanvas.getContext('2d');
      const oc = elements.overviewCanvas;
      const ow = oc.getBoundingClientRect().width || 1200;
      const oh = oc.getBoundingClientRect().height || 400;
      if (oc.width !== ow || oc.height !== oh) {
        oc.width  = Math.max(ow, 400);
        oc.height = Math.max(oh, 200);
      }
    }

    window.addEventListener('resize', () => {
      resizeCanvasToDisplaySize();
      renderMap();
    });
  }

  function resizeCanvasToDisplaySize() {
    const canvas = elements.mapCanvas;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(rect.width, 400);
    const h = Math.max(rect.height, 240);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }

  // ── Event Listeners ─────────────────────────────────────────────────
  function setupEventListeners() {
    // Playback Controls — use /api/replay/* endpoints
    if (elements.btnPlay) elements.btnPlay.addEventListener('click', handlePlay);
    if (elements.btnPause) elements.btnPause.addEventListener('click', handlePause);
    if (elements.btnReset) elements.btnReset.addEventListener('click', handleReset);
    if (elements.btnPrev) elements.btnPrev.addEventListener('click', () => handleStep(-1));
    if (elements.btnNext) elements.btnNext.addEventListener('click', () => handleStep(1));

    // Legacy compat
    if (elements.btnStart) elements.btnStart.addEventListener('click', handlePlay);
    if (elements.btnStop) elements.btnStop.addEventListener('click', handlePause);

    // Scrubber
    if (elements.frameScrubber) {
      elements.frameScrubber.addEventListener('input', (e) => {
        handleSeek(parseInt(e.target.value, 10));
      });
    }

    // Speed Pills
    if (elements.btnSpeed01)  elements.btnSpeed01.addEventListener('click',  () => setSpeed(0.1));
    if (elements.btnSpeed025) elements.btnSpeed025.addEventListener('click', () => setSpeed(0.25));
    if (elements.btnSpeed05)  elements.btnSpeed05.addEventListener('click',  () => setSpeed(0.5));
    if (elements.btnSpeed1)   elements.btnSpeed1.addEventListener('click',   () => setSpeed(1.0));

    // Map Toolbar
    if (elements.btnRefreshMap)  elements.btnRefreshMap.addEventListener('click', fetchMapSnapshot);
    if (elements.btnFitMap)      elements.btnFitMap.addEventListener('click', fitMapToBounds);
    if (elements.btnFitOverview) elements.btnFitOverview.addEventListener('click', renderOverviewMap);
    if (elements.btnZoomIn)      elements.btnZoomIn.addEventListener('click', () => zoomCamera(1.10));
    if (elements.btnZoomOut)     elements.btnZoomOut.addEventListener('click', () => zoomCamera(0.90));
    if (elements.btnToggleColor) elements.btnToggleColor.addEventListener('click', toggleColorMode);

    // Layer Controls
    setupLayerToggle(elements.layerAdaptiveMap, 'adaptiveMap');
    setupLayerToggle(elements.layerPoints, 'points');
    setupLayerToggle(elements.layerTerrainRisk, 'terrainRisk');
    setupLayerToggle(elements.layerCandidateObs, 'candidateObstacles');
    setupLayerToggle(elements.layerChanges, 'changes');

    // Spatial Query Form
    if (elements.btnQuery) elements.btnQuery.addEventListener('click', executeSpatialQuery);

    // Diagnostics
    if (elements.btnDismissDiag) {
      elements.btnDismissDiag.addEventListener('click', () => {
        if (elements.diagnosticsBanner) elements.diagnosticsBanner.classList.add('hidden');
      });
    }

    // Canvas Pan & Zoom Mouse Events
    const canvas = elements.mapCanvas;
    if (!canvas) return;

    canvas.addEventListener('mousedown', (e) => {
      state.isDragging = true;
      state.dragStart = { x: e.clientX, y: e.clientY };
      state.cameraStart = { x: state.camera.x, y: state.camera.y };
    });

    window.addEventListener('mousemove', (e) => {
      if (state.isDragging) {
        const dx = (e.clientX - state.dragStart.x) / state.camera.zoom;
        const dy = (e.clientY - state.dragStart.y) / state.camera.zoom;
        state.camera.x = state.cameraStart.x - dx;
        state.camera.y = state.cameraStart.y + dy;
        renderMap();
      } else {
        handleCanvasHover(e);
      }
    });

    window.addEventListener('mouseup', () => {
      state.isDragging = false;
    });

    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.05 : 0.95;
      zoomCamera(zoomFactor);
    }, { passive: false });

    canvas.addEventListener('click', (e) => {
      const rect = canvas.getBoundingClientRect();
      const clickCanvasX = e.clientX - rect.left;
      const clickCanvasY = e.clientY - rect.top;
      const worldPos = canvasToWorld(clickCanvasX, clickCanvasY);

      if (elements.inputQX) elements.inputQX.value = worldPos.x.toFixed(2);
      if (elements.inputQY) elements.inputQY.value = worldPos.y.toFixed(2);
      executeSpatialQuery();
    });
  }

  function setupLayerToggle(element, layerKey) {
    if (!element) return;
    element.addEventListener('click', () => {
      state.layers[layerKey] = !state.layers[layerKey];
      if (state.layers[layerKey]) {
        element.classList.add('active');
      } else {
        element.classList.remove('active');
      }
      renderMap();
    });
  }

  // ── REST API Calls — use /api/replay/* as primary; /api/stream/* as fallback ──
  async function handlePlay() {
    try {
      const res = await fetch('/api/replay/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fps: 10.0, speed: state.playbackSpeed, mode: 'replay' }),
      });
      const data = await res.json();
      if (data.status === 'started' || data.status === 'already_running') {
        updateStreamState(true, 'PLAYING');
      }
      if (data.last_error) showDiagnosticsError(data.last_error);
      fetchMapSnapshot();
    } catch (err) {
      console.error('Failed to start replay:', err);
      showDiagnosticsError(`Play failed: ${err.message}`);
    }
  }

  async function handlePause() {
    try {
      const res = await fetch('/api/replay/pause', { method: 'POST' });
      const data = await res.json();
      updateStreamState(false, 'PAUSED');
      fetchMapSnapshot();
    } catch (err) {
      console.error('Failed to pause:', err);
    }
  }

  async function handleReset() {
    try {
      const res = await fetch('/api/replay/reset', { method: 'POST' });
      const data = await res.json();
      updateStreamState(false, 'STOPPED');
      state.currentFrameIdx = 0;
      state.currentSnapshot = null;
      if (elements.frameScrubber) elements.frameScrubber.value = 0;
      renderMap();
      updateAllocationFromSnapshot(null);
    } catch (err) {
      console.error('Failed to reset:', err);
    }
  }

  async function handleStep(direction) {
    try {
      const res = await fetch(`/api/replay/step?direction=${direction}`, { method: 'POST' });
      const status = await res.json();
      if (status.current_frame !== undefined) {
        state.currentFrameIdx = status.current_frame;
        if (elements.frameScrubber) elements.frameScrubber.value = status.current_frame;
      }
      fetchMapSnapshot();
    } catch (err) {
      console.error('Failed to step frame:', err);
    }
  }

  async function handleSeek(frameIdx) {
    try {
      const res = await fetch(`/api/replay/seek?frame_idx=${frameIdx}`, { method: 'POST' });
      const status = await res.json();
      state.currentFrameIdx = frameIdx;
      fetchMapSnapshot();
    } catch (err) {
      console.error('Failed to seek frame:', err);
    }
  }

  async function setSpeed(speedVal) {
    state.playbackSpeed = speedVal;
    updateSpeedPillUI(speedVal);
    try {
      await fetch(`/api/replay/speed?speed=${speedVal}`, { method: 'POST' });
    } catch (err) {
      console.error('Failed to set speed:', err);
    }
  }

  function updateSpeedPillUI(speedVal) {
    [elements.btnSpeed01, elements.btnSpeed025, elements.btnSpeed05, elements.btnSpeed1].forEach(btn => {
      if (btn) btn.classList.remove('active');
    });
    if (speedVal === 0.1  && elements.btnSpeed01)  elements.btnSpeed01.classList.add('active');
    if (speedVal === 0.25 && elements.btnSpeed025) elements.btnSpeed025.classList.add('active');
    if (speedVal === 0.5  && elements.btnSpeed05)  elements.btnSpeed05.classList.add('active');
    if (speedVal === 1.0  && elements.btnSpeed1)   elements.btnSpeed1.classList.add('active');
  }

  async function fetchStreamStatus() {
    try {
      const res = await fetch('/api/replay/status');
      const data = await res.json();
      updateStatusUI(data);
      if (data.last_error) showDiagnosticsError(data.last_error);
    } catch (err) {
      console.warn('Status poll failed:', err);
    }
  }

  async function fetchMapSnapshot() {
    try {
      const res = await fetch('/api/map/snapshot');
      const data = await res.json();
      if (data && data.available && Array.isArray(data.cells) && data.cells.length > 0) {
        state.currentSnapshot = data;
        if (!state.autoFitDone) {
          fitMapToBounds();
          state.autoFitDone = true;
        }
        renderMap();
        updateAllocationFromSnapshot(data);
        updateEnvironmentInfoFromSnapshot(data);

        // Update playback info from snapshot
        if (data.playback) {
          updateStatusUI({
            is_streaming: state.isStreaming,
            playback_state: data.playback.state,
            current_frame: data.playback.current_frame,
            total_frames: data.playback.total_frames,
            playback_speed: data.playback.speed,
          });
        }
        if (data.last_error) showDiagnosticsError(data.last_error);
      } else {
        // No map data yet — still render grid + ego marker
        if (!state.currentSnapshot) {
          renderMap();
        }
        updateAllocationFromSnapshot(null);
        // Update playback info even when no cells
        if (data && data.playback) {
          updateStatusUI({
            is_streaming: state.isStreaming,
            playback_state: data.playback.state,
            current_frame: data.playback.current_frame,
            total_frames: data.playback.total_frames,
            playback_speed: data.playback.speed,
          });
        }
        if (data && data.last_error) showDiagnosticsError(data.last_error);
      }
    } catch (err) {
      console.warn('Snapshot fetch failed:', err);
    }
  }

  async function executeSpatialQuery() {
    if (!elements.inputQX || !elements.inputQY) return;
    const x = parseFloat(elements.inputQX.value) || 0.0;
    const y = parseFloat(elements.inputQY.value) || 0.0;

    try {
      const res = await fetch('/api/map/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x: x, y: y }),
      });
      const data = await res.json();

      if (elements.queryResultBox) {
        if (data.is_represented) {
          elements.queryResultBox.innerHTML = `
            <div><strong>Represented Cell Found</strong></div>
            <div>Tier: <span class="highlight">${data.tier}</span> (${data.resolution}m)</div>
            <div>Points: ${data.point_count} | Mean Z: ${data.mean_z?.toFixed(2)}m</div>
            <div>Z Range: [${data.min_z?.toFixed(2)}m, ${data.max_z?.toFixed(2)}m]</div>
          `;
        } else {
          elements.queryResultBox.innerHTML = `<span class="text-muted">No represented cell at (${x.toFixed(2)}, ${y.toFixed(2)}).</span>`;
        }
      }
    } catch (err) {
      console.error('Query failed:', err);
    }
  }

  // ── Diagnostics ─────────────────────────────────────────────────────
  function showDiagnosticsError(msg) {
    if (!msg || !elements.diagnosticsBanner) return;
    elements.diagnosticsBanner.classList.remove('hidden');
    if (elements.diagnosticsText) elements.diagnosticsText.textContent = msg;
  }

  function clearDiagnosticsError() {
    if (elements.diagnosticsBanner) elements.diagnosticsBanner.classList.add('hidden');
  }

  // ── WebSocket Connectivity ──────────────────────────────────────────
  function connectWebSocket() {
    // Derive WS URL dynamically from page location
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || '127.0.0.1:8000';
    const wsUrl = `${protocol}//${host}/ws/stream`;

    console.log(`[WS] Connecting to ${wsUrl} (attempt, backoff=${state.reconnectDelay}ms)`);
    updateWsBadge('reconnecting');

    try {
      state.ws = new WebSocket(wsUrl);

      state.ws.onopen = () => {
        state.wsConnected = true;
        state.reconnectDelay = 500; // reset backoff on success
        updateWsBadge('connected');
        clearDiagnosticsError();
        console.log('[WS] Connected to', wsUrl);

        // Start heartbeat ping every 5s
        if (state.pingInterval) clearInterval(state.pingInterval);
        state.pingInterval = setInterval(() => {
          if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({ command: 'ping', timestamp: Date.now() }));
          }
        }, 5000);
      };

      state.ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          handleWsMessage(msg);
        } catch (e) {
          console.error('[WS] Parse error:', e);
        }
      };

      state.ws.onclose = (evt) => {
        state.wsConnected = false;
        updateWsBadge('disconnected');
        if (state.pingInterval) { clearInterval(state.pingInterval); state.pingInterval = null; }
        console.warn('[WS] Disconnected (code=%d). Scheduling reconnect in %dms', evt.code, state.reconnectDelay);
        scheduleReconnect();
      };

      state.ws.onerror = (err) => {
        state.wsConnected = false;
        updateWsBadge('disconnected');
        console.error('[WS] Error:', err);
      };
    } catch (err) {
      console.error('[WS] Connection error:', err);
      updateWsBadge('disconnected');
      scheduleReconnect();
    }
  }

  function scheduleReconnect() {
    if (state.reconnectTimer) return; // already scheduled
    state.reconnectTimer = setTimeout(() => {
      state.reconnectTimer = null;
      // Exponential backoff: 500, 1000, 2000, 4000, 5000(cap)
      state.reconnectDelay = Math.min(state.reconnectDelay * 2, 5000);
      connectWebSocket();
    }, state.reconnectDelay);
  }

  function handleWsMessage(msg) {
    if (msg.type === 'frame_update') {
      updateTelemetryUI(msg);
      // Update playback state from WS frame_update payload
      if (msg.playback) {
        updateStatusUI({
          is_streaming: true,
          playback_state: msg.playback.state,
          current_frame: msg.playback.current_frame,
          total_frames: msg.playback.total_frames,
          playback_speed: msg.playback.speed,
        });
      }
    } else if (msg.type === 'map_update') {
      // Server signals a new frame is ready — fetch snapshot
      fetchMapSnapshot();
    } else if (msg.type === 'connection_established') {
      if (msg.status) updateStatusUI(msg.status);
      console.log('[WS] connection_established', msg.status);
    } else if (msg.type === 'stream_stopped') {
      updateStreamState(false, 'PAUSED');
      if (msg.status) updateStatusUI(msg.status);
    } else if (msg.type === 'error') {
      console.error('[WS] Server error:', msg.message);
      showDiagnosticsError(`Server: ${msg.message}`);
    } else if (msg.type === 'pong') {
      // Heartbeat OK
    }
  }

  // ── UI Updates ──────────────────────────────────────────────────────
  function updateWsBadge(state_) {
    if (!elements.wsBadge || !elements.wsStatusText) return;
    if (state_ === 'connected') {
      elements.wsBadge.className = 'badge badge-connected';
      elements.wsStatusText.textContent = 'WS Connected';
    } else if (state_ === 'reconnecting') {
      elements.wsBadge.className = 'badge badge-reconnecting';
      elements.wsStatusText.textContent = 'WS Reconnecting…';
    } else {
      elements.wsBadge.className = 'badge badge-disconnected';
      elements.wsStatusText.textContent = 'WS Disconnected';
    }
  }

  function updateStreamState(isStreaming, playbackState) {
    state.isStreaming = isStreaming;
    state.playbackState = playbackState || (isStreaming ? 'PLAYING' : 'STOPPED');

    if (!elements.streamBadge || !elements.streamStatusText) return;
    if (state.playbackState === 'PLAYING') {
      elements.streamBadge.className = 'badge badge-active';
      elements.streamStatusText.textContent = 'PLAYING';
    } else if (state.playbackState === 'PAUSED') {
      elements.streamBadge.className = 'badge badge-idle';
      elements.streamStatusText.textContent = 'PAUSED';
    } else {
      elements.streamBadge.className = 'badge badge-idle';
      elements.streamStatusText.textContent = 'REPLAY IDLE';
    }

    if (elements.mapStatusPill) elements.mapStatusPill.textContent = state.playbackState;
    if (elements.simPlaybackState) elements.simPlaybackState.textContent = state.playbackState;
  }

  function updateStatusUI(data) {
    if (!data) return;
    updateStreamState(data.is_streaming, data.playback_state);

    if (data.current_frame !== undefined) {
      state.currentFrameIdx = data.current_frame;
      if (elements.mFrameId) elements.mFrameId.textContent = data.current_frame;
      if (elements.mFrameCurrent) elements.mFrameCurrent.textContent = data.current_frame;
      if (elements.simFrameIdx) elements.simFrameIdx.textContent = data.current_frame;
      if (elements.frameScrubber) elements.frameScrubber.value = data.current_frame;
    }

    if (data.total_frames !== undefined) {
      state.totalFrames = data.total_frames;
      if (elements.mFrameTotal) elements.mFrameTotal.textContent = data.total_frames;
      if (elements.simFrameTotal) elements.simFrameTotal.textContent = data.total_frames;
      if (elements.frameScrubber) elements.frameScrubber.max = data.total_frames - 1;
    }

    if (data.playback_speed !== undefined) {
      state.playbackSpeed = data.playback_speed;
      if (elements.simPlaybackSpeed) elements.simPlaybackSpeed.textContent = `${data.playback_speed.toFixed(1)}x`;
      updateSpeedPillUI(data.playback_speed);
    }

    if (data.playback_mode !== undefined && elements.vmPlaybackMode) {
      elements.vmPlaybackMode.textContent = data.playback_mode === 'replay' ? 'Recorded Dataset' : 'Synthetic Generator';
    }
  }

  function updateTelemetryUI(msg) {
    if (msg.frame_id !== undefined) {
      if (elements.mFrameId) elements.mFrameId.textContent = msg.frame_id;
      if (elements.mFrameCurrent) elements.mFrameCurrent.textContent = msg.frame_id;
      if (elements.simFrameIdx) elements.simFrameIdx.textContent = msg.frame_id;
      if (elements.frameScrubber) elements.frameScrubber.value = msg.frame_id;
    }

    if (msg.latency_ms !== undefined) {
      if (elements.mLatency) elements.mLatency.innerHTML = `${msg.latency_ms.toFixed(1)} <small>ms</small>`;
      if (elements.simLatency) elements.simLatency.textContent = msg.latency_ms.toFixed(1);
    }

    if (msg.current_fps !== undefined) {
      if (elements.liveFpsBadge) elements.liveFpsBadge.textContent = `${msg.current_fps.toFixed(1)} FPS`;
      if (elements.simFps) elements.simFps.textContent = msg.current_fps.toFixed(1);
    }

    // Temporal stats
    if (msg.temporal) {
      if (elements.mTempNew) elements.mTempNew.textContent = msg.temporal.newly_observed_cells || 0;
      if (elements.mTempChanged) elements.mTempChanged.textContent = msg.temporal.changed_cells || 0;
      if (elements.mTempUnchanged) elements.mTempUnchanged.textContent = msg.temporal.unchanged_cells || 0;
      if (elements.mTempRemoved) elements.mTempRemoved.textContent = msg.temporal.no_longer_observed_cells || 0;
    }

    // Stage timings
    if (msg.stage_timings) {
      if (elements.tPreproc)   elements.tPreproc.textContent   = `${msg.stage_timings.preprocessing_ms || 0} ms`;
      if (elements.tBasemap)   elements.tBasemap.textContent   = `${msg.stage_timings.base_mapping_ms || 0} ms`;
      if (elements.tTerrain)   elements.tTerrain.textContent   = `${msg.stage_timings.terrain_analysis_ms || 0} ms`;
      if (elements.tResselect) elements.tResselect.textContent = `${msg.stage_timings.resolution_selection_ms || 0} ms`;
      if (elements.tMapupdate) elements.tMapupdate.textContent = `${msg.stage_timings.map_update_ms || 0} ms`;
    }

    // Environment from telemetry
    if (msg.environment) {
      if (elements.mEnvCandObs) elements.mEnvCandObs.textContent = msg.environment.candidate_obstacles_count || 0;
      if (elements.simCandObs) elements.simCandObs.textContent = msg.environment.candidate_obstacles_count || 0;
      const risk = msg.environment.terrain_risk || 'LOW';
      if (elements.mEnvTerrainRisk) {
        elements.mEnvTerrainRisk.textContent = risk;
        elements.mEnvTerrainRisk.className = `env-risk-badge risk-${risk.toLowerCase()}`;
      }
      if (elements.simTerrainRisk) elements.simTerrainRisk.textContent = risk;
      const repCells = msg.environment.represented_cells || 0;
      if (elements.mEnvRepCells) elements.mEnvRepCells.textContent = repCells.toLocaleString();
      if (elements.simMapCells) elements.simMapCells.textContent = repCells.toLocaleString();
    }
  }

  function updateEnvironmentInfoFromSnapshot(snapshot) {
    if (!snapshot) return;
    const cells = snapshot.cells || [];
    const totalCells = snapshot.represented_cells || cells.length;
    const candObs = snapshot.candidate_obstacles_count || 0;
    const terrainRisk = snapshot.terrain_risk || 'LOW';
    const rawPts = snapshot.raw_point_count || 0;

    // Compute min/max Z from cells
    let minZ = 999.0;
    let maxZ = -999.0;
    cells.forEach(cell => {
      if (cell.z !== undefined) {
        if (cell.z < minZ) minZ = cell.z;
        if (cell.z > maxZ) maxZ = cell.z;
      }
    });
    if (minZ === 999.0) minZ = 0.0;
    if (maxZ === -999.0) maxZ = 3.5;

    if (elements.mEnvRawPoints) elements.mEnvRawPoints.textContent = rawPts.toLocaleString();
    if (elements.mEnvRepCells)  elements.mEnvRepCells.textContent  = totalCells.toLocaleString();
    if (elements.mEnvCandObs)   elements.mEnvCandObs.textContent   = candObs.toLocaleString();

    if (elements.mEnvTerrainRisk) {
      elements.mEnvTerrainRisk.textContent = terrainRisk;
      elements.mEnvTerrainRisk.className = `env-risk-badge risk-${terrainRisk.toLowerCase()}`;
    }

    if (elements.mEnvElevRange) {
      elements.mEnvElevRange.textContent = `${minZ.toFixed(2)} m → ${maxZ >= 0 ? '+' : ''}${maxZ.toFixed(2)} m`;
    }

    // Status summary panel
    if (elements.simLidarPts) elements.simLidarPts.textContent = rawPts.toLocaleString();
    if (elements.simMapCells) elements.simMapCells.textContent = totalCells.toLocaleString();
    if (elements.simCandObs)  elements.simCandObs.textContent  = candObs;
    if (elements.simTerrainRisk) elements.simTerrainRisk.textContent = terrainRisk;

    if (elements.legendMinZ) elements.legendMinZ.textContent = `${minZ.toFixed(1)}m`;
    if (elements.legendMaxZ) elements.legendMaxZ.textContent = `${maxZ >= 0 ? '+' : ''}${maxZ.toFixed(1)}m`;
  }

  function updateAllocationFromSnapshot(snapshot) {
    if (!snapshot || !snapshot.available || !snapshot.tiers) {
      if (elements.allocNoData) elements.allocNoData.classList.remove('hidden');
      if (elements.allocContent) elements.allocContent.classList.add('hidden');
      return;
    }

    if (elements.allocNoData) elements.allocNoData.classList.add('hidden');
    if (elements.allocContent) elements.allocContent.classList.remove('hidden');

    // Read accurate cell counts from snapshot.tiers (from AdaptiveMap25D.to_snapshot_dict)
    const tiers = snapshot.tiers;
    const totalCells = snapshot.represented_cells || 0;

    const coarseCount    = (tiers['coarse']     || {}).cell_count || 0;
    const mediumCount    = (tiers['medium']     || {}).cell_count || 0;
    const fineCount      = (tiers['fine']       || {}).cell_count || 0;
    const ultrafineCount = (tiers['ultra_fine'] || tiers['ultrafine'] || {}).cell_count || 0;

    const calcPct = (cnt) => totalCells > 0 ? ((cnt / totalCells) * 100).toFixed(1) : '0.0';

    updateTierRow('coarse',    coarseCount,    calcPct(coarseCount),    elements.cntCoarse,    elements.pctCoarse,    elements.barCoarse);
    updateTierRow('medium',    mediumCount,    calcPct(mediumCount),    elements.cntMedium,    elements.pctMedium,    elements.barMedium);
    updateTierRow('fine',      fineCount,      calcPct(fineCount),      elements.cntFine,      elements.pctFine,      elements.barFine);
    updateTierRow('ultrafine', ultrafineCount, calcPct(ultrafineCount), elements.cntUltrafine, elements.pctUltrafine, elements.barUltrafine);

    if (elements.allocTotalCells) elements.allocTotalCells.textContent = totalCells.toLocaleString();
    if (elements.sumCellCount)    elements.sumCellCount.textContent    = totalCells.toLocaleString();
    if (elements.mapCellCount)    elements.mapCellCount.textContent    = totalCells.toLocaleString();

    if (elements.simTierC)  elements.simTierC.textContent  = coarseCount;
    if (elements.simTierM)  elements.simTierM.textContent  = mediumCount;
    if (elements.simTierF)  elements.simTierF.textContent  = fineCount;
    if (elements.simTierUF) elements.simTierUF.textContent = ultrafineCount;
  }

  function updateTierRow(tierKey, count, pct, cntElem, pctElem, barElem) {
    if (cntElem) cntElem.textContent = `${count.toLocaleString()} cells`;
    if (pctElem) pctElem.textContent = `${pct}%`;
    if (barElem) barElem.style.width = `${Math.min(100, parseFloat(pct))}%`;
  }

  // ── Camera Navigation Functions ─────────────────────────────────────
  function fitMapToBounds() {
    if (!state.currentSnapshot || !state.currentSnapshot.bounds) {
      state.camera = { x: 0, y: 0, zoom: 16.0 };
      renderMap();
      return;
    }

    const b = state.currentSnapshot.bounds;
    const worldW = (b.max_x - b.min_x) || 30.0;
    const worldH = (b.max_y - b.min_y) || 30.0;

    state.camera.x = (b.min_x + b.max_x) / 2.0;
    state.camera.y = (b.min_y + b.max_y) / 2.0;

    const canvas = elements.mapCanvas;
    const canvasW = canvas.width || 800;
    const canvasH = canvas.height || 450;

    const zoomX = (canvasW * 0.82) / worldW;
    const zoomY = (canvasH * 0.82) / worldH;
    state.camera.zoom = Math.max(2.0, Math.min(zoomX, zoomY, 40.0));

    renderMap();
  }

  function zoomCamera(factor) {
    state.camera.zoom = Math.max(2.0, Math.min(state.camera.zoom * factor, 80.0));
    renderMap();
  }

  function toggleColorMode() {
    state.colorMode = state.colorMode === 'tier' ? 'elevation' : 'tier';
    if (elements.btnToggleColor) {
      elements.btnToggleColor.textContent = `Color: ${state.colorMode === 'tier' ? 'Resolution' : 'Elevation'}`;
    }
    if (elements.colorModeLabel) {
      elements.colorModeLabel.textContent = state.colorMode === 'tier' ? 'Resolution Tier' : 'Elevation Gradient';
    }
    renderMap();
  }

  function showEmptyOverlay(show) {
    if (elements.mapEmptyOverlay) {
      if (show) elements.mapEmptyOverlay.classList.remove('hidden');
      else elements.mapEmptyOverlay.classList.add('hidden');
    }
  }

  // ── Coordinate Transforms ───────────────────────────────────────────
  function worldToCanvas(wx, wy) {
    const canvas = elements.mapCanvas;
    const cx = canvas.width / 2.0;
    const cy = canvas.height / 2.0;

    const canvasX = cx + (wx - state.camera.x) * state.camera.zoom;
    const canvasY = cy - (wy - state.camera.y) * state.camera.zoom; // Invert Y for top-down
    return { x: canvasX, y: canvasY };
  }

  function canvasToWorld(cx, cy) {
    const canvas = elements.mapCanvas;
    const centerCanvasX = canvas.width / 2.0;
    const centerCanvasY = canvas.height / 2.0;

    const wx = state.camera.x + (cx - centerCanvasX) / state.camera.zoom;
    const wy = state.camera.y - (cy - centerCanvasY) / state.camera.zoom;
    return { x: wx, y: wy };
  }

  // ── Main Map Rendering ──────────────────────────────────────────────
  function renderMap() {
    if (!ctx || !elements.mapCanvas) return;
    const canvas = elements.mapCanvas;

    // Clear background
    ctx.fillStyle = '#04060a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Always draw the grid and ego vehicle — even before map data
    drawGridLines();
    drawRangeRings();

    const hasMapData = state.currentSnapshot &&
                       Array.isArray(state.currentSnapshot.cells) &&
                       state.currentSnapshot.cells.length > 0;

    if (hasMapData) {
      showEmptyOverlay(false);

      // Update map frame pill badge
      if (elements.mapFrameBadge) {
        const fid = state.currentSnapshot.frame_id !== undefined ? state.currentSnapshot.frame_id : 0;
        elements.mapFrameBadge.textContent = `Frame ${fid}`;
      }

      // 1. Render Adaptive Cells
      if (state.layers.adaptiveMap) drawAdaptiveCells();

      // 2. Render LiDAR Points layer
      if (state.layers.points) drawLidarPoints();

      // 3. Render Candidate Obstacle Highlights
      if (state.layers.candidateObstacles) drawCandidateObstacleHighlights();

      // 4. Render Temporal Changes layer
      if (state.layers.changes) drawChangedCells();

    } else {
      // No data yet: show overlay but keep grid/ego visible behind it
      showEmptyOverlay(true);
    }

    // Ego vehicle always rendered on top
    drawEgoMarker();
    
    // Also render overview map
    renderOverviewMap();
  }

  function drawGridLines() {
    const canvas = elements.mapCanvas;
    ctx.lineWidth = 1;

    // Subtle 5m grid
    const stepMeters = 5.0;
    const stepPx = stepMeters * state.camera.zoom;

    if (stepPx >= 15) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
      ctx.beginPath();

      const startWorldX = Math.floor((state.camera.x - canvas.width / (2 * state.camera.zoom)) / stepMeters) * stepMeters;
      const endWorldX   = Math.ceil((state.camera.x  + canvas.width / (2 * state.camera.zoom)) / stepMeters) * stepMeters;

      for (let wx = startWorldX; wx <= endWorldX; wx += stepMeters) {
        const pt = worldToCanvas(wx, 0);
        ctx.moveTo(pt.x, 0);
        ctx.lineTo(pt.x, canvas.height);
      }

      const startWorldY = Math.floor((state.camera.y - canvas.height / (2 * state.camera.zoom)) / stepMeters) * stepMeters;
      const endWorldY   = Math.ceil((state.camera.y  + canvas.height / (2 * state.camera.zoom)) / stepMeters) * stepMeters;

      for (let wy = startWorldY; wy <= endWorldY; wy += stepMeters) {
        const pt = worldToCanvas(0, wy);
        ctx.moveTo(0, pt.y);
        ctx.lineTo(canvas.width, pt.y);
      }
      ctx.stroke();
    }

    // Main World Axes (X & Y = 0)
    const origin = worldToCanvas(0, 0);
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.25)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(origin.x, 0);
    ctx.lineTo(origin.x, canvas.height);
    ctx.moveTo(0, origin.y);
    ctx.lineTo(canvas.width, origin.y);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawRangeRings() {
    // Draw LiDAR range rings at 5m, 10m, 15m, 20m from origin
    const rings = [5, 10, 15, 20];
    const origin = worldToCanvas(0, 0);
    ctx.setLineDash([3, 5]);
    rings.forEach((r, i) => {
      const radiusPx = r * state.camera.zoom;
      if (radiusPx < 5) return;
      ctx.strokeStyle = `rgba(250, 204, 21, ${0.06 + i * 0.02})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(origin.x, origin.y, radiusPx, 0, 2 * Math.PI);
      ctx.stroke();
    });
    ctx.setLineDash([]);
  }

  function drawAdaptiveCells() {
    const cells = state.currentSnapshot.cells;
    const tierPriority = { coarse: 1, medium: 2, fine: 3, ultrafine: 4, ultra_fine: 4 };

    // Sort cells so finer cells draw on top
    const sorted = cells.slice().sort((a, b) => {
      const pa = tierPriority[(a.tier || 'coarse').toLowerCase()] || 1;
      const pb = tierPriority[(b.tier || 'coarse').toLowerCase()] || 1;
      return pa - pb;
    });

    sorted.forEach(cell => {
      const res = cell.resolution || 0.2;
      const cellPx = Math.max(2, res * state.camera.zoom);

      const pt = worldToCanvas(cell.x, cell.y);
      const halfPx = cellPx / 2.0;

      const tKey = (cell.tier || 'coarse').toLowerCase().replace('_', '');
      const tStyle = tierColors[tKey] || tierColors.coarse;

      if (state.colorMode === 'tier') {
        ctx.fillStyle = tStyle.fill;
        ctx.strokeStyle = tStyle.stroke;
      } else {
        const z = cell.z !== undefined ? cell.z : 0.0;
        ctx.fillStyle = getElevationColor(z);
        ctx.strokeStyle = 'rgba(255,255,255,0.15)';
      }

      ctx.fillRect(pt.x - halfPx, pt.y - halfPx, cellPx, cellPx);
      if (cellPx >= 6) {
        ctx.lineWidth = 1;
        ctx.strokeRect(pt.x - halfPx, pt.y - halfPx, cellPx, cellPx);
      }
    });
  }

  function drawLidarPoints() {
    // Draw raw LiDAR points as small glowing dots if available in snapshot
    // Points can be sparse — subsample to max 1000 for performance
    if (!state.currentSnapshot) return;
    const cells = state.currentSnapshot.cells || [];
    // Use cell centers as point approximations when raw points not available
    const MAX_PTS = 800;
    const step = Math.max(1, Math.floor(cells.length / MAX_PTS));
    ctx.fillStyle = 'rgba(255, 255, 255, 0.80)';
    for (let i = 0; i < cells.length; i += step) {
      const cell = cells[i];
      const pt = worldToCanvas(cell.x, cell.y);
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 1.5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  function drawCandidateObstacleHighlights() {
    const cells = state.currentSnapshot.cells || [];
    cells.forEach(cell => {
      if (!cell.is_candidate_obstacle) return;

      const res = cell.resolution || 0.2;
      const cellPx = Math.max(4, res * state.camera.zoom);
      const pt = worldToCanvas(cell.x, cell.y);
      const halfPx = cellPx / 2.0;

      // Red obstacle highlight border
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.9)';
      ctx.strokeRect(pt.x - halfPx - 1, pt.y - halfPx - 1, cellPx + 2, cellPx + 2);

      // Small hazard dot if large enough
      if (cellPx >= 10) {
        ctx.fillStyle = '#ef4444';
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 2, 0, 2 * Math.PI);
        ctx.fill();
      }
    });
  }

  function drawChangedCells() {
    // Highlight newly observed cells (cyan tint) and changed cells (amber)
    const cells = state.currentSnapshot.cells || [];
    // We use refinement_reason as a proxy — actual temporal masks are per-tier grid arrays
    // and not serialized into the cell list, so we show fine/ultra_fine as "active change" areas
    cells.forEach(cell => {
      const tier = (cell.tier || '').toLowerCase();
      if (tier !== 'fine' && tier !== 'ultra_fine' && tier !== 'ultrafine') return;
      if (!state.layers.changes) return;

      const res = cell.resolution || 0.1;
      const cellPx = Math.max(3, res * state.camera.zoom);
      const pt = worldToCanvas(cell.x, cell.y);
      const halfPx = cellPx / 2.0;

      // Cyan tint for high-detail changed areas
      ctx.fillStyle = 'rgba(6, 182, 212, 0.18)';
      ctx.fillRect(pt.x - halfPx, pt.y - halfPx, cellPx, cellPx);
    });
  }

  function drawEgoMarker() {
    if (!ctx) return;
    const ego = worldToCanvas(0, 0);

    // Sensor scan range aura
    const scanRadiusPx = 15.0 * state.camera.zoom;
    ctx.strokeStyle = 'rgba(250, 204, 21, 0.12)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    ctx.arc(ego.x, ego.y, scanRadiusPx, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);

    // Outer glow aura
    ctx.fillStyle = 'rgba(250, 204, 21, 0.18)';
    ctx.beginPath();
    ctx.arc(ego.x, ego.y, 14, 0, 2 * Math.PI);
    ctx.fill();

    // Ego Icon Center (Golden Circle)
    ctx.fillStyle = '#facc15';
    ctx.beginPath();
    ctx.arc(ego.x, ego.y, 6, 0, 2 * Math.PI);
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();

    // Direction Triangle ▲ — pointing up (North / heading 0°)
    ctx.fillStyle = '#facc15';
    ctx.beginPath();
    ctx.moveTo(ego.x, ego.y - 13);
    ctx.lineTo(ego.x - 5, ego.y - 7);
    ctx.lineTo(ego.x + 5, ego.y - 7);
    ctx.closePath();
    ctx.fill();

    // "EGO / SENSOR" Label Pill
    ctx.font = '700 10px "Plus Jakarta Sans", sans-serif';
    ctx.textAlign = 'center';

    const labelText = 'SENSOR / EGO (0, 0)';
    const textWidth = ctx.measureText(labelText).width;
    const pillW = textWidth + 14;
    const pillH = 17;
    const pillX = ego.x - pillW / 2;
    const pillY = ego.y + 12;

    ctx.fillStyle = 'rgba(8, 11, 18, 0.88)';
    ctx.strokeStyle = 'rgba(250, 204, 21, 0.6)';
    ctx.lineWidth = 1;
    roundRect(ctx, pillX, pillY, pillW, pillH, 4, true, true);

    ctx.fillStyle = '#facc15';
    ctx.fillText(labelText, ego.x, pillY + 11.5);
    ctx.textAlign = 'left';
  }

  function getElevationColor(z) {
    // Normalised elevation map: z in [-0.5, 3.5] -> t in [0, 1]
    const norm = Math.max(0.0, Math.min(1.0, (z + 0.5) / 4.0));
    const r = Math.round(30 + norm * 200);
    const g = Math.round(60 + Math.sin(norm * Math.PI) * 160);
    const b = Math.round(200 - norm * 150);
    return `rgba(${r}, ${g}, ${b}, 0.80)`;
  }

  function roundRect(context, x, y, width, height, radius, fill, stroke) {
    context.beginPath();
    context.moveTo(x + radius, y);
    context.lineTo(x + width - radius, y);
    context.quadraticCurveTo(x + width, y, x + width, y + radius);
    context.lineTo(x + width, y + height - radius);
    context.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    context.lineTo(x + radius, y + height);
    context.quadraticCurveTo(x, y + height, x, y + height - radius);
    context.lineTo(x, y + radius);
    context.quadraticCurveTo(x, y, x + radius, y);
    context.closePath();
    if (fill) context.fill();
    if (stroke) context.stroke();
  }

  // ── Hover Tooltip ───────────────────────────────────────────────────
  function handleCanvasHover(e) {
    if (!elements.mapTooltip || !state.currentSnapshot || !state.currentSnapshot.cells) return;
    const rect = elements.mapCanvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const worldPos = canvasToWorld(cx, cy);

    // Find cell near mouse
    const cell = state.currentSnapshot.cells.find(c => {
      const res = c.resolution || 0.2;
      return Math.abs(c.x - worldPos.x) <= res / 2 && Math.abs(c.y - worldPos.y) <= res / 2;
    });

    if (cell) {
      elements.mapTooltip.classList.remove('hidden');
      elements.mapTooltip.style.left = `${cx + 12}px`;
      elements.mapTooltip.style.top  = `${cy + 12}px`;

      let content = `
        <div><strong>Cell (${cell.x.toFixed(2)}, ${cell.y.toFixed(2)})</strong></div>
        <div>Tier: <span class="highlight">${cell.tier}</span> (${cell.resolution}m)</div>
        <div>Mean Z: ${cell.z?.toFixed(2)}m (Points: ${cell.point_count})</div>
        <div>Z Range: [${cell.min_z?.toFixed(2)}m, ${cell.max_z?.toFixed(2)}m]</div>
      `;
      if (cell.is_candidate_obstacle) {
        content += `<div style="color:#ef4444; font-weight:bold; margin-top:2px;">⚠ ${cell.refinement_reason || 'Candidate obstacle region'}</div>`;
      }
      elements.mapTooltip.innerHTML = content;
    } else {
      elements.mapTooltip.classList.add('hidden');
    }
  }

  // Boot app on DOMReady
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }



  // ── Environment Overview Rendering (Secondary Bird's-Eye View) ─────────────
  function renderOverviewMap() {
    if (!overviewCtx || !elements.overviewCanvas) return;
    const canvas = elements.overviewCanvas;
    const octx = overviewCtx;

    // Clear background
    octx.fillStyle = '#04060a';
    octx.fillRect(0, 0, canvas.width, canvas.height);

    const hasMapData = state.currentSnapshot &&
                       Array.isArray(state.currentSnapshot.cells) &&
                       state.currentSnapshot.cells.length > 0;

    if (!hasMapData) {
      if (elements.overviewEmptyOverlay) elements.overviewEmptyOverlay.classList.remove('hidden');
      return;
    } else {
      if (elements.overviewEmptyOverlay) elements.overviewEmptyOverlay.classList.add('hidden');
    }

    // Fixed ego-centric range for overview (e.g. +/- 35 meters)
    let range = 35;
    if (state.currentSnapshot.bounds) {
      const b = state.currentSnapshot.bounds;
      const maxExt = Math.max(Math.abs(b.min_x), Math.abs(b.max_x), Math.abs(b.min_y), Math.abs(b.max_y), 30);
      range = maxExt * 1.15;
    }

    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const overviewZoom = Math.min((canvas.width * 0.45) / range, (canvas.height * 0.45) / range);

    function overviewWorldToCanvas(wx, wy) {
      const x = cx + wx * overviewZoom;
      const y = cy - wy * overviewZoom; // Invert Y (top is forward/North)
      return { x, y };
    }

    // Draw concentric range rings (10m, 20m, 30m)
    octx.lineWidth = 1;
    [10, 20, 30].forEach(r => {
      if (r * overviewZoom < Math.min(canvas.width, canvas.height) / 2) {
        octx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        octx.setLineDash([4, 4]);
        octx.beginPath();
        octx.arc(cx, cy, r * overviewZoom, 0, 2 * Math.PI);
        octx.stroke();
        octx.setLineDash([]);

        // Range text
        octx.fillStyle = 'rgba(143, 163, 192, 0.5)';
        octx.font = '9px "JetBrains Mono", monospace';
        octx.fillText(`${r}m`, cx + r * overviewZoom + 4, cy - 3);
      }
    });

    // Draw coordinate axes
    octx.strokeStyle = 'rgba(59, 130, 246, 0.25)';
    octx.lineWidth = 1;
    octx.beginPath();
    octx.moveTo(0, cy); octx.lineTo(canvas.width, cy); // X axis
    octx.moveTo(cx, 0); octx.lineTo(cx, canvas.height); // Y axis
    octx.stroke();

    // Axis labels
    octx.fillStyle = 'rgba(59, 130, 246, 0.6)';
    octx.font = 'bold 9px "JetBrains Mono", monospace';
    octx.fillText('+Y (Fwd)', cx + 6, 14);
    octx.fillText('+X (Right)', canvas.width - 55, cy - 6);

    const cells = state.currentSnapshot.cells || [];

    // 1. Draw environment background points (white/cyan)
    const step = Math.max(1, Math.floor(cells.length / 400));
    octx.fillStyle = 'rgba(255, 255, 255, 0.65)';
    for (let i = 0; i < cells.length; i += step) {
      const cell = cells[i];
      if (cell.is_candidate_obstacle) continue; // Obstacles drawn separately
      const pt = overviewWorldToCanvas(cell.x, cell.y);
      octx.fillRect(pt.x - 1, pt.y - 1, 2, 2);
    }

    // 2. Draw candidate obstacles (red with glow & connecting distance lines)
    cells.forEach(cell => {
      if (!cell.is_candidate_obstacle) return;
      const pt = overviewWorldToCanvas(cell.x, cell.y);
      const res = cell.resolution || 0.2;
      const cellPx = Math.max(4, res * overviewZoom);

      // Hazard fill
      octx.fillStyle = 'rgba(239, 68, 68, 0.85)';
      octx.beginPath();
      octx.arc(pt.x, pt.y, cellPx, 0, 2 * Math.PI);
      octx.fill();

      // Outer warning ring
      octx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
      octx.lineWidth = 1.5;
      octx.beginPath();
      octx.arc(pt.x, pt.y, cellPx + 3, 0, 2 * Math.PI);
      octx.stroke();

      // Distance line from ego to obstacle if close (< 25m)
      const dist = Math.sqrt(cell.x * cell.x + cell.y * cell.y);
      if (dist < 25.0) {
        octx.strokeStyle = 'rgba(239, 68, 68, 0.25)';
        octx.lineWidth = 1;
        octx.setLineDash([2, 3]);
        octx.beginPath();
        octx.moveTo(cx, cy);
        octx.lineTo(pt.x, pt.y);
        octx.stroke();
        octx.setLineDash([]);
      }
    });

    // 3. Draw Ego Vehicle Marker at center (0,0)
    // Outer pulse ring
    octx.fillStyle = 'rgba(250, 204, 21, 0.2)';
    octx.beginPath();
    octx.arc(cx, cy, 14, 0, 2 * Math.PI);
    octx.fill();

    // Car body (rounded rectangle)
    octx.fillStyle = '#facc15';
    octx.strokeStyle = '#ffffff';
    octx.lineWidth = 1.5;
    const carW = 10, carH = 18;
    octx.beginPath();
    octx.roundRect(cx - carW/2, cy - carH/2, carW, carH, 3);
    octx.fill();
    octx.stroke();

    // Direction arrow (pointing forward/up)
    octx.fillStyle = '#080b12';
    octx.beginPath();
    octx.moveTo(cx, cy - carH/2 + 2);
    octx.lineTo(cx - 3, cy - carH/2 + 6);
    octx.lineTo(cx + 3, cy - carH/2 + 6);
    octx.closePath();
    octx.fill();

    // Vehicle label
    octx.fillStyle = 'rgba(250, 204, 21, 0.95)';
    octx.font = 'bold 9px "JetBrains Mono", monospace';
    octx.textAlign = 'center';
    octx.fillText('EGO', cx, cy + 18);
    octx.textAlign = 'left';
  }

})();
