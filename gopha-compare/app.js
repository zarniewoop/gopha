// ===== Element refs =====
const $ = sel => document.querySelector(sel);
const urlA = $('#urlA');
const urlB = $('#urlB');
const vidA = $('#vidA');
const vidB = $('#vidB');
const playPause = $('#playPause');
const restartBtn = $('#restart');
const seek = $('#seek');
const timeLabel = $('#timeLabel');
const statusEl = $('#status');
const handle = $('#handle');
const knob = $('#knob');
const viewport = $('#viewport');
const offsetAEl = $('#offsetA');
const offsetBEl = $('#offsetB');
const applyOffsetsBtn = $('#applyOffsets');
const permalink = $('#permalink');

// New UI elements (ensure these exist in HTML)
const driftHud = $('#driftHud');            // <div id="driftHud" class="hud"></div>
const bufferBar = $('#bufferBar');          // <div id="bufferBar" class="buffer"><div class="barA"></div><div class="barB"></div></div>
const barA = bufferBar ? bufferBar.querySelector('.barA') : null;
const barB = bufferBar ? bufferBar.querySelector('.barB') : null;
const playOverlay = $('#playOverlay');      // <div id="playOverlay" class="playOverlay"><button class="btn">Play</button></div>
const snapChk = $('#snapFrames');           // <input type="checkbox" id="snapFrames">
const fpsAInput = $('#fpsA');               // <input type="text" id="fpsA" placeholder="fps A (e.g. 29.97)">
const fpsBInput = $('#fpsB');               // <input type="text" id="fpsB" placeholder="fps B (e.g. 25)">
const nudgeA01 = $('#nudgeA01m');           // buttons for nudge offsets
const nudgeA01p = $('#nudgeA01p');
const nudgeA1m = $('#nudgeA1m');
const nudgeA1p = $('#nudgeA1p');
const nudgeB01m = $('#nudgeB01m');
const nudgeB01p = $('#nudgeB01p');
const nudgeB1m = $('#nudgeB1m');
const nudgeB1p = $('#nudgeB1p');

let hlsA = null, hlsB = null;
let offsets = { A:0, B:0 };
let duration = 0;
let seeking = false;
let dragging = false;
let rafId = null;
let lastSync = 0;
let softSyncUntilA = 0;
let softSyncUntilB = 0;
let lastStatusTrim = 0;

// ==== Audio selection state (A | B | none)
let audioSource = 'A';
function applyAudioRouting(){
  if (vidA) vidA.muted = (audioSource!=='A');
  if (vidB) vidB.muted = (audioSource!=='B');
  setStatus(`Audio source: ${audioSource.toUpperCase()}`, 'ok');
}
function setAudioSource(src){
  audioSource = src;
  applyAudioRouting();
}

// ==== Diagnostics: build minimal UI at runtime so no HTML edits needed
let diagPanel, diagBody, diagToggle, copyDiagBtn, diagFab;
(function ensureDiagnosticsUI(){
  // Audio selector buttons area (insert into existing controls if present)
  const controls = document.querySelector('.controls');
  if (controls && !document.querySelector('#audioSelectorGroup')){
    const group = document.createElement('div');
    group.className = 'group';
    group.id = 'audioSelectorGroup';
    group.innerHTML = `
      <span class="pill">Audio:</span>
      <button id="audioSelA" class="btn ghost">A</button>
      <button id="audioSelB" class="btn ghost">B</button>
      <button id="audioSelNone" class="btn ghost">None</button>
    `;
    controls.appendChild(group);
  }

  // Side diagnostics panel
  if (!document.querySelector('#diagPanel')){
    diagPanel = document.createElement('div');
    diagPanel.id = 'diagPanel';
    diagPanel.className = 'diag-panel';
    diagPanel.innerHTML = `
      <div class="diag-header">
        <div class="title">Diagnostics</div>
        <div class="spacer"></div>
        <button id="copyDiag" class="btn ghost">Copy</button>
        <button id="diagToggle" class="btn">Close</button>
      </div>
      <div id="diagBody" class="diag-body"></div>
    `;
    document.body.appendChild(diagPanel);
  } else {
    diagPanel = document.querySelector('#diagPanel');
  }
  diagBody = document.querySelector('#diagBody');
  diagToggle = document.querySelector('#diagToggle');
  copyDiagBtn = document.querySelector('#copyDiag');

  // Floating debug button
  if (!document.querySelector('#diagOpenFab')){
    diagFab = document.createElement('button');
    diagFab.id = 'diagOpenFab';
    diagFab.className = 'diag-fab btn';
    diagFab.textContent = 'Debug';
    document.body.appendChild(diagFab);
  } else {
    diagFab = document.querySelector('#diagOpenFab');
  }

  // Wire events
  const audioSelA = document.querySelector('#audioSelA');
  const audioSelB = document.querySelector('#audioSelB');
  const audioSelNone = document.querySelector('#audioSelNone');

  audioSelA?.addEventListener('click', ()=> setAudioSource('A'));
  audioSelB?.addEventListener('click', ()=> setAudioSource('B'));
  audioSelNone?.addEventListener('click', ()=> setAudioSource('none'));

  diagFab?.addEventListener('click', ()=> diagPanel?.classList.add('open'));
  diagToggle?.addEventListener('click', ()=> diagPanel?.classList.remove('open'));
  copyDiagBtn?.addEventListener('click', ()=>{
    const s = JSON.stringify(gatherStats(), null, 2);
    navigator.clipboard?.writeText(s).then(()=> setStatus('Diagnostics copied','ok')).catch(()=>{});
  });
})();

// ==== Diagnostics error buffer
let diagnosticsErrors = [];
function pushDiagError(msg){
  diagnosticsErrors.push({ t: new Date().toISOString(), msg });
  if (diagnosticsErrors.length > 100) diagnosticsErrors.shift();
}

function fmt(t){
  if(!isFinite(t)||t<0) t=0;
  const m = Math.floor(t/60);
  const s = Math.floor(t%60);
  return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}
function setStatus(msg, level){
  const span = document.createElement('span');
  span.textContent = msg;
  span.className = level || '';
  statusEl.appendChild(span);
  while(statusEl.children.length>8) statusEl.removeChild(statusEl.firstChild);
}
function isHls(u){ return /\.m3u8(\?|$)/i.test(u); }

function attach(src, video, which){
  return new Promise((resolve,reject)=>{
    try{
      if (window.Hls && window.Hls.isSupported() && isHls(src)) {
        const hls = new window.Hls({
          enableWorker:true,
          lowLatencyMode:false,
          backBufferLength: 60,
          liveSyncDurationCount: 3
        });
        let retries = 0;
        const maxRetries = 3;

        hls.loadSource(src);
        hls.attachMedia(video);

        hls.on(window.Hls.Events.MANIFEST_PARSED, ()=>{
          setStatus(`${which}: manifest parsed`, 'ok');
          resolve(hls);
        });

        hls.on(window.Hls.Events.ERROR, (ev,data)=>{
          console.warn(which,'hls.js error',data);
          const fatal = data?.fatal;
          const type = data?.type;
          const details = data?.details || 'hls error';
          setStatus(`${which}: ${details}`, fatal?'err':'warn');
          pushDiagError(`${which} hls: ${details} fatal=${!!fatal} type=${type||'-'}`);

          if (!fatal) return;

          if (type === window.Hls.ErrorTypes.NETWORK_ERROR && retries < maxRetries){
            retries++;
            const backoff = 500 * retries;
            setStatus(`${which}: network retry ${retries}`, 'warn');
            setTimeout(()=> hls.startLoad(), backoff);
            return;
          }

          if (type === window.Hls.ErrorTypes.MEDIA_ERROR){
            setStatus(`${which}: recovering media error`, 'warn');
            try{ hls.recoverMediaError(); }catch{}
            return;
          }

          reject(new Error(`fatal hls error: ${details}`));
        });
      } else {
        video.src = src;
        video.addEventListener('loadedmetadata', ()=> {
          setStatus(`${which}: metadata loaded (native)`, 'ok');
          resolve(null);
        }, { once:true });
        video.addEventListener('error', ()=> {
          pushDiagError(`${which} native: media error event`);
          reject(new Error('native error loading'));
        }, { once:true });
      }

      // Media element diagnostics
      const onMediaEvt = (e)=> pushDiagError(`${which} media: ${e.type}`);
      ['stalled','suspend','waiting','emptied','abort','error'].forEach(evt=>{
        video.addEventListener(evt, onMediaEvt);
      });
    } catch(e){
      reject(e);
    }
  });
}

function destroyHls(hls){
  if(hls){
    try{ hls.destroy(); }catch{}
  }
}

function updateDuration(){
  const dA = isFinite(vidA.duration) ? vidA.duration : 0;
  const dB = isFinite(vidB.duration) ? vidB.duration : 0;
  duration = Math.max(dA, dB);
}

function setBothCurrentTime(t){
  // Optional frame-snapping per stream
  const snap = !!(snapChk && snapChk.checked);
  const fpsA = parseFps(fpsAInput?.value);
  const fpsB = parseFps(fpsBInput?.value);

  let tA = t + offsets.A;
  let tB = t + offsets.B;

  if (snap){
    if (isFinite(vidA.duration) && fpsA>0) tA = snapTime(tA, fpsA);
    if (isFinite(vidB.duration) && fpsB>0) tB = snapTime(tB, fpsB);
  }

  if (isFinite(vidA.duration)) vidA.currentTime = clamp(tA, 0, vidA.duration);
  if (isFinite(vidB.duration)) vidB.currentTime = clamp(tB, 0, vidB.duration);
}

function clamp(v, a, b){ return Math.min(Math.max(v,a),b); }
function parseFps(s){
  if (!s) return NaN;
  const x = Number(s);
  if (Number.isFinite(x)) return x;
  const m = String(s).match(/^(\d+)\s*\/\s*(\d+)$/);
  if (m){ return Number(m[1]) / Number(m[2]); }
  if (String(s).toLowerCase()==='29.97') return 30000/1001;
  if (String(s).toLowerCase()==='59.94') return 60000/1001;
  if (String(s).toLowerCase()==='23.976') return 24000/1001;
  return NaN;
}
function snapTime(t, fps){
  if (!fps || !isFinite(fps)) return t;
  const frame = Math.round(t * fps);
  return frame / fps;
}

function getUnifiedTime(){
  if (isFinite(vidA.currentTime)) return Math.max(0, vidA.currentTime - offsets.A);
  if (isFinite(vidB.currentTime)) return Math.max(0, vidB.currentTime - offsets.B);
  return 0;
}

function computeErrors(){
  const ref = getUnifiedTime();
  const a = isFinite(vidA.currentTime) ? vidA.currentTime : 0;
  const b = isFinite(vidB.currentTime) ? vidB.currentTime : 0;
  return {
    ref,
    a, b,
    aErr: (a - (ref + offsets.A)),
    bErr: (b - (ref + offsets.B))
  };
}

function isBuffered(video, t, slack=0.2){
  try{
    const br = video.buffered;
    for (let i=0;i<br.length;i++){
      const s=br.start(i), e=br.end(i);
      if (t >= s - 0.05 && t <= e - 0.05) return true;
    }
  } catch {}
  return false;
}

// Modified softSync: never rate-nudge the audible stream (A or B). Only snap if needed.
// For the muted stream, use gentler rate adjustments to reduce artifacts.
function softSync(video, err, which){
  // err in seconds relative to where it should be: positive => video ahead
  const now = performance.now();
  const ahead = err > 0;
  const mag = Math.abs(err);

  const target = (which==='A' ? getUnifiedTime()+offsets.A : getUnifiedTime()+offsets.B);
  if (!isBuffered(video, target)) return;

  const hardSnap = 0.18;   // >180ms => snap
  const nudgeLow = 0.05;   // gentle window
  const nudgeHigh = 0.16;

  const isAudible = (audioSource==='A' && which==='A') || (audioSource==='B' && which==='B');

  // Audible stream: avoid playbackRate changes; only snap if notably off.
  if (isAudible){
    if (mag >= hardSnap || video.seeking){
      video.currentTime -= err; // snap
    }
    return;
  }

  // Muted stream behavior (safe to nudge)
  if (mag >= hardSnap || video.seeking){
    video.currentTime -= err;
    return;
  }
  if (mag >= nudgeLow && mag < nudgeHigh){
    const rate = ahead ? 0.992 : 1.008; // gentler ±0.8%
    video.playbackRate = rate;
    const until = now + 320;
    if (which==='A') softSyncUntilA = until; else softSyncUntilB = until;
    setTimeout(()=>{ video.playbackRate = 1.0; }, 340);
  }
}

function syncIfNeeded(){
  const { aErr, bErr } = computeErrors();
  const thresh = 0.06; // if above this, consider soft sync
  if (Math.abs(aErr) > thresh && !vidA.seeking) softSync(vidA, aErr, 'A');
  if (Math.abs(bErr) > thresh && !vidB.seeking) softSync(vidB, bErr, 'B');
}

function updateHud(){
  if (!driftHud) return;
  const { ref, a, b, aErr, bErr } = computeErrors();
  const f = (x)=> (isFinite(x)? x.toFixed(3) : '--');
  driftHud.textContent =
    `t=${f(ref)}s | A=${f(a)} (err ${f(aErr)}s) | B=${f(b)} (err ${f(bErr)}s)`;
}

function updateBufferBars(){
  if (!bufferBar || !barA || !barB || !isFinite(duration) || duration<=0) return;
  const rect = bufferBar.getBoundingClientRect();
  const width = rect.width || 1;
  const upd = (video, bar) => {
    try{
      const br = video.buffered;
      if (br.length===0){ bar.style.width = '0%'; return; }
      const ct = video.currentTime || 0;
      let sel = {start: br.start(0), end: br.end(0)};
      for (let i=0;i<br.length;i++){
        const s=br.start(i), e=br.end(i);
        if (ct >= s-0.05 && ct <= e+0.05){ sel={start:s,end:e}; break; }
      }
      const pStart = clamp(sel.start / duration, 0, 1);
      const pEnd = clamp(sel.end / duration, 0, 1);
      bar.style.left = `${pStart*100}%`;
      bar.style.width = `${Math.max(0,(pEnd-pStart))*100}%`;
    }catch{
      bar.style.width = '0%';
    }
  };
  upd(vidA, barA);
  upd(vidB, barB);
}

function tick(){
  const { ref } = computeErrors();
  updateDuration();

  // Slider
  if (!seeking && isFinite(duration) && duration>0) {
    seek.value = Math.round((ref/duration)*1000);
  }
  // Label
  timeLabel.textContent = `${fmt(ref)} / ${fmt(duration||0)}`;

  // Periodic sync
  const now = performance.now();
  if (now - lastSync > 250) {
    syncIfNeeded();
    lastSync = now;
  }

  // Normalize playback rates after soft sync windows
  if (now > softSyncUntilA && vidA.playbackRate !== 1.0) vidA.playbackRate = 1.0;
  if (now > softSyncUntilB && vidB.playbackRate !== 1.0) vidB.playbackRate = 1.0;

  // Ensure audible stream is always 1.0 to avoid audio warble
  if (audioSource==='A' && vidA.playbackRate !== 1.0) vidA.playbackRate = 1.0;
  if (audioSource==='B' && vidB.playbackRate !== 1.0) vidB.playbackRate = 1.0;

  // HUD + buffers
  updateHud();
  updateBufferBars();

  // Diagnostics
  updateDiagnosticsUI();

  rafId = requestAnimationFrame(tick);
}

function playBoth(){
  normalizeRates();
  applyAudioRouting(); // ensure only one audio active from first frame
  const p1 = vidA.play().catch(()=>{});
  const p2 = vidB.play().catch(()=>{});
  Promise.allSettled([p1,p2]).finally(()=>{
    playPause.textContent='Pause';
    hideOverlay();
  });
}
function pauseBoth(){ vidA.pause(); vidB.pause(); playPause.textContent='Play'; }

function normalizeRates(){
  vidA.playbackRate = 1.0;
  vidB.playbackRate = 1.0;
}

function applyOffsets(){
  const a = parseFloat(offsetAEl.value || '0') || 0;
  const b = parseFloat(offsetBEl.value || '0') || 0;
  offsets.A = a; offsets.B = b;
  const ref = getUnifiedTime();
  setBothCurrentTime(ref);
  setStatus(`Offsets applied A=${a.toFixed(2)}s, B=${b.toFixed(2)}s`,'ok');
  updatePermalink();
}

function nudge(which, delta){
  const el = which==='A' ? offsetAEl : offsetBEl;
  const v = (parseFloat(el.value || '0') || 0) + delta;
  el.value = v.toFixed(2);
  applyOffsets();
}

function updatePermalink(){
  try{
    const u = new URL(location.href);
    u.searchParams.set('a', urlA.value);
    u.searchParams.set('b', urlB.value);
    if (offsetAEl.value) u.searchParams.set('oa', offsetAEl.value);
    if (offsetBEl.value) u.searchParams.set('ob', offsetBEl.value);
    if (snapChk?.checked) u.searchParams.set('snap','1'); else u.searchParams.delete('snap');
    if (fpsAInput?.value) u.searchParams.set('fa', fpsAInput.value);
    if (fpsBInput?.value) u.searchParams.set('fb', fpsBInput.value);
    permalink.href = u.toString();
  }catch{}
}

// Slider (wipe)
function setWipe(percent){
  percent = Math.max(0, Math.min(1, percent));
  const rightInset = (1 - percent) * 100;
  vidB.style.clipPath = `inset(0 ${rightInset}% 0 0)`;
  handle.style.left = `${percent*100}%`;
  knob.style.left = `${percent*100}%`;
}
function initDrag(){
  const rect = () => viewport.getBoundingClientRect();
  const onMove = (clientX) => {
    const r = rect();
    const p = (clientX - r.left) / r.width;
    setWipe(p);
  };
  const down = (e) => {
    e.preventDefault();
    dragging = true;
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    onMove(cx);
    window.addEventListener('mousemove', move);
    window.addEventListener('touchmove', move, {passive:false});
    window.addEventListener('mouseup', up, {once:true});
    window.addEventListener('touchend', up, {once:true});
  };
  const move = (e) => {
    if (!dragging) return;
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    onMove(cx);
  };
  const up = () => {
    dragging = false;
    window.removeEventListener('mousemove', move);
    window.removeEventListener('touchmove', move);
  };
  knob.addEventListener('mousedown', down);
  knob.addEventListener('touchstart', down, {passive:false});
  handle.addEventListener('mousedown', down);
  handle.addEventListener('touchstart', down, {passive:false});
  viewport.addEventListener('mousedown', (e)=>{
    if (e.target===knob || e.target===handle) return;
    const r = viewport.getBoundingClientRect();
    const p = (e.clientX - r.left)/r.width;
    setWipe(p);
  });
  setWipe(0.5);
}

// Seek
seek.addEventListener('input', ()=>{
  if (!isFinite(duration)||duration<=0) return;
  seeking = true;
  let t = (seek.value/1000)*duration;
  setBothCurrentTime(t);
});
seek.addEventListener('change', ()=>{
  seeking = false;
});

playPause.addEventListener('click', ()=>{
  if (vidA.paused && vidB.paused) playBoth();
  else pauseBoth();
});
restartBtn.addEventListener('click', ()=>{
  setBothCurrentTime(0);
  pauseBoth();
  setTimeout(()=> playBoth(), 50);
});
applyOffsetsBtn.addEventListener('click', applyOffsets);

// Offset nudge buttons
[['A', nudgeA01, -0.1], ['A', nudgeA01p, 0.1], ['A', nudgeA1m, -1], ['A', nudgeA1p, 1],
 ['B', nudgeB01m, -0.1], ['B', nudgeB01p, 0.1], ['B', nudgeB1m, -1], ['B', nudgeB1p, 1]
].forEach(([which, el, delta])=>{
  if (!el) return;
  el.addEventListener('click', ()=> nudge(which, delta));
});

// Keyboard
window.addEventListener('keydown', (e)=>{
  if (['INPUT','TEXTAREA'].includes((e.target.tagName||''))) return;
  if (e.code==='Space'){ e.preventDefault(); (vidA.paused&&vidB.paused)?playBoth():pauseBoth(); }
  if (e.key==='k'){ (vidA.paused&&vidB.paused)?playBoth():pauseBoth(); }
  if (e.key==='j'){ const t=getUnifiedTime()-5; setBothCurrentTime(Math.max(0,t)); }
  if (e.key==='l'){ const t=getUnifiedTime()+5; setBothCurrentTime(t); }
});

// Loading
async function load(which){
  const url = (which==='A'? urlA.value.trim(): urlB.value.trim());
  if (!url) { setStatus(`${which}: URL is empty`,'warn'); return; }
  setStatus(`${which}: loading ${url}`);
  try{
    if (which==='A'){
      destroyHls(hlsA);
      vidA.src=''; vidA.load();
      hlsA = await attach(url, vidA, 'A');
    } else {
      destroyHls(hlsB);
      vidB.src=''; vidB.load();
      hlsB = await attach(url, vidB, 'B');
    }
    const v = which==='A'?vidA:vidB;
    await new Promise(res=>{
      if (isFinite(v.duration) && v.duration>0) res();
      else v.addEventListener('loadedmetadata', res, {once:true});
      setTimeout(res, 800);
    });
    updateDuration();
    setStatus(`${which}: loaded`, 'ok');
    updatePermalink();
    normalizeRates();
    applyAudioRouting(); // ensure correct mute state post-load
  }catch(err){
    console.error(err);
    setStatus(`${which}: failed to load (${err?.message||'error'})`,'err');
    pushDiagError(`${which} load failed: ${err?.message||'error'}`);
  }
}

document.getElementById('loadA').addEventListener('click', ()=>load('A'));
document.getElementById('loadB').addEventListener('click', ()=>load('B'));
document.getElementById('loadDemo').addEventListener('click', ()=>{
  urlA.value = 'https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8';
  urlB.value = 'https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8';
  load('A'); load('B');
});

// Play overlay (mobile gesture unlock)
function showOverlay(){ if (playOverlay) playOverlay.classList.add('show'); }
function hideOverlay(){ if (playOverlay) playOverlay.classList.remove('show'); }
playOverlay?.querySelector('button')?.addEventListener('click', ()=>{
  playBoth();
});

// Autoload from query
function parseQuery(){
  try{
    const u = new URL(location.href);
    const a = u.searchParams.get('a');
    const b = u.searchParams.get('b');
    const oa = parseFloat(u.searchParams.get('oa')||'0')||0;
    const ob = parseFloat(u.searchParams.get('ob')||'0')||0;
    const snap = u.searchParams.get('snap')==='1';
    const fa = u.searchParams.get('fa') || '';
    const fb = u.searchParams.get('fb') || '';

    if (a) urlA.value = a;
    if (b) urlB.value = b;
    if (a) load('A');
    if (b) load('B');

    offsetAEl.value = oa;
    offsetBEl.value = ob;
    applyOffsets();

    if (snapChk) snapChk.checked = snap;
    if (fpsAInput) fpsAInput.value = fa;
    if (fpsBInput) fpsBInput.value = fb;
  } catch {}
}

// Diagnostics data collection and rendering
function bufferedRangesToArray(video){
  const out = [];
  try{
    const br = video.buffered;
    for (let i=0;i<br.length;i++) out.push([+(br.start(i).toFixed(3)), +(br.end(i).toFixed(3))]);
  }catch{}
  return out;
}
function hlsInfo(hls){
  if (!hls) return null;
  const c = hls.levels?.[hls.currentLevel] || null;
  return {
    currentLevel: hls.currentLevel,
    nextLevel: hls.nextLevel,
    autoLevelEnabled: hls.autoLevelEnabled,
    bandwidthEstimate: hls.bandwidthEstimate,
    liveLatency: hls.latency,
    levelHeight: c?.height,
    levelWidth: c?.width,
    levelBitrate: c?.bitrate
  };
}
function gatherStats(){
  const { ref, a, b, aErr, bErr } = computeErrors();
  const fpsA = parseFps(fpsAInput?.value);
  const fpsB = parseFps(fpsBInput?.value);
  return {
    time: new Date().toISOString(),
    unifiedTime: +ref.toFixed(3),
    duration: +(duration||0).toFixed(3),
    offsets: { A: offsets.A, B: offsets.B },
    audioSource,
    snap: !!(snapChk?.checked),
    fps: { A: fpsA, B: fpsB },
    A: {
      currentTime: +a.toFixed(3),
      err: +aErr.toFixed(3),
      paused: vidA.paused,
      seeking: vidA.seeking,
      playbackRate: +vidA.playbackRate.toFixed(3),
      readyState: vidA.readyState,
      buffered: bufferedRangesToArray(vidA),
      isBufferedAtTarget: isBuffered(vidA, ref+offsets.A)
    },
    B: {
      currentTime: +b.toFixed(3),
      err: +bErr.toFixed(3),
      paused: vidB.paused,
      seeking: vidB.seeking,
      playbackRate: +vidB.playbackRate.toFixed(3),
      readyState: vidB.readyState,
      buffered: bufferedRangesToArray(vidB),
      isBufferedAtTarget: isBuffered(vidB, ref+offsets.B)
    },
    hls: {
      A: hlsInfo(hlsA),
      B: hlsInfo(hlsB)
    },
    errors: diagnosticsErrors.slice(-20)
  };
}
function renderDiagnostics(stats){
  if (!diagBody) return;
  const esc = (s)=> String(s).replace(/[&<>]/g, c=>({ '&':'&amp;','<':'&lt;','>':'&gt;' }[c]));
  const rows = [];
  rows.push(`<div class="sec"><div class="h">Core</div>
    <div>t=${stats.unifiedTime}s / ${stats.duration}s</div>
    <div>offsets A=${stats.offsets.A}s B=${stats.offsets.B}s</div>
    <div>audio=${stats.audioSource} snap=${stats.snap}</div>
    <div>fps A=${stats.fps.A||'-'} B=${stats.fps.B||'-'}</div>
  </div>`);
  rows.push(`<div class="sec"><div class="h">Stream A</div>
    <div>ct=${stats.A.currentTime}s err=${stats.A.err}s pr=${stats.A.playbackRate} rs=${stats.A.readyState} paused=${stats.A.paused} seeking=${stats.A.seeking} buffered=${esc(JSON.stringify(stats.A.buffered))} atTarget=${stats.A.isBufferedAtTarget}</div>
  </div>`);
  rows.push(`<div class="sec"><div class="h">Stream B</div>
    <div>ct=${stats.B.currentTime}s err=${stats.B.err}s pr=${stats.B.playbackRate} rs=${stats.B.readyState} paused=${stats.B.paused} seeking=${stats.B.seeking} buffered=${esc(JSON.stringify(stats.B.buffered))} atTarget=${stats.B.isBufferedAtTarget}</div>
  </div>`);
  rows.push(`<div class="sec"><div class="h">HLS</div>
    <div>A=${esc(JSON.stringify(stats.hls.A))}</div>
    <div>B=${esc(JSON.stringify(stats.hls.B))}</div>
  </div>`);
  if (stats.errors?.length){
    rows.push(`<div class="sec"><div class="h">Recent errors</div>
      ${stats.errors.map(e=>`<div class="err">${esc(e.t)} — ${esc(e.msg)}</div>`).join('')}
    </div>`);
  }
  diagBody.innerHTML = rows.join('\n');
}
function updateDiagnosticsUI(){
  renderDiagnostics(gatherStats());
}

// Start
initDrag();
parseQuery();
rafId = requestAnimationFrame(tick);
updatePermalink();
showOverlay(); // shown until first play
applyAudioRouting(); // initial mute routing

// Keep permalink fresh
[urlA, urlB, offsetAEl, offsetBEl, snapChk, fpsAInput, fpsBInput].forEach(el=>{
  if (!el) return;
  el.addEventListener('input', updatePermalink);
  el.addEventListener('change', updatePermalink);
});

// Pause on tab hide
document.addEventListener('visibilitychange', ()=>{
  if (document.hidden){ pauseBoth(); }
});