// SPDX-License-Identifier: Apache-2.0
// The live demo's page: the WebRTC call with the laptop app (doom_pipecat.py), and the
// dashboard drawn from the telemetry the GPU side sends (doom_live.py's client view),
// which the laptop forwards over the call's data channel as {"label": "doom", "type": ...}:
//   hello        the schema: labels, colours, the heatmap's colour stops (overlay.schema)
//   tics         per-tic columns {k, s, p, c, d, a} for the two maps (doom_live.tic_column)
//   panel        the panel's fields, five times a second (doom_live.panel_fields)
//   line         one of his lines, with what you said if he is answering you
//   match_start, event (died, frag, match_over), quality (the stream stepped), latency
//   link         the laptop's connection to the GPU side: up or down, its round trip
// Telemetry runs a little ahead of the video (its jitter buffer, ~50-150 ms).
"use strict";

const $ = (id) => document.getElementById(id);
const CAPTION_S = 6; // a spoken line stays over the game this long
const WINDOW_TICS = 350; // the maps show about this many tics (10 s)
const HISTORY = 6000; // tic columns kept, to draw again on a resize
// The heatmap's bands, in units (an action row is 2): behavior, gap, the actions, gap,
// danger, gap, the fresh-decision marks. The activity map: one row per model.
const HEAT_SPEC = (n) => [[3, 1], [1, 0], ...Array(n).fill([2, 1]), [1, 0], [3, 1], [1, 0], [1.5, 1]];

let pc, mic, channel;
let S = null; // the hello
let LUT = [], RGB = {}; // the colour map over p (0-255); palette colours as [r, g, b]
const hist = []; // tic columns, oldest first

const put = (id, text) => {
  const e = $(id);
  if (e.textContent !== text) e.textContent = text;
};
const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
const css = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;

// ── The call ──────────────────────────────────────────────────────────────────
async function start(withMic) {
  $("start").disabled = $("watch").disabled = true;
  (withMic ? $("start") : $("watch")).textContent = "Connecting…";
  try {
    pc = new RTCPeerConnection();
    if (withMic) {
      // The browser's own echo cancellation, noise suppression and gain.
      mic = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      mic.getAudioTracks().forEach((t) => pc.addTrack(t, mic)); // first: audio
    } else {
      pc.addTransceiver("audio", { direction: "recvonly" }); // first: his voice
      $("mute").hidden = true;
    }
    pc.addTransceiver("video", { direction: "recvonly" }); // second: the game
    channel = pc.createDataChannel("app");
    channel.onopen = () => {
      setInterval(() => channel.send("ping"), 1000);
      tell("hello"); // what was sent before the channel opened
    };
    channel.onmessage = (e) => {
      let m;
      try {
        m = JSON.parse(e.data);
      } catch {
        return;
      }
      if (m && m.label === "doom") on(m);
    };
    const remote = new MediaStream();
    $("screen").srcObject = remote;
    pc.ontrack = (e) => {
      remote.addTrack(e.track);
      $("screen").play().catch(() => {});
    };
    pc.onconnectionstatechange = () => put("status", pc.connectionState);
    await pc.setLocalDescription(await pc.createOffer());
    await new Promise((done) => {
      // every candidate in the offer: no trickle
      if (pc.iceGatheringState === "complete") return done();
      pc.onicegatheringstatechange = () => pc.iceGatheringState === "complete" && done();
      setTimeout(done, 3000);
    });
    const r = await fetch("/api/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
    });
    await pc.setRemoteDescription(await r.json());
    $("starts").hidden = true;
    $("bar").hidden = false;
  } catch (err) {
    $("start").disabled = $("watch").disabled = false;
    $("start").textContent = "Start (allow the microphone)";
    $("watch").textContent = "Watch only (no microphone)";
    $("error").textContent = "Could not start: " + err;
  }
}

// Pipecat's RTVI layer reads the data channel: a client-message reaches the pipeline.
const tell = (t, d = {}) =>
  channel &&
  channel.readyState === "open" &&
  channel.send(
    JSON.stringify({ label: "rtvi-ai", type: "client-message", id: String(Date.now()), data: { t, d } }),
  );

// ── What the GPU side sends ───────────────────────────────────────────────────
function on(m) {
  switch (m.type) {
    case "hello":
      return hello(m);
    case "tics":
      return tics(m.cols);
    case "panel":
      return S && panel(m);
    case "line":
      if (m.heard) said("you", m.heard);
      return said("him", m.line);
    case "match_start":
      return newMatch("New match");
    case "event":
      if (m.kind === "match_over") note("Match over");
      return;
    case "quality":
      link.q = m;
      return;
    case "latency":
      link.reply = m.speech_to_audio_ms;
      return;
    case "link":
      link.gpu = m;
      $("linkdown").hidden = m.state === "up";
      put(
        "linkdown",
        {
          connecting: "Connecting to the game…",
          down: "Reconnecting to the game…",
          replaced: "Another window took over the game (reload to take it back)",
        }[m.state] || m.state,
      );
      return;
  }
}

function hello(m) {
  const restarted = S && S.boot !== m.boot;
  S = m;
  document.body.classList.toggle("composited", m.view !== "client");
  for (const [k, v] of Object.entries(m.palette)) {
    document.documentElement.style.setProperty(`--${k}`, v);
    RGB[k] = rgb(v);
  }
  LUT = colormap(m.stops);
  $("ramp").style.background =
    `linear-gradient(to right, ${m.stops.map(([p, c]) => `${c} ${100 * p}%`).join(", ")})`;
  $("ramp").replaceChildren(
    ...[0, 0.05, 0.25, 0.5, 1].map((p) => {
      const s = document.createElement("span");
      s.style.left = `${100 * Math.sqrt(p)}%`;
      s.textContent = String(p);
      return s;
    }),
  );
  put("kind", m.kind ? `${m.kind} adapter` : "");
  put("match-label", `MATCH VS ${m.n_bots} BOTS`);
  $("tic-mark").style.left = `${m.tic_ms}%`; // of a 100 ms bar
  $("tic-label").style.marginLeft = `calc(${m.tic_ms}% + 4px)`;
  put("tic-label", `1 tic ${m.tic_ms.toFixed(1)} ms`);
  if (restarted) newMatch("The game restarted");
  maps.forEach((s) => s.layout());
}

// The colour of each p (0-255): the stops are over sqrt(p), as in overlay.cmap.
function colormap(stops) {
  const xs = stops.map((s) => s[0]),
    cs = stops.map((s) => rgb(s[1]));
  const out = [];
  for (let v = 0; v < 256; v++) {
    const t = Math.sqrt(v / 255);
    let i = 0;
    while (i < xs.length - 2 && t > xs[i + 1]) i++;
    const f = Math.min(1, Math.max(0, (t - xs[i]) / (xs[i + 1] - xs[i])));
    out.push(css(cs[i].map((c, k) => Math.round(c + f * (cs[i + 1][k] - c)))));
  }
  return out;
}

function newMatch(why) {
  hist.length = 0;
  maps.forEach((s) => s.repaint());
  shown.length = 0;
  captions();
  note(why);
}

// ── The maps: which model runs, and the action heatmap ────────────────────────
function tics(cols) {
  hist.push(...cols);
  if (hist.length > HISTORY) hist.splice(0, hist.length - HISTORY);
  if (S) maps.forEach((s) => s.push(cols));
}

// [y0, y1] of each drawn band, in device pixels: whole pixels, so rows stay crisp.
function bands(spec, h) {
  const k = h / spec.reduce((a, [u]) => a + u, 0);
  const out = [];
  let y = 0;
  for (const [u, drawn] of spec) {
    const y0 = Math.round(y * k);
    y += u;
    if (drawn) out.push([y0, Math.round(y * k)]);
  }
  return out;
}

function heatFills(col) {
  const P = S.palette,
    n = S.display_order.length;
  const [bg, am, rd] = [RGB.bg, RGB.amber, RGB.red];
  const pm = col.c[1] / 255,
    ph = col.c[2] / 255;
  const danger = bg.map((b, k) => Math.max(0, Math.min(255, Math.round(b + pm * (am[k] - b) + ph * (rd[k] - b)))));
  return [
    S.colors[col.s] || P.text,
    ...(col.p ? col.p.map((v) => LUT[v]) : Array(n).fill(LUT[0])),
    css(danger),
    col.d ? P.text : P.bg,
  ];
}

const actFills = (col) => S.models.map((m, i) => ((col.a >> i) & 1 ? S.model_colors[m] : S.palette.layer));

class Strip {
  // A map on a canvas at the device's resolution: a label gutter at the left, then one
  // column per tic, newest at the right. It fills from the left, then scrolls: the
  // drawn columns shift left and only the new ones are painted.
  constructor(id, kind) {
    this.cv = $(id);
    this.kind = kind;
    this.ctx = this.cv.getContext("2d", { alpha: false });
    this.win = this.drawn = 0;
    this.dpr = window.devicePixelRatio || 1;
    const ro = new ResizeObserver(([e]) => {
      // The exact device-pixel box when the browser gives one that agrees with the
      // device pixel ratio (an emulated ratio may not), else the CSS box times it.
      const dpr = window.devicePixelRatio || 1,
        r = e.contentRect;
      const box = e.devicePixelContentBoxSize && e.devicePixelContentBoxSize[0];
      const exact = box && Math.abs(box.inlineSize - r.width * dpr) <= 2;
      const w = exact ? box.inlineSize : Math.round(r.width * dpr);
      const h = exact ? box.blockSize : Math.round(r.height * dpr);
      this.dpr = r.width ? w / r.width : dpr;
      if (w !== this.cv.width || h !== this.cv.height) {
        this.cv.width = w;
        this.cv.height = h;
      }
      this.layout();
    });
    try {
      ro.observe(this.cv, { box: "device-pixel-content-box" }); // also on a DPR change
    } catch {
      ro.observe(this.cv);
    }
  }

  layout() {
    if (!S || !this.cv.width || !this.cv.height) return;
    const W = this.cv.width,
      d = this.dpr;
    const gutter = Math.round(Math.min(140, Math.max(92, 0.095 * (W / d)))); // CSS px
    this.gx = Math.round(gutter * d);
    this.cw = Math.max(1, Math.round((W - this.gx) / WINDOW_TICS));
    this.win = Math.floor((W - this.gx) / this.cw);
    if (this.kind === "heat") {
      this.bands = bands(HEAT_SPEC(S.display_order.length), this.cv.height);
      this.gap = 0;
      $("maps").style.setProperty("--gutter", `${gutter}px`);
      put("span", `−${Math.round(this.win / S.tic_hz)} s`);
    } else {
      this.bands = bands(S.models.map(() => [1, 1]), this.cv.height);
      this.gap = Math.max(1, Math.round((0.16 * this.cv.height) / S.models.length));
    }
    this.repaint();
  }

  repaint() {
    if (!this.win) return;
    const c = this.ctx;
    c.fillStyle = S.palette.bg;
    c.fillRect(0, 0, this.cv.width, this.cv.height);
    const shown = hist.slice(-this.win);
    shown.forEach((col, j) => this.paint(col, j));
    this.drawn = shown.length;
    this.labels(shown[shown.length - 1]);
  }

  push(cols) {
    if (!this.win || !cols.length) return;
    const n = cols.length;
    if (this.drawn + n <= this.win) {
      cols.forEach((col, i) => this.paint(col, this.drawn + i));
      this.drawn += n;
    } else if (n >= this.win) {
      return this.repaint();
    } else {
      const sh = (this.drawn + n - this.win) * this.cw,
        x = this.gx,
        w = this.win * this.cw,
        h = this.cv.height;
      this.ctx.drawImage(this.cv, x + sh, 0, w - sh, h, x, 0, w - sh, h);
      cols.forEach((col, i) => this.paint(col, this.win - n + i));
      this.drawn = this.win;
    }
    if (this.kind === "act") this.labels(cols[n - 1]);
  }

  paint(col, j) {
    const c = this.ctx,
      x = this.gx + j * this.cw,
      w = this.cw;
    c.fillStyle = S.palette.bg;
    c.fillRect(x, 0, w, this.cv.height);
    const fills = this.kind === "heat" ? heatFills(col) : actFills(col);
    this.bands.forEach(([y0, y1], i) => {
      c.fillStyle = fills[i];
      c.fillRect(x, y0, w, y1 - y0 - this.gap);
    });
  }

  labels(last) {
    const c = this.ctx,
      d = this.dpr,
      P = S.palette;
    c.fillStyle = P.bg;
    c.fillRect(0, 0, this.gx, this.cv.height);
    c.textBaseline = "middle";
    const x = Math.round(8 * d),
      fit = (this.gx / d - 12) / (11 * 0.62); // 11 characters in the gutter
    const mid = ([y0, y1]) => (y0 + y1) / 2;
    if (this.kind === "heat") {
      const [r0, r1] = this.bands[1],
        row = (r1 - r0) / d;
      const size = Math.min(fit, Math.max(9, Math.min(13, 1.15 * row)));
      const every = Math.ceil((1.1 * size) / row); // rows too thin for a label each
      c.font = `${size * d}px ui-monospace, SFMono-Regular, Menlo, monospace`;
      c.fillStyle = P.text2;
      S.display_order.forEach((a, i) => {
        if (i % every === 0) c.fillText(S.short_labels[a], x, mid(this.bands[i + 1]));
      });
      c.fillStyle = P.help;
      const n = S.display_order.length;
      c.fillText("danger", x, mid(this.bands[n + 1]));
      if ((this.bands[0][1] - this.bands[0][0]) / d >= 8) c.fillText("behavior", x, mid(this.bands[0]));
      return;
    }
    const row = this.cv.height / d / S.models.length;
    const size = Math.min(fit, Math.max(8, Math.min(14, 0.8 * row)));
    c.font = `${size * d}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    S.models.forEach((m, i) => {
      const on = last && (last.a >> i) & 1,
        y = mid(this.bands[i]) - this.gap / 2;
      if (on) {
        c.fillStyle = S.model_colors[m];
        c.fillRect(Math.round(1 * d), Math.round(y - 2.5 * d), Math.round(4 * d), Math.round(5 * d));
      }
      c.fillStyle = on ? S.model_colors[m] : P.help;
      c.fillText(S.model_labels[m], x, y);
    });
  }
}

const maps = [new Strip("act", "act"), new Strip("heat", "heat")];

// ── The panel ─────────────────────────────────────────────────────────────────
function panel(m) {
  const P = S.palette;
  $("dot").style.background = S.colors[m.adapter] || P.text;
  put("style", m.adapter);
  const o = m.order;
  $("order").hidden = !o;
  if (o) {
    put("o-told", o.told);
    put("o-status", o.status);
    $("o-status").style.color = S.order_colors[o.status] || P.text2;
    put("o-why", o.why || "");
  }
  put("ms", m.ms.toFixed(1));
  put("p50", `run p50 ${m.p50.toFixed(1)} ms`);
  put("p99", `run p99 ${m.p99.toFixed(1)} ms`);
  $("budget-fill").style.width = `${Math.min(100, m.ms)}%`; // of 100 ms
  put("action", S.action_labels[m.action] || m.action);
  put("action-key", m.action);
  const crit = S.danger_levels.map((k) => m.critic[k] || 0);
  const tot = crit.reduce((a, b) => a + b, 0) || 1;
  [...$("critic").children].forEach((e, i) => (e.style.width = `${(100 * crit[i]) / tot}%`));
  put("critic-text", S.danger_levels.map((k, i) => `${k} ${Math.round((100 * crit[i]) / tot)}%`).join("    "));
  const p = m.plan;
  put(
    "plan",
    p
      ? `${S.weapon_names[p.slot]} (slot ${p.slot}, ${Math.round(100 * p.p)}%)   ·   holding ${m.weapon}`
      : `holding ${m.weapon}`,
  );
  const st = m.stats;
  put("s-hp", String(m.hp));
  put("s-armor", String(m.armor));
  put("s-frags", String(st.frags));
  put("s-deaths", String(st.deaths));
  put("s-rank", String(st.rank));
  put("s-best", String(st.best_bot[1]));
  $("s-best").title = st.best_bot[0];
  put("input", m.text);
  put("foot", `tic ${m.tick}  ·  ${S.gpu}`);
}

// ── His lines: the conversation, and captions over the game ───────────────────
const shown = []; // [time, who, text], for the captions

function entry(cls, label, text) {
  const e = document.createElement("div");
  e.className = cls;
  if (label) {
    const b = document.createElement("b");
    b.textContent = label;
    e.append(b);
  }
  e.append(text);
  return e;
}

function append(e) {
  const box = $("lines");
  const atEnd = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  box.append(e);
  while (box.children.length > 300) box.firstChild.remove();
  if (atEnd) box.scrollTop = box.scrollHeight;
}

function said(who, text) {
  text = text.replace(/\[[a-z ]+\]\s*/g, "").trim(); // sound tags are voiced, not shown
  if (!text) return;
  const label = who === "you" ? "YOU" : "GRANITE";
  append(entry(who, label, text));
  shown.push([performance.now(), who, text]);
  if (shown.length > 10) shown.shift();
  captions();
}

const note = (text) => append(entry("note", "", text));

function captions() {
  const now = performance.now();
  const recent = shown.filter(([t]) => now - t < 1000 * CAPTION_S).slice(-2);
  $("caps").replaceChildren(
    ...recent.map(([, who, text]) => entry(who, who === "you" ? "YOU:" : "GRANITE:", ` ${text}`)),
  );
}
setInterval(captions, 500);

// ── The link: the call's round trip and bitrate, and the GPU side's ───────────
const link = { gpu: null, q: null, reply: null, prev: null };
$("link").append(document.createElement("i"), document.createElement("span"));

async function linkStats() {
  if (!pc || pc.connectionState !== "connected") return;
  let rtt = null,
    bytes = 0,
    ts = 0,
    fps = null;
  (await pc.getStats()).forEach((s) => {
    if (s.type === "candidate-pair" && s.nominated && s.state === "succeeded" && s.currentRoundTripTime != null)
      rtt = 1000 * s.currentRoundTripTime;
    if (s.type === "inbound-rtp" && s.kind === "video") {
      bytes = s.bytesReceived;
      ts = s.timestamp;
      fps = s.framesPerSecond;
    }
  });
  const kbps = link.prev && ts > link.prev.ts ? (8 * (bytes - link.prev.bytes)) / (ts - link.prev.ts) : null;
  link.prev = { bytes, ts };
  const g = link.gpu,
    q = link.q;
  const parts = [
    `call ${rtt == null ? "–" : Math.round(rtt)} ms`,
    kbps == null ? "" : `${(kbps / 1000).toFixed(2)} Mbps`,
    fps == null ? "" : `${Math.round(fps)} fps`,
    "│",
    !g || g.state !== "up" ? `GPU ${g ? g.state : "–"}` : `GPU ${g.rtt == null ? "–" : g.rtt} ms`,
    q ? `q${q.q}/${q.fps}` : "",
    link.reply == null ? "" : `· reply ${(link.reply / 1000).toFixed(1)} s`,
  ];
  $("link").lastChild.textContent = parts.filter(Boolean).join(" ");
  const slow = (rtt != null && rtt > 250) || (g && g.rtt > 250) || (q && q.level > 0);
  const state = !g || g.state !== "up" ? "red" : slow ? "amber" : "green";
  $("link").firstChild.style.background = `var(--${state})`;
}
setInterval(() => linkStats().catch(() => {}), 1000);

// ── The controls ──────────────────────────────────────────────────────────────
// Leaving (a reload, a closed tab): end the call now, so the laptop's pipeline ends at
// once and not only when the call times out.
window.addEventListener("pagehide", () => pc && pc.close());
$("start").onclick = () => start(true);
$("watch").onclick = () => start(false);
$("mute").onclick = () => {
  const t = mic.getAudioTracks()[0];
  t.enabled = !t.enabled;
  $("mute").textContent = t.enabled ? "Mute mic" : "Unmute mic";
};
$("reset").onclick = () => tell("reset");
$("sfx").onclick = () => {
  // the game's sound, mixed under his voice on the laptop
  tell("sfx");
  $("sfx").textContent = $("sfx").textContent.endsWith("on") ? "Game sound: off" : "Game sound: on";
};
// Its volume: a gain from 0 to 0.6 (the slider's percent; 18 is the default, -15 dB).
$("sfxvol").oninput = () => tell("sfx_gain", { gain: $("sfxvol").value / 100 });
$("pixels").onclick = () => {
  const on = $("screen").classList.toggle("pixelated");
  $("pixels").textContent = on ? "Pixels: sharp" : "Pixels: smooth";
};
$("full").onclick = () =>
  document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
