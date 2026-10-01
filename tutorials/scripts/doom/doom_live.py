# SPDX-License-Identifier: Apache-2.0
"""The live demo's GPU side: one match in real time that you watch and talk to.

The game, the speech-enabled checkpoint (vLLM ``AsyncLLM``: every game adapter,
the narrator, the ASR cascade) and the player's voice run on one GPU node; a
laptop (``doom_pipecat.py``) connects over one websocket, carried by ``ssh -L``,
and turns it into WebRTC for the browser. What you say arrives as one audio
segment per utterance (the laptop's VAD cuts them); the model's ASR transcribes
it inside the narrator's request (:meth:`engine.Game.player_said`), and the
narrator answers. Without you he speaks soon after a salient event or after a
silence (:class:`talk.TalkClock`).

Processes: this one (the decision loop, the websocket server), vLLM's engine
core, the game (:func:`engine.game_worker`, publishing its frames to shared
memory), a renderer that draws each frame as in the videos (:mod:`overlay`)
and JPEG-encodes it at ``--fps``, and the voice (``voice_video.py --serve`` in
a TTS environment: Kokoro by default, Chatterbox with ``--tts``). Drawing stays
off the decision loop.

The websocket, at ``/ws`` (one client at a time; a new one replaces the old):

* down: ``b"J" + JPEG`` (a frame); ``b"A" + line id (4 bytes) + int16 PCM`` (a
  piece of his line); JSON ``{"type": "hello" | "line" | "heard" | "audio_end"
  | "event" | "latency", ...}``.
* up: ``b"U" + int16 PCM, 16 kHz`` (one utterance); JSON ``{"type":
  "speaking"}`` (you started: he holds his remarks and drops what he had not
  said yet), ``{"type": "reset"}`` (a new match).

::

    # his voice: Kokoro's am_michael, 2 semitones down
    python doom_live.py serve --model models/doom-h-alora-narr3-audio --port 8765 \\
        --tts-python kokoro-env/bin/python
    # or Chatterbox-Turbo, cloning a clip
    python doom_live.py serve --model models/doom-h-alora-narr3-audio --port 8765 \\
        --tts turbo --tts-python tts-env/bin/python --voice-ref voices/him.wav
    # no laptop: a scripted check, with a spoken question after a death
    python doom_live.py smoke --url ws://localhost:8765/ws --questions out/b0
"""

from __future__ import annotations

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"  # as engine.py

import argparse
import asyncio
import io
import json
import multiprocessing as mp
import queue
import random
import re
import socket
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from doom_env import TIC_HZ
from engine import AUDIO_HZ, AsyncPolicy, Game, SharedFrame
from expert import BEHAVIORS
from policy import ARMS, CRITIC, NARRATOR, ROUTER
from talk import sound_tag

TTS_AUTHKEY = b"granite-switch-doom-tts"  # voice_video.TTS_AUTHKEY
MODELS = ["base", *BEHAVIORS, ARMS, CRITIC, ROUTER, NARRATOR]
_SENTENCE = re.compile(r"(?<=[.?!])\s+")


# ── The renderer (its own process) ─────────────────────────────────────────────
def render_loop(frames_name: str, inbox, out, opts: dict) -> None:
    """Draw the newest game frame with the telemetry the server sends, as the
    videos do, and send it out as JPEG, ``opts["fps"]`` times a second."""
    from overlay import KINDS, Overlay

    shared = SharedFrame(frames_name)
    view = Overlay(MODELS)
    caps: list[tuple[float, str, str]] = []
    lat: deque[float] = deque(maxlen=1000)
    last_second: deque[float] = deque(maxlen=TIC_HZ)
    info = {
        "instruction": None,
        "adapter": BEHAVIORS[0],
        "kind": KINDS.get(opts["placement"], ""),
        "action": "wait",
        "critic": {},
        "plan": None,
        "stats": {"frags": 0, "deaths": 0, "rank": 1, "best_bot": ["", 0]},
        "hp": 100,
        "armor": 0,
        "weapon": "pistol",
        "text": "",
        "gpu": opts["gpu"],
    }
    last_tick, talk_until = -1, -1
    period, due = 1.0 / opts["fps"], time.perf_counter()
    w, h = view.size
    size = (round(w * opts["scale"]) // 2 * 2, round(h * opts["scale"]) // 2 * 2)
    while True:
        while True:
            try:
                msg = inbox.get_nowait()
            except queue.Empty:
                break
            if msg[0] == "stop":
                return
            if msg[0] == "new":  # a new match
                view, caps, last_tick = Overlay(MODELS), [], -1
                continue
            if msg[0] == "cap":
                caps = caps[-6:] + [(msg[1] / TIC_HZ, msg[2], msg[3])]
                continue
            if msg[0] == "talk":
                talk_until = msg[1]
                continue
            _, tick, d, ms, state, stats = msg  # a decision
            style = info["adapter"]
            for t in range(last_tick + 1, tick):  # tics with no decision (dead, busy)
                view.heat.push({}, style, False, info["critic"])
                view.act.push({"base", NARRATOR} if t <= talk_until else set())
            last_tick = tick
            probs = d[style][1]
            critic = d[CRITIC][1] if CRITIC in d else {}
            view.heat.push(probs, style, True, critic)
            active = {"base", *d} | ({NARRATOR} if tick <= talk_until else set())
            view.act.push(active)
            lat.append(ms)
            last_second.append(ms)
            hud = re.match(
                r"now \S+ \| hp (\d+) armor (\d+) \| .*? \| (\w+) \d+ \|", state
            )
            info.update(
                action=d[style][0],
                critic=critic,
                stats=stats,
                text=state,
                tick=tick,
                ms=float(np.median(last_second)),
                p50=float(np.percentile(lat, 50)),
                p99=float(np.percentile(lat, 99)),
            )
            if ARMS in d:
                info["plan"] = {"slot": int(d[ARMS][0]), "probs": d[ARMS][1]}
            if hud:
                info.update(
                    hp=int(hud.group(1)), armor=int(hud.group(2)), weapon=hud.group(3)
                )
        now = time.perf_counter()
        if now < due:
            time.sleep(min(due - now, 0.005))
            continue
        due = max(due + period, now)
        got = shared.get()
        if got is None or "ms" not in info:
            continue
        tick, frame = got
        img = view.draw(frame, {**info, "tick": tick}, caps, tick / TIC_HZ)
        if size != (w, h):
            img = img.resize(size)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=opts["quality"])
        out.send_bytes(buf.getvalue())


# ── His voice (a process in the TTS environment) ───────────────────────────────
class Voice:
    """``voice_video.py --serve`` in the TTS environment, one line at a time."""

    def __init__(self, args):
        from multiprocessing.connection import Client

        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        cmd = [args.tts_python, str(HERE / "voice_video.py"), "--serve", str(port)]
        cmd += ["--tts", args.tts, "--pitch", str(args.pitch)]
        if args.tts == "kokoro":
            cmd += ["--kokoro-voice", args.kokoro_voice, "--speed", str(args.speed)]
        elif args.voice_ref:
            cmd += ["--ref", str(args.voice_ref)]
        self.proc = subprocess.Popen(cmd)
        t0 = time.time()
        while True:
            try:
                self.conn = Client(("127.0.0.1", port), authkey=TTS_AUTHKEY)
                break
            except ConnectionRefusedError:
                if self.proc.poll() is not None or time.time() - t0 > 600:
                    raise SystemExit("the voice did not start") from None
                time.sleep(1)
        self.lock = asyncio.Lock()

    def _say(self, text: str):
        self.conn.send(("say", text))
        _, pcm, sr, ms = self.conn.recv()
        return pcm, sr, ms

    async def say(self, text: str) -> tuple[bytes, int, float]:
        async with self.lock:
            return await asyncio.get_running_loop().run_in_executor(
                None, self._say, text
            )

    def close(self) -> None:
        self.conn.close()
        self.proc.terminate()


# ── The service ────────────────────────────────────────────────────────────────
class Live:
    def __init__(self, args, pol: AsyncPolicy, voice: Voice | None, gpu: str):
        self.args, self.pol, self.voice = args, pol, voice
        ctx = mp.get_context("spawn")
        self.frames = SharedFrame()
        self.tele = ctx.Queue()  # to the renderer; put() never waits
        jpeg_r, jpeg_w = ctx.Pipe(duplex=False)
        self.jpeg = jpeg_r
        opts = {
            "fps": args.fps,
            "scale": args.scale,
            "quality": args.quality,
            "gpu": gpu,
            "placement": pol.kit.placement,
        }
        self.renderer = ctx.Process(
            target=render_loop,
            args=(self.frames.name, self.tele, jpeg_w, opts),
            daemon=True,
        )
        self.renderer.start()
        self.ws = None
        self.game: Game | None = None
        self.seed = args.seed
        self.line_id = 0
        self.cancelled: set[int] = set()
        self.said_at: deque[float] = deque()  # when each utterance arrived
        self.last_death = None
        self.rng = random.Random(args.seed)  # which lines get a sound tag
        self._sending = False
        self._next_frame: bytes | None = None
        # What happened in the last stats window (see stats()).
        self.n = {"frames": 0, "frame_kb": 0, "voiced": 0, "heard": 0}
        self.ms: list[float] = []
        self._tasks: set[asyncio.Task] = set()

    def spawn(self, coro) -> None:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    def tell(self, msg) -> None:
        self.tele.put(msg)

    async def send(self, data) -> None:
        ws = self.ws
        if ws is None or ws.closed:
            return
        try:
            if isinstance(data, bytes):
                await ws.send_bytes(data)
            else:
                await ws.send_str(json.dumps(data))
        except (ConnectionError, RuntimeError):
            pass

    # Frames: the newest one wins when the link is slow.
    def on_jpeg(self) -> None:
        data = self.jpeg.recv_bytes()
        if self._sending:
            self._next_frame = data
            return
        self._sending = True
        self.spawn(self._send_frames(data))

    async def _send_frames(self, data: bytes) -> None:
        while data is not None:
            await self.send(b"J" + data)
            self.n["frames"] += 1
            self.n["frame_kb"] += len(data) // 1024
            data, self._next_frame = self._next_frame, None
        self._sending = False

    # The match.
    async def new_game(self, start: bool = True) -> None:
        """A new match; ``start=False`` loads it and holds its clock until a
        client connects (loading takes a while)."""
        if self.game is not None and not self.game.done:
            self.game.proc.kill()
        a = self.args
        spec = {
            "seed": self.seed,
            "seconds": a.seconds,
            "bots": a.bots,
            "n_bots": a.n_bots,
            "frames": self.frames.name,
        }
        self.seed += 1
        g = Game(0, spec, self.pol, a.idle_s, autostart=start)
        g.on_decision, g.on_line = self.on_decision, self.on_line
        self.game, self.last_death = g, None
        self.tell(("new",))
        g.proc.start()
        self.spawn(self._play(g))
        print(f"match {spec['seed']} {'started' if start else 'loaded'}", flush=True)

    async def _play(self, g: Game) -> None:
        await g.run()
        if g is not self.game:
            return
        watched = self.ws is not None and not self.ws.closed
        if watched:
            await self.send({"type": "event", "kind": "match_over", "stats": g.stats})
            await asyncio.sleep(5)
        if g is self.game:
            await self.new_game(start=watched)

    async def stats(self, every_s: float = 10.0) -> None:
        """Every ``every_s``: decision latency in the window (as the panel
        measures it: from asking the engine to having the answer, in this
        process), with what else this process did (frames streamed, lines
        voiced, utterances transcribed), and how late this event loop runs."""
        while True:
            t0 = time.perf_counter()
            await asyncio.sleep(every_s)
            lag = time.perf_counter() - t0 - every_s
            ms, self.ms = self.ms, []
            n = dict(self.n)
            self.n = dict.fromkeys(self.n, 0)
            if not ms:
                continue
            print(
                f"[stats] decisions {len(ms) / every_s:4.1f}/s p50 {np.percentile(ms, 50):5.1f} "
                f"ms p90 {np.percentile(ms, 90):5.1f} p99 {np.percentile(ms, 99):5.1f} | "
                f"frames {n['frames'] / every_s:4.1f}/s ({n['frame_kb'] / every_s:5.0f} KB/s) "
                f"| voiced {n['voiced']} heard {n['heard']} | loop lag {1000 * lag:.0f} ms",
                flush=True,
            )

    def on_decision(self, tick: int, d: dict, ms: float) -> None:
        self.ms.append(ms)
        f = self.game.facts or {}
        board = f.get("board") or [["", 0]]
        me = f.get("frags", 0)
        stats = {
            "frags": me,
            "deaths": f.get("deaths", 0),
            "rank": 1 + sum(n > me for _, n in board),
            "best_bot": board[0],
        }
        if self.game.talking:
            self.tell(("talk", tick + 1))
        self.tell(("dec", tick, d, ms, self.game.state or "", stats))
        death = f.get("last_death")
        if death and death != self.last_death:
            self.last_death = death
            self.spawn(self.send({"type": "event", "kind": "died", **death}))

    def on_line(self, x: dict) -> None:
        self.line_id += 1
        lid = self.line_id
        if x["heard"]:
            self.tell(("cap", x["tick"], "player", x["heard"]))
        self.tell(("cap", x["tick"], "bot", x["line"]))
        since = None
        if x["cue"] == "partner" and self.said_at:
            since = time.perf_counter() - self.said_at.popleft()
        self.spawn(self._speak(lid, x, since))

    async def _speak(self, lid: int, x: dict, since: float | None) -> None:
        t0 = time.perf_counter()
        await self.send(
            {
                "type": "line",
                "id": lid,
                **{k: x[k] for k in ("line", "heard", "cue", "ms")},
            }
        )
        if x["heard"]:
            await self.send({"type": "heard", "text": x["heard"]})
        print(
            f"t{x['tick'] / TIC_HZ:6.1f} {x['cue']:<7} {x['ms']:4d} ms"
            + (f" [you: {x['heard']}]" if x["heard"] else "")
            + f" {x['line']}",
            flush=True,
        )
        if self.voice is None:
            return
        text = (
            sound_tag(x["brief"], self.rng) + x["line"]
            if self.args.tts == "turbo"
            else x["line"]
        )
        first = None
        for part in _SENTENCE.split(text):
            if lid in self.cancelled:
                break
            pcm, sr, _ = await self.voice.say(part)
            self.n["voiced"] += 1
            if lid in self.cancelled:
                break
            if first is None:
                first = time.perf_counter()
                await self.send({"type": "audio", "id": lid, "sr": sr})
            await self.send(b"A" + lid.to_bytes(4, "big") + pcm)
        await self.send({"type": "audio_end", "id": lid})
        if since is not None and first is not None:
            lat = {
                "type": "latency",
                "id": lid,
                "speech_to_line_ms": round(1000 * since),
                "speech_to_audio_ms": round(1000 * (since + first - t0)),
                "line_ms": x["ms"],
            }
            print(f"  latency {lat}", flush=True)
            await self.send(lat)

    # The client.
    async def handle(self, request):
        from aiohttp import WSMsgType, web

        ws = web.WebSocketResponse(max_msg_size=64 * 2**20, heartbeat=20)
        await ws.prepare(request)
        if self.ws is not None and not self.ws.closed:
            await self.ws.close()
        self.ws = ws
        print("client connected", flush=True)
        await self.send({"type": "hello", "fps": self.args.fps, "audio_hz": AUDIO_HZ})
        if self.game is None or self.game.done:
            await self.new_game()
        else:
            self.game.start()
        async for msg in ws:
            if msg.type == WSMsgType.BINARY and msg.data[:1] == b"U":
                pcm = np.frombuffer(msg.data[1:], np.int16).astype(np.float32) / 32768
                if len(pcm) >= AUDIO_HZ // 5 and self.game is not None:
                    self.said_at.append(time.perf_counter())
                    self.n["heard"] += 1
                    self.game.player_said(pcm)
            elif msg.type == WSMsgType.TEXT:
                m = json.loads(msg.data)
                if m.get("type") == "speaking" and self.game is not None:
                    self.game.hush(True)
                    self.cancelled.update(range(1, self.line_id + 1))
                elif m.get("type") == "reset":
                    await self.new_game()
        print("client gone", flush=True)
        if ws is self.ws:
            self.ws = None
        return ws


async def serve(args) -> None:
    import torch
    from aiohttp import web

    t0 = time.time()
    pol = AsyncPolicy(
        args.model,
        max_num_seqs=16,
        gpu_mem=args.gpu_mem,
        temperature=args.temperature,
        layout="chat",
    )
    await pol.warmup()
    # The ASR loads on its first clip: a second of quiet noise, now, not on yours.
    hum = np.random.default_rng(0).normal(0, 1e-3, AUDIO_HZ).astype(np.float32)
    ids = pol.tok.encode(
        "t0.2 hp 100 face 90 | clear | did wait\n", add_special_tokens=False
    )
    await pol.talk(ids, "now t1.0 | hp 100 armor 0", "", hum)
    voice = Voice(args) if args.tts_python else None
    gpu = torch.cuda.get_device_name(0).replace("NVIDIA ", "")
    live = Live(args, pol, voice, gpu)
    await live.new_game(start=False)
    live.spawn(live.stats())
    loop = asyncio.get_running_loop()
    loop.add_reader(live.jpeg.fileno(), live.on_jpeg)
    app = web.Application()
    app.router.add_get("/ws", live.handle)
    app.router.add_get("/health", lambda _: web.json_response({"ok": True}))
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, args.host, args.port).start()
    host = socket.getfqdn()
    if args.addr_file:
        args.addr_file.write_text(f"{host} {args.port}\n")
    print(
        f"ready in {time.time() - t0:.0f} s: ws://{host}:{args.port}/ws (on the laptop: "
        f"ssh -L {args.port}:{host}:{args.port} <login node>)",
        flush=True,
    )
    await asyncio.Event().wait()


# ── A scripted client, for a check without the laptop ──────────────────────────
async def smoke(args) -> None:
    """Connect, watch, and talk: "can you hear me" after a few seconds, "who
    killed you" after each of the first deaths, then "what's the score". Prints
    each reply (the transcript, the line, whether a killer was named) and the
    latency from the end of the question to his first audio; saves his audio
    and a frame."""
    import wave

    import aiohttp
    import soundfile as sf
    from scipy.signal import resample_poly

    def pcm16(name: str) -> bytes:
        x, sr = sf.read(str(args.questions / f"q_{name}.wav"), dtype="float32")
        x = x.mean(1) if x.ndim > 1 else x
        x = resample_poly(x, AUDIO_HZ, sr)
        return (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()

    asked: deque = deque()
    audio: dict[int, list[bytes]] = {}
    rates, lines, frames = {}, {}, []
    t_start, deaths, plan = time.perf_counter(), 0, [(5, "hear"), (60, "score")]
    replies = []
    args.out.mkdir(parents=True, exist_ok=True)
    async with (
        aiohttp.ClientSession() as s,
        s.ws_connect(args.url, max_msg_size=64 * 2**20) as ws,
    ):

        async def ask(name, killer=None):
            await ws.send_bytes(b"U" + pcm16(name))
            asked.append((name, killer, time.perf_counter()))

        async for msg in ws:
            el = time.perf_counter() - t_start
            if plan and el >= plan[0][0]:
                await ask(plan.pop(0)[1])
            if msg.type == aiohttp.WSMsgType.BINARY:
                if msg.data[:1] == b"J":
                    frames.append((el, len(msg.data)))
                    if len(frames) == 200:
                        (args.out / "frame.jpg").write_bytes(msg.data[1:])
                elif msg.data[:1] == b"A":
                    lid = int.from_bytes(msg.data[1:5], "big")
                    if lid in lines and "first" not in lines[lid]:
                        lines[lid]["first"] = time.perf_counter()
                    audio.setdefault(lid, []).append(msg.data[5:])
                continue
            m = json.loads(msg.data)
            if m["type"] == "event" and m["kind"] == "died" and deaths < 2:
                deaths += 1
                await asyncio.sleep(1.5)  # respawned
                await ask("who_killed", m["by"])
            elif m["type"] == "line":
                lines[m["id"]] = m
                print(
                    f"[{el:5.1f}s] {m['cue']:<7} {m['line']}"
                    + (f"  (heard {m['heard']!r})" if m["heard"] else ""),
                    flush=True,
                )
                if m["heard"] and asked:
                    name, killer, t_ask = asked.popleft()
                    replies.append((name, killer, t_ask, m))
            elif m["type"] == "audio":
                rates[m["id"]] = m["sr"]
            elif m["type"] == "latency":
                print(f"         server latency {m}", flush=True)
            if el > args.seconds or (len(replies) >= 4 and not plan):
                break
    for lid, parts in audio.items():
        with wave.open(str(args.out / f"line_{lid}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rates.get(lid, 24000))
            w.writeframes(b"".join(parts))
    span = frames[-1][0] - frames[0][0] if len(frames) > 1 else 1
    print(
        f"== {len(frames)} frames, {len(frames) / span:.1f} fps, {np.mean([b for _, b in frames]) / 1024:.0f} KB each"
    )
    for name, killer, t_ask, m in replies:
        first = m.get("first")
        named = killer is not None and killer.lower() in m["line"].lower()
        print(
            f"  {name:<10} heard {m['heard']!r} -> {m['line']}"
            + (
                f"  [killer {killer}: {'named' if named else 'NOT named'}]"
                if name == "who_killed"
                else ""
            )
            + (
                f"  first audio {1000 * (first - t_ask):.0f} ms after the question"
                if first
                else ""
            )
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="The game, the model and the voice")
    s.add_argument("--model", required=True, help="Composed with audio (--asr-model)")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--addr-file", type=Path, help="Write 'host port' here")
    s.add_argument("--seconds", type=float, default=600.0, help="Match length")
    s.add_argument("--bots", default="default")
    s.add_argument("--n-bots", type=int, default=7)
    s.add_argument("--seed", type=int, default=5)
    s.add_argument("--idle-s", type=float, default=12.0, help="Silence before a remark")
    s.add_argument("--temperature", type=float, default=1.0, help="Style, planner")
    s.add_argument("--gpu-mem", type=float, default=0.45)
    s.add_argument("--fps", type=float, default=20.0, help="Stream frame rate")
    s.add_argument("--scale", type=float, default=1.0, help="Stream size (1: 1000x898)")
    s.add_argument("--quality", type=int, default=70, help="JPEG quality")
    s.add_argument("--tts-python", help="The TTS environment's python (none: silent)")
    s.add_argument(
        "--tts",
        choices=("kokoro", "turbo", "standard"),
        default="kokoro",
        help="kokoro (fast, its own voices) or Chatterbox turbo / standard",
    )
    s.add_argument("--kokoro-voice", default="am_michael")
    s.add_argument("--speed", type=float, default=1.0, help="kokoro: pace")
    s.add_argument(
        "--voice-ref", type=Path, help="Chatterbox: the clip his voice clones"
    )
    s.add_argument(
        "--pitch", type=float, help="Semitones (default: -2 for kokoro, else 0)"
    )
    k = sub.add_parser("smoke", help="A scripted client")
    k.add_argument("--url", default="ws://localhost:8765/ws")
    k.add_argument("--questions", type=Path, required=True, help="q_<name>.wav files")
    k.add_argument("--seconds", type=float, default=150.0)
    k.add_argument("--out", type=Path, default=Path("out/smoke"))
    args = ap.parse_args()
    if args.cmd == "serve" and args.pitch is None:
        args.pitch = -2.0 if args.tts == "kokoro" else 0.0
    asyncio.run(serve(args) if args.cmd == "serve" else smoke(args))


if __name__ == "__main__":
    main()
