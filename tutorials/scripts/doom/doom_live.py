# SPDX-License-Identifier: Apache-2.0
"""The live demo's GPU side: one match in real time that you watch and talk to.

The game, the speech-enabled checkpoint (vLLM ``AsyncLLM``: every game adapter,
the narrator, the ASR cascade) and the player's voice run on one GPU node; a
laptop (``doom_pipecat.py``) connects over one websocket, carried by ``ssh -L``,
and turns it into WebRTC for the browser. What you say arrives as one audio
segment per utterance (the laptop's VAD cuts them); the model's ASR transcribes
it inside the orders adapter's request (:meth:`engine.Game.player_said`): an
order (stop, turn, ram the wall, switch guns, play it safe, ...: :mod:`orders`)
goes to the game at once, and the panel shows it; then the narrator answers,
from his own conversation with you and the game state (the output of his
``get_game_state`` call, :func:`talk.game_state`, which says what you told him
and whether he is doing it). Without you he speaks soon after a salient event,
after a silence, or when an order hurts or ends (:class:`talk.TalkClock`).
The run log (``--log-dir``) keeps that state with every line, so any answer can
be checked afterwards (:mod:`probes`).

Processes: this one (the decision loop, the websocket server), vLLM's engine
core, the game (:func:`engine.game_worker`, publishing its frames to shared
memory), a renderer that JPEG-encodes the newest frame at ``--fps``, and the
voice (``voice_video.py --serve`` in a TTS environment: Kokoro by default,
Chatterbox with ``--tts``). In the ``client`` view (the default) the stream is
the game's screen alone, and the renderer sends the telemetry the page draws
the panel and the maps from; in ``wide`` and ``classic`` it draws them into
the frame as the videos do (:mod:`overlay`). Drawing stays off the decision
loop. The stream's quality and frame rate follow the link
(:class:`LinkGovernor`).

A match runs while someone watches. ``GRACE_S`` after the last client leaves,
it is replaced by a fresh one, held until the next client connects: opening
the page starts a match; a reload or a reconnect sooner picks it up.

The websocket, at ``/ws`` (one client at a time; a new one replaces the old,
told ``{"type": "replaced"}`` and closed with code ``REPLACED``, so it does not
reconnect):

* down: ``b"J" + JPEG`` (a frame); ``b"A" + line id (4 bytes) + int16 PCM`` (a
  piece of his line); ``b"S" + int16 PCM`` (the game's own sound, mono at the
  hello's ``sfx_hz``, every 40 ms; ``--no-sound``: none); JSON ``{"type":
  "hello" | "line" | "heard" | "audio" | "audio_end" | "event" | "latency" |
  "quality" | "pong" | "replaced", ...}`` (the hello carries the page's
  schema; a line, the game state he answered from; events: ``died``, with the
  killer; ``frag``; ``match_over``; quality: the stream stepped down or up),
  and in the client view ``{"type": "tics" | "panel" | "match_start", ...}``
  (:func:`tic_column`, :func:`panel_fields`).
* up: ``b"U" + int16 PCM, 16 kHz`` (one utterance); JSON ``{"type":
  "speaking"}`` (you started: he holds his remarks and drops what he had not
  said yet), ``{"type": "reset"}`` (a new match), ``{"type": "ack"}`` (a frame
  arrived: the next may be sent; ``{"type": "acking"}`` first says acks will
  come), ``{"type": "ping", "t": ...}`` (answered with a pong carrying ``t``).

::

    # his voice: Kokoro's am_michael, 2 semitones down
    python doom_live.py serve --model models/doom-h-alora-narr4-audio --port 8765 \\
        --tts-python kokoro-env/bin/python
    # or Chatterbox-Turbo, cloning a clip
    python doom_live.py serve --model models/doom-h-alora-narr4-audio --port 8765 \\
        --tts turbo --tts-python tts-env/bin/python --voice-ref voices/him.wav
    # no laptop: a scripted check, with spoken questions (q_<name>.wav): some
    # referring back ("who was that", two lines after a death or a frag),
    # probes of the state (q_probe_<type>.wav), each answer checked against the
    # state he was given, and spoken orders (q_order_<name>.wav), each checked
    # in the game and against his reply
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
from collections import Counter, deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from conversation import CONV_EXCHANGES, Conversation
from doom_env import BOT_SETS, TIC_HZ
from engine import AUDIO_HZ, GAME_HZ, AsyncPolicy, Game, SharedFrame, SharedSound
from expert import BEHAVIORS
from policy import ARMS, CRITIC, NARRATOR, ORDERS, WARM_STATE
from talk import sound_tag

TTS_AUTHKEY = b"granite-switch-doom-tts"  # voice_video.TTS_AUTHKEY
MODELS = ["base", *BEHAVIORS, ARMS, CRITIC, ORDERS, NARRATOR]
ORDER_SHOWN_S = 4.0  # an order over stays on the panel this long
# Nobody watching this long: the match is replaced by a fresh one, held for the
# next client. Coming back sooner (a reload, the tunnel blipping) picks it up.
GRACE_S = 20.0
# Frames sent and not yet acked, at most (a client that acks): 20 fps up to a 150 ms
# round trip; 1 at the governor's lowest step (a frame takes long to cross there).
IN_FLIGHT = 3
BOOT = f"{os.getpid()}-{int(time.time())}"  # this process, in every hello
REPLACED = 4000  # the websocket close code to a client a newer one replaced
_SENTENCE = re.compile(r"(?<=[.?!])\s+")


# ── The renderer (its own process) ─────────────────────────────────────────────
VIEWS = ("client", "classic", "wide")
TICS_EVERY_S = 0.1  # the client view: a batch of per-tic columns this often
PANEL_EVERY_S = 0.2  # ... and the panel's fields this often


def stream_size(view: str, scale: float) -> tuple[int, int]:
    """The streamed frame's size: the view's, scaled, even (for the codec);
    the ``client`` view's is the game's own screen."""
    from overlay import W_GAME, H, Overlay

    w, h = (W_GAME, H) if view == "client" else Overlay(MODELS, view).size
    return round(w * scale) // 2 * 2, round(h * scale) // 2 * 2


def tic_column(
    tick: int, style: str, probs: dict, critic: dict, decided: bool, ran: set
) -> dict:
    """One tic of the maps, for the client view's page: ``s`` the behavior,
    ``p`` the action probabilities in ``DISPLAY_ORDER`` (0-255; None on a tic
    with no decision), ``c`` the critic's low/mid/high (0-255), ``d`` a fresh
    decision, ``a`` the models that ran (bit i: ``MODELS[i]``)."""
    from doom_env import DISPLAY_ORDER
    from policy import DANGER_LEVELS

    return {
        "k": tick,
        "s": style,
        "p": [round(255 * probs.get(a, 0.0)) for a in DISPLAY_ORDER]
        if decided
        else None,
        "c": [round(255 * critic.get(c, 0.0)) for c in DANGER_LEVELS],
        "d": int(decided),
        "a": sum(1 << i for i, m in enumerate(MODELS) if m in ran),
    }


def render_loop(frames_name: str, inbox, out, opts: dict) -> None:
    """Send the newest game frame as JPEG, ``opts["fps"]`` times a second at
    ``opts["quality"]`` (an inbox ``("quality", q, fps)`` changes both), and
    what the server sends about each decision, each tagged on ``out``.

    The ``client`` view sends the game's screen alone (``b"J"``), and the
    telemetry for the page to draw: a batch of :func:`tic_column` every
    ``TICS_EVERY_S`` (``b"T"``) and the panel's fields every ``PANEL_EVERY_S``
    (``b"P"``; the server keeps only the newest). The ``classic`` and ``wide``
    views draw it all into the frame, as the videos do (:mod:`overlay`)."""
    from overlay import KINDS, Overlay
    from PIL import Image

    shared = SharedFrame(frames_name)
    client = opts["view"] == "client"

    def overlay():
        return None if client else Overlay(MODELS, opts["view"])

    view = overlay()
    caps: list[tuple[float, str, str]] = []  # the captions; the wide view's chat
    cols: list[dict] = []  # the client view's columns not sent yet
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

    def push(t: int, probs: dict, style: str, decided: bool, crit: dict, ran: set):
        if client:
            cols.append(tic_column(t, style, probs, crit, decided, ran))
        else:
            view.heat.push(probs, style, decided, crit)
            view.act.push(ran)

    def tell(kind: bytes, msg: dict) -> None:
        out.send_bytes(kind + json.dumps(msg, separators=(",", ":")).encode())

    last_tick, talk_until, read_at = -1, -1, set()
    quality, period = opts["quality"], 1.0 / opts["fps"]
    due = tics_due = panel_due = time.perf_counter()
    size = stream_size(opts["view"], opts["scale"])
    while True:
        while True:
            try:
                msg = inbox.get_nowait()
            except queue.Empty:
                break
            if msg[0] == "stop":
                return
            if msg[0] == "new":  # a new match
                view, caps, cols, last_tick = overlay(), [], [], -1
                if client:
                    tell(b"T", {"type": "match_start"})
                continue
            if msg[0] == "quality":  # the link's state: the server's call
                quality, period = msg[1], 1.0 / msg[2]
                continue
            if msg[0] == "cap":
                caps = caps[-15:] + [(msg[1] / TIC_HZ, msg[2], msg[3])]
                continue
            if msg[0] == "talk":
                talk_until = msg[1]
                continue
            if msg[0] == "read":  # the orders adapter read the watcher's words
                read_at.add(max(msg[1], last_tick + 1))  # not on a column drawn
                continue
            _, tick, d, ms, state, stats, style, order = msg  # a decision
            info["adapter"] = style
            # One request per adapter asked (the base model only when it writes a
            # line itself, in a checkpoint without the narrator).
            talker = {opts["talker"]}

            def ran(t: int, asked: set) -> set:
                return (
                    asked
                    | (talker if t <= talk_until else set())
                    | ({ORDERS} if t in read_at else set())
                )

            for t in range(last_tick + 1, tick):  # tics with no decision (dead, busy)
                push(t, {}, style, False, info["critic"], ran(t, set()))
            last_tick = tick
            probs = d[style][1]
            critic = d[CRITIC][1] if CRITIC in d else {}
            push(tick, probs, style, True, critic, ran(tick, set(d)))
            read_at = {t for t in read_at if t > tick}
            info["order"] = order
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
        if client and now >= tics_due:
            tics_due = max(tics_due + TICS_EVERY_S, now)
            if cols:
                tell(b"T", {"type": "tics", "cols": cols})
                cols = []
        if client and now >= panel_due:
            panel_due = max(panel_due + PANEL_EVERY_S, now)
            if "ms" in info:
                tell(b"P", panel_fields(info))
        if now < due:
            time.sleep(min(due - now, 0.005))
            continue
        due = max(due + period, now)
        got = shared.get()
        if got is None or "ms" not in info:
            continue
        tick, frame = got
        if client:
            img = Image.fromarray(frame)
        else:
            img = view.draw(frame, {**info, "tick": tick}, caps, tick / TIC_HZ)
        if size != img.size:
            img = img.resize(size)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        out.send_bytes(b"J" + buf.getvalue())


def panel_fields(info: dict) -> dict:
    """The panel's fields (what :meth:`overlay.Panel.draw` shows), for the page;
    the weapon planner as its slot and that slot's probability."""
    plan = info["plan"]
    return {
        "type": "panel",
        **{
            k: info[k]
            for k in ("adapter", "action", "order", "stats", "hp", "armor", "weapon")
        },
        **{k: round(info[k], 2) for k in ("ms", "p50", "p99")},
        "tick": info["tick"],
        "critic": {k: round(v, 3) for k, v in info["critic"].items()},
        "plan": plan
        and {
            "slot": plan["slot"],
            "p": round(plan["probs"].get(str(plan["slot"]), 0.0), 3),
        },
        "text": info["text"],
    }


# ── The link: the stream's quality ─────────────────────────────────────────────
LOWER_STEPS = ((60, 15), (45, 10))  # JPEG quality and fps, below the top's


class LinkGovernor:
    """The stream's JPEG quality and frame rate, stepped from how the link
    keeps up, judged once a window (:meth:`judge`; :meth:`Live.adapt` calls it
    every 2 s): from each frame's time from being sent to the client's ack,
    and how many frames were replaced by a newer one before they could be sent
    (the newest wins).

    A window is slow when the median send-to-ack exceeds ``slow_ms`` or more
    than ``slow_replaced`` of the frames were replaced: one step down. It is
    healthy below ``fast_ms`` and ``fast_replaced``: after ``patience`` healthy
    windows in a row, one step up. A step up that turns slow within
    ``patience`` windows doubles the patience (up to ``max_patience``), so a
    link at its limit does not swing up and down. At the lowest step, one
    frame at a time is in flight (:attr:`in_flight`): fewer frames, but less
    for his voice to wait behind."""

    def __init__(
        self,
        steps: list[tuple[int, float]],
        slow_ms: float = 150.0,
        fast_ms: float = 100.0,
        slow_replaced: float = 0.5,
        fast_replaced: float = 0.2,
        patience: int = 3,
        max_patience: int = 30,
    ):
        self.steps = steps
        self.slow_ms, self.fast_ms = slow_ms, fast_ms
        self.slow_replaced, self.fast_replaced = slow_replaced, fast_replaced
        self.first_patience, self.max_patience = patience, max_patience
        self.reset()

    def reset(self) -> None:
        """A new client: the top step, and nothing measured."""
        self.level, self.patience = 0, self.first_patience
        self.healthy, self.since_up = 0, None
        self.acks: list[float] = []
        self.made = self.sent = 0

    @property
    def step(self) -> tuple[int, float]:
        return self.steps[self.level]

    @property
    def in_flight(self) -> int:
        """Frames that may be sent and not yet acked, at this step."""
        lowest = self.level > 0 and self.level == len(self.steps) - 1
        return 1 if lowest else IN_FLIGHT

    def judge(self) -> tuple[str, dict] | None:
        """End a window: ``("down" | "up", what was measured)`` when the step
        changed (:attr:`step` is the new one), else None."""
        acks, made, sent = self.acks, self.made, self.sent
        self.acks, self.made, self.sent = [], 0, 0
        if made == 0:  # nothing streamed (no client, no match)
            return None
        ms = float(np.median(acks)) if acks else None
        replaced = max(0.0, 1.0 - sent / made)
        seen = {"ack_ms": ms and round(ms), "replaced": round(replaced, 2)}
        if self.since_up is not None:
            self.since_up += 1
        if (ms is not None and ms > self.slow_ms) or replaced > self.slow_replaced:
            self.healthy = 0
            if self.since_up is not None and self.since_up <= self.patience:
                self.patience = min(2 * self.patience, self.max_patience)
            self.since_up = None
            if self.level + 1 < len(self.steps):
                self.level += 1
                return "down", seen
            return None
        if (ms is None or ms < self.fast_ms) and replaced < self.fast_replaced:
            self.healthy += 1
            if self.healthy >= self.patience and self.level > 0:
                self.level -= 1
                self.healthy, self.since_up = 0, 0
                return "up", seen
        else:
            self.healthy = 0
        return None


def quality_steps(quality: int, fps: float) -> list[tuple[int, float]]:
    """The governor's steps: ``(quality, fps)`` first, then the lower ones."""
    return [(quality, fps)] + [
        (q, f) for q, f in LOWER_STEPS if q < quality and f < fps
    ]


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
        self.args, self.pol, self.voice, self.gpu = args, pol, voice, gpu
        ctx = mp.get_context("spawn")
        self.frames = SharedFrame()
        self.tele = ctx.Queue()  # to the renderer; put() never waits
        rendered_r, rendered_w = ctx.Pipe(duplex=False)
        self.rendered = rendered_r  # from the renderer: frames and telemetry
        opts = {
            "fps": args.fps,
            "view": args.view,
            "scale": args.scale,
            "quality": args.quality,
            "gpu": gpu,
            "placement": pol.kit.placement,
            "talker": pol.kit.talker or "base",  # who writes his lines
        }
        self.renderer = ctx.Process(
            target=render_loop,
            args=(self.frames.name, self.tele, rendered_w, opts),
            daemon=True,
        )
        self.renderer.start()
        # The game's sound, as the worker writes it every tic, and how far it was sent.
        self.sound = None if args.no_sound else SharedSound()
        self.sound_mark = 0
        self.ws = None
        self.game: Game | None = None
        self.seed = args.seed
        self.line_id = 0
        self.cancelled: set[int] = set()
        self.said_at: deque[float] = deque()  # when each utterance arrived
        self.last_death, self.frags = None, 0  # for the client's events
        self.rng = random.Random(args.seed)  # which lines get a sound tag
        self._sending = False
        self._next_frame: bytes | None = None
        self._acking, self._unacked = False, 0
        self._acked = asyncio.Event()
        self._sent_at: deque[float] = deque()  # the unacked frames' send times
        self.gov = LinkGovernor(quality_steps(args.quality, args.fps))
        # The client view's telemetry: the tic batches in order, the panel newest-wins.
        self._tele: deque[str] = deque()
        self._panel: str | None = None
        self._tele_sending = False
        # What happened in the last stats window (see stats()).
        self.n = {"frames": 0, "frame_kb": 0, "tele_b": 0, "voiced": 0, "heard": 0}
        self.ms: list[float] = []
        self._tasks: set[asyncio.Task] = set()
        self._left = 0  # how many times the last client has left
        self.log = None  # this run's lines and matches, as JSON rows
        if args.log_dir:
            args.log_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            self.log = open(args.log_dir / f"live_{stamp}.jsonl", "a")
            print(f"logging lines to {self.log.name}", flush=True)

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
            elif isinstance(data, str):  # JSON already
                await ws.send_str(data)
            else:
                await ws.send_str(json.dumps(data))
        except (ConnectionError, RuntimeError):
            pass

    def watched(self) -> bool:
        return self.ws is not None and not self.ws.closed

    # What the renderer sends, by its tag: a frame (b"J"), a batch of tic
    # columns (b"T") or the panel (b"P"). Frames: the newest one wins when the
    # link is slow. A client that acks each frame ({"type": "ack"}) has at most
    # LinkGovernor.in_flight unacked: the link's buffers (the ssh tunnel's) never
    # fill with frames, so his voice, the game's sound and the telemetry, on
    # the same stream, do not wait behind them. The telemetry goes at once (it
    # is small): the tic batches all, in order; of the panels, the newest.
    def on_render(self) -> None:
        data = self.rendered.recv_bytes()
        kind = data[:1]
        if kind == b"J":
            self.gov.made += 1
            self._next_frame = data[1:]
            if not self._sending:
                self._sending = True
                self.spawn(self._send_frames())
            return
        if not self.watched():
            return
        if kind == b"P":
            self._panel = data[1:].decode()
        else:
            self._tele.append(data[1:].decode())
        if not self._tele_sending:
            self._tele_sending = True
            self.spawn(self._send_tele())

    async def _send_frames(self) -> None:
        while self._next_frame is not None:
            if self._acking and self._unacked >= self.gov.in_flight:
                self._acked.clear()
                try:
                    await asyncio.wait_for(self._acked.wait(), 2.0)
                except TimeoutError:  # acks stopped: do not stall the stream
                    self._unacked = 0
                    self._sent_at.clear()
                    self.gov.acks.append(2000.0)
                continue
            data, self._next_frame = self._next_frame, None
            self._sent_at.append(time.perf_counter())
            await self.send(b"J" + data)
            self._unacked += 1
            self.gov.sent += 1
            self.n["frames"] += 1
            self.n["frame_kb"] += len(data) // 1024
        self._sending = False

    def on_ack(self) -> None:
        self._acking = True
        self._unacked = max(0, self._unacked - 1)
        if self._sent_at:
            self.gov.acks.append(1000 * (time.perf_counter() - self._sent_at.popleft()))
        self._acked.set()

    async def _send_tele(self) -> None:
        while self._tele or self._panel is not None:
            if self._tele:
                data = self._tele.popleft()
            else:
                data, self._panel = self._panel, None
            await self.send(data)
            self.n["tele_b"] += len(data)
        self._tele_sending = False

    async def adapt(self, every_s: float = 2.0) -> None:
        """Every ``every_s``: the stream's quality for the link, as the
        governor judges it; the renderer and the client are told of a change."""
        while True:
            await asyncio.sleep(every_s)
            if not self.watched():
                continue
            got = self.gov.judge()
            if got is None:
                continue
            way, seen = got
            q, fps = self.gov.step
            self.tell(("quality", q, fps))
            print(f"stream {way}: JPEG q{q} at {fps:g} fps ({seen})", flush=True)
            await self.send(
                {"type": "quality", "q": q, "fps": fps, "level": self.gov.level, **seen}
            )

    # The match.
    async def new_game(self, start: bool = True) -> None:
        """A new match; ``start=False`` loads it and holds its clock until a
        client connects (loading takes a while)."""
        old = self.game
        if old is not None:
            # Nothing of the old match is said any more: not a line it is still
            # writing, nor the queued lines his voice has not reached.
            old.on_line = old.on_decision = old.on_read = None
            self.cancelled.update(range(1, self.line_id + 1))
            self.said_at.clear()
            if not old.done:
                old.proc.kill()
        a = self.args
        spec = {
            "seed": self.seed,
            "seconds": a.seconds,
            "bots": a.bots,
            "n_bots": a.n_bots,
            "frames": self.frames.name,
            "sound": self.sound.name if self.sound else None,
        }
        self.seed += 1
        g = Game(0, spec, self.pol, a.idle_s, autostart=start, conv_n=a.conv_exchanges)
        g.on_decision, g.on_line, g.on_read = (
            self.on_decision,
            self.on_line,
            self.on_read,
        )
        self.game, self.last_death, self.frags = g, None, 0
        self.record(
            {
                "type": "match",
                "seed": spec["seed"],
                "model": a.model,
                "talker": self.pol.kit.talker or "base",
                "conv_exchanges": a.conv_exchanges,
            }
        )
        self.tell(("new",))
        g.proc.start()
        self.spawn(self._play(g))
        print(f"match {spec['seed']} {'started' if start else 'loaded'}", flush=True)

    async def _play(self, g: Game) -> None:
        await g.run()
        if g is not self.game:
            return
        watched = self.watched()
        if watched:
            await self.send({"type": "event", "kind": "match_over", "stats": g.stats})
            await asyncio.sleep(5)
        if g is self.game:
            await self.new_game(start=watched)

    async def stream_sound(self, every_s: float = 0.04) -> None:
        """The game's sound since the last send, every ``every_s``, to the client
        (dropped while none is connected: it resumes at the live edge)."""
        while True:
            await asyncio.sleep(every_s)
            pcm, self.sound_mark = self.sound.since(self.sound_mark)
            if len(pcm) and self.watched():
                await self.send(b"S" + pcm.tobytes())

    async def stats(self, every_s: float = 10.0) -> None:
        """Every ``every_s``: decision latency in the window (as the panel
        measures it: from asking the engine to having the answer, in this
        process), with what else this process did (frames streamed, at which
        JPEG quality and fps; telemetry; lines voiced, utterances
        transcribed), and how late this event loop runs."""
        while True:
            t0 = time.perf_counter()
            await asyncio.sleep(every_s)
            lag = time.perf_counter() - t0 - every_s
            ms, self.ms = self.ms, []
            n = dict(self.n)
            self.n = dict.fromkeys(self.n, 0)
            if not ms:
                continue
            q, fps = self.gov.step
            print(
                f"[stats] decisions {len(ms) / every_s:4.1f}/s p50 {np.percentile(ms, 50):5.1f} "
                f"ms p90 {np.percentile(ms, 90):5.1f} p99 {np.percentile(ms, 99):5.1f} | "
                f"frames {n['frames'] / every_s:4.1f}/s ({n['frame_kb'] / every_s:5.0f} KB/s, "
                f"q{q}/{fps:g}) | telemetry {n['tele_b'] / 1024 / every_s:4.1f} KB/s "
                f"| voiced {n['voiced']} heard {n['heard']} | loop lag {1000 * lag:.0f} ms",
                flush=True,
            )

    def on_decision(self, tick: int, d: dict, ms: float) -> None:
        self.ms.append(ms)
        g = self.game
        f = g.facts or {}
        board = f.get("board") or [["", 0]]
        me = f.get("frags", 0)
        stats = {
            "frags": me,
            "deaths": f.get("deaths", 0),
            "rank": 1 + sum(n > me for _, n in board),
            "best_bot": board[0],
        }
        if g.talking:
            self.tell(("talk", tick + 1))
        o = f.get("order")
        shown = o and (
            o["status"] == "doing" or tick - o["end"] <= ORDER_SHOWN_S * TIC_HZ
        )
        order = {k: o[k] for k in ("told", "status", "why")} if shown else None
        style = next(a for a in d if a in BEHAVIORS)  # the one this decision asked
        self.tell(("dec", tick, d, ms, g.state or "", stats, style, order))
        death = f.get("last_death")
        if death and death != self.last_death:
            self.last_death = death
            self.spawn(self.send({"type": "event", "kind": "died", **death}))
        if me > self.frags:
            self.frags = me
            self.spawn(self.send({"type": "event", "kind": "frag", "frags": me}))

    def record(self, row: dict) -> None:
        """One row of this run's log (``--log-dir``): a match, or a line with
        what the narrator was told (the game state, and the moment's events
        his conversation keeps; the conversation is the lines before it, back
        to the match's start, up to ``conv_exchanges``)."""
        if self.log is not None:
            row = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **row}
            self.log.write(json.dumps(row) + "\n")
            self.log.flush()

    def on_read(self, tick: int, order: dict) -> None:
        """The orders adapter read what you said (an order or not)."""
        self.tell(("read", tick))
        if order["kind"] != "none":
            print(f"  order {order['kind']!r} from {order['heard']!r}", flush=True)

    def on_line(self, x: dict) -> None:
        keys = ("tick", "cue", "heard", "order", "line", "ms", "state", "moment")
        self.record({"type": "line", **{k: x[k] for k in (*keys, "brief")}})
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
                **{k: x[k] for k in ("line", "heard", "cue", "ms", "state", "order")},
            }
        )
        if x["heard"]:
            await self.send({"type": "heard", "text": x["heard"]})
        o = x.get("order") or {}
        told = (
            f" [order {o['kind']}: {o.get('status')}, {o.get('ms')} ms]"
            if o.get("kind") not in (None, "none")
            else ""
        )
        print(
            f"t{x['tick'] / TIC_HZ:6.1f} {x['cue']:<7} {x['ms']:4d} ms"
            + (f" [you: {x['heard']}]" if x["heard"] else "")
            + told
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
    def hello(self) -> dict:
        """The first message to a client: the stream (its view and frame size,
        the audio rates), and for the client view's page, what it draws the
        rest with (:func:`overlay.schema`). ``boot`` is this process's: a
        client that reconnects can tell a restart (line ids start again)."""
        from overlay import KINDS, schema

        a = self.args
        return {
            "type": "hello",
            "boot": BOOT,
            "view": a.view,
            "size": list(stream_size(a.view, a.scale)),
            "fps": a.fps,
            "audio_hz": AUDIO_HZ,
            "sfx_hz": GAME_HZ,
            "kind": KINDS.get(self.pol.kit.placement, ""),
            "gpu": self.gpu,
            "n_bots": a.n_bots,
            **schema(MODELS),
        }

    async def handle(self, request):
        from aiohttp import WSMsgType, web

        ws = web.WebSocketResponse(max_msg_size=64 * 2**20, heartbeat=20)
        await ws.prepare(request)
        old = self.ws
        if old is not None and not old.closed:
            # Replaced: told so (in band, as the close code may not arrive), it does
            # not come back.
            try:
                await old.send_str(json.dumps({"type": "replaced"}))
            except (ConnectionError, RuntimeError):
                pass
            await old.close(code=REPLACED, message=b"another client connected")
        self.ws = ws
        # This client's frame acks, and the stream's quality: from the top again.
        self._acking, self._unacked = False, 0
        self._sent_at.clear()
        self._tele.clear()
        if self.gov.level:
            self.tell(("quality", *self.gov.steps[0]))
        self.gov.reset()
        print("client connected", flush=True)
        await self.send(self.hello())
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
                elif m.get("type") == "ack":  # a frame arrived
                    self.on_ack()
                elif m.get("type") == "acking":  # it will ack: from the first frame
                    self._acking = True
                elif m.get("type") == "ping":  # the link's delay, as the client sees it
                    await self.send({"type": "pong", "t": m.get("t")})
        print("client gone", flush=True)
        if ws is self.ws:
            self.ws = None
            self._left += 1
            self.spawn(self._when_left(self.game, self._left))
        return ws

    async def _when_left(self, g: Game | None, left: int) -> None:
        """The last client left (the ``left``-th time): unless one comes back
        within ``GRACE_S``, the match ends and a fresh one is loaded, held until
        the next client: who opens the page starts a match, not joins one played
        to nobody. A match still held is fresh already."""
        await asyncio.sleep(GRACE_S)
        if left != self._left or self.watched():  # came back (and maybe left again)
            return
        if g is not None and g is self.game and not g.done and g.autostart:
            print(f"nobody watching for {GRACE_S:.0f} s: a fresh match", flush=True)
            await self.new_game(start=False)


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
        base_talk=args.base_talk,
        max_model_len=args.max_model_len,
    )
    await pol.warmup()
    # The ASR loads on its first clip: a second of quiet noise, now, not on yours.
    hum = np.random.default_rng(0).normal(0, 1e-3, AUDIO_HZ).astype(np.float32)
    await pol.talk(Conversation(), WARM_STATE, hum)
    if pol.kit.orders:  # where your words go first
        await pol.order(hum)
    voice = Voice(args) if args.tts_python else None
    gpu = torch.cuda.get_device_name(0).replace("NVIDIA ", "")
    live = Live(args, pol, voice, gpu)
    await live.new_game(start=False)
    live.spawn(live.stats())
    live.spawn(live.adapt())
    if live.sound is not None:
        live.spawn(live.stream_sound())
    loop = asyncio.get_running_loop()
    loop.add_reader(live.rendered.fileno(), live.on_render)
    app = web.Application()
    app.router.add_get("/ws", live.handle)
    # The stream's frame size too: the laptop's video track is made that size.
    size = stream_size(args.view, args.scale)
    app.router.add_get(
        "/health", lambda _: web.json_response({"ok": True, "size": list(size)})
    )
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
# Spoken probes of the game state for the smoke test: q_probe_<type>.wav.
SMOKE_PROBES = (
    "frags",
    "deaths",
    "health",
    "weapon",
    "ammo",
    "leader",
    "time_left",
    "rank",
    "armor",
)
# Spoken orders for the smoke test: q_order_<name>.wav -> the order it gives.
SMOKE_ORDERS = {
    "stop": "stop",
    "left": "left",
    "ram": "ram",
    "shotgun": "weapon",
    "safe": "cautious",
    "gun": "fetch",
    "rambo": "hunt",
}


async def smoke(args) -> None:
    """Connect, watch, and talk: "can you hear me" after a few seconds, "what's
    the score" later. Two back-references, asked two of his lines after the
    event (so it is in the conversation, the past tool outputs): "who was that"
    after the first death (the reply should name the killer) and after a frag
    with no death since (the victim is not known: it should name nobody). "who
    killed you" right after the next death. Then the spoken probes
    (q_probe_<type>.wav, :data:`SMOKE_PROBES`), one every 12 s, then the
    spoken orders (q_order_<name>.wav, :data:`SMOKE_ORDERS`), one every 10 s.
    Every answer to a question about the game is checked against the state he
    was given (:func:`probes.verify`, the state the server sends with his
    line); every order, that the orders adapter read it, the game took it
    (its status in his state) and his reply's stance matches
    (:func:`probes.order_stance`). Prints each reply (the transcript, the
    line, the verdict) and the latency from the end of the question to his
    first audio; saves his audio, a frame and smoke.json."""
    import wave

    import aiohttp
    import probes
    import soundfile as sf
    from scipy.signal import resample_poly

    def pcm16(name: str) -> bytes:
        x, sr = sf.read(str(args.questions / f"q_{name}.wav"), dtype="float32")
        x = x.mean(1) if x.ndim > 1 else x
        x = resample_poly(x, AUDIO_HZ, sr)
        return (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()

    bots = {n for names in BOT_SETS.values() for n in names}
    back = (args.questions / "q_who_was_that.wav").exists()
    if not back:
        print("no q_who_was_that.wav: no back-reference questions", flush=True)
    asked: deque = deque()
    audio: dict[int, list[bytes]] = {}
    rates, lines, frames = {}, {}, []
    t_start, deaths, plan = time.perf_counter(), 0, [(5, "hear"), (60, "score")]
    spoken = [t for t in SMOKE_PROBES if (args.questions / f"q_probe_{t}.wav").exists()]
    plan += [(75 + 12 * i, f"probe_{t}") for i, t in enumerate(spoken)]
    if not spoken:
        print("no q_probe_<type>.wav: no spoken probes", flush=True)
    said = [k for k in SMOKE_ORDERS if (args.questions / f"q_order_{k}.wav").exists()]
    t_orders = 75 + 12 * len(spoken) + 6
    plan += [(t_orders + 10 * i, f"order_{k}") for i, k in enumerate(said)]
    if not said:
        print("no q_order_<name>.wav: no spoken orders", flush=True)
    t_end = t_orders + 10 * len(said)
    # Back-references waiting to be asked: [what, the killer or None, lines to wait].
    recall: dict[str, list] = {}
    done: set[str] = set() if back else {"death", "frag"}
    death_at = None  # how many lines he had said at the last death
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
            if m["type"] == "event" and m["kind"] == "died":
                death_at = len(lines)
                named = m["by"] not in (None, "yourself")
                if "frag" in recall:  # "who was that" would be about the death now
                    del recall["frag"]
                    done.discard("frag")
                if "death" in recall:  # another death first: ask about this one
                    if named:
                        recall["death"] = ["back_death", m["by"], 2]
                    else:
                        del recall["death"]
                        done.discard("death")
                elif "death" not in done and named:
                    recall["death"] = ["back_death", m["by"], 2]
                    done.add("death")
                elif deaths < 1:
                    deaths += 1
                    await asyncio.sleep(1.5)  # respawned
                    await ask("who_killed", m["by"])
            elif m["type"] == "event" and m["kind"] == "frag":
                # No death in the conversation (its last 8 lines) to mix it up with.
                quiet = death_at is None or len(lines) - death_at > 8
                if "frag" not in done and not recall and quiet and el > 20:
                    recall["frag"] = ["back_frag", None, 2]
                    done.add("frag")
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
                for k, r in list(recall.items()):
                    r[2] -= 1
                    if r[2] == 0:
                        del recall[k]
                        await ws.send_bytes(b"U" + pcm16("who_was_that"))
                        asked.append((r[0], r[1], time.perf_counter()))
            elif m["type"] == "audio":
                rates[m["id"]] = m["sr"]
            elif m["type"] == "latency":
                print(f"         server latency {m}", flush=True)
            finished = not plan and not recall and not asked and deaths >= 1
            if el > args.seconds or (finished and done == {"death", "frag"}):
                break
            if not plan and not asked and el > t_end + 14 and deaths >= 1:
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
    # What each question asks about the game state, checked against the state
    # he was given for the line that answered it.
    as_probe = {"who_killed": "killer_now", "score": "score", "back_frag": "victim"}
    verdicts, out, ordered = Counter(), [], Counter()
    for name, killer, t_ask, m in replies:
        first = m.get("first")
        said = sorted(n for n in bots if re.search(rf"\b{n}\b", m["line"]))
        check, ptype = "", as_probe.get(name)
        if name.startswith("probe_"):
            ptype = name[len("probe_") :]
        if name == "back_death":
            ok = killer in said
            check = f"  [killer {killer}: {'named' if ok else 'NOT named'}]"
        state = m.get("state")
        verdict = None
        if ptype and state:
            if ptype in probes.allowed(state) or ptype in ("victim", "score"):
                verdict, why = probes.verify({"type": ptype}, m["line"], state)
                check += f"  [{ptype}: {verdict.upper()}; {probes.answer_text({'type': ptype}, state)}]"
                verdicts[verdict] += 1
            else:
                check += f"  [{ptype}: not askable in this state]"
        if name.startswith("order_"):
            want = SMOKE_ORDERS[name[len("order_") :]]
            o = m.get("order") or {}
            took = o.get("kind") == want and o.get("status") is not None
            stance, why = probes.order_stance(m["line"], state or {})
            if took and o.get("status") == "cant":
                stance = stance and probes.says_why(m["line"], state["order"])
            ordered["carried out" if took else "not carried out"] += 1
            ordered["stance right" if took and stance else "stance wrong"] += took
            check += (
                f"  [order {want}: read {o.get('kind')}, {o.get('status')}, "
                f"{o.get('ms')} ms to the game; stance {'ok' if stance else 'WRONG ' + why}]"
            )
        ok_claims = probes.claims(m["line"], state)[0] if state else None
        if ok_claims is False:
            check += "  [CLAIMS: " + probes.claims(m["line"], state)[1] + "]"
        out.append(
            {
                "asked": name,
                "heard": m["heard"],
                "line": m["line"],
                "verdict": verdict,
                "claims": ok_claims,
            }
        )
        print(
            f"  {name:<14} heard {m['heard']!r} -> {m['line']}"
            + check
            + (
                f"  first audio {1000 * (first - t_ask):.0f} ms after the question"
                if first
                else ""
            )
        )
    print(f"SMOKE probes: {dict(verdicts)}", flush=True)
    print(f"SMOKE orders: {dict(ordered)}", flush=True)
    (args.out / "smoke.json").write_text(
        json.dumps({"verdicts": verdicts, "orders": ordered, "replies": out}, indent=1)
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
    s.add_argument(
        "--base-talk",
        action="store_true",
        help="The base model writes his lines (on the narrator's prompt), not the "
        "narrator adapter",
    )
    s.add_argument(
        "--conv-exchanges",
        type=int,
        default=CONV_EXCHANGES,
        help="Exchanges the narrator's conversation keeps",
    )
    s.add_argument(
        "--max-model-len", type=int, default=16384, help="Room for a long conversation"
    )
    s.add_argument("--log-dir", type=Path, help="Write each run's lines here (JSONL)")
    s.add_argument("--temperature", type=float, default=1.0, help="Style, planner")
    s.add_argument("--gpu-mem", type=float, default=0.45)
    s.add_argument(
        "--fps", type=float, default=20.0, help="Stream frame rate (on a good link)"
    )
    s.add_argument("--no-sound", action="store_true", help="Not the game's own sound")
    s.add_argument(
        "--view",
        choices=VIEWS,
        default="client",
        help="client: the game alone, 640x480, the page draws the rest from the "
        "telemetry; drawn here instead: wide (16:9, 1920x1080) or classic "
        "(1000x910, as the videos)",
    )
    s.add_argument(
        "--scale", type=float, default=1.0, help="Stream size, times the view's"
    )
    s.add_argument(
        "--quality", type=int, default=75, help="JPEG quality (on a good link)"
    )
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
    k.add_argument("--seconds", type=float, default=300.0)
    k.add_argument("--out", type=Path, default=Path("out/smoke"))
    args = ap.parse_args()
    if args.cmd == "serve" and args.pitch is None:
        args.pitch = -2.0 if args.tts == "kokoro" else 0.0
    asyncio.run(serve(args) if args.cmd == "serve" else smoke(args))


if __name__ == "__main__":
    main()
