# SPDX-License-Identifier: Apache-2.0
"""The live demo's page and call: a browser call with the Doom player.

Pipecat serves the page and the WebRTC call: the browser sends your microphone
(its own echo cancellation, noise suppression and gain applied) and plays the
game's video, his voice and the game's own sound (mixed under his voice, lower
while he speaks: :class:`GameSound`). The game, the model and his voice are the
GPU side (``doom_live.py serve``), one websocket away. Pipecat's VAD (Silero)
cuts what you say into utterances; each one goes up as a single audio segment,
which the model's own ASR transcribes inside the narrator's request. No
speech-to-text, LLM or text-to-speech service runs here. When you start talking
over him, his line stops (an interruption) and the GPU side drops what he had
not said yet.

The page at ``/`` (``static/live.html``, ``live.js``, ``live.css``) shows the
game's video with the dashboard around it, drawn in the browser from the
telemetry (crisp at any size): the panel, which model runs, the action
heatmap, captions and the conversation; with a mute button, the game's sound,
a new-match button and the link's state. The GPU side's ``wide`` and
``classic`` views send it all drawn into the video instead; the page then
shows the video alone. Pipecat's prebuilt client is still at ``/client``.

A small environment of its own (not the vLLM one)::

    uv venv doom-pipecat && uv pip install --python doom-pipecat/bin/python \\
        "pipecat-ai[webrtc,silero]" pipecat-ai-small-webrtc-prebuilt aiohttp pillow

As a deployment would run it: on the GPU node itself, beside the GPU side, so
only the call crosses the network, compressed (VP8, with the browser's jitter
buffer and congestion control). Over HTTPS (the microphone needs it), offering
the one address browsers reach (``--ice-address``), and, where the network cuts
UDP flows (the one between a laptop and the cluster does: TLS passes), with a
TURN relay over TLS beside it (coturn) that the browser sends the call's media
through, and only through it (``/ice`` tells the page)::

    python doom_pipecat.py --https --port 8443 --cert-dir certs/ \\
        --ice-address <the node's address> --turn "turns:<node>:5349?transport=tcp"

Or on a laptop, the GPU side through ``ssh -L`` (the frames then cross the
network as JPEG)::

    ssh -N -L 8765:<gpu node>:8765 <login node>   # the GPU side
    python doom_pipecat.py                         # open http://localhost:7860/

The browser allows the microphone on ``localhost`` without a certificate; from
another machine it needs HTTPS, here with a self-signed certificate (in
``--cert-dir``, made with openssl if there is none; the browser asks you to
accept it once).
"""

from __future__ import annotations

import argparse
import asyncio
import io
import ipaddress
import json
import os
import socket
import subprocess
import time
import uuid
from pathlib import Path

import aiohttp
import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from PIL import Image
from pipecat.audio.mixers.base_audio_mixer import BaseAudioMixer
from pipecat.audio.utils import create_stream_resampler, is_silence
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    MixerControlFrame,
    MixerEnableFrame,
    OutputImageRawFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
    TTSAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVIClientMessageFrame
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

MIC_HZ = 16_000  # what the ASR takes
PRE_ROLL_S = 0.3  # audio kept from before the VAD's start, so no word is clipped
SFX_HZ = 22_050  # the game's sound as the GPU side sends it (its hello says so)
# What the GPU side sends that the page draws (over the call's data channel).
TO_PAGE = {
    "hello",
    "tics",
    "panel",
    "match_start",
    "line",
    "event",
    "quality",
    "latency",
}
RETRY_S = (0.5, 5.0)  # reconnecting to the GPU side: first wait, longest wait
REPLACED = 4000  # the GPU side's close code when a newer call took over (doom_live)


class GameSound(BaseAudioMixer):
    """The game's own sound, mixed into the call's audio under his voice.

    The GPU side sends it every 40 ms (``b"S"``); it is resampled to the call's
    rate and kept in a small jitter buffer: playback starts once ``PRIME_S`` is
    buffered (again after running dry), and anything past ``MAX_S`` is dropped
    from the old end, so it stays within about a tenth of a second of the
    picture. While he speaks it plays at ``duck`` times its gain."""

    PRIME_S, MAX_S = 0.1, 0.3

    def __init__(self, gain: float = 0.18, duck: float = 0.3):
        self.gain, self.duck, self.on = gain, duck, gain > 0
        self.rate = 0
        self.buf = bytearray()
        self.primed = False
        self.resampler = create_stream_resampler()
        self.out_bytes = 0  # the call's audio written through here (DoomLink logs it)

    async def start(self, sample_rate: int):
        self.rate = sample_rate

    async def stop(self):
        self.buf.clear()

    async def process_frame(self, frame: MixerControlFrame):
        if isinstance(frame, MixerEnableFrame):
            self.on = frame.enable
            self.buf.clear()

    async def feed(self, pcm: bytes, rate: int = SFX_HZ) -> None:
        if not self.on or not self.rate:
            return
        self.buf += await self.resampler.resample(pcm, rate, self.rate)
        cap = 2 * int(self.MAX_S * self.rate)
        if len(self.buf) > cap:
            del self.buf[: len(self.buf) - cap]

    async def mix(self, audio: bytes) -> bytes:
        n = len(audio)
        self.out_bytes += n
        if not self.on or not n:
            return audio
        if not self.primed:
            if len(self.buf) < 2 * int(self.PRIME_S * self.rate):
                return audio
            self.primed = True
        take = bytes(self.buf[:n])
        del self.buf[:n]
        if len(take) < n:  # ran dry: wait for the buffer again
            self.primed = False
            take += b"\x00" * (n - len(take))
        voice = not is_silence(audio)
        g = self.gain * (self.duck if voice else 1.0)
        out = np.frombuffer(audio, np.int16).astype(np.int32)
        out += (np.frombuffer(take, np.int16) * g).astype(np.int32)
        return np.clip(out, -32768, 32767).astype(np.int16).tobytes()


class DoomLink(FrameProcessor):
    """The pipeline's link to the GPU side. Up: each utterance (from the VAD's
    start to its stop, with a little audio from before the start), and a
    "speaking" notice at the start. Down: the frames (JPEG, decoded here) as
    video, his lines as TTS audio, so an interruption stops them, the game's
    sound to the mixer (``sound``), and to the page, over the data channel
    (``{"label": "doom", ...}``), what it draws (:data:`TO_PAGE`: the hello's
    schema, the telemetry, his lines) and the link's state (``{"type":
    "link", "state": "up" | "down" | "replaced", "rtt": ms}``). The connection
    is kept: when it drops, it is made again (:meth:`_keep_link`), unless a
    newer call replaced this one there."""

    def __init__(self, url: str, sound: GameSound | None = None):
        super().__init__()
        self.url, self.sound = url, sound
        self._sfx_hz = SFX_HZ
        self._session: aiohttp.ClientSession | None = None
        self._ws = None
        self._task = self._pinger = None
        self._speaking = self._bot_speaking = False
        self._pre: list[bytes] = []
        self._utt: list[bytes] = []
        self._rates: dict[int, int] = {}
        self._last_line = 0
        self._dropped_upto = 0  # lines talked over: their late audio is dropped
        self._rtt: list[float] = []  # the link's round trips (ping, pong), ms
        self._voice_bytes = 0  # his voice pushed into the call since the last log
        self._hello: dict | None = None  # the GPU side's latest, for the page
        self._replaced = False  # a newer call took the GPU side: do not reconnect
        self._link: dict = {"type": "link", "state": "connecting"}

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self._connect()
            return
        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._close()
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, RTVIClientMessageFrame):  # the page's buttons (RTVI)
            if frame.type == "reset":
                # His lines from the old match are not voiced any more.
                self._dropped_upto = self._last_line
                await self._send({"type": "reset"})
                if self._bot_speaking:
                    await self.broadcast_interruption()
                logger.info("new match")
            elif frame.type == "sfx" and self.sound is not None:
                await self.sound.process_frame(
                    MixerEnableFrame(enable=not self.sound.on)
                )
                logger.info(f"game sound {'on' if self.sound.on else 'off'}")
            elif frame.type == "sfx_gain" and self.sound is not None:  # the slider
                self.sound.gain = min(1.0, max(0.0, float((frame.data or {})["gain"])))
            elif frame.type == "hello":  # the page's channel opened: what it missed
                if self._hello is not None:
                    await self._to_page(self._hello)
                await self._to_page(self._link)
            return
        if isinstance(frame, InputAudioRawFrame):  # the microphone: up, not out
            if self._speaking:
                self._utt.append(frame.audio)
            else:
                self._pre.append(frame.audio)
                keep = int(PRE_ROLL_S * frame.sample_rate * 2)
                while sum(map(len, self._pre)) > keep and len(self._pre) > 1:
                    self._pre.pop(0)
            return
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._speaking, self._utt, self._pre = True, list(self._pre), []
            self._dropped_upto = self._last_line
            await self._send({"type": "speaking"})
            if self._bot_speaking:
                await self.broadcast_interruption()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._speaking = False
            pcm, self._utt = b"".join(self._utt), []
            if await self._send(b"U" + pcm):
                logger.info(f"utterance sent: {len(pcm) / (2 * MIC_HZ):.1f} s")
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        await self.push_frame(frame, direction)

    async def _send(self, msg: dict | bytes) -> bool:
        """To the GPU side, if connected (False if not, or it just dropped)."""
        ws = self._ws
        if ws is None or ws.closed:
            return False
        try:
            if isinstance(msg, bytes):
                await ws.send_bytes(msg)
            else:
                await ws.send_str(json.dumps(msg))
        except (ConnectionError, RuntimeError, aiohttp.ClientError):
            return False
        return True

    async def _to_page(self, msg: dict) -> None:
        await self.push_frame(
            OutputTransportMessageUrgentFrame(message={"label": "doom", **msg})
        )

    async def _set_link(self, state: str, rtt: float | None = None) -> None:
        self._link = {"type": "link", "state": state, "rtt": rtt}
        await self._to_page(self._link)

    async def _connect(self) -> None:
        self._session = aiohttp.ClientSession()
        self._task = self.create_task(self._keep_link(), "doom_link")
        self._pinger = self.create_task(self._ping(), "doom_ping")

    async def _keep_link(self) -> None:
        """Connect to the GPU side, and again whenever the connection drops (the
        tunnel blips, the service restarts): after ``RETRY_S[0]``, doubling up to
        ``RETRY_S[1]``. The match goes on there meanwhile; the page is told. Not
        when the GPU side closed it for a newer call (it says ``replaced``, and
        closes with ``REPLACED``)."""
        wait, said = RETRY_S[0], False
        while True:
            try:
                self._ws = await asyncio.wait_for(
                    self._session.ws_connect(
                        self.url, max_msg_size=64 * 2**20, heartbeat=10
                    ),
                    15,
                )
            except (aiohttp.ClientError, OSError, TimeoutError) as e:
                if not said:
                    logger.error(
                        f"cannot reach the GPU side at {self.url} ({e or type(e).__name__}); "
                        "is ssh -L up? Trying again."
                    )
                    said = True
                if self._link["state"] != "down":
                    await self._set_link("down")
                await asyncio.sleep(wait)
                wait = min(2 * wait, RETRY_S[1])
                continue
            logger.info(f"connected to {self.url}")
            wait, said = RETRY_S[0], False
            await self._send({"type": "acking"})  # frames wait for acks from the first
            await self._set_link("up")
            try:
                await self._receive()
            except (aiohttp.ClientError, OSError) as e:
                logger.warning(f"the link to the GPU side failed: {e}")
            replaced = self._replaced or self._ws.close_code == REPLACED
            self._ws = None
            if replaced:  # a newer call has the game: this one leaves it
                logger.warning("another call connected to the GPU side; this one stops")
                await self._set_link("replaced")
                return
            await self._set_link("down")
            logger.warning("the GPU side's connection closed; reconnecting")
            await asyncio.sleep(wait)

    async def _ping(self, every_s: float = 1.0) -> None:
        """A ping a second, on the stream the frames and his voice come down; its
        round trip (to the page with each pong; logged every 10) is how far
        behind the link runs. With it, every 10, the call's audio: what was
        written into it (none: the call's audio stopped here) and his voice."""
        t0 = time.perf_counter()
        while True:
            await asyncio.sleep(every_s)
            await self._send({"type": "ping", "t": time.perf_counter()})
            if len(self._rtt) >= 10:
                r = sorted(self._rtt)
                dt, t0 = time.perf_counter() - t0, time.perf_counter()
                out = ""
                if self.sound is not None:
                    out = (
                        f"call audio out {self.sound.out_bytes / 1024 / dt:.0f} KB/s, "
                    )
                    self.sound.out_bytes = 0
                logger.info(
                    f"link round trip p50 {r[len(r) // 2]:.0f} ms, max {r[-1]:.0f} ms | "
                    f"{out}his voice in {self._voice_bytes / 1024 / dt:.0f} KB/s"
                )
                self._rtt, self._voice_bytes = [], 0

    async def _close(self) -> None:
        if self._pinger is not None:
            await self.cancel_task(self._pinger)
            self._pinger = None
        if self._task is not None:
            await self.cancel_task(self._task)
            self._task = None
        if self._ws is not None:
            await self._ws.close()
        if self._session is not None:
            await self._session.close()

    async def _receive(self) -> None:
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                kind, data = msg.data[:1], msg.data[1:]
                if kind == b"J":
                    await self._send({"type": "ack"})  # the GPU side may send the next
                    img = await asyncio.to_thread(_decode, data)
                    await self.push_frame(
                        OutputImageRawFrame(
                            image=img.tobytes(), size=img.size, format="RGB"
                        )
                    )
                elif kind == b"S":
                    if self.sound is not None:
                        await self.sound.feed(data, self._sfx_hz)
                elif kind == b"A":
                    lid = int.from_bytes(data[:4], "big")
                    if lid > self._dropped_upto:
                        self._voice_bytes += len(data) - 4
                        await self.push_frame(
                            TTSAudioRawFrame(
                                audio=data[4:],
                                sample_rate=self._rates.get(lid, 24_000),
                                num_channels=1,
                                context_id=str(lid),
                            )
                        )
                continue
            if msg.type != aiohttp.WSMsgType.TEXT:
                break
            m = json.loads(msg.data)
            if m["type"] == "hello":
                self._sfx_hz = m.get("sfx_hz", SFX_HZ)
                if self._hello is not None and m.get("boot") != self._hello.get("boot"):
                    # The GPU side restarted: its line ids start again.
                    self._last_line = self._dropped_upto = 0
                    self._rates.clear()
                self._hello = m
            elif m["type"] == "pong" and m.get("t") is not None:
                rtt = 1000 * (time.perf_counter() - m["t"])
                self._rtt.append(rtt)
                await self._set_link("up", round(rtt))
            elif m["type"] == "line":
                self._last_line = max(self._last_line, m["id"])
                heard = f" (you: {m['heard']})" if m.get("heard") else ""
                logger.info(f"him: {m['line']}{heard}")
                m = {k: v for k, v in m.items() if k != "state"}  # not for the page
            elif m["type"] == "audio":
                self._rates[m["id"]] = m["sr"]
            elif m["type"] == "latency":
                logger.info(
                    f"end of your speech to his voice: {m['speech_to_audio_ms']} ms "
                    f"(line written after {m['speech_to_line_ms']} ms)"
                )
            elif m["type"] == "replaced":  # before the close (its code may not come)
                self._replaced = True
            elif m["type"] == "quality":
                logger.info(f"the stream: JPEG q{m['q']} at {m['fps']} fps ({m})")
            elif m["type"] == "event" and m["kind"] == "match_over":
                logger.info(f"match over: {m.get('stats')}")
            if m["type"] in TO_PAGE:
                await self._to_page(m)


def _decode(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


async def stream_size(args) -> tuple[int, int]:
    """The GPU side's frame size (its health check says), else ``--width`` x
    ``--height``: the video track is made that size (no resizing here)."""
    url = args.cluster.replace("ws://", "http://").rsplit("/", 1)[0] + "/health"
    try:
        async with aiohttp.ClientSession() as s, s.get(url, timeout=5) as r:
            w, h = (await r.json())["size"]
            return int(w), int(h)
    except (aiohttp.ClientError, TimeoutError, KeyError, ValueError):
        return args.width, args.height


async def run_call(conn: SmallWebRTCConnection, args) -> None:
    sound = GameSound(args.sfx_gain, args.sfx_duck) if args.sfx_gain > 0 else None
    width, height = await stream_size(args)
    logger.info(f"video {width}x{height}")
    transport = SmallWebRTCTransport(
        webrtc_connection=conn,
        params=TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=MIC_HZ,
            audio_out_enabled=True,
            audio_out_mixer=sound,
            video_out_enabled=True,
            video_out_is_live=True,
            video_out_width=width,
            video_out_height=height,
            video_out_framerate=args.fps,
        ),
    )
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=args.stop_secs))
    )
    pipeline = Pipeline(
        [transport.input(), vad, DoomLink(args.cluster, sound), transport.output()]
    )
    task = PipelineTask(pipeline)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        await task.cancel()

    await PipelineRunner(handle_sigint=False).run(task)


def certificate(host: str, folder: Path) -> tuple[Path, Path]:
    """A self-signed certificate for this machine's names and addresses."""
    cert, key = folder / "cert.pem", folder / "key.pem"
    if cert.exists() and key.exists():
        return cert, key
    folder.mkdir(parents=True, exist_ok=True)
    names = {"localhost", socket.gethostname()}
    ips = {"127.0.0.1"}
    for addr in socket.gethostbyname_ex(socket.gethostname())[2]:
        ips.add(addr)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # no packet is sent: picks the LAN address
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    if host not in ("0.0.0.0", "::"):
        try:
            ipaddress.ip_address(host)
            ips.add(host)
        except ValueError:
            names.add(host)
    san = ",".join(
        [*(f"DNS:{n}" for n in sorted(names)), *(f"IP:{i}" for i in sorted(ips))]
    )
    cmd = ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "30"]
    cmd += ["-keyout", str(key), "-out", str(cert), "-subj", "/CN=doom-demo"]
    cmd += ["-addext", f"subjectAltName={san}"]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"made a self-signed certificate for {san} -> {cert}")
    return cert, key


def main() -> None:
    from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cluster", default="ws://localhost:8765/ws", help="The GPU side")
    ap.add_argument("--host", help="Default: localhost (0.0.0.0 with --https)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--https", action="store_true", help="For a browser elsewhere")
    ap.add_argument(
        "--cert-dir", type=Path, default=Path.home() / ".cache" / "doom-pipecat"
    )
    ap.add_argument(
        "--width", type=int, default=640, help="If the GPU side does not say"
    )
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument(
        "--video-kbps",
        type=int,
        default=1500,
        help="Video bitrate cap; the browser's congestion control adapts below it "
        "(the game alone: the page draws the rest)",
    )
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument(
        "--ice-address",
        help="Offer the call on this address only: a server with many interfaces "
        "(a GPU node's InfiniBand, a container bridge) offers the one browsers reach",
    )
    ap.add_argument("--ice-log", action="store_true", help="Log the ICE checks")
    ap.add_argument(
        "--turn",
        help="A TURN relay the browser sends the call's media through, and only "
        "through (turns:<host>:5349?transport=tcp: TLS, which networks that cut "
        "UDP flows let pass)",
    )
    ap.add_argument("--turn-user", default="doom")
    ap.add_argument(
        "--turn-pass",
        default=os.environ.get("TURN_PASS"),
        help="The relay's password (default: $TURN_PASS)",
    )
    ap.add_argument(
        "--stop-secs", type=float, default=0.45, help="Silence that ends an utterance"
    )
    ap.add_argument(
        "--sfx-gain",
        type=float,
        default=0.18,
        help="The game's sound as a gain (0.18: -15 dB; 0: none); the page's slider sets it",
    )
    ap.add_argument(
        "--sfx-duck", type=float, default=0.3, help="... times this while he speaks"
    )
    args = ap.parse_args()
    host = args.host or ("0.0.0.0" if args.https else "localhost")

    # aiortc's encoders cap the bitrate at module level; the browser's
    # congestion control still adapts below the cap.
    import aiortc.codecs.h264 as h264
    import aiortc.codecs.vpx as vpx

    for codec in (vpx, h264):
        codec.MAX_BITRATE = 1000 * args.video_kbps
        codec.DEFAULT_BITRATE = min(codec.MAX_BITRATE, 1_000_000)
    if args.ice_address:
        # aiortc has no setting for it: its ICE (aioice) offers every local address.
        import aioice.ice

        aioice.ice.get_host_addresses = lambda use_ipv4, use_ipv6: [args.ice_address]
    if args.ice_log:
        import logging

        logging.basicConfig(format="%(asctime)s %(name)s %(message)s")
        logging.getLogger("aioice.ice").setLevel(logging.INFO)

    app = FastAPI()
    calls = SmallWebRTCRequestHandler()
    app.mount("/client", SmallWebRTCPrebuiltUI)  # Pipecat's own page, a small tile
    static = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static))  # the page's script, styles

    @app.middleware("http")
    async def no_stale_page(request: Request, call_next):
        # The browser checks with us on every load (a 304 if unchanged), so an
        # updated page is never run from its cache.
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/", include_in_schema=False)
    async def root():  # the game and the dashboard, full window
        return FileResponse(static / "live.html")

    @app.get("/ice", include_in_schema=False)
    async def ice():
        # How the browser reaches the call: with a relay, through it only (no direct
        # UDP for a firewall to cut); else directly.
        config: dict = {"iceServers": []}
        if args.turn:
            config = {
                "iceServers": [
                    {
                        "urls": [args.turn],
                        "username": args.turn_user,
                        "credential": args.turn_pass,
                    }
                ],
                "iceTransportPolicy": "relay",
            }
        return JSONResponse(config, headers={"Cache-Control": "no-store"})

    @app.post("/start")
    async def start(_request: Request):
        return {"sessionId": str(uuid.uuid4())}  # the call starts with its offer

    async def offer(request: Request, background: BackgroundTasks):
        body = await request.json()
        if request.method == "PATCH":
            await calls.handle_patch_request(
                SmallWebRTCPatchRequest(
                    pc_id=body["pc_id"],
                    candidates=[IceCandidate(**c) for c in body.get("candidates", [])],
                )
            )
            return {"status": "success"}

        async def connected(conn: SmallWebRTCConnection):
            background.add_task(run_call, conn, args)

        return await calls.handle_web_request(
            SmallWebRTCRequest(
                sdp=body["sdp"],
                type=body["type"],
                pc_id=body.get("pc_id"),
                restart_pc=body.get("restart_pc"),
            ),
            connected,
        )

    app.add_api_route("/api/offer", offer, methods=["POST", "PATCH"])
    app.add_api_route(
        "/sessions/{session_id}/api/offer", offer, methods=["POST", "PATCH"]
    )

    ssl = {}
    scheme = "http"
    if args.https:
        cert, key = certificate(host, args.cert_dir)
        ssl, scheme = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}, "https"
    shown = "localhost" if host in ("0.0.0.0", "localhost") else host
    print(f"open {scheme}://{shown}:{args.port}/ (the GPU side: {args.cluster})")
    uvicorn.run(app, host=host, port=args.port, **ssl)


if __name__ == "__main__":
    main()
