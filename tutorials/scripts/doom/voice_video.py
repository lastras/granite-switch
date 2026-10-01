# SPDX-License-Identifier: Apache-2.0
"""Give a recorded Doom video its voices: speak the transcript record_video.py wrote.

Every line in ``<video>.talk.json`` is spoken at the moment it was said, the
player's in the default voice and the watcher's pitched up (or, with
``--partner-voice``, in another model's default voice), so the two are told
apart; a line starts after the previous one ends. The game's own sound
(``<video>.sfx.wav``, from ``record_video.py --sfx``) goes under the voices,
turned down while a line plays. The track is then muxed into the video. Two Chatterbox models: ``standard`` has the expressivity controls
(``--exaggeration``: emotional intensity, 0.5 neutral; ``--cfg-weight``: lower is
slower, more deliberate) and cannot voice sound tags, so they are dropped;
``turbo`` is about twice as fast, ignores those controls and voices tags such as
[chuckle]. ``--ref clip.wav`` makes the player's voice a clone of the clip's (for
Turbo, e.g. a line of the standard voice at ``--pitch -3``: the same voice at
Turbo's speed).

``--tts kokoro`` uses Kokoro-82M instead (``pip install kokoro``): one of its
own voices (``--kokoro-voice``, e.g. ``am_michael``; no cloning, no sound tags),
five to ten times faster than Turbo, which is why the live demo uses it.

``--serve PORT`` voices lines for the live demo instead (:mod:`doom_live`), one
request at a time over a local connection.

Runs in an environment with Chatterbox (``pip install chatterbox-tts``) or
Kokoro, each of which pins its own torch, so not the vLLM one::

    python voice_video.py out/clip.mp4 --out out/clip_voiced.mp4
    python voice_video.py --serve 7000 --tts turbo --ref voices/him.wav
    python voice_video.py --serve 7000 --tts kokoro --kokoro-voice am_michael --pitch -2
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

TAG = re.compile(r"\[[a-z ]+\]\s*")  # [chuckle], [sigh], ...
KOKORO_HZ = 24_000


def pitch_up(wav: np.ndarray, factor: float = 1.15) -> np.ndarray:
    """A second voice from the same model: shorter, higher."""
    n = int(len(wav) / factor)
    x = np.linspace(0, len(wav) - 1, n)
    return np.interp(x, np.arange(len(wav)), wav).astype(np.float32)


def shift(wav: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """The same voice lower (or higher), at the same pace."""
    import librosa

    return librosa.effects.pitch_shift(wav, sr=sr, n_steps=semitones).astype(np.float32)


def game_sound(
    path: Path,
    sr: int,
    n: int,
    spoken: list[tuple[int, int]],
    gain_db: float,
    duck_db: float,
) -> np.ndarray:
    """The game's stereo sound at ``sr``, ``n`` samples, ``gain_db`` down, and
    ``duck_db`` further down (with 0.1 s ramps) wherever a line is spoken."""
    import librosa

    fx, fsr = sf.read(path, dtype="float32", always_2d=True)
    fx = librosa.resample(fx.T, orig_sr=fsr, target_sr=sr).T[:n]
    fx = np.pad(fx, ((0, n - len(fx)), (0, 0)))
    level = np.ones(n, np.float32)
    for a, b in spoken:
        level[a:b] = 10 ** (duck_db / 20)
    ramp = int(0.1 * sr)
    level = np.convolve(level, np.ones(ramp) / ramp, "same").astype(np.float32)
    return fx * (10 ** (gain_db / 20)) * level[:, None]


def ffmpeg_exe(given: str | None) -> str:
    if given:
        return given
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


class Voices:
    """The voices, each model loaded once. ``say(kind, text)`` voices a line:
    Chatterbox ``standard`` (with the expressivity controls; its sound tags
    dropped) or ``turbo`` (tags voiced), or ``kokoro`` (Kokoro-82M in its voice
    ``kokoro_voice``, at ``speed``; tags dropped). ``pitch``: semitones, same
    pace. :meth:`clone` makes a Chatterbox ``kind`` speak in a reference clip's
    voice from then on (``default=True`` in :meth:`say` still gets its own)."""

    def __init__(
        self,
        device: str = "cuda",
        exaggeration: float = 0.8,
        cfg_weight: float = 0.3,
        temperature: float = 0.8,
        kokoro_voice: str = "am_michael",
        speed: float = 1.0,
    ):
        self.device = device
        self.kokoro_voice, self.speed = kokoro_voice, speed
        self.opts = {
            "standard": {
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
                "temperature": temperature,
            },
            "turbo": {"temperature": temperature},
        }
        self.models: dict = {}
        self._conds: dict = {}  # kind -> (its own voice, the cloned one)

    def model(self, kind: str):
        if kind not in self.models and kind == "kokoro":
            from kokoro import KPipeline

            self.models[kind] = KPipeline(
                lang_code="a", device=self.device, repo_id="hexgrad/Kokoro-82M"
            )
        if kind not in self.models:
            if kind == "turbo":
                from chatterbox.tts_turbo import ChatterboxTurboTTS as Model
            else:
                from chatterbox.tts import ChatterboxTTS as Model
            self.models[kind] = Model.from_pretrained(device=self.device)
        return self.models[kind]

    def sr(self, kind: str) -> int:
        return KOKORO_HZ if kind == "kokoro" else self.model(kind).sr

    def clone(self, kind: str, ref: str) -> None:
        m = self.model(kind)
        own = m.conds
        m.prepare_conditionals(ref)
        self._conds[kind] = (own, m.conds)

    def say(
        self, kind: str, text: str, pitch: float = 0.0, default: bool = False
    ) -> np.ndarray:
        m = self.model(kind)
        if kind in self._conds:
            m.conds = self._conds[kind][0 if default else 1]
        if kind != "turbo":
            text = TAG.sub("", text).strip()
        if kind == "kokoro":
            parts = [
                a for _, _, a in m(text, voice=self.kokoro_voice, speed=self.speed)
            ]
            wav = np.concatenate(
                [np.zeros(0, np.float32)] + [np.asarray(p) for p in parts]
            )
        else:
            wav = m.generate(text, **self.opts[kind]).squeeze().cpu().numpy()
        wav = wav.astype(np.float32)
        return shift(wav, self.sr(kind), pitch) if pitch else wav


TTS_AUTHKEY = b"granite-switch-doom-tts"


def serve(args) -> None:
    """Voice lines for the live demo (doom_live.py) until the connection
    closes: ``("say", text)`` -> ``("audio", int16 PCM bytes, rate, ms)``."""
    import time
    from multiprocessing.connection import Listener

    voices = Voices(
        args.device,
        args.exaggeration,
        args.cfg_weight,
        args.temperature,
        args.kokoro_voice,
        args.speed,
    )
    if args.ref and args.tts != "kokoro":  # Kokoro has its own voices only
        voices.clone(args.tts, str(args.ref))
    # With the pitch: librosa compiles its resampler on first use (seconds).
    voices.say(args.tts, "Warming up.", args.pitch)
    sr = voices.sr(args.tts)
    with Listener(("127.0.0.1", args.serve), authkey=TTS_AUTHKEY) as listener:
        print(f"voicing on 127.0.0.1:{args.serve} ({args.tts}, {sr} Hz)", flush=True)
        conn = listener.accept()
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                return
            t0 = time.perf_counter()
            wav = voices.say(args.tts, msg[1], args.pitch)
            pcm = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            conn.send(("audio", pcm, sr, (time.perf_counter() - t0) * 1000))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video", type=Path, nargs="?")
    ap.add_argument("--talk", type=Path, help="Default: <video>.talk.json")
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--serve",
        type=int,
        metavar="PORT",
        help="Voice lines for the live demo on this local port (doom_live.py)",
    )
    ap.add_argument(
        "--ref", type=Path, help="A clip whose voice the player's voice clones"
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gap", type=float, default=0.15, help="Seconds between lines")
    ap.add_argument("--ffmpeg", help="ffmpeg binary (default: imageio-ffmpeg's)")
    ap.add_argument(
        "--tts", choices=("standard", "turbo", "kokoro"), default="standard"
    )
    ap.add_argument(
        "--kokoro-voice", default="am_michael", help="kokoro: one of its voices"
    )
    ap.add_argument("--speed", type=float, default=1.0, help="kokoro: pace")
    ap.add_argument("--exaggeration", type=float, default=0.8, help="standard only")
    ap.add_argument("--cfg-weight", type=float, default=0.3, help="standard only")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument(
        "--pitch",
        type=float,
        default=0.0,
        help="Semitones to shift the player's voice (e.g. -3: deeper, same pace)",
    )
    ap.add_argument(
        "--sfx",
        type=Path,
        help="The game's sound (default: <video>.sfx.wav from record_video.py --sfx, "
        "if there)",
    )
    ap.add_argument("--sfx-gain", type=float, default=-10.0, help="dB")
    ap.add_argument(
        "--duck",
        type=float,
        default=-14.0,
        help="dB on the game sound while a line plays",
    )
    ap.add_argument(
        "--partner-voice",
        choices=("pitched", "standard", "turbo"),
        default="pitched",
        help="The watcher's voice: the player's pitched up, or a model's own default "
        "voice (Chatterbox-Turbo's is an American woman's)",
    )
    args = ap.parse_args()
    if args.serve:
        serve(args)
        return
    if args.video is None or args.out is None:
        ap.error("a video and --out are required (or --serve)")

    talk = json.loads((args.talk or args.video.with_suffix(".talk.json")).read_text())
    voices = Voices(
        args.device,
        args.exaggeration,
        args.cfg_weight,
        args.temperature,
        args.kokoro_voice,
        args.speed,
    )
    if args.ref and args.tts != "kokoro":
        voices.clone(args.tts, str(args.ref))
    sr = voices.sr(args.tts)
    dur = talk["frames"] / talk["fps"]
    track = np.zeros(int((dur + 10) * sr), np.float32)
    spoken: list[tuple[int, int]] = []  # sample ranges with a line playing
    t_free = 0.0
    for ev in talk["lines"]:
        text = ev["text"]
        if not TAG.sub("", text).strip():
            continue
        if ev["who"] == "player" and args.partner_voice != "pitched":
            kind = args.partner_voice  # in that model's own voice
            wav = voices.say(kind, text, default=True)
            if voices.sr(kind) != sr:
                import librosa

                wav = librosa.resample(wav, orig_sr=voices.sr(kind), target_sr=sr)
        elif ev["who"] == "player":
            wav = pitch_up(voices.say(args.tts, text))
        else:
            wav = voices.say(args.tts, text, args.pitch)
        start = max(ev["t"], t_free)
        if start >= dur:  # lines queued past the end of the video are dropped
            break
        i = int(start * sr)
        if i + len(wav) > len(track):
            track = np.concatenate(
                [track, np.zeros(i + len(wav) - len(track), np.float32)]
            )
        track[i : i + len(wav)] += wav
        spoken.append((i, i + len(wav)))
        t_free = start + len(wav) / sr + args.gap
        print(f"{start:6.1f}s  {ev['who']:<6} {ev['text']}", flush=True)
    track = track[: int(dur * sr)]
    mix = np.stack([track, track], axis=1)
    sfx_path = args.sfx or args.video.with_suffix(".sfx.wav")
    if sfx_path.exists():
        mix += game_sound(sfx_path, sr, len(track), spoken, args.sfx_gain, args.duck)
    wav_path = args.out.with_suffix(".wav")
    sf.write(wav_path, np.clip(mix, -1.0, 1.0), sr)
    subprocess.run(
        [
            ffmpeg_exe(args.ffmpeg),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(args.video),
            "-i",
            str(wav_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-shortest",
            str(args.out),
        ],
        check=True,
    )
    print(f"wrote {args.out} ({len(talk['lines'])} lines, {dur:.0f} s)")


if __name__ == "__main__":
    main()
