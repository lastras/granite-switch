# SPDX-License-Identifier: Apache-2.0
"""Record the model playing, with a telemetry panel, as an MP4.

A script of segments drives the video. Each segment is either an adapter name
or a free-text instruction (read by the orders adapter: a style order switches
the style adapter; the video plays no maneuvers), plus seconds of game time::

    python record_video.py --model models/doom-round0 --out out/granite_doom.mp4 \
        --segment "fighter" 20 \
        --segment "stop fighting and grab health" 20 \
        --segment "collect all the loot" 20

Every action in the video comes from the composed checkpoint, one token per tic,
and so do the critic's danger (every tic) and the weapon plan (every 0.5 s): all
three adapters read the same history in one engine step. The panel shows the
measured wall-clock time of those steps (a 1 s rolling median, so it is
readable) against the 28.6 ms tic. The strip below the game is the full action
distribution as a heatmap, one row per action and one column per tic, filling
and then sliding, with the critic's danger under it. The game runs synchronously
and the video plays at 35 fps, so it is real game speed.

``--policy expert`` records the scripted teacher instead (no GPU needed; its
distribution is one-hot), which is how the layout is checked on a laptop.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conversation import Conversation, Exchange
from doom_env import AUDIO_HZ, MATCH_TICS, TIC_HZ, TIC_MS, DoomEnv
from expert import BEHAVIORS, PLAN_EVERY_TICS
from history import History
from overlay import KINDS, Overlay
from policy import ARMS, CRITIC, NARRATOR, ORDERS, make_policy, state_text
from talk import (
    IDLE_S,
    EventLog,
    EventPartner,
    TalkClock,
    Tracker,
    brief,
    game_state,
    moment_events,
    sound_tag,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--policy", choices=("vllm", "expert"), default="vllm")
    ap.add_argument("--model", help="Composed checkpoint (for --policy vllm)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--segment",
        nargs=2,
        action="append",
        metavar=("ADAPTER_OR_INSTRUCTION", "SECONDS"),
        required=True,
    )
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument(
        "--temperature", type=float, default=1.0, help="Style and planner sampling"
    )
    ap.add_argument("--layout", default="log", help="The game adapters' prompt layout")
    ap.add_argument(
        "--no-hud",
        action="store_true",
        help="No status bar (collect.py and the engine have one)",
    )
    ap.add_argument("--no-critic", action="store_true", help="Do not ask the critic")
    ap.add_argument("--eager", action="store_true", help="No CUDA graphs")
    ap.add_argument("--cudagraph-mode", default="FULL")
    ap.add_argument("--fa2", action="store_true", help="FlashAttention 2, not 3")
    ap.add_argument(
        "--talk",
        action="store_true",
        help="The player talks on its own: after salient events and silences",
    )
    ap.add_argument(
        "--idle-s", type=float, default=IDLE_S, help="Silence before a remark"
    )
    ap.add_argument(
        "--player",
        action="append",
        default=[],
        metavar="SECONDS:TEXT",
        help="What the watcher says, and when (video seconds); repeatable",
    )
    ap.add_argument(
        "--partner-events",
        action="store_true",
        help="A watcher who reacts to what happens (talk.EventPartner)",
    )
    ap.add_argument(
        "--sfx",
        action="store_true",
        help="Keep the game's sound (<out>.sfx.wav; voice_video.py mixes it in)",
    )
    args = ap.parse_args()

    import imageio.v2 as imageio

    if args.policy == "vllm":
        import torch

        where = torch.cuda.get_device_name(0).replace("NVIDIA ", "")
        pol = make_policy(
            "vllm",
            args.model,
            warmup=100,
            temperature=args.temperature,
            layout=args.layout,
            enforce_eager=args.eager,
            cudagraph_mode=args.cudagraph_mode,
            attention={"flash_attn_version": 2} if args.fa2 else None,
        )
    else:
        where = "scripted teacher, no model"
        pol = make_policy("expert")
    env = DoomEnv(
        seed=args.seed,
        resolution="640X480",
        hud=not args.no_hud,
        timeout_tics=MATCH_TICS,  # a 10-minute match, as the narrator's data
        audio=args.sfx,
    )
    sfx: list[np.ndarray] = []  # the game's sound, one tic per frame
    silent_tic = np.zeros((AUDIO_HZ // TIC_HZ, 2), np.int16)
    obs = env.reset(seed=args.seed)
    hist = History(getattr(pol, "tok", None))  # the game log
    conv = Conversation()  # what the narrator reads, with the game state
    tracker = Tracker(match_s=MATCH_TICS / TIC_HZ)  # the game state's facts
    log, last_ex = EventLog(), None  # the match's events; his last exchange's tick
    fired = tracker.update(obs)
    clock = TalkClock(args.idle_s) if args.talk else None
    partner = EventPartner(random.Random(args.seed)) if args.partner_events else None
    # Who writes spoken lines: the narrator adapter, or the base model without one.
    talker = getattr(pol, "talker", None) or "base"
    models = ["base", *BEHAVIORS, ARMS, CRITIC, ORDERS]
    view = Overlay(models + ([NARRATOR] if talker != "base" else []))
    talk_until, routed = -1, False  # the frame the line writer runs through
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(args.out), fps=TIC_HZ, codec="libx264", quality=8, macro_block_size=1
    )
    lat: deque[float] = deque(maxlen=1000)
    last_second: deque[float] = deque(maxlen=TIC_HZ)
    all_ms = []
    script = sorted(
        (float(t), text) for t, text in (x.split(":", 1) for x in args.player)
    )
    caps: list[tuple[float, str, str]] = []  # (video s, who, text)
    frame_i = 0
    tag_rng = random.Random(args.seed)
    live, fist = 0, 0  # tics alive, and of those holding the fist
    kind = KINDS.get(getattr(pol, "placement", "base"), "")
    adapter = BEHAVIORS[0]
    for what, secs in args.segment:
        info: dict = {"instruction": None}
        if what in BEHAVIORS:
            adapter = what
        else:
            r = pol.order(what)
            if r.kind in BEHAVIORS:  # a style order; the rest keep the style
                adapter = r.kind
            routed = True
            info = {
                "instruction": what,
                "routed": r.kind,
                "route_p": r.prob,
                "route_ms": r.ms,
            }
        print(f"segment: {what!r} -> {adapter} for {secs}s", flush=True)
        action, critic, plan = "wait", {}, None
        for _ in range(int(float(secs) * TIC_HZ)):
            weapon = None
            active: set[str] = {ORDERS} if routed else set()  # models run this tic
            routed = False
            log.add(fired)
            if clock is not None:
                clock.event(fired)
            if partner is not None:
                partner.event(fired)
            if obs.dead:  # respawning: nothing to decide
                view.heat.push({}, False, critic)
            else:
                want = [adapter] if args.no_critic else [adapter, CRITIC]
                if obs.tick % PLAN_EVERY_TICS == 0:
                    want.append(ARMS)
                # One request per adapter; vLLM computes their shared prefix once.
                active |= set(want)
                decs = pol.decide_many(obs, tuple(want), hist)
                d = decs[adapter]
                action = d.action
                critic = decs[CRITIC].probs if CRITIC in decs else {}
                if ARMS in decs:
                    weapon = int(decs[ARMS].action)
                    plan = {"slot": weapon, "probs": decs[ARMS].probs}
                view.heat.push(d.probs, True, critic)
                last_second.append(d.ms)
                lat.append(d.ms)
                all_ms.append(d.ms)
                now = frame_i / TIC_HZ
                player = script.pop(0)[1] if script and script[0][0] <= now else None
                if player is None and partner is not None:
                    player = partner.poll(obs.tick)
                # On its own: soon after a salient event, or after a silence, as
                # the engine (and the narrator's data) does.
                cue = None
                if clock is not None and player is None and len(hist.entries) >= 5:
                    cue = clock.due(obs.tick)
                if hasattr(pol, "narrate") and (player or cue):
                    facts = tracker.facts()
                    b = brief(hist.entries, state_text(obs), facts)
                    gs = game_state(state_text(obs), facts, log.events, adapter)
                    t_talk = time.perf_counter()
                    line = pol.narrate([(conv, gs, player)])[0]
                    # Shown as running for as many tics as writing the line took.
                    talk_ms = (time.perf_counter() - t_talk) * 1000
                    talk_until = frame_i + max(1, math.ceil(talk_ms / TIC_MS)) - 1
                    # Tags: voiced, not in the conversation.
                    voiced = sound_tag(b, tag_rng) + line
                    if line:
                        moment = moment_events(log.since(last_ex, obs.tick))
                        conv.add(Exchange(obs.tick, moment, player, line))
                        last_ex = obs.tick
                    if clock is not None:
                        clock.said(obs.tick, line, reply=player is not None)
                    if player:
                        caps.append((now, "player", player))
                    caps.append((now, "bot", voiced))
                    print(
                        f"t{now:5.1f}  {'[' + player + '] ' if player else ''}-> {line}"
                        f"\n         ({b})"
                    )
            if frame_i <= talk_until:
                active.add(talker)
            view.act.push(active)
            frame = env.frame()
            text, hud = obs.text, (obs.hp, obs.armor, obs.weapon)
            if not obs.dead:
                live += 1
                fist += obs.weapon == "fist"
            hist.observe(obs, action)
            obs = env.step(action, weapon=weapon)
            if obs.done:
                obs = env.reset()
                hist.reset()
                conv = Conversation()
                tracker = Tracker(match_s=MATCH_TICS / TIC_HZ)
                log, last_ex = EventLog(), None
            fired = tracker.update(obs)
            a = np.fromiter(lat, dtype=np.float64)
            img = view.draw(
                frame,
                {
                    **info,
                    "adapter": adapter,
                    "kind": kind,
                    "ms": float(np.median(last_second)),
                    "p50": float(np.percentile(a, 50)),
                    "p99": float(np.percentile(a, 99)),
                    "action": action,
                    "critic": critic,
                    "plan": plan,
                    "stats": env.stats.as_dict(),
                    "hp": hud[0],
                    "armor": hud[1],
                    "weapon": hud[2],
                    "text": text,
                    "tick": obs.tick,
                    "gpu": where,
                },
                caps,
                frame_i / TIC_HZ,
            )
            writer.append_data(np.asarray(img))
            if args.sfx:  # this frame's tic of game sound
                sound = env.audio() if env.audio() is not None else silent_tic
                sfx.append(np.asarray(sound, np.int16).reshape(-1, 2))
            frame_i += 1
    writer.close()
    env.close()
    if args.sfx:
        import wave

        with wave.open(str(args.out.with_suffix(".sfx.wav")), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(AUDIO_HZ)
            w.writeframes(np.concatenate(sfx).tobytes())
    if caps:
        talk = [{"t": round(t, 3), "who": who, "text": text} for t, who, text in caps]
        args.out.with_suffix(".talk.json").write_text(
            json.dumps({"fps": TIC_HZ, "frames": frame_i, "lines": talk}, indent=1)
        )
    a = np.asarray(all_ms)
    st = env.stats.as_dict()
    print(
        f"match: frags {st['frags']} deaths {st['deaths']} rank {st['rank']}; "
        f"holding the fist {100 * fist / max(1, live):.0f}% of live tics"
    )
    print(
        f"wrote {args.out}: {a.size} decisions, p50 {np.percentile(a, 50):.2f} ms, "
        f"p99 {np.percentile(a, 99):.2f} ms, max {a.max():.2f} ms"
    )


if __name__ == "__main__":
    main()
