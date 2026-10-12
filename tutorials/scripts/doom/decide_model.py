# SPDX-License-Identifier: Apache-2.0
"""The decision adapters' checkpoint: compose it, check it against PEFT, time it, play with it.

Tutorial step 1's composed side: ``decide.py`` writes the data (the RL teacher
at a compilable boundary), ``train_alora.py --records`` trains ``move`` and
``weapon`` (aLoRA), and this script makes and measures the one checkpoint that
serves both. Every prompt is :mod:`game_chat`'s: the match's conversation in
Granite's chat template, the look, and the adapter's control token on the
generation prompt (:meth:`game_chat.GameChat.prompt_ids`). Under the aLoRA the
conversation runs on base weights, so its KV is cached once and shared by both
adapters and by every tic: each decision prefills only the look's turn.

* ``compose``: the two trained adapters, staged and composed
  (``build_model.stage`` / ``build_model.compose``).
* ``verify``: the composed checkpoint in vLLM against PEFT, on each adapter's
  held-out records (``heldout_preds.jsonl``: argmax agreement >= 99%, mean
  total-variation distance); then live against cold: one match's prompts sent
  one tic at a time (a warm prefix cache), against the same prompts in batches
  with the cache cleared.
* ``bench``: the decisions' latency in the live regime, on held-out matches
  (``decide.py``'s records, never trained on), replayed in order:
  every recorded tic's requests in one engine step, as live play sends them
  (the move; on a planner tic, the weapon too); steady tics, planner tics and
  tics after a push apart.
* ``play``: whole matches, lockstep (each tic waits for its decisions): the
  student (``--policy model``), or the teacher in the same harness, at the
  same settings and seeds (``--policy teacher``), the fair comparison.

Runs in the demo's environment (vLLM, granite_switch; no Mellea)::

    python decide_model.py compose --runs runs/s1 --base ibm-granite/granite-4.1-3b \\
        --out models/decide-s1
    python decide_model.py verify --runs runs/s1 --model models/decide-s1
    python decide_model.py bench --model models/decide-s1 --rows data/s1/bench.jsonl.gz
    python decide_model.py play --model models/decide-s1 --matches 12 --bots cast
    python decide_model.py play --policy teacher --teacher runs/teacher_rl0_706M.pt \\
        --matches 12 --bots cast
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from collections import Counter
from pathlib import Path

import game_chat
import numpy as np
from doom_env import ACTIONS, TIC_HZ, DoomEnv, isolate_workdir
from expert import PLAN_EVERY_TICS
from game_chat import MOVE, OUTPUTS, RESOLUTION, STYLE, WEAPON, GameChat, look
from talk import Tracker


# ── The checkpoint in vLLM ─────────────────────────────────────────────────────
class Decider:
    """The composed checkpoint, served in-process by vLLM, with the decision
    prompts' pieces, each adapter's vocabulary and its sampling (one token,
    from the allowed outputs, at ``temperature``: 1.0, as the student was
    calibrated to a stochastic teacher; its distribution comes with it)."""

    def __init__(
        self,
        model: str,
        *,
        max_num_seqs: int = 16,
        gpu_mem: float = 0.5,
        temperature: float = 1.0,
        max_model_len: int = 16384,
        enforce_eager: bool = False,
    ):
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from policy import engine_kwargs, output_token_ids
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            **engine_kwargs(
                model,
                max_num_seqs=max_num_seqs,
                gpu_memory_utilization=gpu_mem,
                max_model_len=max_model_len,
                enforce_eager=enforce_eager,
            )
        )
        self.tok = self.llm.get_tokenizer()
        where = game_chat.placement(self.tok)
        if where != "alora":
            raise SystemExit(f"{model}: the decision adapters are {where}, not aLoRA")
        self.pieces = game_chat.Pieces(self.tok)
        self.vocab = {a: output_token_ids(self.tok, w) for a, w in OUTPUTS.items()}
        self.words = {a: {i: w for w, i in v.items()} for a, v in self.vocab.items()}
        self.sp = {
            a: SamplingParams(
                max_tokens=1,
                temperature=temperature,
                allowed_token_ids=list(v.values()),
                logprobs=len(v),
            )
            for a, v in self.vocab.items()
        }

    def run(self, asks: list[tuple[list[int], str]]) -> list[dict]:
        """One engine call: each ``(prompt ids, adapter)``'s word, its
        distribution, and the prompt's fresh and cached tokens."""
        from policy import output_dist
        from vllm.inputs import TokensPrompt

        outs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=p) for p, _ in asks],
            [self.sp[a] for _, a in asks],
            use_tqdm=False,
        )
        res = []
        for (p, a), o in zip(asks, outs):
            cached = o.num_cached_tokens or 0
            res.append(
                {
                    "word": self.words[a][o.outputs[0].token_ids[0]],
                    "probs": dict(output_dist(o, self.words[a])),
                    "fresh": len(p) - cached,
                    "cached": cached,
                }
            )
        return res


def asks_of(chat: GameChat, lk: dict, tick: int) -> list[tuple[list[int], str]]:
    """A tic's requests: the move; on a weapon planner's tic, the weapon too."""
    out = [(chat.prompt_ids(lk, MOVE), MOVE)]
    if tick % PLAN_EVERY_TICS == 0:
        out.append((chat.prompt_ids(lk, WEAPON), WEAPON))
    return out


# ── compose ────────────────────────────────────────────────────────────────────
def compose(args) -> None:
    """``compose``: ``<runs>/move`` and ``<runs>/weapon``, staged (aLoRA) and
    composed into one checkpoint."""
    import build_model

    target = args.base.rstrip("/").split("/")[-1]
    stage_root = args.out.parent / f"{args.out.name}-stage"
    paths = []
    for name in OUTPUTS:
        d = args.runs / name
        if not (d / "adapter_config.json").exists():
            raise SystemExit(f"missing trained adapter: {d}")
        paths.append(build_model.stage(d, name, stage_root, target, "alora"))
    build_model.compose(paths, args.base, args.out)
    print(f"\ncomposed {', '.join(OUTPUTS)} -> {args.out}")


# ── verify ─────────────────────────────────────────────────────────────────────
MARGIN = 0.05  # PEFT's top-1 minus top-2 probability, for a "clear" decision


def tv(p: dict, q: dict) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def argmax(d: dict) -> str:
    return max(d, key=d.get)


def verify(args) -> dict:
    """``verify``: the composed checkpoint against PEFT on each adapter's
    held-out records (the prompt PEFT read, its generation prompt given the
    adapter's control token), then live against cold (:func:`live_check`)."""
    dec = Decider(args.model)
    report = {}
    for name in OUTPUTS:
        rows = [json.loads(x) for x in open(args.runs / name / "heldout_preds.jsonl")]
        rows = rows[: args.limit]
        head = dec.pieces.gen(None)
        asks = []
        for r in rows:
            ids = r["prompt_ids"]
            assert ids[-len(head) :] == head, "a prompt ends with the generation prompt"
            asks.append((ids[: -len(head)] + dec.pieces.gen(name), name))
        got = []
        for i in range(0, len(asks), 64):
            got += dec.run(asks[i : i + 64])
        same = [argmax(g["probs"]) == r["peft"] for g, r in zip(got, rows)]
        clear = []  # where PEFT's top-1 leads its second by MARGIN or more
        for s, r in zip(same, rows):
            top = sorted(r["peft_probs"].values(), reverse=True)
            if top[0] - top[1] >= MARGIN:
                clear.append(s)
        dist = [tv(g["probs"], r["peft_probs"]) for g, r in zip(got, rows)]
        teacher = [tv(g["probs"], r["teacher"]) for g, r in zip(got, rows)]
        agree = sum(same) / max(1, len(same))
        report[name] = {
            "n": len(rows),
            "agree_with_peft": round(agree, 4),
            "mean_tv_to_peft": round(float(np.mean(dist)), 4),
            "agree_when_peft_margin_ge": {
                "margin": MARGIN,
                "n": len(clear),
                "agree": round(sum(clear) / max(1, len(clear)), 4),
            },
            "mean_tv_to_teacher": round(float(np.mean(teacher)), 4),
            "pass": agree >= 0.99,
        }
        print(
            f"{name:<7} n={len(rows):>5}  composed==PEFT {100 * agree:6.2f}%  mean TV "
            f"{np.mean(dist):.4f} (to the teacher {np.mean(teacher):.4f})  "
            f"{'PASS' if agree >= 0.99 else 'FAIL'}",
            flush=True,
        )
    report["live"] = live_check(dec, args.seconds)
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))
    return report


def live_check(dec: Decider, seconds: float, seed: int = 5) -> dict:
    """One match's decisions sent as live play sends them (one game, each
    tic's requests in one engine step, the prefix cache warm from the tic
    before), against the same prompts sent cold (the cache cleared, batches
    of 32). The two must agree: a serving fault that shows only in small
    cached steps over a long conversation fails here and nowhere else. The
    student drives, so the states are its own."""
    env = DoomEnv(
        seed=seed,
        resolution=RESOLUTION,
        hud=True,
        timeout_tics=int(seconds * TIC_HZ),
        bots="cast",
    )
    obs = env.reset(seed=seed)
    tracker = Tracker(match_s=seconds)
    fired = tracker.update(obs)
    chat = GameChat(game_chat.players_of(tracker.facts()), dec.pieces)
    asked, live = [], []
    while not obs.done:
        chat.observe(fired, obs.tick)
        if obs.dead:
            obs = env.step("wait")
        else:
            asks = asks_of(chat, look(obs), obs.tick)
            got = dec.run(asks)
            asked += asks
            live += got
            slot = int(got[1]["word"]) if len(got) > 1 else None
            obs = env.step(got[0]["word"], weapon=slot)
        fired = [] if obs.done else tracker.update(obs)
    env.close()
    dec.llm.reset_prefix_cache()
    cold = []
    for i in range(0, len(asked), 32):
        cold += dec.run(asked[i : i + 32])
    rep = {}
    for name in OUTPUTS:
        pairs = [(x, y) for (_, a), x, y in zip(asked, live, cold) if a == name]
        d = [tv(x["probs"], y["probs"]) for x, y in pairs]
        agree = sum(argmax(x["probs"]) == argmax(y["probs"]) for x, y in pairs)
        agree /= max(1, len(pairs))
        rep[name] = {
            "n": len(pairs),
            "mean_tv": round(float(np.mean(d)), 4),
            "agree": round(agree, 4),
            "pass": float(np.mean(d)) < 0.02 and agree >= 0.97,
        }
        print(
            f"live one-game steps vs cold batches  {name:<7} n={len(pairs):>4}  "
            f"mean TV {np.mean(d):.4f}  same argmax {100 * agree:5.1f}%  "
            f"{'PASS' if rep[name]['pass'] else 'FAIL'}",
            flush=True,
        )
    return rep


# ── bench ──────────────────────────────────────────────────────────────────────
def bench(args) -> dict:
    """``bench``: held-out matches replayed in order, ``--games`` at once (one
    engine step per recorded tic of each, all the games' requests together),
    each tic's requests rebuilt from its record (both adapters read the same
    prompt but its last token). Latency is per step, wall clock; tics are
    told apart by what the step held: a weapon request too (planner), or a
    push since the game's step before (the conversation grew)."""
    dec = Decider(args.model, max_num_seqs=max(16, 2 * args.games))
    by_ep: dict[str, dict[int, list]] = {}
    for r in game_chat.iter_rows(args.rows):
        by_ep.setdefault(r["episode"], {}).setdefault(r["meta"]["t"], []).append(r)
    eps = sorted(by_ep)
    # Per game: its recorded tics in order, each with its records.
    games = [sorted(by_ep[eps[i % len(eps)]].items()) for i in range(args.games)]
    chats = [GameChat([], dec.pieces) for _ in games]
    n_steps = min(len(g) for g in games)
    if args.steps:
        n_steps = min(n_steps, args.steps)
    rows = []
    for k in range(n_steps):
        asks, kinds = [], []
        for chat, g in zip(chats, games):
            tick, recs = g[k]
            before = len(chat.messages)
            lk = chat.follow(recs[0]["prompt"]["messages"])
            a = asks_of(chat, lk, tick)
            asks += a
            if len(chat.messages) > before:
                kinds.append("push")
            else:
                kinds.append("planner" if len(a) > 1 else "steady")
        t0 = time.perf_counter()
        got = dec.run(asks)
        ms = (time.perf_counter() - t0) * 1000
        if k >= args.warm:
            kind = next(x for x in ("push", "planner", "steady") if x in kinds)
            rows.append(
                {
                    "kind": kind,
                    "ms": ms,
                    "fresh": sum(x["fresh"] for x in got),
                    "cached": sum(x["cached"] for x in got),
                    "prompt": max(len(p) for p, _ in asks),
                }
            )
    out = {"games": args.games, "steps": len(rows), "by_kind": {}}
    print(f"{args.games} games, {len(rows)} steps (each: one engine call)")
    for kind in ("steady", "planner", "push", "all"):
        rs = [r for r in rows if kind in ("all", r["kind"])]
        if not rs:
            continue
        ms = [r["ms"] for r in rs]
        out["by_kind"][kind] = s = {
            "n": len(rs),
            "p50_ms": round(float(np.percentile(ms, 50)), 2),
            "p99_ms": round(float(np.percentile(ms, 99)), 2),
            "fresh_tokens": round(float(np.mean([r["fresh"] for r in rs])), 1),
            "cached_tokens": round(float(np.mean([r["cached"] for r in rs])), 1),
            "prompt_tokens_max": max(r["prompt"] for r in rs),
        }
        print(
            f"  {kind:<8} n={s['n']:>5}  p50 {s['p50_ms']:6.2f} ms  p99 {s['p99_ms']:6.2f} "
            f"ms  fresh {s['fresh_tokens']:6.1f}  cached {s['cached_tokens']:8.1f} "
            f"(a tic: {1000 / TIC_HZ:.1f} ms)"
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1))
    return out


# ── play ───────────────────────────────────────────────────────────────────────
def _play_worker(conn, match: dict, seconds: float, teacher: str | None) -> None:
    """One match. With ``teacher``, the RL teacher plays it here, sampling (as
    ``decide.py`` plays). Else: every tic's look and events go to the
    main process (``("tic", tick, look | None, fired)``; None while dead, when
    no decision is needed), and the reply is the move and the slot."""
    isolate_workdir()
    env = DoomEnv(
        seed=match["seed"],
        resolution=RESOLUTION,
        hud=True,
        timeout_tics=int(seconds * TIC_HZ),
        bots=match["bots"],
        n_bots=match["n_bots"],
    )
    obs = env.reset(seed=match["seed"])
    tracker = Tracker(match_s=seconds)
    fired = tracker.update(obs)
    rl = rng = None
    if teacher:
        from rl_teacher import RLPolicy

        rl, rng = RLPolicy(teacher), random.Random(match["seed"])
    conn.send(("begin", game_chat.players_of(tracker.facts())))
    while not obs.done:
        if rl is None:
            conn.send(("tic", obs.tick, None if obs.dead else look(obs), fired))
        if obs.dead:
            obs = env.step("wait")
        elif rl is not None:
            pm, pw = rl.step(obs, STYLE)
            a = ACTIONS[rng.choices(range(len(ACTIONS)), weights=pm)[0]]
            slot = None
            if obs.tick % PLAN_EVERY_TICS == 0:
                slot = rl.weapon_slot(pw, obs, True, rng)
            obs = env.step(a, weapon=slot)
        else:
            a, slot = conn.recv()
            obs = env.step(a, weapon=slot)
        fired = [] if obs.done else tracker.update(obs)
    stats = env.stats.as_dict()
    env.close()
    conn.send(("done", stats))


def play(args) -> list[dict]:
    """``play``: ``--matches`` ten-minute fighter matches at once, seeds from
    ``--seed`` (as ``rl_teacher.py eval``'s), each in its own process; with
    ``--policy model`` every live tic of every match is decided in one engine
    step (lockstep: no tic waits on the clock), each game's conversation kept
    here. The stats rows and ``collect.summarize``'s table."""
    from collect import summarize

    dec = None
    if args.policy == "model":
        dec = Decider(args.model, max_num_seqs=max(16, 2 * args.matches))
    teacher = str(Path(args.teacher).resolve()) if args.policy == "teacher" else None
    ctx = mp.get_context("spawn")
    conns, procs = [], []
    for i in range(args.matches):
        m = {"seed": args.seed + i, "bots": args.bots, "n_bots": args.n_bots}
        parent, child = ctx.Pipe()
        p = ctx.Process(target=_play_worker, args=(child, m, args.seconds, teacher))
        p.start()
        conns.append(parent)
        procs.append(p)
    chats: dict = {}
    seed_of = {c: args.seed + i for i, c in enumerate(conns)}
    rows, active, t0 = [], list(conns), time.time()
    played: Counter = Counter()
    while active:
        pending = []  # (conn, tick, look)
        for c in list(active):
            while True:
                msg = c.recv()
                if msg[0] == "begin":
                    chats[c] = GameChat(msg[1], dec.pieces if dec else None)
                    continue
                if msg[0] == "done":
                    rows.append(
                        {
                            "behavior": STYLE,
                            "bots": args.bots,
                            "ep": seed_of[c],
                            "seed": seed_of[c],
                            "policy": args.policy,
                            **msg[1],
                        }
                    )
                    active.remove(c)
                    r = rows[-1]
                    print(
                        f"[{len(rows)}/{args.matches}] seed {r['seed']} frags {r['frags']:>3} "
                        f"deaths {r['deaths']:>3} margin {r['margin']:>+4} rank {r['rank']} "
                        f"({time.time() - t0:.0f} s)",
                        flush=True,
                    )
                    break
                _, tick, lk, fired = msg
                chats[c].observe(fired, tick)
                if lk is not None:
                    pending.append((c, tick, lk))
                    break
        if not pending:
            continue
        asks, owner = [], []
        for c, tick, lk in pending:
            a = asks_of(chats[c], lk, tick)
            asks += a
            owner += [c] * len(a)
        got = dec.run(asks)
        reply: dict = {}
        for c, (_, adapter), g in zip(owner, asks, got):
            reply.setdefault(c, {})[adapter] = g["word"]
            played[(adapter, g["word"])] += 1
        for c, _, _ in pending:
            r = reply[c]
            c.send((r[MOVE], int(r[WEAPON]) if WEAPON in r else None))
    for p in procs:
        p.join()
    rows.sort(key=lambda r: r["seed"])
    print(summarize(rows))
    if played:
        moves = Counter({w: n for (a, w), n in played.items() if a == MOVE})
        print(f"moves played: {dict(moves.most_common(10))}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=1))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compose", help="Compose the trained move and weapon adapters")
    v = sub.add_parser("verify", help="The composed checkpoint against PEFT")
    b = sub.add_parser("bench", help="Decision latency on held-out matches")
    pl = sub.add_parser("play", help="Lockstep matches: the student or the teacher")
    c.add_argument("--runs", type=Path, required=True, help="Holds move/ and weapon/")
    c.add_argument("--base", default="ibm-granite/granite-4.1-3b")
    c.add_argument("--out", type=Path, required=True)
    v.add_argument("--runs", type=Path, required=True)
    v.add_argument("--limit", type=int, default=1500)
    v.add_argument("--seconds", type=float, default=20.0, help="live_check's match")
    for p in (v, b, pl):
        p.add_argument("--model", help="The composed checkpoint")
        p.add_argument("--json", type=Path)
    b.add_argument("--rows", type=Path, nargs="+", required=True, help="decide.py rows")
    b.add_argument("--games", type=int, default=1)
    b.add_argument("--steps", type=int, default=0, help="At most so many (0: all)")
    b.add_argument("--warm", type=int, default=20, help="Steps left out at the start")
    pl.add_argument("--policy", choices=("model", "teacher"), default="model")
    pl.add_argument("--teacher", default="runs/teacher_rl0_706M.pt")
    pl.add_argument("--matches", type=int, default=12)
    pl.add_argument("--seconds", type=float, default=600.0)
    pl.add_argument("--bots", default="cast")
    pl.add_argument("--n-bots", type=int, default=7)
    pl.add_argument("--seed", type=int, default=5000)
    args = ap.parse_args()
    if args.cmd in ("verify", "bench") or (
        args.cmd == "play" and args.policy == "model"
    ):
        if not args.model:
            raise SystemExit("--model is required")
    {"compose": compose, "verify": verify, "bench": bench, "play": play}[args.cmd](args)


if __name__ == "__main__":
    main()
