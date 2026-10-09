# SPDX-License-Identifier: Apache-2.0
"""The RL writer: rl_teacher's recurrent PPO, writing a word with frags.

``ibm_writing.py``'s set piece as a training task (``rl_teacher.TASKS["writer"]``):
the harness lays a word on MAP02's north-west band, and a frag he makes
within ``RADIUS_M`` of an open atom writes it. The objective is the time to
write the whole word, which the scripted writer showed is mostly the time he
spends dying (``docs/DOOM_IBM_WRITING.md``).

* **What the actor reads:** rl0's inputs (``rl_teacher.obs_vector``, as the
  fighter) and the harness's offer (:func:`writer_features`): the atom he is
  on his way to (the nearest open one, kept while open, as the scripted
  writer's), its place in his own frame to a tenth of a metre, the next two,
  his velocity, how much of the word is left. Never the floor or the bots'
  positions: those, and the walking distance to the atom, are the critic's
  (:func:`writer_priv`).
* **The reward** (:class:`WriterShaping`): per atom written, a bonus for the
  word, a cost per second until it is written (dead seconds included), a death
  price on top of the deathmatch's (a death costs about 11 s of writing, past
  the discount's horizon), the walking distance to the atom as a potential
  (shaping that cannot change the best policy), and the deathmatch's own
  shaping at half weight, so he keeps fighting.
* **The words he trains on:** three random capitals (IBM a tenth of the time),
  at a random place along the band, with part of the word already written, so
  every stage of a word is seen.
* **The start:** rl0's weights (``--init``), the new inputs' weights zero, so
  he starts out playing exactly as rl0; kickstarted toward the scripted
  writer's moves (``ibm_writing.ScriptedWriter``), decaying to nothing.
* **The evaluation:** IBM in 10-minute matches, seeds from 0, against
  ``ibm_writing.py match``'s scripted writer on the same seeds.

::

    python rl_writer.py train --out runs/writer0 --init runs/teacher_rl0_706M.pt \\
        --envs 256 --workers 96 --device cuda
    python rl_writer.py eval --ckpt runs/writer0/latest.pt --seeds 0-47 --workers 48
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import multiprocessing as mp
import os
import random
import string
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rl_teacher as T
from doom_env import (
    _ACTION_INDEX,
    ACTIONS,
    MATCH_TICS,
    PRIV_DIM,
    TIC_HZ,
    UNITS_PER_M,
    WEAPON_SLOTS,
    DoomEnv,
    isolate_workdir,
    priv_features,
)
from expert import PLAN_EVERY_TICS
from ibm_writing import RADIUS_M, Canvas, Floor, ScriptedWriter, result_row, summarize

# The style rl0 reads while he writes: its fighter, his starting point.
STYLE = "fighter"
N_NEXT = 2  # the atoms after the one he is going to, in his view
WRITER_DIM = 5 + 2 + 2 * N_NEXT + 2
OBS_DIM = T.OBS_DIM + WRITER_DIM
PRIV_EXTRA = 2
RUN_MS = 17.0  # his running speed, m/s: velocities are read in these units


@dataclass(frozen=True)
class WriterShaping:
    """The writer's reward, besides the deathmatch's shaping (``fight`` times
    ``rl_teacher.STYLE_SHAPING["fighter"]``)."""

    atom: float = 2.0  # per atom written
    word: float = 5.0  # the word written (the episode ends)
    per_s: float = -0.05  # per second until the word is written, dead ones too
    death: float = -1.5  # per death, on top of the deathmatch's -0.75
    approach: float = 0.05  # per metre walked toward his atom (a potential)
    fight: float = 0.5  # the deathmatch's shaping, at this weight


SHAPING = WriterShaping()


# ── What he reads ──────────────────────────────────────────────────────────────
def _frame(obs, x: float, y: float) -> tuple[float, float]:
    """(x, y) in his frame: metres ahead of him and to his left."""
    dx, dy = (x - obs.priv.x) / UNITS_PER_M, (y - obs.priv.y) / UNITS_PER_M
    a = math.radians(obs.priv.angle)
    return dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) + dy * math.cos(a)


def next_atoms(canvas: Canvas, target: int) -> list[int]:
    """The open atoms nearest the one he is going to: where he goes after."""
    t = canvas.pts[target]
    rest = [i for i in canvas.open() if i != target]
    return sorted(rest, key=lambda i: math.hypot(*(canvas.pts[i] - t)))[:N_NEXT]


def writer_features(obs, canvas: Canvas, target: int | None, vel) -> np.ndarray:
    """The harness's offer, as the actor reads it: the atom he is going to
    (ahead and left of him, tenths of 10 m; its distance; to within 3 m,
    finer; whether he is on it), his velocity (ahead, left, in running
    speeds), the next atoms, how much of the word is left, and whether he is
    writing at all (zero once the word is written)."""
    x = np.zeros(WRITER_DIM, np.float32)
    if target is None:
        return x
    f, left = _frame(obs, *canvas.pts[target])
    d = math.hypot(f, left)
    x[0:5] = (
        np.clip(f / 10, -4, 4),
        np.clip(left / 10, -4, 4),
        min(d / 10, 4),
        min(d, 3) / 3,
        float(d <= RADIUS_M),
    )
    x[5:7] = vel[0] / RUN_MS, vel[1] / RUN_MS
    for k, i in enumerate(next_atoms(canvas, target)):
        nf, nl = _frame(obs, *canvas.pts[i])
        x[7 + 2 * k : 9 + 2 * k] = np.clip(nf / 10, -4, 4), np.clip(nl / 10, -4, 4)
    x[-2] = len(canvas.open()) / len(canvas.pts)
    x[-1] = 1.0
    return x


def writer_obs(obs, canvas, target, vel) -> np.ndarray:
    return np.concatenate(
        [T.obs_vector(obs, STYLE), writer_features(obs, canvas, target, vel)]
    )


def writer_priv(obs, frac: float, floor: Floor, canvas: Canvas, target) -> np.ndarray:
    """The critic's extra inputs: the walking distance to his atom (50 m
    units, capped) and how much of the word is left."""
    walk = 0.0
    if target is not None:
        walk = min(
            floor.path_m((obs.priv.x, obs.priv.y), tuple(canvas.pts[target])) / 50, 2
        )
    left = len(canvas.open()) / len(canvas.pts)
    return np.concatenate(
        [priv_features(obs, frac), np.asarray([walk, left], np.float32)]
    )


class Velocity:
    """His velocity in his own frame (m/s, ahead and left), from his position
    tic to tic; zero at a respawn."""

    def __init__(self):
        self.last = None

    def __call__(self, obs) -> tuple[float, float]:
        if obs.dead or "respawn" in obs.events or self.last is None:
            self.last = (obs.priv.x, obs.priv.y)
            return 0.0, 0.0
        px, py = self.last
        self.last = (obs.priv.x, obs.priv.y)
        dx, dy = (obs.priv.x - px) / UNITS_PER_M, (obs.priv.y - py) / UNITS_PER_M
        a = math.radians(obs.priv.angle)
        k = TIC_HZ
        return k * (dx * math.cos(a) + dy * math.sin(a)), k * (
            -dx * math.sin(a) + dy * math.cos(a)
        )


# ── Training: one env ──────────────────────────────────────────────────────────
def training_word(rng: random.Random) -> str:
    return (
        "IBM"
        if rng.random() < 0.1
        else "".join(rng.choices(string.ascii_uppercase, k=3))
    )


class WriterSlot:
    """One training env (the ``rl_teacher._Slot`` interface): a word per
    episode, the reward, his inputs, the scripted writer's labels."""

    def __init__(self, cfg: T.Config, seed: int, ks=None):
        self.cfg, self.rng, self.seed = cfg, random.Random(seed), seed
        self.env = DoomEnv(
            seed=seed,
            resolution=cfg.resolution,
            timeout_tics=int(cfg.episode_s * TIC_HZ),
            respawn_s=cfg.respawn_s,
            item_rules=cfg.item_rules,
        )
        self.episode = 0
        self.floor = None
        self.new_episode()

    def new_episode(self) -> None:
        self.episode += 1
        n = self.rng.randint(self.cfg.n_bots_min, self.cfg.n_bots_max)
        self.obs = self.env.reset(
            seed=self.seed * 1000 + self.episode, bots=T.BOT_SETS[self.cfg.bots][:n]
        )
        if self.floor is None:  # MAP02 is the same every episode
            self.floor = Floor(self.env._segments, (self.obs.priv.x, self.obs.priv.y))
        self.floor._fields.clear()
        self.canvas = Canvas(
            self.floor, training_word(self.rng), offset=self.rng.uniform(-1, 1)
        )
        pre = self.rng.uniform(0, 0.7)  # part of the word already written
        for i in range(len(self.canvas.pts)):
            if self.rng.random() < pre and len(self.canvas.open()) > 1:
                self.canvas.written[i] = {"pre": True}
        self.writer = ScriptedWriter(self.floor, self.canvas)
        self.vel = Velocity()
        self.ret, self.atoms, self.done_word = 0.0, 0, False
        self._skip_dead()

    def _target(self):
        return self.writer.spot(self.obs) if self.canvas.open() else None

    def _potential(self) -> float:
        t = self._target()
        if t is None:
            return 0.0
        return -SHAPING.approach * self.floor.path_m(
            (self.obs.priv.x, self.obs.priv.y), tuple(self.canvas.pts[t])
        )

    def _tic_reward(self, prev, obs) -> float:
        """One tic's reward, but the potential's."""
        r = SHAPING.fight * T.shaped_reward(prev, obs, T.STYLE_SHAPING[STYLE])
        r += SHAPING.per_s / TIC_HZ
        r += SHAPING.death * obs.events.count("died")
        for _ in range(obs.events.count("frag")):
            if self.canvas.frag(obs.priv.x, obs.priv.y, obs.tick) is not None:
                r += SHAPING.atom
                self.atoms += 1
                if not self.canvas.open():
                    r += SHAPING.word
                    self.done_word = True
        return r

    def _skip_dead(self) -> tuple[float, int]:
        """Dead tics need no decision: step through them, their cost kept."""
        r, n = 0.0, 0
        while self.obs.dead and not self.obs.done:
            prev = self.obs
            self.obs = self.env.step("wait")
            n += 1
            if not self.obs.done:
                r += self._tic_reward(prev, self.obs)
        return r, n

    def step(self, move: int, weapon: int):
        prev, phi0 = self.obs, self._potential()
        slot = None
        if prev.tick % PLAN_EVERY_TICS == 0:
            slot = prev.slot if weapon == T.KEEP else WEAPON_SLOTS[weapon]
        self.obs = self.env.step(ACTIONS[move], weapon=slot)
        r, elapsed = 0.0, 1
        if not self.obs.done:
            r = self._tic_reward(prev, self.obs)
            dr, dead = self._skip_dead()
            r += dr
            elapsed += dead
            # The approach, as a potential: gamma * phi(next) - phi(now).
            r += self.cfg.gamma**elapsed * self._potential() - phi0
        self.ret += r
        if self.obs.done or self.done_word:
            info = {
                "style": "writer",
                "return": round(self.ret, 3),
                "atoms": self.atoms,
                "word_done": float(self.done_word),
                "left": len(self.canvas.open()),
                "seconds": round(self.obs.tick / TIC_HZ, 1),
                **{
                    k: v
                    for k, v in self.env.stats.as_dict().items()
                    if k in ("frags", "deaths")
                },
            }
            self.new_episode()
            return r, True, info, elapsed
        return r, False, None, elapsed

    def views(self, labels: bool = True):
        frac = self.obs.tick / max(1, self.env.game.get_episode_timeout())
        target = self._target()
        vel = self.vel(self.obs)
        move = np.zeros(T.N_MOVE, np.float32)
        weapon = np.zeros(T.N_WEAPON, np.float32)
        if labels:
            move[_ACTION_INDEX[self.writer.act(self.obs)]] = 1.0
            s = self.writer.weapon(self.obs)
            weapon[T.KEEP if s == self.obs.slot else WEAPON_SLOTS.index(s)] = 1.0
        return (
            writer_obs(self.obs, self.canvas, target, vel),
            writer_priv(self.obs, frac, self.floor, self.canvas, target),
            move,
            weapon,
        )


# ── Evaluation: IBM, as the scripted writer writes it ──────────────────────────
class WriterPolicy:
    """A trained writer for one match: its move and weapon distributions per tic."""

    def __init__(self, ckpt: str | Path, device: str = "cpu"):
        torch, _ = T._torch()
        torch.set_num_threads(1)
        self.torch, self.device = torch, device
        self.agent, self.cfg, _ = T.load(ckpt, device)
        self.h = self.agent.initial_state(1, device)
        self.first = True

    def step(self, x: np.ndarray):
        torch = self.torch
        with torch.no_grad():
            z, self.h = self.agent.sequence(
                torch.as_tensor(x, device=self.device).view(1, 1, -1),
                torch.tensor([[1.0 if self.first else 0.0]], device=self.device),
                self.h,
            )
            pm = torch.softmax(self.agent.pi_move(z[0]), -1)[0].cpu().numpy()
            pw = torch.softmax(self.agent.pi_weapon(z[0]), -1)[0].cpu().numpy()
        self.first = False
        return pm, pw


def writer_match(args: tuple) -> dict:
    """One match of a trained writer: ``(ckpt, seed, word, seconds, n_bots,
    sample)`` -> ``ibm_writing.result_row``. Built as ``ibm_writing``'s
    scripted match is (the default resolution, the default bots)."""
    ckpt, seed, word, seconds, n_bots, sample = args
    isolate_workdir()
    pol, rng = WriterPolicy(ckpt), random.Random(seed)
    env = DoomEnv(seed=seed, n_bots=n_bots, timeout_tics=int(seconds * TIC_HZ) + 10)
    obs = env.reset(seed=seed)
    floor = Floor(env._segments, (obs.priv.x, obs.priv.y))
    canvas = Canvas(floor, word)
    spot = ScriptedWriter(floor, canvas)  # only its choice of atom (the harness's)
    vel = Velocity()
    done_at, frags, deaths = None, 0, 0
    while not obs.done and obs.tick < seconds * TIC_HZ:
        if obs.dead:
            vel(obs)
            obs = env.step("wait")
        else:
            target = spot.spot(obs) if canvas.open() else None
            pm, pw = pol.step(writer_obs(obs, canvas, target, vel(obs)))
            m = (
                rng.choices(range(T.N_MOVE), weights=pm)[0]
                if sample
                else int(pm.argmax())
            )
            slot = None
            if obs.tick % PLAN_EVERY_TICS == 0:
                i = (
                    rng.choices(range(T.N_WEAPON), weights=pw)[0]
                    if sample
                    else int(pw.argmax())
                )
                slot = obs.slot if i == T.KEEP else WEAPON_SLOTS[i]
            obs = env.step(ACTIONS[m], weapon=slot)
        for _ in range(obs.events.count("frag")):
            frags += 1
            if (
                canvas.frag(obs.priv.x, obs.priv.y, obs.tick) is not None
                and not canvas.open()
            ):
                done_at = done_at or obs.tick
        deaths += obs.events.count("died")
    env.close()
    return result_row(f"rl:{Path(ckpt).name}", seed, canvas, done_at, frags, deaths)


def eval_tasks(ckpt, cfg: T.Config) -> list[tuple]:
    seconds = min(cfg.eval_seconds, MATCH_TICS / TIC_HZ)
    return [(str(ckpt), s, "IBM", seconds, 7, True) for s in range(cfg.eval_matches)]


T.TASKS["writer"] = T.Task(
    functools.partial(T._env_worker, slot=WriterSlot),
    OBS_DIM,
    PRIV_DIM + PRIV_EXTRA,
    eval_tasks,
    writer_match,
    summarize,
)


# ── The command line ───────────────────────────────────────────────────────────
WRITER_DEFAULTS = {
    "task": "writer",
    "envs": 256,
    "workers": 96,
    "episode_s": 240.0,
    "total_steps": 200_000_000,
    "ks_coef": 0.5,
    "ks_steps": 20_000_000,
    "eval_every": 50,
    "eval_matches": 24,
    "eval_workers": 24,
}


def main() -> None:
    from ibm_writing import seeds_arg

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="Train (rl_teacher's PPO, the writer's task)")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--device", default="cuda")
    t.add_argument("--resume", type=Path)
    t.add_argument("--smoke", action="store_true", help="Tiny run for a laptop")
    for f in T.Config.__dataclass_fields__.values():
        t.add_argument(
            f"--{f.name.replace('_', '-')}", type=type(f.default), default=None
        )
    e = sub.add_parser("eval", help="IBM matches with a checkpoint")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--seeds", default="0-47")
    e.add_argument("--word", default="IBM")
    e.add_argument("--workers", type=int, default=24)
    e.add_argument("--argmax", action="store_true")
    e.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.cmd == "train":
        cfg = replace(T.Config(), **WRITER_DEFAULTS)
        if args.smoke:
            cfg = replace(
                cfg,
                envs=4,
                workers=2,
                rollout=32,
                total_steps=4 * 32 * 3,
                episode_s=20.0,
                eval_every=2,
                eval_matches=1,
                eval_workers=1,
                eval_seconds=20.0,
                ks_steps=4 * 32 * 2,
            )
        overrides = {
            k: getattr(args, k)
            for k in T.Config.__dataclass_fields__
            if getattr(args, k, None) is not None
        }
        cfg = replace(cfg, **overrides)
        if cfg.init:
            cfg = replace(cfg, init=str(Path(cfg.init).resolve()))
        T.train(cfg, args.out.resolve(), args.device, args.resume)
        return
    ckpt = str(Path(args.ckpt).resolve())
    tasks = [
        (ckpt, s, args.word, MATCH_TICS / TIC_HZ, 7, not args.argmax)
        for s in seeds_arg(args.seeds)
    ]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = sorted(pool.map(writer_match, tasks), key=lambda r: r["seed"])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(summarize(rows)))


if __name__ == "__main__":
    main()
