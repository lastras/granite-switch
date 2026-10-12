# SPDX-License-Identifier: Apache-2.0
"""Record the RL teacher's every move, for Granite to learn: two compilable functions.

``Player.move`` is a compilable function (``mellea.stdlib.compile``, the
Mellea fork's). Its signature is what the student reads: the match so far as
a conversation in Granite's chat template, then the tic's look
(:mod:`game_chat`). Its body is the teacher, an RL policy whose memory is
``self``, which the student never sees. ``Player.weapon`` is the same for the
weapon.

The teacher plays each match. Every call is recorded, and every other answer
is added to it as another completion, each scored by the teacher's
probability: the student is trained to match the whole distribution
(``train_alora.py --records``).

    python decide.py --out data/s1/decide/shard_0.jsonl.gz --shard 0/32
"""

import argparse
import json
import random
import zlib
from functools import partial
from pathlib import Path
from typing import Literal

from decide_tools import code_version, teacher_id
from doom_env import ACTIONS
from expert import PLAN_EVERY_TICS
from game_chat import FORMAT, MOVE, STYLE, TOOLS, WEAPON, WEAPON_TOKENS, Match
from mellea.stdlib.compile import Completion, Score, annotate, compilable, emulate
from mellea.stdlib.context import ChatContext
from rl_teacher import RLPolicy

EVERY = 7  # a move call every 7th tic (5 a second); the teacher draws its own between


class Player:
    """The RL teacher in one match. ``self`` is its memory: the student reads
    the conversation instead."""

    def __init__(self, teacher: str, seed: int):
        self.rl, self.rng = RLPolicy(teacher), random.Random(seed)

    def see(self, obs) -> None:
        """Every live tic, as the teacher was trained: its distributions now."""
        self.tick, self.dist = obs.tick, self.rl.labels(obs, STYLE)

    def draw(self, kind: str) -> str:
        d = self.dist[kind]
        return self.rng.choices(list(d), weights=list(d.values()))[0]

    @compilable
    def move(self, chat: ChatContext) -> Literal[ACTIONS]:
        """Your move this tic."""
        annotate(t=self.tick, dist=self.dist[MOVE])
        return self.draw(MOVE)

    @compilable
    def weapon(self, chat: ChatContext) -> Literal[WEAPON_TOKENS]:
        """The weapon to hold now."""
        annotate(t=self.tick, dist=self.dist[WEAPON])
        return self.draw(WEAPON)


def alternatives(record, episode) -> list[Completion]:
    """Every other answer: with the one returned, the teacher's whole distribution."""
    said = record.completions[0].value
    return [Completion(w) for w in record.meta["dist"] if w != said]


def score(record, completion, episode) -> Score:
    """The teacher's probability of an answer: the student's soft label."""
    return Score(value=record.meta["dist"][completion.value])


def chat_of(match: Match) -> ChatContext:
    """The match's conversation with this tic's look, as the calls take it."""
    return ChatContext.from_openai(match.messages(), tools=TOOLS)


def episode(seed: int, teacher: str, seconds: float = 600.0, run: int = 0) -> None:
    """One match against the cast, the teacher playing it."""
    with Match(seed, seconds) as match:
        player = Player(teacher, seed_of(seed, run))
        while not match.over:
            if match.obs.dead:  # no decision: the respawn passes
                match.step()
                continue
            player.see(match.obs)
            tick, slot = match.obs.tick, None
            if tick % EVERY == 0:
                move = player.move(chat_of(match))
            else:
                move = player.draw(MOVE)  # the teacher's own, not recorded
            if tick % PLAN_EVERY_TICS == 0:
                slot = player.weapon(chat_of(match))
            match.step(move, slot)


def seed_of(*key) -> int:
    return zlib.crc32(repr(key).encode())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="A .jsonl.gz shard")
    ap.add_argument("--shard", default="0/1", help="K/N: every N-th match from K")
    ap.add_argument("--matches", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=920000, help="The first match's seed")
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--teacher", default="runs/teacher_rl0_706M.pt")
    ap.add_argument("--seed", type=int, default=0, help="The teacher's draws")
    args = ap.parse_args()
    seeds = range(args.seed0, args.seed0 + args.matches)
    run = partial(episode, teacher=args.teacher, seconds=args.seconds, run=args.seed)
    summary = emulate(
        [(json.dumps(["cast", s]), partial(run, s)) for s in seeds],
        args.out,
        shard=args.shard,
        score=score,
        alternatives=alternatives,
        meta={
            "code": code_version(),
            "format": FORMAT,
            "teacher": teacher_id(args.teacher),
            **{k: getattr(args, k) for k in ("seconds", "seed", "seed0")},
        },
    )
    if summary["failed"]:
        raise SystemExit(f"{len(summary['failed'])} matches failed; run again")


if __name__ == "__main__":
    main()
