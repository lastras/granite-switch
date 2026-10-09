# SPDX-License-Identifier: Apache-2.0
"""Writing a word with frags: the stencil, the floor, the atoms, the scripted writer.

The set piece of ``docs/DOOM_IBM_WRITING.md``, after the 1989 xenon-atom IBM:
the harness lays a word on the map as a stencil of spots ("atoms") and offers
the player the open ones; a frag he makes while standing within
:data:`RADIUS_M` of an open atom writes it, and the set narrows until the
word is written.

* :data:`FONT`: the capital letters, five rows each (an ``X`` is an atom).
* :class:`Floor`: the floor he can stand on (a grid of half-metre cells clear
  of every blocking line) and shortest paths over it.
* :class:`Canvas`: one word laid on the floor (by default in the map's biggest
  open band, three times wider than tall: MAP02's north-west room), and which
  atoms are written.
* :class:`ScriptedWriter`: the scripted player that writes it, the RL writer's
  kickstart teacher (``rl_writer.py``): the scripted fighter (``expert.py``),
  which walks to the nearest open atom, holds it and frags from there; armed
  with no more than the pistol, he first gets a gun (the variant that cut
  deaths by a quarter, ``docs/DOOM_IBM_WRITING.md``). It reads ground truth
  (his exact position, the floor); the model will not.

::

    python ibm_writing.py match --seeds 0-47 --workers 80 --out runs/ibm/scripted.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doom_env import _PLAYER_RADIUS, MATCH_TICS, TIC_HZ, UNITS_PER_M, Seen
from expert import Expert, steer, turn

# ── The letters ────────────────────────────────────────────────────────────────
FONT = {  # rows top to bottom; X: an atom
    "A": [".X.", "X.X", "XXX", "X.X", "X.X"],
    "B": ["XXX.", "X..X", "XXX.", "X..X", "XXX."],
    "C": [".XX", "X..", "X..", "X..", ".XX"],
    "D": ["XX.", "X.X", "X.X", "X.X", "XX."],
    "E": ["XXX", "X..", "XX.", "X..", "XXX"],
    "F": ["XXX", "X..", "XX.", "X..", "X.."],
    "G": [".XX.", "X...", "X.XX", "X..X", ".XX."],
    "H": ["X.X", "X.X", "XXX", "X.X", "X.X"],
    "I": ["X", "X", "X", "X", "X"],
    "J": ["..X", "..X", "..X", "X.X", ".X."],
    "K": ["X..X", "X.X.", "XX..", "X.X.", "X..X"],
    "L": ["X..", "X..", "X..", "X..", "XXX"],
    "M": ["X...X", "XX.XX", "X.X.X", "X...X", "X...X"],
    "N": ["X..X", "XX.X", "X.XX", "X..X", "X..X"],
    "O": [".XX.", "X..X", "X..X", "X..X", ".XX."],
    "P": ["XXX.", "X..X", "XXX.", "X...", "X..."],
    "Q": [".XX.", "X..X", "X..X", "X.X.", ".X.X"],
    "R": ["XXX.", "X..X", "XXX.", "X.X.", "X..X"],
    "S": [".XX", "X..", ".X.", "..X", "XX."],
    "T": ["XXX", ".X.", ".X.", ".X.", ".X."],
    "U": ["X.X", "X.X", "X.X", "X.X", "XXX"],
    "V": ["X.X", "X.X", "X.X", "X.X", ".X."],
    "W": ["X...X", "X...X", "X.X.X", "XX.XX", "X...X"],
    "X": ["X.X", "X.X", ".X.", "X.X", "X.X"],
    "Y": ["X.X", "X.X", ".X.", ".X.", ".X."],
    "Z": ["XXX", "..X", ".X.", "X..", "XXX"],
}
ROWS = 5
SPACING_M = 3.0  # at most this between neighbouring atoms (closer if the word is wide)
MARGIN_M = 1.0  # the atoms this far inside the band, off its walls
RADIUS_M = 1.5  # a frag this close to an open atom writes it (drawn on its site)
CELL = 16  # map units per floor cell (half a metre)


def stencil(word: str) -> tuple[list[tuple[str, int, int]], int]:
    """The word's atoms, ``(letter, row, column)``, and its width in columns
    (a blank column between letters)."""
    atoms, col = [], 0
    for letter in word.upper():
        rows = FONT[letter]
        for r, line in enumerate(rows):
            atoms += [(letter, r, col + c) for c, ch in enumerate(line) if ch == "X"]
        col += len(rows[0]) + 1
    return atoms, col - 1


# ── The floor ──────────────────────────────────────────────────────────────────
class Floor:
    """The floor he can stand on: CELL-unit cells at least his radius (plus 4
    units) from every blocking line, reachable from ``start``; and shortest
    paths over them (breadth-first, cached per goal cell)."""

    def __init__(self, segs: np.ndarray, start: tuple[float, float]):
        self.x0, self.y0 = segs[:, [0, 2]].min(), segs[:, [1, 3]].min()
        x1, y1 = segs[:, [0, 2]].max(), segs[:, [1, 3]].max()
        self.nx = int((x1 - self.x0) // CELL) + 1
        self.ny = int((y1 - self.y0) // CELL) + 1
        X, Y = np.meshgrid(
            self.x0 + (np.arange(self.nx) + 0.5) * CELL,
            self.y0 + (np.arange(self.ny) + 0.5) * CELL,
        )
        near = np.full(X.shape, np.inf)
        for ax, ay, bx, by in segs:
            dx, dy = bx - ax, by - ay
            t = np.clip(
                ((X - ax) * dx + (Y - ay) * dy) / (dx * dx + dy * dy or 1.0), 0, 1
            )
            near = np.minimum(near, np.hypot(X - (ax + t * dx), Y - (ay + t * dy)))
        clear = near > _PLAYER_RADIUS + 4
        self.free = np.zeros_like(clear)
        q = deque([self.cell(*start)])
        self.free[q[0]] = True
        while q:
            i, j = q.popleft()
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if (
                    0 <= a < self.ny
                    and 0 <= b < self.nx
                    and clear[a, b]
                    and not self.free[a, b]
                ):
                    self.free[a, b] = True
                    q.append((a, b))
        self._fields: dict[tuple[int, int], np.ndarray] = {}

    def cell(self, x: float, y: float) -> tuple[int, int]:
        i = min(self.ny - 1, max(0, int((y - self.y0) // CELL)))
        return i, min(self.nx - 1, max(0, int((x - self.x0) // CELL)))

    def xy(self, i: int, j: int) -> tuple[float, float]:
        return self.x0 + (j + 0.5) * CELL, self.y0 + (i + 0.5) * CELL

    def band(self, aspect: float) -> tuple[float, float, float, float]:
        """The biggest free rectangle at least ``aspect`` times wider than
        tall: (x, y, width, height) in map units."""
        best = None
        for h in range(4, self.ny):
            w = math.ceil(aspect * h)
            if w > self.nx:
                break
            rows = np.ones((self.ny - h + 1, self.nx), bool)
            for k in range(h):
                rows &= self.free[k : k + self.ny - h + 1]
            run = np.zeros(rows.shape[0], int)
            for j in range(self.nx):
                run = np.where(rows[:, j], run + 1, 0)
                hit = np.flatnonzero(run >= w)
                if len(hit):
                    i = hit[0]
                    best = (
                        self.x0 + (j - w + 1) * CELL,
                        self.y0 + i * CELL,
                        w * CELL,
                        h * CELL,
                    )
                    break
        return best

    def field(self, goal: tuple[float, float]) -> np.ndarray:
        """Steps from every free cell to ``goal``'s cell (-1: unreachable)."""
        g = self.cell(*goal)
        if g not in self._fields:
            d = np.full(self.free.shape, -1, int)
            d[g] = 0
            q = deque([g])
            while q:
                i, j = q.popleft()
                for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if (
                        0 <= a < self.ny
                        and 0 <= b < self.nx
                        and self.free[a, b]
                        and d[a, b] < 0
                    ):
                        d[a, b] = d[i, j] + 1
                        q.append((a, b))
            self._fields[g] = d
        return self._fields[g]

    def _on(self, d: np.ndarray, frm: tuple[float, float]) -> tuple[int, int]:
        c = self.cell(*frm)
        if d[c] < 0:  # off the grid (pressed against a wall): the nearest free cell
            ii, jj = np.nonzero(d >= 0)
            k = np.argmin((ii - c[0]) ** 2 + (jj - c[1]) ** 2)
            c = (ii[k], jj[k])
        return c

    def path_m(self, frm: tuple[float, float], goal: tuple[float, float]) -> float:
        """The walking distance from ``frm`` to ``goal``, in metres (four-way
        steps, so a little over the true shortest path)."""
        d = self.field(goal)
        return d[self._on(d, frm)] * CELL / UNITS_PER_M

    def waypoint(
        self, frm: tuple[float, float], goal: tuple[float, float], ahead: int = 8
    ):
        """A point about ``ahead`` cells along the shortest path to ``goal``."""
        d = self.field(goal)
        c = self._on(d, frm)
        for _ in range(ahead):
            i, j = c
            nxt = [
                (a, b)
                for a, b in (
                    (i + 1, j),
                    (i - 1, j),
                    (i, j + 1),
                    (i, j - 1),
                    (i + 1, j + 1),
                    (i - 1, j - 1),
                    (i + 1, j - 1),
                    (i - 1, j + 1),
                )
                if 0 <= a < self.ny
                and 0 <= b < self.nx
                and d[a, b] >= 0
                and self.free[a, j]
                and self.free[i, b]
            ]
            best = min(nxt, key=lambda c2: d[c2], default=None)
            if best is None or d[best] >= d[c]:
                break
            c = best
        return self.xy(*c)


# ── The word on the floor ──────────────────────────────────────────────────────
class Canvas:
    """A word laid on the floor: its atoms (map units), and which are written.

    The atoms fill ``band`` (default: the floor's biggest 3:1 band) at most
    SPACING_M apart, MARGIN_M inside it; a word narrower than the band sits at
    ``offset`` (-1: its left end, 0: the middle, 1: its right end)."""

    def __init__(self, floor: Floor, word: str = "IBM", band=None, offset: float = 0.0):
        cells, cols = stencil(word)
        bx, by, bw, bh = band or floor.band(3.0)
        margin = MARGIN_M * UNITS_PER_M
        sx = min(SPACING_M * UNITS_PER_M, (bw - 2 * margin) / max(1, cols - 1))
        sy = min(SPACING_M * UNITS_PER_M, (bh - 2 * margin) / (ROWS - 1))
        width, height = (cols - 1) * sx, (ROWS - 1) * sy
        slack = bw - 2 * margin - width
        cx, cy = bx + bw / 2 + offset * slack / 2, by + bh / 2
        self.word = word.upper()
        self.letter = [a[0] for a in cells]
        self.pts = np.array(
            [(cx - width / 2 + c * sx, cy + height / 2 - r * sy) for _, r, c in cells]
        )
        self.spacing_m = (sx / UNITS_PER_M, sy / UNITS_PER_M)
        self.written: list[dict | None] = [None] * len(self.pts)

    def open(self) -> list[int]:
        return [i for i, w in enumerate(self.written) if w is None]

    def near_open(self, x: float, y: float, radius_m: float = RADIUS_M) -> int | None:
        """The open atom nearest (x, y) within ``radius_m``, if any."""
        best, bd = None, radius_m * UNITS_PER_M
        for i in self.open():
            d = math.hypot(self.pts[i][0] - x, self.pts[i][1] - y)
            if d <= bd:
                best, bd = i, d
        return best

    def frag(self, x: float, y: float, tick: int) -> int | None:
        """A frag he made standing at (x, y): the atom it writes, if any."""
        i = self.near_open(x, y)
        if i is not None:
            self.written[i] = {"x": x, "y": y, "tick": tick}
        return i


def bearing_to(px: float, py: float, heading: float, x: float, y: float) -> float:
    """The bearing of (x, y) from (px, py) facing ``heading``, in degrees,
    positive to the right (as ``Seen.bearing``)."""
    return -(
        (math.degrees(math.atan2(y - py, x - px)) - heading + 180.0) % 360.0 - 180.0
    )


# ── The scripted writer ────────────────────────────────────────────────────────
SELF_DEFENSE_HP = 50  # below this he shoots whatever is in front of him
CLOSE_M = 3.0  # ... and any bot this close
HEAL_HP = 35  # below this, with no bot in sight, he fetches health
ARM_DETOUR_M = 10  # on the way to an atom, he picks up what is this close
# Armed with no more than the pistol, he looks for a gun this long after a respawn.
REARM_TICS = 15 * TIC_HZ
REARM_SLOT = 3  # a gun: this slot or above, with ammo (3: the shotgun)


class ScriptedWriter:
    """The scripted fighter, writing: walk to the nearest open atom, hold it and
    frag from there; re-arm first after a respawn; back to plain fighting when
    the word is written. Reads ground truth (``obs.priv``)."""

    def __init__(self, floor: Floor, canvas: Canvas):
        self.ex, self.floor, self.canvas = Expert(), floor, canvas
        self.why = ""  # what the last decision was
        self._spot: int | None = None
        self._born = 0

    def spot(self, obs) -> int | None:
        """The open atom he is writing: the one he chose while it is open; else
        the nearest open one."""
        cur = self._spot
        if cur is None or self.canvas.written[cur] is not None:
            open_ = self.canvas.open()
            if not open_:
                return None
            p = np.array([obs.priv.x, obs.priv.y])
            cur = min(open_, key=lambda i: math.hypot(*(self.canvas.pts[i] - p)))
            self._spot = cur
        return cur

    def act(self, obs) -> str:
        if obs.dead:
            return "wait"
        if "respawn" in obs.events:
            self._born = obs.tick
        foes = [o for o in obs.seen if o.kind == "enemy"]
        t = self.ex._target(foes)
        i = self.spot(obs)
        if i is None:  # written: back to fighting
            self.why = "written"
            return self.ex.act(obs, "fighter")
        gx, gy = self.canvas.pts[i]
        px, py = obs.priv.x, obs.priv.y
        off = math.hypot(gx - px, gy - py) / UNITS_PER_M  # metres from the spot
        on = off <= RADIUS_M
        if t is not None and (on or obs.hp < SELF_DEFENSE_HP or t.dist < CLOSE_M):
            self.why = "fire from the atom" if on else "fire: defend"
            b = t.b
            if abs(b) > 12:
                return turn(b)
            if abs(b) > 3:
                return "al" if b < 0 else "ar"
            if off > 0.6:  # drifting: fire while stepping back toward the spot
                ab = bearing_to(px, py, obs.priv.angle, gx, gy)
                moves = {"charge": 0, "cr": 90, "bf": 180, "cl": -90}
                return min(moves, key=lambda m: abs((ab - moves[m] + 180) % 360 - 180))
            return "fire"  # still: strafing spoils his aim
        if t is None and obs.hp < HEAL_HP:
            heal = self.ex._items(obs, ("health",))
            if heal:
                self.why = "heal"
                return self.ex._go_to(obs, heal[0])
        armed = any(s >= REARM_SLOT and n > 0 for s, n in obs.arms.items())
        if t is None and not armed and obs.tick - self._born < REARM_TICS:
            guns = self.ex._items(obs, ("weapon",))
            self.why = "rearm"
            return (
                self.ex._go_to(obs, guns[0]) if guns else self.ex.act(obs, "collector")
            )
        if t is None and off > RADIUS_M:  # on the way: what the fighter would pick up
            items = [
                o
                for o in self.ex._items(obs, ("weapon", "ammo", "armor", "health"))
                if o.dist <= ARM_DETOUR_M and self.ex._useful(obs, o)
            ]
            if items:
                self.why = f"pick up {items[0].label}"
                return self.ex._go_to(obs, items[0])
        if off > 0.5:
            self.why = "to the atom"
            return self._walk(obs, (gx, gy))
        self.why = "hold the atom"
        return "right" if t is None else turn(t.b)  # look around for the next bot

    def weapon(self, obs) -> int:
        return self.ex.weapon(obs)

    def _walk(self, obs, goal) -> str:
        px, py = obs.priv.x, obs.priv.y
        wx, wy = self.floor.waypoint((px, py), goal)
        b = bearing_to(px, py, obs.priv.angle, wx, wy)
        return steer(
            obs, Seen(-1, "", "", "", b, math.hypot(wx - px, wy - py) / UNITS_PER_M)
        )


# ── A match ────────────────────────────────────────────────────────────────────
def scripted_match(args: tuple) -> dict:
    """One match of the scripted writer: ``(seed, word, seconds, n_bots)`` ->
    the atoms written (tick, where), when the word was done, frags, deaths."""
    from doom_env import DoomEnv, isolate_workdir

    seed, word, seconds, n_bots = args
    isolate_workdir()
    env = DoomEnv(seed=seed, n_bots=n_bots, timeout_tics=int(seconds * TIC_HZ) + 10)
    obs = env.reset(seed=seed)
    floor = Floor(env._segments, (obs.priv.x, obs.priv.y))
    canvas = Canvas(floor, word)
    writer = ScriptedWriter(floor, canvas)
    done_at, frags, deaths = None, 0, 0
    while not obs.done and obs.tick < seconds * TIC_HZ:
        obs = env.step(writer.act(obs), writer.weapon(obs))
        for _ in range(obs.events.count("frag")):
            frags += 1
            if (
                canvas.frag(obs.priv.x, obs.priv.y, obs.tick) is not None
                and not canvas.open()
            ):
                done_at = done_at or obs.tick
        deaths += obs.events.count("died")
    env.close()
    return result_row("scripted", seed, canvas, done_at, frags, deaths)


def result_row(policy: str, seed: int, canvas: Canvas, done_at, frags, deaths) -> dict:
    """A match's row: what was written, and when the word was done."""
    return {
        "policy": policy,
        "seed": seed,
        "word": canvas.word,
        "atoms": len(canvas.pts),
        "written": sum(w is not None for w in canvas.written),
        "done_min": None if done_at is None else round(done_at / TIC_HZ / 60, 3),
        "frags": frags,
        "deaths": deaths,
        "writes": [w for w in canvas.written],
    }


def summarize(rows: list[dict]) -> dict:
    """Matches' rows in a few numbers: atoms, words done, the finish (an
    unfinished match counts as its full length), deaths."""
    fins = sorted(r["done_min"] or MATCH_TICS / TIC_HZ / 60 for r in rows)
    q = lambda p: fins[int(p * (len(fins) - 1))]  # noqa: E731
    atoms = float(np.mean([r["written"] for r in rows]))
    deaths = float(np.mean([r["deaths"] for r in rows]))
    return {
        "n": len(rows),
        "atoms": round(atoms, 2),
        "done": sum(r["done_min"] is not None for r in rows),
        "finish_p25": round(q(0.25), 2),
        "finish_median": round(q(0.5), 2),
        "finish_p75": round(q(0.75), 2),
        "deaths": round(deaths, 2),
        "deaths_per_atom": round(deaths / atoms, 3) if atoms else None,
        "frags": round(float(np.mean([r["frags"] for r in rows])), 2),
    }


def seeds_arg(s: str) -> list[int]:
    a, _, b = s.partition("-")
    return list(range(int(a), int(b) + 1)) if b else [int(a)]


def main() -> None:
    import multiprocessing as mp

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("match", help="The scripted writer's matches, in parallel")
    m.add_argument("--seeds", default="0-47")
    m.add_argument("--word", default="IBM")
    m.add_argument("--seconds", type=float, default=MATCH_TICS / TIC_HZ)
    m.add_argument("--bots", type=int, default=7)
    m.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    m.add_argument("--out", type=Path)
    args = ap.parse_args()
    tasks = [(s, args.word, args.seconds, args.bots) for s in seeds_arg(args.seeds)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = sorted(pool.map(scripted_match, tasks), key=lambda r: r["seed"])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(summarize(rows)))


if __name__ == "__main__":
    main()
