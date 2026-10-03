# SPDX-License-Identifier: Apache-2.0
"""The partner's orders, and the game carrying them out.

The person watching can tell the player what to do: stop, turn, back up, ram
the wall, fire, switch weapons, play it safe or go get them. The orders
adapter (``policy.ORDERS``) reads their words (live: their speech, which the
checkpoint's ASR transcribes inside the adapter's own request) and answers one
token, the order (:data:`ORDERS`; ``none`` for anything that is not one). No
new way of playing is trained:

* a **maneuver** (stop, left, right, around, back, ram, fire) is a few tics of
  the game adapters' own actions (``doom_env.ACTIONS``), in code, played
  instead of the style adapter's action while it runs;
* a **weapon** order holds the gun asked for in place of the weapon planner's
  pick for a while;
* a **style** order switches the style adapter (fighter, cautious,
  collector);
* a **goal** (fetch, hunt, explore) takes many decisions until it is reached:
  the scripted player's own code plays it (``expert.py``: its item seeking,
  fighting and roaming, from what is on screen), until he picks up what he was
  sent for, frags someone, or has looked around; or until it gives up. He
  cannot be sent after one bot: nothing names the bot on screen, so "go after
  Rambo" hunts whoever comes, and the obituary says afterwards who it was.
  Places ("go through the door") are no order: the map has no doors, and the
  state no map.

He obeys every order and complains when it costs him; he refuses only what
would likely kill him (:func:`refusal`: a maneuver at 25 health or less while
under fire), before he starts or halfway through. :class:`Orders` runs in the
game worker with every observation: it decides, plays the maneuver and
reports. Its facts join the match tracker's (``talk.Tracker``), so the
narrator's game state says what he was told and whether he is doing it (the
same JSON live and in the dataset); its events (an order given, ended, hurting
him) join the match's events, and two of them make him speak on his own
(``talk.TalkClock``: ``order_hurts``, ``order_done``).

Game-free (an observation is read by its fields, nothing imports ViZDoom), so
the dataset writer can import it::

    python orders.py check                    # every maneuver and refusal, synthetic
    python orders.py match --seconds 90       # a model-free match, scripted orders
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import probes

TIC_HZ = 35  # doom_env.TIC_HZ
STYLES = ("fighter", "cautious", "collector")  # expert.BEHAVIORS
# The orders adapter's output: one token, each order's first token distinct
# (policy.order_token_ids checks it against the tokenizer).
ORDERS: dict[str, str] = {
    "stop": "stand still, not firing, until told to go, another order, or 12 s",
    "go": "drop the order he is carrying out and play on",
    "left": "turn left in place, 90 degrees",
    "right": "turn right in place, 90 degrees",
    "around": "turn around in place, 180 degrees",
    "back": "back up until a wall is 1 m behind him, or 1.5 s",
    "ram": "run into the wall ahead and keep pushing for 1 s",
    "fire": "fire where he faces for 1 s",
    "weapon": "switch to the gun named (none named: his best loaded other gun)",
    "fighter": "play aggressive: hunt the bots",
    "cautious": "play it safe: avoid damage, heal",
    "collector": "grab the loot: items, ammo, weapons",
    "fetch": "go get one thing he lacks: a gun (any, or the one named), health, "
    "armor or ammo, until he picks it up, or 20 s",
    "hunt": "go after the bots (or one by name, though he cannot tell which is "
    "which) until he frags one, or 30 s",
    "explore": "go look around: run through open space for 8 s",
    "none": "not an order: a question, chatter, or something he cannot do",
}
ORDER_WORDS: tuple[str, ...] = tuple(ORDERS)
MANEUVERS = ("stop", "left", "right", "around", "back", "ram", "fire")
GOALS = ("fetch", "hunt", "explore")
# What would likely kill him at low health under fire: standing, turning his
# back, running into a wall. Firing and switching guns never hurt.
RISKY = ("stop", "left", "right", "around", "back", "ram")
# The action each maneuver plays (doom_env.ACTIONS).
PLAY = {
    "stop": "wait",
    "left": "left",
    "right": "right",
    "around": "left",
    "back": "back",
    "ram": "forward",
    "fire": "fire",
}
# How the state words what he was told.
TOLD = {
    "stop": "stop",
    "go": "go",
    "left": "turn left",
    "right": "turn right",
    "around": "turn around",
    "back": "back up",
    "ram": "ram the wall",
    "fire": "fire",
    "weapon": "switch weapons",
    "fighter": "play aggressive",
    "cautious": "play it safe",
    "collector": "grab the loot",
    "fetch": "get a gun",  # by target: get the shotgun, get health, ...
    "hunt": "hunt a bot",  # hunt Rambo
    "explore": "explore",
}
GUNS = {  # slot -> how he says it (talk.SAY by slot)
    1: "fist",
    2: "pistol",
    3: "shotgun",
    4: "chaingun",
    5: "rocket launcher",
    6: "plasma rifle",
    7: "BFG",
}
SLOT_OF = {g: s for s, g in GUNS.items()}
LOW_AMMO = {2: 20, 3: 4, 4: 20, 5: 3, 6: 20, 7: 40}  # talk.LOW_AMMO

STOP_S = 12.0  # a stop not called off ends by itself
TURN_DEG = {"left": 90.0, "right": 90.0, "around": 180.0}
TURN_STEP = 10.0  # doom_env.TURN_DEG: degrees per tic
BACK_S, BACK_WALL_M = 1.5, 1
RAM_S, BONK_S = 4.0, 1.0  # at most this long to find the wall; then push this long
FIRE_S = 1.0
WEAPON_S = 10.0  # the gun asked for overrides the planner this long
FETCH_S, HUNT_S, EXPLORE_S = 20.0, 30.0, 8.0  # a goal given up after this long
CLOSE_M = 6  # on a goal, a bot this close (or one hitting him) is fought first
ITEMS = ("weapon", "health", "armor", "ammo")  # what a fetch is for, but a gun
REFUSE_HP = 25  # at or below this, under fire, a risky order is refused
UNDER_FIRE_M = 10  # a bot this close counts as under fire
HURTS_HP = 25  # health lost while obeying: he says so ...
HURTS = ("stop", "back", "ram")  # ... obeying what holds him still or blind that long
# An end worth a remark of his own: the bonk, a stop run out, a goal over.
REMARK_ENDS = ("stop", "ram", "fetch", "hunt")
PLAYER = "AI"  # doom_env.PLAYER_NAME: his name in the obituaries

DONE, DOING, REFUSED, CANT, CANCELLED = "done", "doing", "refused", "cant", "cancelled"


def _wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def under_fire(obs) -> bool:
    """Damage in the last second, or a bot within :data:`UNDER_FIRE_M`."""
    return obs.hit > 0 or any(
        o.kind == "enemy" and o.dist <= UNDER_FIRE_M for o in obs.seen
    )


def refusal(kind: str, obs) -> str | None:
    """Why he will not do it (None: he will). The one rule: a risky maneuver at
    :data:`REFUSE_HP` health or less while under fire."""
    if kind in RISKY and obs.hp <= REFUSE_HP and under_fire(obs):
        return f"{obs.hp} health and under fire"
    return None


def weapon_arg(said: str) -> int | None:
    """The slot of the gun an order names (the first one said), if any."""
    guns = probes.weapons_said(said or "")
    return SLOT_OF[guns[0]] if guns else None


def fetch_arg(said: str) -> int | str:
    """What a fetch is for: a gun named (its slot), else health, armor or
    ammo, else any gun ("weapon")."""
    gun = weapon_arg(said)
    if gun is not None and gun > 1:
        return gun
    items = probes.items_said(said or "")
    for k in ("health", "armor"):
        if k in items:
            return k
    if items & set(probes.AMMO_KINDS) or probes._AMMO_ANY.search((said or "").lower()):
        return "ammo"
    return "weapon"


def order_arg(kind: str, said: str) -> int | str | None:
    """An order's argument: a weapon order's gun slot, a fetch's target
    (:func:`fetch_arg`), the bot a hunt names; None for the rest."""
    if kind == "weapon":
        return weapon_arg(said)
    if kind == "fetch":
        return fetch_arg(said)
    if kind == "hunt":
        names = sorted(probes.bot_names(said or "", ignore_case=True))
        return names[0] if names else None
    return None


def best_other(obs) -> int | None:
    """The best gun he is not holding with enough ammo for a few shots; else
    with any (the state's best_loaded_weapon rule)."""
    for enough in (True, False):
        for s in sorted(obs.arms, reverse=True):
            if (
                s in LOW_AMMO
                and s != obs.slot
                and obs.arms[s] >= (LOW_AMMO[s] if enough else 1)
            ):
                return s
    return None


class ExpertPlanner:
    """How a goal is played: the scripted player's own code (``expert.py``),
    from what is on screen, a bot close by (or hitting him) fought first.
    Imported on first use: only the game's processes play goals."""

    def __init__(self):
        from doom_env import WEAPON_ITEM_SLOT, pickable
        from expert import Expert, roam

        self.ex, self.roam = Expert(), roam
        self.slot_of, self.pickable = WEAPON_ITEM_SLOT, pickable

    def _threat(self, obs) -> str | None:
        m = self.ex._target([s for s in obs.seen if s.kind == "enemy"])
        if m is not None and (m.dist <= CLOSE_M or obs.hit > 0):
            return self.ex._fight(obs, "collector", m)
        return None

    def fetch(self, obs, target: int | str) -> str:
        def wanted(s) -> bool:
            if isinstance(target, int):
                return s.kind == "weapon" and self.slot_of.get(s.cls) == target
            if target == "weapon":
                return s.kind == "weapon" and self.pickable(
                    s.cls, s.kind, obs.hp, obs.armor, obs.arms
                )
            return s.kind == target

        items = sorted(
            (s for s in self.ex._items(obs) if wanted(s)), key=lambda s: s.dist
        )
        if a := self._threat(obs):
            return a
        return self.ex._go_to(obs, items[0]) if items else self.roam(obs)

    def hunt(self, obs) -> str:
        return self.ex.act(obs, "fighter")

    def explore(self, obs) -> str:
        return self._threat(obs) or self.roam(obs)


class Orders:
    """The partner's orders in one match, carried out (the game worker's).

    :meth:`give` when an order arrives (with the latest observation): he
    obeys it, refuses it or cannot do it, at once; :meth:`update` with every
    observation, dead or alive, in order (``talk.Tracker`` calls it): the
    maneuver's progress, its end, what it cost him, and the events since the
    last call; :meth:`act` before each step: the action and weapon slot to
    play. ``style``: the style adapter playing (a style order switches it).
    ``log``: every order, as it went (given, played, ended). ``planner``: who
    plays a goal (:class:`ExpertPlanner` by default, made on the first goal)."""

    def __init__(self, style: str = STYLES[0], planner=None):
        self.style = style
        self.planner = planner
        self.cur: dict | None = None  # the latest order
        self.n = 0
        self.log: list[dict] = []
        self._queued: list[dict] = []  # events since the last update
        self._hp: int | None = None  # the last live tic's health, heading
        self._face: int | None = None

    # ── Orders in ──────────────────────────────────────────────────────────────
    def give(self, kind: str, arg: int | None, said: str, obs) -> str:
        """A new order (``kind`` in :data:`ORDERS`, not ``none``; ``arg``: a
        weapon order's slot, :func:`order_arg`; ``said``: the partner's words)
        at observation ``obs``. Returns its status: ``doing``, ``done`` (a style
        order or "go": done at once), ``refused`` or ``cant``."""
        if kind not in TOLD:
            raise ValueError(f"not an order: {kind!r}")
        self.n += 1
        t = obs.tick
        prev = (
            self.cur if self.cur is not None and self.cur["status"] == DOING else None
        )
        if prev is not None:
            self._end(
                prev, t, CANCELLED, "told to go" if kind == "go" else "a new order"
            )
        o = {
            "n": self.n,
            "kind": kind,
            "arg": arg,
            "said": said,
            "told": TOLD[kind],
            "tick": t,
            "status": DOING,
            "why": None,
            "end": None,
            "start": None,  # the first tic it played
            "lost": 0,
            "turned": 0.0,
            "hit_wall": None,
            "bonk": None,
            "hurt": False,
            "acts": {},
            "got": None,  # a goal's: what he picked up, whom he fragged
        }
        status, why = self._decide(o, obs, prev)
        o["status"], o["why"] = status, why
        if status != DOING:
            o["end"] = t
        self.cur = o
        self.log.append(o)
        self._queued.append(
            {
                "kind": "order",
                "order": kind,
                "told": o["told"],
                "status": status,
                "why": why,
                "said": said,
            }
        )
        return status

    def _decide(self, o: dict, obs, prev: dict | None) -> tuple[str, str | None]:
        kind = o["kind"]
        if kind in STYLES:
            was, self.style = self.style, kind
            return DONE, ("already playing that way" if was == kind else None)
        if kind == "go":
            if prev is None:
                return DONE, "no order to call off"
            o["ended"] = prev["told"]
            return DONE, None
        if obs.dead:
            return CANT, "dead, respawning"
        if kind == "fetch":
            return self._fetch(o, obs)
        if kind == "hunt":
            o["told"] = f"hunt {o['arg']}" if o["arg"] else TOLD["hunt"]
            return DOING, None
        if kind == "explore":
            return DOING, None
        if kind == "weapon":
            slot = o["arg"] if o["arg"] is not None else best_other(obs)
            if slot is None:
                return CANT, "no other gun loaded"
            gun = GUNS[slot]
            o["slot"], o["told"] = slot, f"switch to the {gun}"
            if slot not in obs.arms:
                return CANT, f"no {gun}"
            if slot > 1 and obs.arms[slot] <= 0:
                return CANT, f"no ammo for the {gun}"
            if slot == obs.slot:
                return DONE, f"already holding the {gun}"
            return DOING, None
        why = refusal(kind, obs)
        return (REFUSED, why) if why else (DOING, None)

    def _fetch(self, o: dict, obs) -> tuple[str, str | None]:
        """A fetch: off to get it, or not (already has it: a gun he owns is
        switched to instead)."""
        t = o["arg"] or "weapon"
        o["arg"], o["arms0"] = t, sorted(obs.arms)
        if isinstance(t, int):
            gun = GUNS[t]
            o["told"] = f"get the {gun}"
            if t in obs.arms:  # he has one: he takes it out
                if obs.arms[t] <= 0:
                    return CANT, f"has the {gun}, no ammo for it"
                if t == obs.slot:
                    return DONE, f"already holding the {gun}"
                o["slot"] = t
                return DOING, f"already had the {gun}: switching to it"
            return DOING, None
        o["told"] = {"weapon": "get a gun"}.get(t, f"get {t}")
        if t == "weapon" and all(s in obs.arms for s in LOW_AMMO if s > 2):
            return CANT, "already has every gun"
        if t == "health" and obs.hp >= 100:
            return CANT, "health already full"
        if t == "armor" and obs.armor >= 100:
            return CANT, "armor already full"
        return DOING, None

    # ── Every tic ──────────────────────────────────────────────────────────────
    def update(self, obs) -> list[dict]:
        """The order's progress at ``obs``; returns the events since the last
        call (each a ``talk.Tracker`` event without its tick and index)."""
        out, self._queued = self._queued, []
        o = self.cur
        if o is not None and o["status"] == DOING and obs.tick > o["tick"]:
            self._progress(o, obs)
            out += self._queued
            self._queued = []
        self._hp = None if obs.dead else obs.hp
        self._face = None if obs.dead else obs.face
        return out

    def _progress(self, o: dict, obs) -> None:
        if obs.dead:
            o["died"] = True
            self._end(o, obs.tick, CANCELLED, "died")
            return
        if self._hp is not None and obs.hp < self._hp:
            o["lost"] += self._hp - obs.hp
        if o["lost"] >= HURTS_HP and not o["hurt"] and o["kind"] in HURTS:
            o["hurt"] = True
            self._queued.append(
                {
                    "kind": "order_hurts",
                    "order": o["kind"],
                    "told": o["told"],
                    "lost": o["lost"],
                    "health": obs.hp,
                }
            )
        k = o["kind"]
        el = (obs.tick - o["tick"]) / TIC_HZ
        if k in TURN_DEG:
            if self._face is not None:
                o["turned"] += abs(_wrap180(obs.face - self._face))
            need = TURN_DEG[k] - TURN_STEP / 2
            if o["turned"] >= need or el >= 2 * TURN_DEG[k] / TURN_STEP / TIC_HZ:
                self._end(o, obs.tick, DONE)
        elif k == "stop" and el >= STOP_S:
            self._end(o, obs.tick, DONE)
        elif k == "back" and (obs.walls[3] <= BACK_WALL_M or el >= BACK_S):
            self._end(o, obs.tick, DONE)
        elif k == "ram":
            if o["bonk"] is None and obs.walls[1] == 0:
                o["bonk"], o["hit_wall"] = obs.tick, True
            if o["bonk"] is not None and (obs.tick - o["bonk"]) / TIC_HZ >= BONK_S:
                self._end(o, obs.tick, DONE)
            elif o["bonk"] is None and el >= RAM_S:
                o["hit_wall"] = False
                self._end(o, obs.tick, DONE, "no wall in reach")
        elif k == "fire" and el >= FIRE_S:
            self._end(o, obs.tick, DONE)
        elif (k == "weapon" or o.get("slot")) and el >= WEAPON_S:
            self._end(o, obs.tick, DONE)
        elif k in GOALS and not o.get("slot"):
            self._goal(o, obs, el)
        if o["status"] == DOING and (why := refusal(k, obs)):
            self._end(o, obs.tick, REFUSED, why)  # he quits halfway

    def _goal(self, o: dict, obs, el: float) -> None:
        """A goal reached (he picked it up, fragged someone), or given up."""
        k, t = o["kind"], o["arg"]
        if k == "fetch":
            if isinstance(t, int):
                got = GUNS[t] if t in obs.arms else None
            elif t == "weapon":
                new = set(obs.arms) - set(o["arms0"])
                got = GUNS[max(new)] if new and "got weapon" in obs.events else None
            else:
                got = t if f"got {t}" in obs.events else None
            if got:
                o["got"] = got
                self._end(o, obs.tick, DONE)
            elif el >= FETCH_S:
                self._end(o, obs.tick, DONE, "found none")
        elif k == "hunt":
            if "frag" in obs.events:
                mine = [
                    x["victim"]
                    for x in getattr(obs, "obits", ())
                    if x["killer"] == PLAYER and x["victim"] != PLAYER
                ]
                o["got"] = mine[0] if mine else "a bot"
                self._end(o, obs.tick, DONE)
            elif el >= HUNT_S:
                self._end(o, obs.tick, DONE, "no frag")
        elif el >= EXPLORE_S:
            self._end(o, obs.tick, DONE)

    def _end(self, o: dict, tick: int, status: str, why: str | None = None) -> None:
        o["status"], o["why"], o["end"] = status, why, tick
        # His own remark on it: the bonk, a long stop over, quitting halfway.
        # Not a cancel: the partner just spoke (or he died), and that is answered.
        remark = status == REFUSED or (status == DONE and o["kind"] in REMARK_ENDS)
        e = {
            "kind": "order_end",
            "order": o["kind"],
            "told": o["told"],
            "status": status,
            "why": why,
            "lost": o["lost"],
            "s": round((tick - o["tick"]) / TIC_HZ, 1),
            "remark": remark,
        }
        if o["hit_wall"] is not None:
            e["hit_wall"] = o["hit_wall"]
        if o["got"]:
            e["got"] = o["got"]
        self._queued.append(e)

    def act(self, obs, action: str, slot: int | None) -> tuple[str, int | None]:
        """What to play at ``obs``: the style adapter's ``action`` and the
        weapon planner's ``slot``, unless an order overrides them."""
        o = self.cur
        if o is None or o["status"] != DOING or obs.dead:
            return action, slot
        if o["start"] is None:
            o["start"] = obs.tick
        if o.get("slot"):  # a weapon order, or a fetch of a gun he had
            return action, o["slot"]
        if o["kind"] in GOALS:
            if self.planner is None:
                self.planner = ExpertPlanner()
            k = o["kind"]
            a = (
                self.planner.fetch(obs, o["arg"])
                if k == "fetch"
                else getattr(self.planner, k)(obs)
            )
        else:
            a = PLAY[o["kind"]]
        o["acts"][a] = o["acts"].get(a, 0) + 1
        return a, slot

    # ── Out ────────────────────────────────────────────────────────────────────
    def facts(self) -> dict | None:
        """The latest order, JSON-able (None before the first), and the style
        he plays since (``talk.game_state`` renders both)."""
        o = self.cur
        if o is None:
            return None
        keep = ("n", "kind", "told", "status", "why", "tick", "end", "lost")
        d = {k: o[k] for k in keep}
        for k in ("hit_wall", "died", "ended", "got"):
            if o.get(k) is not None:
                d[k] = o[k]
        d["style"] = self.style
        return d


# ── Checks ─────────────────────────────────────────────────────────────────────
class _Sim:
    """A stand-in game for the checks: heading, wall clearances and health
    follow the actions played (10 degrees a turn tic, half a metre a running
    tic), with an optional attacker."""

    def __init__(self, front: float = 9.0, rear: float = 9.0, hp: int = 100):
        from types import SimpleNamespace

        self.ns = SimpleNamespace
        self.tick, self.face, self.front, self.rear = 0, 90, front, rear
        self.hp, self.hit, self.dead = hp, 0, False
        self.arms, self.slot = {1: 0, 2: 50, 3: 8}, 2
        self.bot: float | None = None  # a bot's distance, if one is in view
        self.dmg = 0  # damage per tic
        self.armor, self.events, self.obits = 0, (), ()  # this tic's

    def obs(self):
        seen = [] if self.bot is None else [self.ns(kind="enemy", dist=self.bot)]
        walls = (9, min(9, int(self.front)), 9, min(9, int(self.rear)))
        return self.ns(
            tick=self.tick,
            hp=self.hp,
            hit=self.hit,
            dead=self.dead,
            seen=seen,
            walls=walls,
            face=self.face % 360,
            arms=dict(self.arms),
            slot=self.slot,
            armor=self.armor,
            events=tuple(self.events),
            obits=tuple(self.obits),
        )

    def step(self, action: str, slot: int | None) -> None:
        self.tick += 1
        self.events, self.obits = (), ()
        self.face += {"left": 10, "right": -10}.get(action, 0)
        if action == "forward":
            self.front = max(0.0, self.front - 0.5)
        if action == "back":
            self.rear = max(0.0, self.rear - 0.4)
        if slot is not None and slot in self.arms:
            self.slot = slot
        if self.dmg:
            self.hp, self.hit = max(0, self.hp - self.dmg), self.dmg
            self.dead = self.hp == 0


def _run(sim: _Sim, orders: Orders, tics: int, style_act: str = "forward"):
    """``tics`` tics of play; returns the actions played and the events."""
    played, events = [], []
    for _ in range(tics):
        ob = sim.obs()
        a, s = orders.act(ob, style_act, None)
        played.append(a)
        sim.step(a, s)
        events += orders.update(sim.obs())
    return played, events


def check() -> None:
    """Every maneuver and refusal on synthetic observations."""

    def fresh(**kw):
        sim, o = _Sim(**kw), Orders()
        o.update(sim.obs())
        return sim, o

    # Turns: 90 degrees in 9 tics, 180 in 18, then the style adapter again.
    for kind, n in (("left", 9), ("right", 9), ("around", 18)):
        sim, o = fresh()
        assert o.give(kind, None, kind, sim.obs()) == DOING
        played, ev = _run(sim, o, n + 3)
        want = PLAY[kind]
        assert played[:n] == [want] * n and played[n:] == ["forward"] * 3, (
            kind,
            played,
        )
        assert o.cur["status"] == DONE and o.cur["turned"] == TURN_DEG[kind], o.cur
        assert [e["kind"] for e in ev] == ["order", "order_end"], ev
        assert not ev[-1]["remark"], "a turn's end needs no remark"
    # Ram: forward until the front clearance is 0, then 1 s more; the bonk.
    sim, o = fresh(front=4.0)
    o.give("ram", None, "ram the wall", sim.obs())
    played, ev = _run(sim, o, 60)
    k = played.index("forward", 0)
    assert k == 0 and o.cur["hit_wall"] and o.cur["status"] == DONE, o.cur
    bonk = o.cur["bonk"]
    assert sim.front == 0 and o.cur["end"] - bonk == round(BONK_S * TIC_HZ), o.cur
    end = ev[-1]
    assert end["kind"] == "order_end" and end["remark"] and end["hit_wall"], end
    # Ram with no wall in reach: 4 s, then it gives up.
    sim, o = fresh(front=200.0)
    o.give("ram", None, "ram it", sim.obs())
    _run(sim, o, 150)
    assert o.cur["hit_wall"] is False and o.cur["why"] == "no wall in reach", o.cur
    # Stop: wait (no firing) for 12 s, then a remark; "go" calls it off sooner.
    sim, o = fresh()
    o.give("stop", None, "stop", sim.obs())
    played, ev = _run(sim, o, int(STOP_S * TIC_HZ) + 2)
    assert set(played[: int(STOP_S * TIC_HZ)]) == {"wait"}, set(played)
    assert ev[-1]["status"] == DONE and ev[-1]["remark"], ev[-1]
    sim, o = fresh()
    o.give("stop", None, "stop", sim.obs())
    _run(sim, o, 40)
    assert o.give("go", None, "ok go", sim.obs()) == DONE
    assert o.cur["ended"] == "stop" and o.log[0]["status"] == CANCELLED, o.log
    played, _ = _run(sim, o, 3)
    assert played == ["forward"] * 3
    assert o.give("go", None, "go", sim.obs()) == DONE and o.cur["why"], o.cur
    # Back: until the rear wall is 1 m away (or 1.5 s); fire: 1 s.
    sim, o = fresh(rear=3.0)
    o.give("back", None, "back up", sim.obs())
    played, _ = _run(sim, o, 30)
    assert o.cur["status"] == DONE and int(sim.rear) <= BACK_WALL_M, (sim.rear, o.cur)
    assert played.count("back") < BACK_S * TIC_HZ
    sim, o = fresh()
    o.give("fire", None, "shoot", sim.obs())
    played, _ = _run(sim, o, 40)
    assert played.count("fire") == round(FIRE_S * TIC_HZ), played
    # Weapons: the gun named, held 10 s; one he lacks, or without ammo: cant.
    sim, o = fresh()
    assert (
        o.give("weapon", weapon_arg("switch to the shotgun"), "x", sim.obs()) == DOING
    )
    assert o.cur["told"] == "switch to the shotgun"
    _run(sim, o, 5)
    assert sim.slot == 3
    assert o.give("weapon", weapon_arg("use the bfg"), "x", sim.obs()) == CANT
    assert o.cur["why"] == "no BFG", o.cur
    sim.arms[4] = 0
    assert o.give("weapon", weapon_arg("chaingun"), "x", sim.obs()) == CANT
    assert o.cur["why"] == "no ammo for the chaingun", o.cur
    assert o.give("weapon", weapon_arg("shotgun"), "x", sim.obs()) == DONE  # holding it
    sim.slot = 2
    assert o.give("weapon", None, "switch guns", sim.obs()) == DOING
    assert o.cur["slot"] == 3, "no gun named: his best loaded other gun"
    # Refusals: a risky maneuver at 25 health or less, under fire; never firing.
    sim, o = fresh(hp=20)
    assert o.give("stop", None, "stop", sim.obs()) == DOING, "not under fire"
    sim.hit = 5
    assert o.give("stop", None, "stop", sim.obs()) == REFUSED
    assert o.cur["why"] == "20 health and under fire", o.cur
    assert o.give("fire", None, "fire", sim.obs()) == DOING
    sim.hit, sim.bot = 0, 6.0
    assert o.give("ram", None, "ram", sim.obs()) == REFUSED, "a bot 6 m away"
    sim.bot = 15.0
    assert o.give("left", None, "left", sim.obs()) == DOING
    # Halfway: stopped under fire, it hurts (a remark), then he quits.
    sim, o = fresh(hp=60)
    o.give("stop", None, "stop", sim.obs())
    sim.dmg = 2
    _, ev = _run(sim, o, 30)
    kinds = [e["kind"] for e in ev]
    assert kinds == ["order", "order_hurts", "order_end"], kinds
    assert ev[1]["lost"] >= HURTS_HP and ev[2]["status"] == REFUSED and ev[2]["remark"]
    # Dying while obeying cancels it; an order while dead cannot be done.
    sim, o = fresh(hp=40)
    o.give("stop", None, "stop", sim.obs())
    sim.dmg = 50  # one rocket
    _, ev = _run(sim, o, 5)
    assert o.log[0]["status"] == CANCELLED and o.log[0]["died"], o.log[0]
    assert not ev[-1]["remark"], "the death is its own news"
    assert o.give("left", None, "left", sim.obs()) == CANT
    # Style orders switch the style adapter, at once.
    sim, o = fresh()
    assert o.give("cautious", None, "play it safe", sim.obs()) == DONE
    assert o.style == "cautious" and o.facts()["style"] == "cautious"
    assert o.give("cautious", None, "careful", sim.obs()) == DONE and o.cur["why"]
    played, _ = _run(sim, o, 2)
    assert played == ["forward", "forward"], "a style order plays no maneuver"
    check_goals()
    print(
        "OK: turns (90 in 9 tics, 180 in 18), ram to the wall and the bonk, ram "
        "with no wall, stop and go, back, fire, weapons (named, missing, empty, "
        "held, best other), refusals before and halfway, order_hurts, death, style; "
        "goals: fetch (a gun, one he has, health given up), hunt (who it got), explore"
    )


class _Planner:
    """A stand-in for ExpertPlanner: what it was asked to play."""

    def __init__(self):
        self.asked: list[tuple] = []

    def fetch(self, obs, target):
        self.asked.append(("fetch", target))
        return "fl"

    def hunt(self, obs):
        self.asked.append(("hunt",))
        return "fire"

    def explore(self, obs):
        self.asked.append(("explore",))
        return "forward"


def check_goals() -> None:
    """Goals: what the words ask for, played by the planner until reached (a
    pickup, a frag) or given up; a gun he has is switched to; what cannot be
    fetched; whom a hunt got."""
    for said, want in (
        ("grab a gun", "weapon"),
        ("go get the rocket launcher", 5),
        ("find some health", "health"),
        ("get armor", "armor"),
        ("grab some shells", "ammo"),
        ("we need ammo", "ammo"),
    ):
        assert fetch_arg(said) == want, (said, fetch_arg(said), want)
    assert order_arg("hunt", "go after rambo") == "Rambo"
    assert order_arg("hunt", "go kill something") is None

    def fresh(**kw):
        sim, pl = _Sim(**kw), _Planner()
        o = Orders(planner=pl)
        o.update(sim.obs())
        return sim, o, pl

    # A gun, any: the planner's moves until a new gun is picked up.
    sim, o, pl = fresh()
    assert o.give("fetch", "weapon", "grab a gun", sim.obs()) == DOING
    assert o.cur["told"] == "get a gun"
    played, ev = _run(sim, o, 30)
    assert set(played) == {"fl"} and pl.asked[0] == ("fetch", "weapon"), played
    sim.arms[5], sim.events = 5, ("got weapon",)
    ev = o.update(sim.obs())
    assert o.cur["status"] == DONE and o.cur["got"] == "rocket launcher", o.cur
    assert ev[-1]["remark"] and ev[-1]["got"] == "rocket launcher", ev
    assert o.facts()["got"] == "rocket launcher"
    # A gun he has: he takes it out instead; one he cannot hold any more of.
    sim, o, pl = fresh()
    assert o.give("fetch", 3, "get the shotgun", sim.obs()) == DOING
    assert o.cur["why"].startswith("already had the shotgun"), o.cur
    _run(sim, o, 3)
    assert sim.slot == 3 and not pl.asked, "switched, nothing fetched"
    sim.hp = 100
    assert o.give("fetch", "health", "find health", sim.obs()) == CANT
    assert o.cur["why"] == "health already full"
    # Health: given up after FETCH_S without one.
    sim.hp = 60
    o.give("fetch", "health", "find health", sim.obs())
    _, ev = _run(sim, o, int(FETCH_S * TIC_HZ) + 2)
    assert o.cur["status"] == DONE and o.cur["why"] == "found none", o.cur
    assert ev[-1]["remark"] and "got" not in ev[-1], ev[-1]
    # A hunt, one bot named: it plays the fighter until a frag; the obituary
    # says who it was (not whom he was sent after).
    sim, o, pl = fresh()
    o.give("hunt", "Rambo", "go after rambo", sim.obs())
    assert o.cur["told"] == "hunt Rambo"
    _run(sim, o, 20)
    sim.events = ("frag",)
    sim.obits = ({"victim": "Leone", "killer": PLAYER, "weapon": "shotgun"},)
    o.update(sim.obs())
    assert o.cur["status"] == DONE and o.cur["got"] == "Leone", o.cur
    assert {a for a, *_ in pl.asked} == {"hunt"}
    # Explore: the planner's roaming for EXPLORE_S, no remark at the end.
    sim, o, pl = fresh()
    o.give("explore", None, "go look around", sim.obs())
    played, ev = _run(sim, o, int(EXPLORE_S * TIC_HZ) + 3)
    assert played[: int(EXPLORE_S * TIC_HZ)] == ["forward"] * int(EXPLORE_S * TIC_HZ)
    assert o.cur["status"] == DONE and not ev[-1]["remark"], ev[-1]
    json.dumps([o.log, o.facts()])  # the log and facts stay JSON


# ── A model-free match with a scripted partner ─────────────────────────────────
SCRIPT = (
    "stop right there",
    "turn left",
    "ok go",
    "turn around",
    "ram the wall",
    "switch to the shotgun",
    "back up",
    "play it safe",
    "shoot",
    "turn right",
    "go get them",
    "grab a gun",
    "go after rambo",
    "go look around",
    "what is the score",  # not an order
)


def match(args) -> None:
    """One match with no model, through collect.py's teacher path (as the
    narrator's order data is collected): the scripted player plays, a
    scripted partner says :data:`SCRIPT` (one line every ``--every`` s), the
    keyword stand-in (``policy.keyword_orders``) reads each, and the orders
    are carried out. Prints the actions played around each order and the
    narrator's ``order`` state on the tic after it."""
    from collections import Counter

    import collect
    from policy import keyword_orders
    from talk import EventLog, game_state

    sched = []
    for i, text in enumerate(SCRIPT):
        got = keyword_orders(text)
        tick = round((i + 1) * args.every * TIC_HZ)
        print(f"t{tick / TIC_HZ:5.1f} partner: {text!r} -> {got.kind}")
        if got.kind != "none":
            sched.append([tick, got.kind, order_arg(got.kind, text), text, "at"])
    task = {
        "behavior": STYLES[0],
        "ep": 0,
        "seed": args.seed,
        "timeout": int(args.seconds * TIC_HZ),
        "bots": "default",
        "n_bots": 7,
        "row_every": 1,
        "teacher": "expert",
        "dart": 0.0,
        "beta": 0.0,
        "policy": "teacher",
        "keep_rows": True,
        "video": None,
        "orders": sched,
    }
    res = collect._teacher_episode(task)
    rows = {r["t"]: r for r in res["rows"]}
    log = EventLog()
    for r in res["rows"]:
        log.add(r["facts"]["news"])
    print()
    for o in res["orders"]:
        t0, t1 = o["tick"], o["end"] if o["end"] is not None else o["tick"]
        # What the rows say was played while it ran, and the state on the tic after.
        played = Counter(rows[t]["act"] for t in range(t0, t1) if t in rows)
        after = next((rows[t] for t in range(t0 + 1, t0 + 40) if t in rows), None)
        st = (
            game_state(after["state"], after["facts"], log.events, STYLES[0])
            if after
            else {}
        )
        extra = "" if o["hit_wall"] is None else f" hit_wall={o['hit_wall']}"
        print(
            f"t{t0 / TIC_HZ:5.1f} {o['told']:<22} {o['status']:<9} "
            f"{(t1 - t0) / TIC_HZ:4.1f} s  played {o['acts']} (rows: {dict(played)})"
            f"  why={o['why']} lost={o['lost']}{extra}"
            f"\n        state order: {json.dumps(st.get('order'))}"
        )
        if o["kind"] in PLAY and o["start"] is not None:
            assert set(played) <= {PLAY[o["kind"]]}, f"{o['told']}: rows {played}"
    assert len(res["orders"]) == len(sched), "every order reached the game"
    for o in res["orders"]:
        if o["kind"] in MANEUVERS and o["start"] is not None:
            assert o["acts"].get(PLAY[o["kind"]]), f"{o['told']}: never played"
    print(f"OK: {len(sched)} orders carried out")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="Every maneuver and refusal, synthetic")
    m = sub.add_parser("match", help="A model-free match with scripted orders")
    m.add_argument("--seconds", type=float, default=90.0)
    m.add_argument("--every", type=float, default=6.0, help="Seconds between orders")
    m.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    if args.cmd == "check":
        check()
    else:
        match(args)


if __name__ == "__main__":
    main()
