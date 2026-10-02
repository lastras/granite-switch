# SPDX-License-Identifier: Apache-2.0
"""The player's own voice: what it is told before it speaks, and when it speaks.

The narrator does not read the game log the game adapters read (terse, ``t12.4
hp 64 face 135 | bot +10 8m | did cl | frag``, and only 10 s of it). It reads
its own conversation with the person watching (:mod:`conversation`), and before
each line it calls the ``get_game_state`` tool; :func:`game_state` is what the
tool returns, JSON with no prose in it::

    {"time": "2:31", "time_left": "7:29",
     "you": {"frags": 12, "deaths": 5, "rank": 2, "players": 8, "health": 64, ...},
     "scoreboard": {"Rambo": 13, "you": 12, "Leone": 9, ...},
     "last_death": {"killer": "Rambo", "your_weapon": "BFG", "seconds_ago": 4},
     "recent_events": [{"time": "2:27", "type": "death", "killer": "Rambo", ...}, ...],
     ...}

The facts come from a :class:`Tracker`, fed every tic with the observation (its
scoreboard included): frags and the weapon held, deaths and the killer (the bot
whose frag count rose on the tic the player died), streaks and droughts, close
calls, lead changes, every pickup (a weapon by name; health, armor and each kind
of ammo with the amount), time left. Each event it fires carries an index, so
the match's events can be gathered once each into an :class:`EventLog`: live,
from the events every tic sends; in the dataset, from the rows' recent news.
The same tracker runs in collect.py's workers, in the engine's game worker and
in record_video, so the narrator is trained and served on the same JSON from
the same facts; ``python talk.py check`` replays a match through both paths
and compares them.

The older *brief* (:func:`brief`, the moment in plain sentences) is still
rendered for the videos' sound tags and the round-3 prompts.

When to speak (:class:`TalkClock`): soon after a salient event (a death, a lead
change, a streak, a close call, the first frag in a while, a new weapon: at
least ``MIN_GAP_S`` after the last line; a plain frag, ``FRAG_GAP_S``), after
``IDLE_S`` of silence, and whenever the person watching speaks.

The line it speaks joins the narrator's conversation, with what happened since
his last line (:func:`moment_events`) and what the partner said; the game log
never holds it.
"""

from __future__ import annotations

import re
from collections import deque

from conversation import clock
from doom_env import TIC_HZ, WEAPON_NAMES

# ── Match facts ────────────────────────────────────────────────────────────────
NEWS_S = 4.0  # an event stays under "Just now" this long
STREAK_S, STREAK_N = 10.0, 3  # this many frags within STREAK_S is a streak
DROUGHT_S = 45.0  # a frag after this long without one ends a drought
# A close call: CLOSE_DROP health lost within CLOSE_S, down to CLOSE_HP or less,
CLOSE_DROP, CLOSE_HP, CLOSE_S = 40, 40, 2.0
CLOSE_CONFIRM_S = 1.0  # ... and still alive this much later: a close call
CLOSE_COOLDOWN_S = 5.0
KILLER_TICS = 2  # the killer's frag count rises on the death tic; a little slack
# How the weapons are said (the state line's names are one token each).
SAY = {
    "fist": "fist",
    "pistol": "pistol",
    "shotgun": "shotgun",
    "chaingun": "chaingun",
    "launcher": "rocket launcher",
    "plasma": "plasma rifle",
    "bfg": "BFG",
}
# The ammo each gun fires (the pistol and the chaingun share bullets, the plasma
# rifle and the BFG cells).
AMMO_OF = {
    2: "bullets",
    3: "shells",
    4: "bullets",
    5: "rockets",
    6: "cells",
    7: "cells",
}


class Tracker:
    """The match's memory: what the 10 s history cannot hold.

    Call :meth:`update` with every observation, dead or alive, in order (the
    one from ``reset`` too); it returns the events that tic produced, each with
    its tick, its kind and ``i``, its index in the match. ``facts()`` is a small
    JSON-able snapshot: collect.py stores it with every row and the engine's
    game worker sends it with every state; :func:`game_state` and
    :func:`brief` render it. ``match_s``: the match length, for the time left.
    """

    def __init__(self, match_s: float | None = None):
        self.match_s = match_s
        self.tick = 0
        self.frags = self.deaths = 0
        self.bots: dict[str, int] = {}
        self.weapon = "pistol"  # held on the last live tic
        self.news: list[dict] = []  # events of the last NEWS_S, oldest first
        self.frag_ticks: list[int] = []
        self.killed_by: dict[str, int] = {}
        self.last_death: dict | None = None
        self._n = 0  # events fired so far
        self._boards: deque[dict[str, int]] = deque(maxlen=KILLER_TICS + 1)
        self._death: dict | None = None  # waiting for the killer's frag count
        self._hp: deque[tuple[int, int]] = deque()
        self._close: dict | None = None  # a big drop, waiting to be survived
        self._last_close = -(10**9)
        self._leading = False
        self._arms: set[int] = {1, 2}
        self._hud: tuple[int, int, dict[int, int]] | None = None  # the last live tic's
        self._last_frag = 0

    def update(self, obs) -> list[dict]:
        tick = self.tick = obs.tick
        bots = {name: f for name, f in obs.priv.scoreboard[1:]}
        ev = obs.events
        fired: list[dict] = []

        def fire(kind: str, **kw) -> None:
            fired.append({"tick": tick, "kind": kind, "i": self._n, **kw})
            self._n += 1

        for _ in range(ev.count("frag")):
            gap = (tick - self._last_frag) / TIC_HZ
            self._last_frag = tick
            self.frag_ticks.append(tick)
            drought = {"first_in_s": round(gap)} if gap >= DROUGHT_S else {}
            fire("frag", weapon=SAY[obs.weapon], **drought)
            if gap >= DROUGHT_S:
                fire("drought_ended", gap=round(gap))
            n = self._streak()
            if n >= STREAK_N:
                fire("streak", n=n)
        if "died" in ev:
            self._death = {
                "tick": tick,
                "before": self._boards[0] if self._boards else dict(bots),
                "weapon": SAY[self.weapon],
                "suicide": "suicide" in ev,
            }
            self._close = None
        if self._death is not None:
            d = self._death
            risen = {n: f - d["before"].get(n, 0) for n, f in bots.items()}
            risen = {n: r for n, r in risen.items() if r > 0}
            if d["suicide"] or risen or tick - d["tick"] >= KILLER_TICS:
                by = (
                    "yourself"
                    if d["suicide"]
                    else (max(risen, key=risen.get) if risen else None)
                )
                again = 1
                if by and self.last_death and self.last_death["by"] == by:
                    again = self.last_death["again"] + 1
                if by and by != "yourself":
                    self.killed_by[by] = self.killed_by.get(by, 0) + 1
                self.last_death = {
                    "tick": tick,
                    "by": by,
                    "weapon": d["weapon"],
                    "again": again,
                }
                fire("died", by=by, weapon=d["weapon"], again=again)
                self._death = None
        if "respawn" in ev:
            fire("respawn")
        if "got weapon" in ev:
            for slot in sorted(set(obs.arms) - self._arms):
                fire("weapon", name=SAY[WEAPON_NAMES[slot]])
        self._pickups(obs, fire)
        if not obs.dead:
            self.weapon = obs.weapon
            self._arms = set(obs.arms)
        self._hud = None if obs.dead else (obs.hp, obs.armor, dict(obs.arms))
        self._close_call(obs, fire)

        self.frags, self.deaths = obs.frags, obs.deaths
        best = max(bots.values(), default=0)
        leading = self.frags > best
        if leading != self._leading:
            if leading:
                fire("took_lead")
            else:
                top = max(bots, key=bots.get)
                fire("lost_lead", by=top, tied=bots[top] == self.frags)
        self._leading = leading
        self.bots = bots
        self._boards.append(dict(bots))
        keep = int(NEWS_S * TIC_HZ)
        self.news = [e for e in self.news if tick - e["tick"] <= keep] + fired
        return fired

    def _streak(self) -> int:
        span = STREAK_S * TIC_HZ
        return sum(1 for t in self.frag_ticks if self.tick - t <= span)

    def _pickups(self, obs, fire) -> None:
        """Health, armor and ammo picked up, with the amount (the HUD's rise
        since the last live tic; a weapon is its own event, its ammo with it).
        The amount is left out when damage on the same tic hides it."""
        got = {e[4:] for e in obs.events if e.startswith("got ")}
        if obs.dead or self._hud is None or not got:
            return
        hp, armor, arms = self._hud
        for item, gain in (("health", obs.hp - hp), ("armor", obs.armor - armor)):
            if item in got:
                fire("pickup", item=item, **({"amount": gain} if gain > 0 else {}))
        if "ammo" in got:
            gains: dict[str, int] = {}
            for slot, ammo in obs.arms.items():
                if slot in AMMO_OF and slot in arms:
                    kind = AMMO_OF[slot]
                    gains[kind] = max(gains.get(kind, 0), ammo - arms[slot])
            for kind, gain in sorted(gains.items()):
                if gain > 0:
                    fire("pickup", item=kind, amount=gain)

    def _close_call(self, obs, fire) -> None:
        """A big drop in health that the player lives through."""
        tick = obs.tick
        if obs.dead:
            self._hp.clear()
            return
        self._hp.append((tick, obs.hp))
        while tick - self._hp[0][0] > CLOSE_S * TIC_HZ:
            self._hp.popleft()
        if self._close is None:
            drop = max(h for _, h in self._hp) - obs.hp
            if (
                drop >= CLOSE_DROP
                and obs.hp <= CLOSE_HP
                and tick - self._last_close >= CLOSE_COOLDOWN_S * TIC_HZ
            ):
                self._close = {"tick": tick, "lost": drop, "low": obs.hp}
            return
        c = self._close
        c["low"] = min(c["low"], obs.hp)
        if tick - c["tick"] >= CLOSE_CONFIRM_S * TIC_HZ:
            fire("close_call", lost=c["lost"], low=c["low"])
            self._last_close = tick
            self._close = None

    def facts(self) -> dict:
        t = self.tick / TIC_HZ
        board = sorted(self.bots.items(), key=lambda kv: (-kv[1], kv[0]))
        return {
            "t": round(t, 2),
            "left": None if self.match_s is None else max(0, round(self.match_s - t)),
            "frags": self.frags,
            "deaths": self.deaths,
            "board": [[n, f] for n, f in board],
            "streak": self._streak(),
            "since_frag": round((self.tick - self._last_frag) / TIC_HZ),
            "last_death": dict(self.last_death) if self.last_death else None,
            "killed_by": dict(self.killed_by),
            "news": [dict(e) for e in self.news],
        }


class EventLog:
    """A match's events in order, each once (by its index ``i``), gathered
    from what the tracker fired (live: every tic's events) or from the rows'
    recent news (the dataset: rows come every 0.2 s, news holds 4 s)."""

    def __init__(self):
        self.events: list[dict] = []
        self._seen: set[int] = set()

    def add(self, events) -> None:
        for e in events:
            if e["i"] not in self._seen:
                self._seen.add(e["i"])
                self.events.append(e)

    def since(self, after: int | None, upto: int) -> list[dict]:
        """The events after tick ``after`` (None: from the start) up to ``upto``."""
        lo = -1 if after is None else after
        return [e for e in self.events if lo < e["tick"] <= upto]


# ── The game state (the get_game_state tool's output) ──────────────────────────
LOG_S = 90  # recent_events: this far back
KEEP_FRAGS = KEEP_SUPPLIES = 5  # ... but only the latest frags and supply pickups
MOMENT_EVENTS = 8  # a past moment keeps at most this many events
# Ammo below which a weapon is nearly empty (the BFG spends 40 cells a shot).
LOW_AMMO = {2: 20, 3: 4, 4: 20, 5: 3, 6: 20, 7: 40}
STATE_NOTES = (
    "your_weapon is your own weapon at the time; a bot's weapon and whom you "
    "fragged are never reported; times are match time"
)

_ARMS = re.compile(r"\barms ((?:\d:\d+ ?)+)")
_HELD = re.compile(r"\| (\w+) (\d+) \| arms")
_NUM = {k: re.compile(rf"\b{k} (-?\d+)") for k in ("hp", "armor", "hit")}
_FOES = re.compile(r"\bbot ([+-]\d+) (\d+)m")


def _owned(state: str) -> dict[int, int]:
    """Owned gun slots (the fist left out) -> ammo, from the state line."""
    arms = _ARMS.search(state)
    out = {}
    for part in arms.group(1).split() if arms else ():
        slot, ammo = part.split(":")
        out[int(slot)] = int(ammo)
    return out


def _best_loaded(owned: dict[int, int]) -> str:
    """The best gun with enough ammo for a few shots; else the best with any;
    else the fist."""
    for enough in (True, False):
        for s in sorted(owned, reverse=True):
            if s in LOW_AMMO and owned[s] >= (LOW_AMMO[s] if enough else 1):
                return SAY[WEAPON_NAMES[s]]
    return "fist"


def _side(bearing: int) -> str:
    return "ahead" if abs(bearing) <= 20 else ("left" if bearing < 0 else "right")


def _ago(now: int, tick: int) -> int:
    return round((now - tick) / TIC_HZ)


def event_json(e: dict) -> dict | None:
    """A tracker event as the tool reports it (None: not reported)."""
    k = e["kind"]
    if k == "died":
        d = {
            "type": "death",
            "killer": e["by"] or "unknown",
            "your_weapon": e["weapon"],
        }
        if e["again"] >= 2:
            d["in_a_row"] = e["again"]
        return d
    if k == "frag":
        d = {"type": "frag", "your_weapon": e["weapon"]}
        if "first_in_s" in e:
            d["first_in_s"] = e["first_in_s"]
        return d
    if k == "weapon":
        return {"type": "pickup", "item": e["name"]}
    if k == "pickup":
        d = {"type": "pickup", "item": e["item"]}
        if "amount" in e:
            d["amount"] = e["amount"]
        return d
    if k == "took_lead":
        return {"type": "lead", "leader": "you"}
    if k == "lost_lead":
        d = {"type": "lead", "leader": e["by"]}
        if e["tied"]:
            d["tied_with_you"] = True
        return d
    if k == "close_call":
        return {"type": "close_call", "health_lost": e["lost"], "health_left": e["low"]}
    if k == "streak" and e["n"] in STREAK_LEVELS:
        return {"type": "streak", "frags_in_10s": e["n"]}
    return None


def _thin(events: list[dict], keep_frags: int, keep_supplies: int) -> list[dict]:
    """Every reported event, but only the latest frags and supply pickups
    (health, armor, ammo), oldest first."""
    rep = [(e, j) for e in events if (j := event_json(e)) is not None]
    frags = [e["i"] for e, j in rep if j["type"] == "frag"][-keep_frags:]
    supply = [e["i"] for e, _ in rep if e["kind"] == "pickup"][-keep_supplies:]
    keep = set(frags) | set(supply)
    return [
        e
        for e, j in rep
        if e["i"] in keep or (j["type"] != "frag" and e["kind"] != "pickup")
    ]


def moment_events(events: list[dict]) -> list[dict]:
    """What a past moment keeps (the events since the line before it): what
    happened, which stays true afterwards, and none of the readings of how
    things stood. At most MOMENT_EVENTS, oldest first."""
    kept = _thin(events, MOMENT_EVENTS, MOMENT_EVENTS)
    if len(kept) > MOMENT_EVENTS:  # the supplies go first, then the frags
        for kind in ("pickup", "frag"):
            while len(kept) > MOMENT_EVENTS and any(e["kind"] == kind for e in kept):
                kept.remove(next(e for e in kept if e["kind"] == kind))
    return [event_json(e) for e in kept]


def game_state(state: str, facts: dict, events: list[dict]) -> dict:
    """What ``get_game_state`` returns at a moment: ``state`` (the state line,
    :func:`policy.state_text`), ``facts`` (:meth:`Tracker.facts`) and the
    match's events so far (:class:`EventLog`; later ones are ignored)."""
    now = round(facts["t"] * TIC_HZ)
    me = facts["frags"]
    board = facts["board"]
    nums = {k: int(m.group(1)) for k, r in _NUM.items() if (m := r.search(state))}
    held = _HELD.search(state)
    owned = _owned(state)
    you: dict = {
        "frags": me,
        "deaths": facts["deaths"],
        "rank": 1 + sum(n > me for _, n in board),
        "players": len(board) + 1,
        "health": nums.get("hp"),
        "armor": nums.get("armor"),
    }
    if held:
        weapon = SAY.get(held.group(1), held.group(1))
        you["holding"] = {"weapon": weapon}
        if weapon != "fist":
            you["holding"]["ammo"] = int(held.group(2))
    you["weapons"] = {SAY[WEAPON_NAMES[s]]: a for s, a in sorted(owned.items())}
    you["best_loaded_weapon"] = _best_loaded(owned)
    you["frags_last_10s"] = facts["streak"]
    at = next((i for i, (_, f) in enumerate(board) if f <= me), len(board))
    scores = dict([*board[:at], ("you", me), *board[at:]])  # best first
    foes = _FOES.findall(state.split("| see", 1)[1]) if "| see" in state else []
    out = {
        "time": clock(now),
        "time_left": None if facts["left"] is None else clock(facts["left"] * TIC_HZ),
        "you": you,
        "scoreboard": scores,
        "bots_in_view": [
            {"side": _side(int(b)), "distance_m": int(d)} for b, d in foes
        ],
    }
    d = facts["last_death"]
    if d:
        out["last_death"] = {
            "killer": d["by"] or "unknown",
            "your_weapon": d["weapon"],
            "seconds_ago": _ago(now, d["tick"]),
        }
        if d["again"] >= 2:
            out["last_death"]["in_a_row"] = d["again"]
    out["killed_by"] = dict(sorted(facts["killed_by"].items(), key=lambda kv: -kv[1]))
    recent = [e for e in events if now - LOG_S * TIC_HZ < e["tick"] <= now]
    picks = [e for e in recent if e["kind"] in ("weapon", "pickup")]
    if picks:
        out["last_pickup"] = {
            **event_json(picks[-1]),
            "seconds_ago": _ago(now, picks[-1]["tick"]),
        }
        del out["last_pickup"]["type"]
    out["recent_events"] = [
        {"time": clock(e["tick"]), **event_json(e)}
        for e in _thin(recent, KEEP_FRAGS, KEEP_SUPPLIES)
    ]
    out["notes"] = STATE_NOTES
    return out


# ── The brief ──────────────────────────────────────────────────────────────────
BRIEF_ENTRIES = 15  # the last 3 s of history (the brief without facts)
LOW_HP = 35


def _count(n: int, what: str) -> str:
    return f"one {what}" if n == 1 else f"{n} {what}s"


def _times(n: int) -> str:
    return {1: "once", 2: "twice"}.get(n, f"{n} times")


def _span(s: float) -> str:
    return f"{round(s / 60)} minutes" if s >= 90 else f"{round(s)} seconds"


def _log_news(entries: list[str]) -> list[str]:
    """What just happened, from the history alone (a brief without facts)."""
    recent = "".join(entries[-BRIEF_ENTRIES:])
    news = []
    frags = len(re.findall(r"\| frag\b", recent))
    if frags:
        news.append(f"you fragged {_count(frags, 'bot') if frags > 1 else 'a bot'}")
    if "| died" in recent:
        news.append("you got killed")
    if "| respawn" in recent:
        news.append("you respawned with just a pistol")
    for k in dict.fromkeys(re.findall(r"\| got (\w+)", recent)):
        news.append({"weapon": "you picked up a new weapon"}.get(k, f"you grabbed {k}"))
    return news


def _death_text(e: dict) -> str:
    by = e["by"]
    if by is None:
        text = "you died"
    elif by == "yourself":
        text = "you killed yourself"
    elif e["again"] >= 2:
        text = f"{by} killed you again, {_times(e['again'])} in a row"
    else:
        text = f"{by} killed you"
    return f"{text}; you had the {e['weapon']}"


def _lead_text(e: dict) -> str:
    if e["kind"] == "took_lead":
        return "you took the lead"
    return (
        f"{e['by']} tied you for the lead" if e["tied"] else f"{e['by']} took the lead"
    )


def _fact_news(facts: dict) -> tuple[list[str], str | None]:
    """The headline and the rest of what just happened, most salient first;
    and the latest lead change (the match line notes it). A death drops the
    pickups and close calls before it (the guns went with it), and a gun picked
    up since the respawn drops "just a pistol"."""
    items = facts["news"]
    deaths = [e["tick"] for e in items if e["kind"] == "died"]
    if deaths:
        gone = ("weapon", "close_call", "respawn")
        items = [e for e in items if e["tick"] >= deaths[-1] or e["kind"] not in gone]
    last: dict[str, dict] = {}
    for e in items:
        last[e["kind"]] = e
    if "respawn" in last and any(
        e["kind"] == "weapon" and e["tick"] > last["respawn"]["tick"] for e in items
    ):
        del last["respawn"]
    news = []
    if "died" in last:
        news.append(_death_text(last["died"]))
    frags = [e for e in items if e["kind"] == "frag"]
    if frags:
        text = (
            "you fragged a bot" if len(frags) == 1 else f"you fragged {len(frags)} bots"
        )
        if deaths and frags[-1]["tick"] < deaths[-1]:
            text = "before that, " + text
        text += f" with the {frags[-1]['weapon']}"
        if "streak" in last:
            text += f", {last['streak']['n']} in {round(STREAK_S)} seconds"
        elif "drought_ended" in last:
            text += f", your first in {_span(last['drought_ended']['gap'])}"
        news.append(text)
    if "close_call" in last:
        news.append(
            f"you survived a close call, down to {last['close_call']['low']} health"
        )
    if "weapon" in last:
        names = dict.fromkeys(e["name"] for e in items if e["kind"] == "weapon")
        news.append("you picked up the " + " and the ".join(names))
    if "respawn" in last:
        news.append("you respawned with just a pistol")
    leads = [e for e in items if e["kind"] in ("took_lead", "lost_lead")]
    lead = _lead_text(leads[-1]) if leads else None
    if lead and not news:  # the lead change is the news
        return [lead], None
    return news, lead


def _match_line(facts: dict, lead: str | None, told: set[str]) -> str:
    me, board = facts["frags"], facts["board"]
    parts = [f"you {me}"]
    if board:
        name, n = board[0]
        if n > me:
            parts.append(f"{name} {n} leads")
        elif n == me:
            parts.append(f"{name} {n}, tied")
        else:
            parts.append(f"next {name} {n}")
    s = "Match: " + ", ".join(parts)
    if lead:
        s += f" ({lead})"
    rank = 1 + sum(n > me for _, n in board)
    deaths = _count(facts["deaths"], "death") if facts["deaths"] else "no deaths"
    s += f"; rank {rank} of {len(board) + 1}; {deaths}"
    if facts["streak"] >= 2 and "frag" not in told:
        s += f"; {facts['streak']} frags in the last {round(STREAK_S)} seconds"
    if facts["since_frag"] >= DROUGHT_S:
        s += f"; no frag for {_span(facts['since_frag'])}"
    d = facts["last_death"]
    if d and "died" not in told and d["by"] not in (None, "yourself"):
        ago = facts["t"] - d["tick"] / TIC_HZ
        if ago <= 60:
            s += f"; {d['by']} killed you {_span(ago)} ago"
    if facts["killed_by"]:
        name = max(facts["killed_by"], key=facts["killed_by"].get)
        n = facts["killed_by"][name]
        if n >= 3:
            s += f"; {name} has killed you {_count(n, 'time')}"
    if facts["left"] is not None:
        s += f"; {_span(facts['left'])} left"
    return s + "."


def brief(entries: list[str], state: str, facts: dict | None = None) -> str:
    """The moment in short plain sentences: what just happened (headline
    first), the match (from ``facts``, a :meth:`Tracker.facts` snapshot) and
    where things stand. Without facts, what just happened comes from the last
    3 s of ``entries`` and there is no match line."""
    nums = {k: int(m.group(1)) for k, r in _NUM.items() if (m := r.search(state))}
    if facts is None:
        news, lead, told = _log_news(entries), None, set()
    else:
        news, lead = _fact_news(facts)
        told = {e["kind"] for e in facts["news"]}
    if nums.get("hit", 0) > 0:
        news.append(f"you took {nums['hit']} damage in the last second")

    now = []
    hp = nums.get("hp")
    if hp is not None:
        now.append(f"health {hp}" + (" (low)" if hp < LOW_HP else ""))
    held = _HELD.search(state)
    arms = _ARMS.search(state)
    owned = {}
    if arms:
        for part in arms.group(1).split():
            slot, ammo = part.split(":")
            owned[int(slot)] = int(ammo)
    if held:
        now.append(
            f"holding the {SAY.get(held.group(1), held.group(1))} with "
            f"{held.group(2)} ammo"
        )
    guns = {s: a for s, a in owned.items() if s in LOW_AMMO}
    if guns and all(a < LOW_AMMO[s] for s, a in guns.items()):
        now.append("low on ammo for every gun")
    best = [
        WEAPON_NAMES[s] for s in sorted(guns, reverse=True) if guns[s] >= LOW_AMMO[s]
    ]
    if best and held and held.group(1) != best[0]:
        now.append(f"your best loaded gun is the {SAY[best[0]]}")
    foes = _FOES.findall(state.split("| see", 1)[-1]) if "| see" in state else []
    if foes:
        b, d = int(foes[0][0]), int(foes[0][1])
        side = "ahead" if abs(b) <= 20 else ("to the left" if b < 0 else "to the right")
        now.append(f"{_count(len(foes), 'bot')} in view, nearest {d} m {side}")
    else:
        now.append("no bot in view")
    parts = []
    if news:
        parts.append("Just now: " + "; ".join(news) + ".")
    if facts is not None:
        parts.append(_match_line(facts, lead, told))
    parts.append("Right now: " + "; ".join(now) + ".")
    return " ".join(parts)


def sound_tag(brief_text: str, rng) -> str:
    """A sound for Chatterbox-Turbo to voice before a line, chosen from what just
    happened. Left to the model, one opened every line."""
    head = brief_text.split("Right now:")[0]
    if re.search(
        r"killed you|you died|you got killed|killed yourself", head.split("Match:")[0]
    ):
        return "[sigh] " if rng.random() < 0.6 else ""
    if "you fragged" in head:
        return "[chuckle] " if rng.random() < 0.35 else ""
    if "(low)" in brief_text:
        return "[sigh] " if rng.random() < 0.3 else ""
    return ""


# ── When to speak ──────────────────────────────────────────────────────────────
# Events the player speaks soon after; a plain frag only after a longer silence
# (a strong player frags every few seconds).
MAJOR = (
    "died",
    "took_lead",
    "lost_lead",
    "streak",
    "close_call",
    "drought_ended",
    "weapon",
)
# A streak is news only as it grows past these (a strong player is on one most
# of the time).
STREAK_LEVELS = (3, 5, 8, 12, 20)
MIN_GAP_S = 3.0
FRAG_GAP_S = 8.0
IDLE_S = 12.0
HOLD_S = 1.5  # an event not yet spoken to stays a reason to speak this long


class TalkClock:
    """When the player speaks on its own. Feed :meth:`event` every tic with the
    tracker's events (dead or alive); ask :meth:`due` on a live tic; call
    :meth:`said` when a line is spoken (a reply to the watcher too). The dataset
    (:func:`match_moments`) and the live game use the same clock."""

    def __init__(self, idle_s: float = IDLE_S):
        self.idle_s = idle_s
        self.last = 0  # tick of the last line; the match start counts as one
        self.pending: list[tuple[int, str]] = []

    def event(self, events: list[dict]) -> None:
        self.pending += [
            (e["tick"], e["kind"] if e.get("n", 0) in (0, *STREAK_LEVELS) else "frag")
            for e in events
        ]

    def due(self, tick: int) -> dict | None:
        """``{"cue": "event", "events": kinds}`` or ``{"cue": "idle", ...}``, or
        None."""
        self.pending = [(t, k) for t, k in self.pending if tick - t <= HOLD_S * TIC_HZ]
        since = (tick - self.last) / TIC_HZ
        kinds = sorted({k for _, k in self.pending})
        if (since >= MIN_GAP_S and any(k in MAJOR for k in kinds)) or (
            since >= FRAG_GAP_S and "frag" in kinds
        ):
            return {"cue": "event", "events": kinds}
        if since >= self.idle_s:
            return {"cue": "idle", "events": kinds}
        return None

    def said(self, tick: int) -> None:
        self.last = tick
        self.pending = []


class EventPartner:
    """A stand-in for the person watching, for videos and tests: after some
    events it says something about them, the way a person would (who got you,
    after a death), and asks about the match in a quiet stretch. Lines are as
    speech recognition writes them. Feed :meth:`event` every tic; :meth:`poll`
    on a live tic returns what it says now, if anything."""

    LINES = {
        "died": (
            "who got you",
            "who killed you",
            "what happened there",
            "ouch who was that",
        ),
        "frag": ("nice", "nice shot", "who was that", "got him"),
        "streak": ("you're on fire", "that's a lot of kills in a row"),
        "took_lead": ("are you winning now", "wait are you in first"),
        "lost_lead": ("who's winning", "did you just lose the lead"),
        "close_call": ("that was close", "are you okay"),
        "idle": (
            "what's the score",
            "are you winning",
            "where is everybody",
            "who's in first",
        ),
    }
    RATE = {
        "died": 0.7,
        "frag": 0.2,
        "streak": 0.6,
        "took_lead": 0.5,
        "lost_lead": 0.6,
        "close_call": 0.5,
    }

    def __init__(self, rng, gap_s: float = 10.0, idle_s: float = 20.0):
        self.rng, self.gap, self.idle = rng, gap_s * TIC_HZ, idle_s * TIC_HZ
        self.last = 0  # tick it last spoke
        self.due: tuple[int, str] | None = None

    def event(self, events: list[dict]) -> None:
        for e in events:
            k = e["kind"]
            if (
                k in self.RATE
                and self.due is None
                and e["tick"] - self.last >= self.gap
                and self.rng.random() < self.RATE[k]
            ):
                delay = int(self.rng.uniform(0.6, 1.6) * TIC_HZ)
                self.due = (e["tick"] + delay, self.rng.choice(self.LINES[k]))

    def poll(self, tick: int) -> str | None:
        if self.due is None and tick - self.last >= self.idle:
            self.due = (tick, self.rng.choice(self.LINES["idle"]))
        if self.due is None or tick < self.due[0]:
            return None
        text, self.due, self.last = self.due[1], None, tick
        return text


def pick_runs(moments: list[dict], n: int, runs: int) -> list[dict]:
    """``n`` of a match's moments, in ``runs`` unbroken stretches spread over
    the match (its start, middle, end, ...): late-match situations (time
    running out, a long drought, a nemesis) are covered, and within a stretch
    every line follows the one before, as it would live."""
    if len(moments) <= n:
        return moments
    size = n // runs
    starts = [round(i * (len(moments) - size) / max(1, runs - 1)) for i in range(runs)]
    return [m for s in starts for m in moments[s : s + size]]


def match_moments(
    entries: list[str], rows: list[dict], idle_s: float = IDLE_S, limit: int = 0
) -> list[dict]:
    """The moments a live game speaks at on its own, in order (the
    :class:`TalkClock` over the match's rows; ``limit``: the first so many, 0
    for all). ``rows``: one match's collect.py rows (tick, history length,
    state, tracker facts); each moment carries its cue, the tool's output
    (``tool``, :func:`game_state`), what happened since the moment before
    (``moment``, :func:`moment_events`), the brief, the facts and the last
    log lines."""
    talk, log, out, prev, last = TalkClock(idle_s), EventLog(), [], -1, None
    for r in sorted(rows, key=lambda r: r["t"]):
        news = r["facts"]["news"]
        log.add(news)
        talk.event([e for e in news if prev < e["tick"] <= r["t"]])
        prev = r["t"]
        n = r["hist_n"]
        if n < 5 or (cue := talk.due(r["t"])) is None:
            continue
        out.append(
            {
                "t": r["t"],
                "hist_n": n,
                "state": r["state"],
                **cue,
                "tool": game_state(r["state"], r["facts"], log.events),
                "moment": moment_events(log.since(last, r["t"])),
                "facts": r["facts"],
                "brief": brief(entries[:n], r["state"], r["facts"]),
                "recent": [e.strip() for e in entries[max(0, n - 3) : n]],
            }
        )
        talk.said(r["t"])
        last = r["t"]
        if limit and len(out) >= limit:
            break
    return out


# ── Checks ─────────────────────────────────────────────────────────────────────
def check_tracker() -> None:
    """The tracker on a scripted match: the killer from the scoreboard delta, a
    lead taken and lost, a streak, a drought ended, a close call, pickups with
    their amounts; then the tool's JSON at the end, and a past moment's."""
    from types import SimpleNamespace

    board = {"Rambo": 0, "Leone": 0}
    state = {
        "frags": 0,
        "deaths": 0,
        "hp": 100,
        "armor": 0,
        "dead": False,
        "arms": {1: 0, 2: 50},
    }

    def obs(tick, events=(), **kw):
        state.update(kw)
        sb = [("AI", state["frags"]), *sorted(board.items(), key=lambda kv: -kv[1])]
        return SimpleNamespace(
            tick=tick,
            events=tuple(events),
            dead=state["dead"],
            hp=state["hp"],
            armor=state["armor"],
            weapon="pistol",
            arms=dict(state["arms"]),
            frags=state["frags"],
            deaths=state["deaths"],
            priv=SimpleNamespace(scoreboard=sb),
        )

    tr, seen = Tracker(match_s=600), []
    script = {  # tick -> what happens on it
        # The first frag, 57 s in: a drought ended, and the lead.
        2000: lambda: (["frag"], {"frags": 1}),
        2100: lambda: (["frag"], {"frags": 2}),
        2200: lambda: (["frag"], {"frags": 3}),  # three in 10 s: a streak
        2300: lambda: (["got weapon"], {"arms": {1: 0, 2: 50, 5: 10}}),
        2400: lambda: ([], {"hp": 30}),  # 70 lost at once, survived: a close call
        2500: lambda: (["got health"], {"hp": 100}),
        2550: lambda: (["got ammo"], {"arms": {1: 0, 2: 60, 5: 10}}),  # a clip
        2560: lambda: (["got armor"], {"armor": 100}),
    }
    for tick in range(0, 3300):
        events, kw = script.get(tick, lambda: ([], {}))()
        if tick in (2600, 2700, 2800):  # Rambo frags someone: 1, 2, then 3
            board["Rambo"] += 1
        if tick == 3000:  # Rambo kills the player: 4 to 3, Rambo leads
            board["Rambo"] += 1
            events, kw = ["died"], {"deaths": 1, "dead": True, "hp": 0}
        if tick == 3035:
            events, kw = ["respawn"], {"dead": False, "hp": 100, "arms": {1: 0, 2: 50}}
        if tick == 3200:
            board["Rambo"] += 1
            events, kw = ["died"], {"deaths": 2, "dead": True, "hp": 0}
        seen += tr.update(obs(tick, events, **kw))
    kinds = [(e["tick"], e["kind"]) for e in seen]
    want = [
        (2000, "frag"),
        (2000, "drought_ended"),
        (2000, "took_lead"),
        (2100, "frag"),
        (2200, "frag"),
        (2200, "streak"),
        (2300, "weapon"),
        (2435, "close_call"),
        (2500, "pickup"),
        (2550, "pickup"),
        (2560, "pickup"),
        (2800, "lost_lead"),
        (3000, "died"),
        (3035, "respawn"),
        (3200, "died"),
    ]
    assert kinds == want, f"tracker events\n got  {kinds}\n want {want}"
    assert [e["i"] for e in seen] == list(range(len(seen))), "event indices"
    died = [e for e in seen if e["kind"] == "died"]
    assert died[0]["by"] == "Rambo" and died[1]["again"] == 2, died
    assert next(e for e in seen if e["kind"] == "lost_lead")["tied"], "3-3 is a tie"
    picks = [(e["item"], e.get("amount")) for e in seen if e["kind"] == "pickup"]
    assert picks == [("health", 70), ("bullets", 10), ("armor", 100)], picks
    line = "hp 100 armor 0 | pistol 50 | arms 2:50 | see bot -40 7m, medikit +3 2m"
    b = brief([], line, tr.facts())
    assert b.startswith("Just now: Rambo killed you again, twice in a row"), b
    assert "Rambo 5 leads" in b and "Rambo has killed you" not in b, b

    log = EventLog()
    for k in range(0, len(seen), 3):  # in pieces, overlapping, as rows' news are
        log.add(seen[max(0, k - 2) : k + 3])
    assert log.events == seen, "the event log keeps each event once, in order"
    js = game_state(line, tr.facts(), log.events)
    you = js["you"]
    assert (you["frags"], you["deaths"], you["rank"], you["players"]) == (3, 2, 2, 3)
    assert (you["health"], you["armor"], you["holding"]) == (
        100,
        0,
        {"weapon": "pistol", "ammo": 50},
    ), you
    assert you["weapons"] == {"pistol": 50} and you["best_loaded_weapon"] == "pistol"
    assert list(js["scoreboard"].items()) == [("Rambo", 5), ("you", 3), ("Leone", 0)]
    assert js["bots_in_view"] == [{"side": "left", "distance_m": 7}], js
    assert js["last_death"] == {
        "killer": "Rambo",
        "your_weapon": "pistol",
        "seconds_ago": 3,
        "in_a_row": 2,
    }, js["last_death"]
    assert js["killed_by"] == {"Rambo": 2} and js["time"] == "1:34", js
    assert js["last_pickup"] == {"item": "armor", "amount": 100, "seconds_ago": 21}
    types = [e["type"] for e in js["recent_events"]]
    assert types.count("frag") == 3 and types.count("death") == 2, types
    assert js["recent_events"][0] == {
        "time": "0:57",
        "type": "frag",
        "your_weapon": "pistol",
        "first_in_s": 57,
    }, js["recent_events"][0]
    tied = {"time": "1:20", "type": "lead", "leader": "Rambo", "tied_with_you": True}
    assert tied in js["recent_events"], js["recent_events"]
    # A past moment: what happened since the line before (here, the deaths).
    past = moment_events(log.since(2900, 3299))
    assert past == [
        {"type": "death", "killer": "Rambo", "your_weapon": "pistol"},
        {"type": "death", "killer": "Rambo", "your_weapon": "pistol", "in_a_row": 2},
    ], past
    many = [
        {"tick": t, "kind": "pickup", "item": "bullets", "amount": 5, "i": t}
        for t in range(20)
    ]
    many.append(
        {
            "tick": 20,
            "kind": "died",
            "by": "Leone",
            "weapon": "BFG",
            "again": 1,
            "i": 20,
        }
    )
    short = moment_events(many)
    assert len(short) == MOMENT_EVENTS and short[-1]["type"] == "death", short
    import json

    print(
        "OK: tracker events, killer, streak, drought, close call, lead, pickups; "
        f"the tool's JSON ({len(json.dumps(js))} chars):\n  {json.dumps(js)}"
    )


def check_parity(seconds: float = 40.0, seed: int = 5) -> None:
    """The dataset path against the live path on one match, with the scripted
    player. Dataset: collect.py's worker (rows with the tracker's facts, a
    JSON round trip), then :func:`match_moments`. Live: engine.py's game worker
    plays the same actions on the wall clock, and its messages drive the talk
    clock as the engine's Game does, and its events fill the event log. Every
    state and every fact, and at every moment the brief, the tool's JSON and
    the past moment's events, must be identical."""
    import json
    import multiprocessing as mp

    import collect
    import engine

    task = {
        "behavior": "fighter",
        "ep": 0,
        "seed": seed,
        "timeout": int(seconds * TIC_HZ),
        "bots": "default",
        "n_bots": 7,
        "row_every": 1,
        "teacher": "expert",
        "dart": 0.0,
        "beta": 0.0,
        "policy": "teacher",
        "keep_rows": True,
        "video": None,
    }
    res = json.loads(json.dumps(collect._teacher_episode(task)))
    rows, entries = res["rows"], res["history"]["entries"]
    data = match_moments(entries, rows)
    plan = {r["t"]: (r["act"], r["weapon"]) for r in rows}

    ctx = mp.get_context("spawn")
    conn, child = ctx.Pipe()
    spec = {"seed": seed, "seconds": seconds, "bots": "default", "n_bots": 7}
    proc = ctx.Process(target=engine.game_worker, args=(child, spec), daemon=True)
    proc.start()
    got: dict[int, tuple] = {}
    hist, talk, live, log, last = [], TalkClock(), [], EventLog(), None
    while True:
        msg = conn.recv()
        if msg[0] == "ready":
            conn.send(("start",))
            continue
        if msg[0] == "done":
            conn.send(("bye",))
            break
        _, tick, state, entry, _events, facts, fired = msg
        if entry is not None:
            hist.append(entry)
        talk.event(fired)
        log.add(fired)
        if state is None:
            continue
        act, slot = plan.get(tick, ("wait", None))
        conn.send(("act", tick, act, None if slot is None else int(slot)))
        got[tick] = (state, json.loads(json.dumps(facts)))
        if len(hist) >= 5 and (cue := talk.due(tick)) is not None:
            tool = json.loads(json.dumps(game_state(state, facts, log.events)))
            past = moment_events(log.since(last, tick))
            live.append((tick, cue["cue"], brief(hist, state, facts), tool, past))
            talk.said(tick)
            last = tick
    proc.join(10)
    bad = [r["t"] for r in rows if got.get(r["t"]) != (r["state"], r["facts"])]
    assert not bad, f"{len(bad)} of {len(rows)} tics differ, first at tick {bad[0]}"
    want = [(m["t"], m["cue"], m["brief"], m["tool"], m["moment"]) for m in data]
    for a, b in zip(live, want):
        assert a == b, f"moment at tick {b[0]} differs\n live {a}\n data {b}"
    assert len(live) == len(want), f"{len(live)} live moments, {len(want)} in the data"
    print(
        f"OK: {len(rows)} tics, same state and facts; {len(data)} moments, same "
        "cue, brief, tool JSON and past events on both paths, e.g.\n  "
        + (json.dumps(data[-1]["tool"]) if data else "")
    )


def pool(args) -> None:
    """Lines the base model says at random moments of recorded play (the talk
    augmentation pool, train_alora.py --talk-lines)."""
    import json
    import random

    from history import History
    from policy import VLLMPolicy

    streams = {}
    for line in open(args.data / "fighter_history.jsonl"):
        h = json.loads(line)
        streams[h["ep"]] = h["entries"]
    rows = [json.loads(x) for x in open(args.data / "fighter.jsonl")]
    rows = [r for r in rows if r["hist_n"] >= 10]
    random.Random(0).shuffle(rows)
    rows = rows[: args.n]
    pol = VLLMPolicy(args.model, max_num_seqs=128, warmup=2)
    games, briefs = [], []
    for r in rows:
        entries = streams[r["ep"]][: r["hist_n"]]
        games.append((History.replay(entries, pol.tok).ids, r["state"]))
        briefs.append(brief(entries, r["state"], r.get("facts")))
    lines = []
    for i in range(0, len(games), 256):
        lines += pol.talk(games[i : i + 256], briefs[i : i + 256], temperature=0.9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r, b, t in zip(rows, briefs, lines):
            if t.strip():
                f.write(
                    json.dumps({"ep": r["ep"], "t": r["t"], "brief": b, "line": t})
                    + "\n"
                )
    print(f"{sum(1 for t in lines if t.strip())} lines -> {args.out}")


def main() -> None:
    import argparse
    import json
    import random
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Spoken-line tools for the Doom demo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("pool", help="Lines from the base model (augmentation pool)")
    pl.add_argument("--model", required=True, help="Any composed checkpoint")
    pl.add_argument("--data", type=Path, required=True, help="A collect.py dir")
    pl.add_argument("--n", type=int, default=3000)
    pl.add_argument("--out", type=Path, required=True)
    mo = sub.add_parser("moments", help="Speaking moments of recorded matches")
    mo.add_argument(
        "--data", type=Path, nargs="+", required=True, help="collect.py dirs"
    )
    mo.add_argument("--style", default="fighter")
    mo.add_argument("--matches", type=int, default=400)
    mo.add_argument(
        "--per-match", type=int, default=0, help="Moments per match (0: every one)"
    )
    mo.add_argument(
        "--runs", type=int, default=3, help="Unbroken stretches per match (pick_runs)"
    )
    mo.add_argument(
        "--idle-s", type=float, default=IDLE_S, help="Silence before a remark"
    )
    mo.add_argument("--seed", type=int, default=0)
    mo.add_argument(
        "--split",
        help="NAME=N,...: the shuffled matches cut into files <out>_<NAME>.jsonl "
        "(e.g. write=150,pool=30,heldout=20), split by match",
    )
    mo.add_argument("--out", type=Path, required=True)
    ck = sub.add_parser("check", help="The tracker, and dataset vs live briefs")
    ck.add_argument("--seconds", type=float, default=40.0)
    ck.add_argument("--no-parity", action="store_true", help="Tracker only (no game)")
    args = ap.parse_args()

    if args.cmd == "check":
        check_tracker()
        if not args.no_parity:
            check_parity(args.seconds)
        return
    if args.cmd == "pool":
        pool(args)
        return
    matches = []
    for d in args.data:
        rows_by: dict = {}
        for line in open(d / f"{args.style}.jsonl"):
            r = json.loads(line)
            rows_by.setdefault(r["ep"], []).append(r)
        for line in open(d / f"{args.style}_history.jsonl"):
            h = json.loads(line)
            if h["ep"] in rows_by:
                matches.append((str(d), h["ep"], h["entries"], rows_by[h["ep"]]))
    random.Random(args.seed).shuffle(matches)
    matches = matches[: args.matches]
    parts = [("", len(matches))]
    if args.split:
        parts = [(k, int(v)) for k, v in (p.split("=") for p in args.split.split(","))]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    at = 0
    for name, size in parts:
        path = args.out.with_name(f"{args.out.stem}_{name}.jsonl") if name else args.out
        n, cues = 0, {}
        with open(path, "w") as f:
            for d, ep, entries, rows in matches[at : at + size]:
                ms = match_moments(entries, rows, idle_s=args.idle_s)
                if args.per_match:
                    ms = pick_runs(ms, args.per_match, args.runs)
                for m in ms:
                    cues[m["cue"]] = cues.get(m["cue"], 0) + 1
                row = {"data": d, "style": args.style, "ep": ep, "moments": ms}
                f.write(json.dumps(row) + "\n")
                n += len(ms)
        print(f"{len(matches[at : at + size])} matches, {n} moments ({cues}) -> {path}")
        at += size


if __name__ == "__main__":
    main()
