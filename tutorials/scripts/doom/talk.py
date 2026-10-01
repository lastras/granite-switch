# SPDX-License-Identifier: Apache-2.0
"""The player's own voice: what it is told before it speaks, and when it speaks.

A talk request reads the same prompt every game adapter reads, up to the end of
the state (so it reuses their prefilled KV), then closes the user turn with a
*brief* of the moment in plain words, and the narrator writes the line. The
history is terse (``t12.4 hp 64 face 135 | bot +10 8m | did cl | frag``) and
holds only 10 s, so the brief carries what the model could not work out:

    Just now: Rambo killed you again, twice in a row; you had the BFG. Match: you
    12, Rambo 13 leads (Rambo took the lead); rank 2 of 8; 5 deaths; 6 minutes
    left. Right now: health 100; holding the pistol with 50 ammo; no bot in view.

The facts come from a :class:`Tracker`, fed every tic with the observation (its
scoreboard included): frags and the weapon held, deaths and the killer (the bot
whose frag count rose on the tic the player died), streaks and droughts, close
calls, lead changes, weapon pickups, time left. The brief is rendered from them
in code, in microseconds. The same tracker runs in collect.py's workers, in the
engine's game worker and in record_video, so the narrator is trained and served
on briefs rendered the same way from the same facts; ``python talk.py check``
replays a match through both paths and compares them.

When to speak (:class:`TalkClock`): soon after a salient event (a death, a lead
change, a streak, a close call, the first frag in a while, a new weapon: at
least ``MIN_GAP_S`` after the last line; a plain frag, ``FRAG_GAP_S``), after
``IDLE_S`` of silence, and whenever the person watching speaks.

The line it speaks goes back into the history (:func:`policy.spoken_entry`), so
every adapter's next prompt contains it.
"""

from __future__ import annotations

import re
from collections import deque

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


class Tracker:
    """The match's memory, for the brief: what the 10 s history cannot hold.

    Call :meth:`update` with every observation, dead or alive, in order (the
    one from ``reset`` too); it returns the events that tic produced. ``facts()``
    is a small JSON-able snapshot: :func:`brief` renders it, collect.py stores
    it with every row and the engine's game worker sends it with every state.
    ``match_s``: the match length, for the time left.
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
        self._boards: deque[dict[str, int]] = deque(maxlen=KILLER_TICS + 1)
        self._death: dict | None = None  # waiting for the killer's frag count
        self._hp: deque[tuple[int, int]] = deque()
        self._close: dict | None = None  # a big drop, waiting to be survived
        self._last_close = -(10**9)
        self._leading = False
        self._arms: set[int] = {1, 2}
        self._last_frag = 0

    def update(self, obs) -> list[dict]:
        tick = self.tick = obs.tick
        bots = {name: f for name, f in obs.priv.scoreboard[1:]}
        ev = obs.events
        fired: list[dict] = []

        def fire(kind: str, **kw) -> None:
            fired.append({"tick": tick, "kind": kind, **kw})

        for _ in range(ev.count("frag")):
            gap = (tick - self._last_frag) / TIC_HZ
            self._last_frag = tick
            self.frag_ticks.append(tick)
            fire("frag", weapon=SAY[obs.weapon])
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
        if not obs.dead:
            self.weapon = obs.weapon
            self._arms = set(obs.arms)
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


# ── The brief ──────────────────────────────────────────────────────────────────
BRIEF_ENTRIES = 15  # the last 3 s of history (the brief without facts)
# Ammo below which a weapon is nearly empty (the BFG spends 40 cells a shot).
LOW_AMMO = {2: 20, 3: 4, 4: 20, 5: 3, 6: 20, 7: 40}
LOW_HP = 35

_ARMS = re.compile(r"\barms ((?:\d:\d+ ?)+)")
_HELD = re.compile(r"\| (\w+) (\d+) \| arms")
_NUM = {k: re.compile(rf"\b{k} (-?\d+)") for k in ("hp", "armor", "hit")}
_FOES = re.compile(r"\bbot ([+-]\d+) (\d+)m")


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
    state, tracker facts); each moment carries its cue, brief, facts and last
    log lines."""
    clock, out, prev = TalkClock(idle_s), [], -1
    for r in sorted(rows, key=lambda r: r["t"]):
        clock.event([e for e in r["facts"]["news"] if prev < e["tick"] <= r["t"]])
        prev = r["t"]
        n = r["hist_n"]
        if n < 5 or (cue := clock.due(r["t"])) is None:
            continue
        out.append(
            {
                "t": r["t"],
                "hist_n": n,
                "state": r["state"],
                **cue,
                "facts": r["facts"],
                "brief": brief(entries[:n], r["state"], r["facts"]),
                "recent": [e.strip() for e in entries[max(0, n - 3) : n]],
            }
        )
        clock.said(r["t"])
        if limit and len(out) >= limit:
            break
    return out


# ── Checks ─────────────────────────────────────────────────────────────────────
def check_tracker() -> None:
    """The tracker on a scripted match: the killer from the scoreboard delta, a
    lead taken and lost, a streak, a drought ended, a close call, a pickup."""
    from types import SimpleNamespace

    board = {"Rambo": 0, "Leone": 0}
    state = {"frags": 0, "deaths": 0, "hp": 100, "dead": False, "arms": {1: 0, 2: 50}}

    def obs(tick, events=(), **kw):
        state.update(kw)
        sb = [("AI", state["frags"]), *sorted(board.items(), key=lambda kv: -kv[1])]
        return SimpleNamespace(
            tick=tick,
            events=tuple(events),
            dead=state["dead"],
            hp=state["hp"],
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
        2500: lambda: ([], {"hp": 100}),
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
        (2800, "lost_lead"),
        (3000, "died"),
        (3035, "respawn"),
        (3200, "died"),
    ]
    assert kinds == want, f"tracker events\n got  {kinds}\n want {want}"
    died = [e for e in seen if e["kind"] == "died"]
    assert died[0]["by"] == "Rambo" and died[1]["again"] == 2, died
    assert next(e for e in seen if e["kind"] == "lost_lead")["tied"], "3-3 is a tie"
    b = brief([], "hp 100 armor 0 | pistol 50 | arms 2:50 | see nothing", tr.facts())
    assert b.startswith("Just now: Rambo killed you again, twice in a row"), b
    assert "Rambo 5 leads" in b and "Rambo has killed you" not in b, b
    print("OK: tracker events, killer, streak, drought, close call, lead\n  " + b)


def check_parity(seconds: float = 40.0, seed: int = 5) -> None:
    """The dataset path against the live path on one match, with the scripted
    player. Dataset: collect.py's worker (rows with the tracker's facts, a
    JSON round trip), then :func:`match_moments`. Live: engine.py's game worker
    plays the same actions on the wall clock, and its messages drive the talk
    clock as the engine's Game does. Every state, every fact and every brief at
    every moment must be identical."""
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
    hist, clock, live = [], TalkClock(), []
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
        clock.event(fired)
        if state is None:
            continue
        act, slot = plan.get(tick, ("wait", None))
        conn.send(("act", tick, act, None if slot is None else int(slot)))
        got[tick] = (state, json.loads(json.dumps(facts)))
        if len(hist) >= 5 and (cue := clock.due(tick)) is not None:
            live.append((tick, cue["cue"], brief(hist, state, facts)))
            clock.said(tick)
    proc.join(10)
    bad = [r["t"] for r in rows if got.get(r["t"]) != (r["state"], r["facts"])]
    assert not bad, f"{len(bad)} of {len(rows)} tics differ, first at tick {bad[0]}"
    want = [(m["t"], m["cue"], m["brief"]) for m in data]
    assert live == want, f"moments differ\n live {live[:3]}\n data {want[:3]}"
    print(
        f"OK: {len(rows)} tics, same state and facts; {len(data)} moments, same "
        f"cue and brief on both paths, e.g.\n  " + (data[-1]["brief"] if data else "")
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
    mo.add_argument("--per-match", type=int, default=60)
    mo.add_argument(
        "--runs", type=int, default=3, help="Unbroken stretches per match (pick_runs)"
    )
    mo.add_argument(
        "--idle-s", type=float, default=IDLE_S, help="Silence before a remark"
    )
    mo.add_argument("--seed", type=int, default=0)
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    cues: dict[str, int] = {}
    with open(args.out, "w") as f:
        for d, ep, entries, rows in matches[: args.matches]:
            ms = match_moments(entries, rows, idle_s=args.idle_s)
            ms = pick_runs(ms, args.per_match, args.runs)
            for m in ms:
                cues[m["cue"]] = cues.get(m["cue"], 0) + 1
            f.write(
                json.dumps({"data": d, "style": args.style, "ep": ep, "moments": ms})
                + "\n"
            )
            n += len(ms)
    print(
        f"{min(len(matches), args.matches)} matches, {n} moments ({cues}) -> {args.out}"
    )


if __name__ == "__main__":
    main()
