# SPDX-License-Identifier: Apache-2.0
"""Every rule a narrator line must pass, in one place: the checks in code and the judge's questions.

A line is checked against the turn it is said at (:class:`Turn`): the game state
he read, his earlier lines, what his partner just said, the question or order it
answers, the topic of a remark, the example the writer was shown. Each check in
code is a small function ``name(line, turn) -> (ok, reason)``; its docstring is
the rule (the writer is shown it), its reason the repair (the writer is told it
when the line fails). :func:`line_checks` says which apply to a turn.

* **Form:** :func:`length`, :func:`sentences`, :func:`plain`, :func:`calm`,
  :func:`mild`, :func:`original`, :func:`stock`, :func:`first_person`,
  :func:`no_unknown`, :func:`not_list`, :func:`status_opening`, :func:`numbers`.
* **Variety,** against his own recent lines: :func:`fresh_opening`,
  :func:`no_motif`, :func:`not_repeat`, :func:`not_example`,
  :func:`even_still`, :func:`still_again`, :func:`weapon_again`.
* **Truth,** against the game state: :func:`claims` (:func:`probes.claims`),
  :func:`answer` (:func:`probes.verify`), :func:`says_why`, :func:`stale_bot`.
* **The judge** (gpt-oss-120b), for what takes reading: :data:`JUDGE`'s
  questions, chosen by :func:`judge_questions`, asked in one call through
  :func:`judge_line`, a Mellea generative stub whose answer is typed (a list
  of :class:`Verdict`).

:func:`requirements` makes them Mellea requirements for one line
(``narrator_data.py`` writes with them); :func:`failures` runs the code checks
alone (``rft.py``, ``eval_probes.py``, ``test_partner.py``).

What a remark is about: :func:`news`, the most salient thing that just happened,
and :func:`story_options`, the match's storylines, each as a text code writes
from the game state; :func:`remark_flags` measures remarks
(``test_partner.py remarks``).

Round 10 (``narrator_v10.py``) keeps the truth checks and asks less of the
form: :func:`line_checks_v10` (lengths by the turn's act, no comedy-shaped
rules) and :data:`JUDGE_V10` (:func:`judge_questions_v10`: his voice as a
conversation, the act, originality; no "funny").

Mellea is imported only where the judge or the requirements are built
(:func:`judge`, :func:`ask`, :func:`judged`, :func:`requirements`), so the code
checks run anywhere::

    python checks.py check                                # each check passing and failing
    python checks.py check --rows data/r9/narrator/*.jsonl  # each check's failure rate on rows
"""

import argparse
import functools
import inspect
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import probes
from conversation import past_output, tool_text
from pydantic import BaseModel

# ── Word lists ─────────────────────────────────────────────────────────────────
STRONG = ("fuck", "shit", "cunt", "motherf", "bitch", "nigg", "fag", "retard", "whore")
# Distinctive phrases of well-known film dialogue.
FILM = (
    *("royale", "say what again", "ezekiel", "do you speak it", "big mac"),
    *("zed's dead", "tasty burger", "cornerstone of any nutritious"),
    *("path of the righteous", "get medieval", "pretty please with sugar"),
    *("that's a bingo", "winston wolf", "i'll be back", "hasta la vista"),
    *("feel lucky, punk", "make my day", "say hello to my little"),
)
# Phrases earlier datasets repeated hundreds of times.
STOCK = (
    *("let's see", "about to learn", "feels like", "quiet now", "humming"),
    *("meters", "job well done", "just the way i like"),
)
# A status report's first word ("Health's low...", "BFG's warm...").
STATUS = {
    *("health", "health's", "armor", "armor's", "ammo", "bfg", "bfg's", "pistol"),
    *("pistol's", "chaingun", "chaingun's", "shotgun", "shotgun's", "plasma"),
    *("rocket", "rockets", "another", "time", "situation"),
}
# Words too common to make a motif ("still" is not one of them: it was his habit).
MOTIF_STOP = set(
    "that this with have from they them what when your just like there their about "
    "been were will would could should into than then over some only also here where "
    "which while more most much very really right back down even ever it's i'm that's "
    "don't can't won't you're he's let's".split()
)
NUMBER = re.compile(
    r"\d|\b(zero|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|dozen)\b",
    re.I,
)
SPEAKER = re.compile(r"\s*(he|granite|partner|game|player)\s*:", re.I)
WEAPON = re.compile(
    r"\b(bfg|pistol|chaingun|chain gun|shotgun|plasma|rocket|launcher|fist|chainsaw)",
    re.I,
)
EVEN_STILL = re.compile(
    r"\beven (?:with|after|though|if|now|at|down|without|so)\b[^.?!]*\bstill\b", re.I
)
RECENT = 8  # his last lines a new one must not open as, nor repeat a word of 3 times
REPEAT_WINDOW = 5  # ... nor nearly copy
WEAPON_WINDOW = 4  # ... a weapon only if fewer than 2 of these named one
STALE_S = 30  # a remark names a bot only if the game told of it this recently
# The question types a weapon is the point of.
WEAPON_TYPES = ("weapon", "weapons", "best_gun", "ammo", "ammo_of", "killer_weapon")


# Typographic punctuation, as plain ASCII (the dashes stay: calm reads them).
_PLAIN = str.maketrans(
    {
        **dict.fromkeys("\u2019\u2018", "'"),
        **dict.fromkeys("\u201c\u201d", '"'),
        **dict.fromkeys("\u2011\u2010", "-"),
        "\u00a0": " ",
        "\u2026": "...",
    }
)


def clean(text: str) -> str:
    """The spoken line in a model's output: no reasoning (up to ``</think>``),
    no quote marks, its first line, plain ASCII punctuation."""
    text = re.sub(r"(?s)<think>.*?</think>", "", str(text)).translate(_PLAIN)
    text = text.split("</think>")[-1]
    lines = [x.strip().strip('"').strip() for x in text.splitlines()]
    return next((x for x in lines if x), "")


def words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def jaccard(a: str, b: str) -> float:
    x, y = set(words(a)), set(words(b))
    return len(x & y) / len(x | y) if x and y else 0.0


def opening(line: str) -> str:
    """A line's first three words."""
    return " ".join(words(line)[:3])


def stems(line: str) -> set[str]:
    """The stems (6 letters) of a line's content words."""
    return {w[:6] for w in words(line) if len(w) >= 4 and w not in MOTIF_STOP}


def secs(clock: str) -> int:
    m, s = clock.split(":")
    return 60 * int(m) + int(s)


# ── The turn a line is said at ─────────────────────────────────────────────────
@dataclass
class Turn:
    """What a line is checked against.

    ``state``: the game state he read (``talk.game_state``); ``prev``: his
    earlier lines, oldest first; ``past``: the events of the conversation's
    past exchanges; ``player``: what his partner just said (None: a remark);
    ``utype``: what kind of thing that was (``probe``, ``order``, ``request``,
    ...); ``probe``: the question about the game state it asks; ``order``: the
    partner's order it gives (the state's ``order``); ``topic``: a remark's
    topic (:func:`news`, :func:`story_options`); ``example``: the example line
    the writer was shown; ``conversation``: the conversation as text, for the
    judge (:func:`conversation_text`). Round 10's: ``act``, what the turn was
    to mainly do (``narrator_v10_prompts.ACTS``; for an answer or an order,
    the act after it, if any); ``thread``, an earlier line of his to pick up;
    ``alone``, an answer or an order's reply that is to stand alone;
    ``shown``, example lines the teacher was shown, which a line must not
    copy; ``rib``, the one of the cast the turn ribs, if any; ``about``, the
    one of the cast his partner asked about, if any."""

    state: dict
    prev: list[str] = field(default_factory=list)
    past: list[list[dict]] = field(default_factory=list)
    player: str | None = None
    utype: str | None = None
    probe: dict | None = None
    order: dict | None = None
    topic: dict | None = None
    example: str | None = None
    conversation: str = ""
    act: str | None = None
    thread: str | None = None
    alone: bool = False
    shown: list[str] = field(default_factory=list)
    rib: str | None = None
    about: str | None = None

    @property
    def kind(self) -> str:
        return "reply" if self.player else "remark"


def conversation_text(exchanges) -> str:
    """Past exchanges (``conversation.Exchange``) as the writer and the judge
    read them, oldest first: the partner's words, the game's output at his line
    (its time and what had happened since the line before), his line."""
    out = []
    for ex in exchanges:
        if ex.player:
            out.append(f'Partner: "{ex.player}"')
        out.append(f"Game: {tool_text(past_output(ex))}")
        out.append(f'Granite: "{ex.line}"')
    return "\n".join(out) or "(nothing yet)"


def numeric(probe: dict | None) -> bool:
    """Whether a question asks for a number (its answer may hold numbers)."""
    if probe is None:
        return False
    if probe["type"] == "challenge":
        return probe["field"] not in ("killer", "leader")
    return probe["type"] in probes.NUMERIC


def refusing(t: Turn) -> bool:
    return t.order is not None and t.order["status"] == "refused"


def weapon_point(t: Turn) -> bool:
    """Whether a weapon is the point of the line: the partner named one, the
    question asks about one, the order names one, or the topic is one."""
    said = " ".join((t.player or "", (t.order or {}).get("told", "")))
    return (
        bool(probes.weapons_said(said))
        or (t.probe is not None and t.probe["type"] in WEAPON_TYPES)
        or bool((t.topic or {}).get("weapon"))
    )


def known_bots(t: Turn) -> set[str]:
    """The bots the game just told him about: in the topic, the events of the
    last STALE_S, the last death or frag if that recent, the storylines, his
    partner's words and the order."""
    st = t.state
    now = secs(st["time"])
    out = set((t.topic or {}).get("bots", ()))
    for e in st.get("recent_events", ()):
        if now - secs(e["time"]) <= STALE_S:
            out |= {e.get(k) for k in ("killer", "victim", "leader")}
    for k, who in (("last_death", "killer"), ("last_frag", "victim")):
        if (st.get(k) or {}).get("seconds_ago", STALE_S + 1) <= STALE_S:
            out.add(st[k][who])
    out |= probes.bot_names(json.dumps(st.get("storylines") or {}))
    said = " ".join((t.player or "", (t.order or {}).get("told", "")))
    out |= probes.bot_names(said, ignore_case=True)
    return out - {None, "you", "unknown"}


# ── Form ───────────────────────────────────────────────────────────────────────
def bounds(t: Turn) -> tuple[int, int]:
    if t.probe is not None or t.order is not None:
        return 1, 16
    return (4, 16) if t.kind == "reply" else (4, 14)


def length(line: str, t: Turn):
    """4 to 14 words for a remark, 4 to 16 for a reply, 1 to 16 for an answer to a question or an order."""
    lo, hi = bounds(t)
    n = len(line.split())
    return lo <= n <= hi, f"The line has {n} words; it must have {lo} to {hi}."


def sentences(line: str, t: Turn):
    """At most three short sentences."""
    n = len([s for s in re.split(r"[.?!]+", line) if s.strip()])
    return n <= 3, f"The line has {n} sentences; say it in at most three."


def plain(line: str, t: Turn):
    """Just the spoken words: no brackets, asterisks, quote marks or speaker label; and Doom has no reloading."""
    ok = not re.search(r"[\[\]*#\"]|reload", line, re.I) and not SPEAKER.match(line)
    return ok, (
        "No brackets, asterisks, quote marks or speaker label, just the line; "
        "and Doom has no reloading."
    )


def calm(line: str, t: Turn):
    """Calm punctuation: no exclamation marks, and no dash when one of his last two lines had one."""
    dashes = "\u2014\u2013"
    dash = any(d in line for d in dashes) and any(
        d in p for p in t.prev[-2:] for d in dashes
    )
    return "!" not in line and not dash, (
        "Calm punctuation: no exclamation marks, and no dash (a recent line had "
        "one); use a period or a comma."
    )


def mild(line: str, t: Turn):
    """Mild language: no strong swearing or slurs."""
    bad = [s for s in STRONG if s in line.lower()]
    return not bad, "Keep the language mild: no strong swearing or slurs."


def original(line: str, t: Turn):
    """Original: never a famous film line."""
    hit = [s for s in FILM if s in line.lower()]
    return not hit, "That echoes a famous film line; write an original one."


def stock(line: str, t: Turn):
    """None of the stock phrases earlier datasets wore out (let's see, feels like, ...)."""
    hit = [s for s in STOCK if s in line.lower()]
    said = hit[0] if hit else ""
    return not hit, f'"{said}" is a stock phrase in these lines; put it another way.'


def first_person(line: str, t: Turn):
    """He speaks for himself, as I and me: the game state's "you" is him."""
    bad = re.search(
        r"\b(?:you have|you've|you are|you're|you got|you just|you're currently|"
        r"you are currently|your)\s+(?:\w+\s+){0,2}?(?:frags?|kills?|deaths?|died|"
        r"health|armou?r|ammo|holding|rank|in first|in second|place|weapons?|guns?|"
        r"fragged|picked|streak|leading|ahead)\b"
        r"|\byou\s+(?:just\s+)?(?:fragged|died|picked up|grabbed|respawned|took the "
        r"lead|lost the lead|switched)\b",
        line.lower(),
    )
    said = bad.group(0) if bad else ""
    return not bad, (
        f'He speaks for himself, as I and me ("{said}" reads the game state\'s '
        '"you" as his partner).'
    )


def no_unknown(line: str, t: Turn):
    """Never the game state's word "unknown", and no disclaimer about what the game does not say unless asked."""
    low = line.lower()
    asked = t.probe is not None and t.probe["type"] == "killer_weapon"
    bad = re.search(r"\bunknown\b", low) or (
        not asked
        and re.search(
            r"\b(?:don'?t|do not) know (?:his|her|their|the|its|what) weapons?\b"
            r"|\bweapons? (?:not known|never (?:known|reported|said))\b",
            low,
        )
    )
    return not bad, (
        'Say it as he would: never "unknown" (the game state\'s word), and no '
        "disclaimer about what the game does not say unless asked."
    )


def not_list(line: str, t: Turn):
    """One thought in a natural sentence, not a status report: at most two facts from the game state."""
    parts = [p for p in re.split(r"[,;\u2014\u2013]|\s-\s|[.?!]\s+", line) if p.strip()]
    sizes = sorted(len(p.split()) for p in parts)
    if t.probe is not None and t.probe["type"] in ("weapons", "top_n", "nth"):
        # The answer is a list of names (guns, players), not of their rows.
        if t.probe["type"] != "weapons" and (
            len(probes.number_spans(line, ones=False)) >= 2 or probes.weapons_said(line)
        ):
            return False, (
                "Name the players only, as he would say them: no frags or weapon "
                "for each, that reads the scoreboard aloud."
            )
        return True, ""
    # Short pieces stating facts read as a status report ("Health low, shotgun,
    # second place"), not short pieces alone ("Turning. If nobody's there, we
    # talk."); a refusal's reason comes in fragments ("Eleven health, under fire").
    facts = probes.facts_said(line)
    short = len(parts) >= 3 and sizes[len(sizes) // 2] <= 3
    listy = short and facts >= 2 and not refusing(t)
    many = facts >= (4 if t.probe is not None or refusing(t) else 3)
    return not (listy or many), (
        "That reads as a status report. Say one thought, in a natural sentence, "
        "with at most two facts from the game state."
    )


def status_opening(line: str, t: Turn):
    """Never a status report's opening (Health's..., Shotgun's..., Another...): open with his angle."""
    w = words(line)
    first = w[0] if w else ""
    echo = set(words(t.player or ""))
    ok = bool(w) and (first not in STATUS or first.split("'")[0] in echo)
    return (
        ok,
        f'Do not open with "{first}": no status report; open with his angle on it.',
    )


def numbers(line: str, t: Turn):
    """No numbers, in digits or words: he says one only when asked for it."""
    m = NUMBER.search(line)
    said = m.group(0) if m else ""
    return m is None, f'No numbers in digits or words ("{said}"): he never says them.'


# ── Variety: against his own recent lines ──────────────────────────────────────
def fresh_opening(line: str, t: Turn):
    """Never opens the way one of his last 8 lines opened (their first three words)."""
    o = opening(line)
    hit = next((p for p in t.prev[-RECENT:] if opening(p) == o), None)
    return (
        hit is None,
        f'It opens the way an earlier line did ("{hit}"); open another way.',
    )


def _motif(line: str, t: Turn, times: int):
    skip = {n.lower()[:6] for n in probes.BOT_NAMES} | stems(t.player or "")
    recent = Counter(s for p in t.prev[-RECENT:] for s in stems(p))
    rep = sorted(s for s in stems(line) - skip if recent[s] >= times)
    return not rep, (
        f"It repeats what he keeps saying ({', '.join(rep)}, in {times} or more of "
        "his last 8 lines). Say something new."
    )


def no_motif(line: str, t: Turn):
    """No word he keeps saying: none that is in 3 or more of his last 8 lines (names and his partner's words aside)."""
    return _motif(line, t, 3)


def not_repeat(line: str, t: Turn):
    """Not a near-copy of one of his last 5 lines (at most half its words shared)."""
    hit = next((p for p in t.prev[-REPEAT_WINDOW:] if jaccard(line, p) > 0.5), None)
    return hit is None, f'Too close to an earlier line ("{hit}"); say something new.'


def not_example(line: str, t: Turn):
    """Not a copy of the example line: his own words."""
    ok = jaccard(line, t.example or "") <= 0.4
    return ok, "Too close to the example; write your own line."


def even_still(line: str, t: Turn):
    """Never the "Even with X, I'm still Y" template."""
    return not EVEN_STILL.search(line), (
        'That is the "Even with X, I\'m still Y" template; say it another way.'
    )


def still_again(line: str, t: Turn):
    """No "still" when one of his last 8 lines had it: the word he wore out."""
    had = any(re.search(r"\bstill\b", p, re.I) for p in t.prev[-RECENT:])
    return not (had and re.search(r"\bstill\b", line, re.I)), (
        'He said "still" in one of his last lines; say it without "still".'
    )


def weapon_again(line: str, t: Turn):
    """No weapon named when 2 of his last 4 lines named one, unless a weapon is the point."""
    named = sum(bool(WEAPON.search(p)) for p in t.prev[-WEAPON_WINDOW:])
    m = WEAPON.search(line)
    return not (m and named >= 2), (
        f'Leave the weapon out ("{m.group(0) if m else ""}"): his recent lines keep '
        "naming his guns. Talk about the match instead."
    )


# ── Truth: against the game state ──────────────────────────────────────────────
def claims(line: str, t: Turn):
    """Every fact in it is in the game state: names, numbers, weapons, who killed whom, the standings, the order."""
    ok, why = probes.claims(line, t.state, t.past, t.player or "")
    return ok, f"{why} Every fact you say must be in the game state."


def answer(line: str, t: Turn):
    """Answers his partner's question first, exactly as the game state says."""
    verdict, why = probes.verify(t.probe, line, t.state)
    right = probes.answer_text(t.probe, t.state)
    return verdict == probes.CORRECT, f"{why} The game state says: {right}"


def says_why(line: str, t: Turn):
    """Says why he can't do what his partner told him."""
    return probes.says_why(line, t.order), (
        f"Say why you can't: {t.order['why']}, in your own words."
    )


def stale_bot(line: str, t: Turn):
    """Names only bots the game just told him about: the news, the last 30 seconds, the storylines, his partner's words, the order."""
    named = probes.bot_names(line) & set(probes.board(t.state))
    stale = sorted(named - known_bots(t))
    return not stale, (
        f"Nothing has happened with {', '.join(stale)} lately; talk about what the "
        "game just told you instead."
    )


def line_checks(t: Turn) -> list:
    """The code checks that apply to a line said at ``t``."""
    out = [length, sentences, plain, calm, mild, original, stock, first_person]
    out += [no_unknown, not_list, fresh_opening, no_motif, not_repeat, even_still]
    out.append(still_again)
    out.append(claims)
    if t.probe is None and t.order is None:
        out.append(status_opening)
    if not numeric(t.probe) and not refusing(t):
        out.append(numbers)
    if t.example:
        out.append(not_example)
    if not weapon_point(t):
        out.append(weapon_again)
    if t.probe is not None:
        out.append(answer)
    if t.order is not None and t.order["status"] == "cant":
        out.append(says_why)
    if t.kind == "remark":
        out.append(stale_bot)
    return out


def rule(check) -> str:
    """A check's rule: its docstring."""
    return inspect.getdoc(check)


def failures(line: str, t: Turn) -> list[tuple[str, str]]:
    """The code checks ``line`` fails at ``t``, as (name, reason)."""
    return _failures(line, t, line_checks(t))


def failures_v10(line: str, t: Turn) -> list[tuple[str, str]]:
    """The round-10 code checks ``line`` fails at ``t``, as (name, reason)."""
    return _failures(line, t, line_checks_v10(t))


def _failures(line: str, t: Turn, fs) -> list[tuple[str, str]]:
    out = []
    for f in fs:
        ok, why = f(line, t)
        if not ok:
            out.append((f.__name__, why))
    return out


# ── The judge ──────────────────────────────────────────────────────────────────
# Each question: what the writer is told it asks, and what the judge is asked.
JUDGE = {
    "true": (
        "nothing false about the game",
        "Does the line avoid saying anything false about the game, against the game "
        "state now or earlier as the conversation tells it (kills, deaths, who killed "
        "whom, the bots' kills of each other, the score and the race, damage, pickups, "
        "weapons, health, ammo, the storylines)? The state names the bots he fragged "
        "when the game knows them (a frag's victim, last_frag); bots never take his "
        "weapons (a death drops them). A line with no game facts at all is YES; "
        "jokes, opinions, plans and flavor details are fine.",
    ),
    "grounded": (
        "nothing invented about the bots",
        "Does the line avoid stating as fact what the game does not tell him about "
        "the bots: what a bot is doing, thinking or feeling right now, where a named "
        "bot is, which bot he sees (the bots in view are never identified)? An "
        "obvious joke, an opinion, or a threat about what he will do is fine (YES).",
    ),
    "consistent": (
        "consistent with what he said before",
        "Is it consistent with what he said earlier (no contradicting his own "
        "earlier lines; a callback only to something actually said)? Disagreeing "
        "with his partner is fine.",
    ),
    "coherent": (
        "the next thing he would say in this conversation",
        "Does it fit the conversation so far, as the next thing he would say to his "
        "partner? When the partner refers back to something earlier, does it answer "
        "from the conversation?",
    ),
    "voice": (
        "his voice: a calm, dry, deadpan professional",
        "Does it sound like a calm, dry, deadpan professional from a 1990s crime "
        "movie talking to his partner (not a soldier, a sports announcer or a "
        "cheerleader)?",
    ),
    "funny": (
        "genuinely funny or sharp",
        "Is it genuinely funny or sharp (a real turn of phrase or angle), not a "
        "generic quip?",
    ),
    "specific": (
        "about this moment of this match",
        "Is it about this moment: what just happened, or something particular to "
        "this match (the race with a named bot, the bots' war, a streak, a drought, "
        "who killed him)? A line that would fit almost any moment of any match is NO.",
    ),
    "about": (
        "about what he was given to talk about",
        "He was given this to talk about: {topic}. Is the line about that? It need "
        "not say every fact in it, nor any number.",
    ),
    "answers": (
        "a reply to what his partner said",
        "Does it respond to what the partner actually said, playing off their words "
        "or their point?",
    ),
    "fends_off": (
        "the request acknowledged and fended off",
        "Does he acknowledge the partner's request but turn it down or put it off? "
        "Agreeing to do it is NO.",
    ),
    "takes_order": (
        "the order taken as the game says he did",
        "His partner just told him to {told}. The game says: {status}. Does his reply "
        "fit that, in character (going along with a deadpan grumble or a jab, "
        "refusing because it would get him killed, or saying why he can't), without "
        "saying he does something he does not?",
    ),
}
ORDER_STATUS = {
    "doing": "he is doing it",
    "done": "it is done",
    "refused": "he refused ({why})",
    "cant": "he cannot ({why})",
    "cancelled": "it was called off ({why})",
}
# Replies that need not be about the moment ("specific" is not asked).
SOCIAL = ("greeting", "identity", "smalltalk", "request", "followup")


def judge_questions(t: Turn) -> dict[str, str]:
    """The judge's questions for a line said at ``t``, by name."""
    names = ["true", "grounded", "consistent", "coherent", "voice"]
    if t.probe is None:
        names.append("funny")
    if t.kind == "remark" or t.utype not in (*SOCIAL, "probe"):
        names.append("specific")
    if t.kind == "remark" and t.topic:
        names.append("about")
    if t.kind == "reply":
        names.append("answers")
    if t.utype == "request":
        names.append("fends_off")
    if t.order is not None:
        names.append("takes_order")
    status = ""
    if t.order is not None:
        st = ORDER_STATUS.get(t.order["status"], "{why}")
        status = st.format(why=t.order.get("why"))
    fill = {
        "topic": (t.topic or {}).get("text", ""),
        "told": (t.order or {}).get("told", ""),
        "status": status,
    }
    return {k: JUDGE[k][1].format(**fill) for k in names}


class Verdict(BaseModel):
    question: str
    yes: bool
    reason: str


def judge_line(
    line: str, said_to: str, game_state: str, conversation: str, questions: dict
) -> list[Verdict]:
    """Judge one line that Granite says out loud in a Doom deathmatch against bots.

    Granite is a calm, dry professional out of a 1990s crime movie; his partner
    sits next to him, watching the screen. ``game_state`` is the output of his
    get_game_state call, which he had just read ("you" in it is him; times are
    match time). ``conversation`` is what had been said and had happened
    before, oldest first. ``said_to`` is what his partner had just said to him,
    as speech recognition wrote it (empty: he speaks on his own).

    Answer every question in ``questions``, in order: ``question`` is its name
    there, ``yes`` whether the answer is yes, ``reason`` a short reason (for a
    no, what is wrong with the line)."""


def judge_line_v10(
    line: str, said_to: str, game_state: str, conversation: str, questions: dict
) -> list[Verdict]:
    """Judge one line that Granite says out loud in a Doom deathmatch against bots.

    Granite is a hired gun doing a job, the deathmatch; his partner of many
    years sits next to him, watching the screen, and the two of them talk about
    the match and about anything else. ``game_state`` is the output of his
    get_game_state call, which he had just read ("you" in it is him; times are
    match time). ``conversation`` is what had been said and had happened
    before, oldest first. ``said_to`` is what his partner had just said to him,
    as speech recognition wrote it (empty: he speaks on his own).

    Answer every question in ``questions``, in order: ``question`` is its name
    there, ``yes`` whether the answer is yes, ``reason`` a short reason (for a
    no, what is wrong with the line)."""


@functools.cache
def judge(v10: bool = False):
    """:func:`judge_line` (round 10: :func:`judge_line_v10`) as a Mellea
    generative stub (Mellea imported here, on first use)."""
    from mellea import generative

    return generative(judge_line_v10 if v10 else judge_line)


def ask(session, t: Turn, line: str) -> dict[str, Verdict]:
    """The judge's answers on ``line`` said at ``t``, by question, asked on the
    judge's Mellea ``session`` (a question it left out has no answer)."""
    return ask_questions(session, t, line, judge_questions(t))


def ask_questions(
    session, t: Turn, line: str, questions: dict, v10: bool = False
) -> dict[str, Verdict]:
    """The judge's answers on ``line`` said at ``t`` to ``questions`` (name ->
    question), by name (a question it left out has no answer); ``v10``: the
    round-10 judge (:func:`judge_line_v10`)."""
    from mellea.backends import ModelOption

    try:
        out = judge(v10)(
            session,
            line=line,
            said_to=t.player or "",
            game_state=tool_text(t.state),
            conversation=t.conversation,
            questions=questions,
            model_options={
                ModelOption.THINKING: "low",
                ModelOption.TEMPERATURE: 0.0,
                ModelOption.MAX_NEW_TOKENS: 4000,  # its reasoning included
                ModelOption.SEED: 0,
            },
        )
    except ValueError:  # its answer was not the JSON asked for: no answers
        return {}
    got = {v.question: v for v in out}
    if set(got) != set(questions) and len(out) == len(questions):
        got = dict(zip(questions, out))  # renamed, but in order
    return {k: got[k] for k in questions if k in got}


def judged(session, t: Turn, verdicts: dict | None = None):
    """The judge's questions on a line as one Mellea requirement: it passes
    when every answer is yes; the reasons for the noes are the repair.
    ``verdicts``: each judged line's answers (question -> yes)."""
    from mellea.stdlib.requirements import Requirement, simple_validate

    questions = judge_questions(t)

    def fn(line: str):
        got = ask(session, t, line)
        if verdicts is not None:
            verdicts[line] = {k: v.yes for k, v in got.items()}
        no = [
            f"{k}: {got[k].reason}" if k in got else f"{k}: not answered"
            for k in questions
            if not (k in got and got[k].yes)
        ]
        return not no, "The judge said no. " + " ".join(no)

    desc = "A judge must find it: " + "; ".join(JUDGE[k][0] for k in questions) + "."
    return Requirement(desc, validation_fn=simple_validate(lambda x: fn(clean(x))))


def requirements(t: Turn, judge_session=None, verdicts: dict | None = None) -> list:
    """The checks of a line said at ``t``, as Mellea requirements: each code
    check (its rule the description the writer is shown, its reason the
    repair) and, with a judge's session, the judge's questions (:func:`judged`)."""
    from mellea.stdlib.requirements import Requirement, simple_validate

    reqs = [
        Requirement(
            rule(f), validation_fn=simple_validate(lambda x, f=f: f(clean(x), t))
        )
        for f in line_checks(t)
    ]
    if judge_session is not None:
        reqs.append(judged(judge_session, t, verdicts))
    return reqs


# ── Round 10: the checks of a conversational turn ──────────────────────────────
# Words and sentences by the turn's act (narrator_v10_prompts.ACTS): a brief
# turn is a word or a few; telling the match runs longer.
SHORT_ACTS = ("react", "needle", "menace", "ask", "echo", "rib")
TELL_ACTS = ("tell", "plan", "narrate", "setup")
LONG_ACTS = ("riff", "story", "smalltalk", "argue", "explain", "resume")


def bounds_v10(t: Turn) -> tuple[int, int, int]:
    """(fewest words, most words, most sentences) for a round-10 turn."""
    act = t.act
    if t.probe is not None or t.order is not None:
        return (1, 40, 6) if act in LONG_ACTS else (1, 25, 4)
    if t.thread and act not in ("brief", *SHORT_ACTS):
        return 4, 30, 4
    if act == "brief":
        return 1, 6, 2
    if act in TELL_ACTS:
        return 4, 30, 4
    if act in LONG_ACTS:
        return 8, 40, 6
    return 3, 25, 3


def length_v10(line: str, t: Turn):
    """As many words as the turn wants: a brief turn 1 to 6; a reaction, a rib, a question or a jab 3 to 25; telling the match or planning 4 to 30; an answer or an order's reply 1 to 25."""
    lo, hi, _ = bounds_v10(t)
    n = len(line.split())
    return lo <= n <= hi, f"The line has {n} words; this turn wants {lo} to {hi}."


def sentences_v10(line: str, t: Turn):
    """No more sentences than the turn's act wants (2 for a brief turn, up to 4 for telling the match)."""
    _, _, most = bounds_v10(t)
    n = len([x for x in re.split(r"[.?!]+", line) if x.strip()])
    return n <= most, f"The line has {n} sentences; this turn wants at most {most}."


def no_motif_v10(line: str, t: Turn):
    """No word he keeps saying: none that is in 4 or more of his last 8 lines (names and his partner's words aside)."""
    return _motif(line, t, 4)


def _cast_words() -> dict[str, set[str]]:
    """Each of the cast's own thing, as word stems (conversation.CAST)."""
    from conversation import CAST

    common = stems(
        "still never every before after thinks claims swears really makes match"
    )
    return {n: stems(needle) - common for n, _, needle in CAST}


def fresh_rib(line: str, t: Turn):
    """No rib he just made: not the same person's thing again within his last 5 lines."""
    for name, thing in _cast_words().items():
        if name == t.about:  # his partner asked about them: their thing is the answer
            continue
        if not re.search(rf"\b{name}\b", line) or not stems(line) & thing:
            continue
        hit = next(
            (
                p
                for p in t.prev[-REPEAT_WINDOW:]
                if re.search(rf"\b{name}\b", p) and stems(p) & thing
            ),
            None,
        )
        if hit:
            return False, (
                f'He just ribbed {name} about that ("{hit}"); let it rest, or find '
                "a new angle on them."
            )
    return True, ""


# What a bot is doing right now: the game never says (only what happened, the
# score and the storylines). Its moves in the arena, said of a bot: chasing him,
# firing, on his tail; not the cast's own things (bragging, baking), nor his.
_MOVES = (
    "chasing|firing|shooting|stalking|hunting|following|aiming|swinging|spraying|"
    "hiding|camping|lurking|closing|gaining|trying|scrambling|gunning|catching|"
    "sneaking|flanking|circling|lagging|inching|waiting|running|charging|"
    "pushing|creeping|tailing|targeting|looking|searching|missing|dodging"
)
_NOW = re.compile(
    rf"\b(?:{_MOVES})\b"
    r"|\b(?:on my (?:tail|heels|six)|breathing down my neck|after me|behind me|"
    r"right on my doorstep)\b",
    re.I,
)


def bot_now(line: str, t: Turn):
    """Nothing about what a bot is doing right now in the arena (chasing him, firing, on his tail): the game never says."""
    for clause in re.split(r"[.,;!?—-]|\band\b|\bbut\b|\bwhile\b", line):
        if not probes.bot_names(clause) or re.search(r"\bI\b|\bI'm\b", clause):
            continue
        if re.match(rf"\s*(?:{_MOVES})\b", clause, re.I):  # his own move, "I" elided
            continue
        m = _NOW.search(clause)
        if m:
            return False, (
                f'"{clause.strip()}": what a bot is doing right now is not in the game '
                "state; say what it did (the kills, the score, the storylines)."
            )
    return True, ""


def answer_alone(line: str, t: Turn):
    """An answer that is to stand alone stands alone: at most 15 words, and none of the cast named but those the answer is about."""
    want = probes.bot_names(probes.answer_text(t.probe, t.state)) if t.probe else set()
    want |= {t.about} if t.about else set()  # asked about one of the cast: that one
    extra = sorted(probes.bot_names(line) - want - {"you"})
    n = len(line.split())
    ok = n <= 15 and not extra
    return ok, (
        f"This turn is the answer alone: {n} words"
        + (f", and it brings in {', '.join(extra)}" if extra else "")
        + ". Say the answer and stop."
    )


def _bits(text: str) -> set[str]:
    """A text's words of three letters or more, stemmed (possessives and plurals off)."""
    out = set()
    for w in words(text):
        w = re.sub(r"'s?$", "", w)
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        if len(w) >= 3:
            out.add(w[:6])
    return out


_COMMON = None


def _common() -> set[str]:
    global _COMMON
    if _COMMON is None:
        _COMMON = (
            _bits(
                "the and you your but just now that this his her him with for not out "
                "off got get was are has had one all again tonight guess looks frag "
                "fragged killed down dead match lead leading time back"
            )
            | _bits(" ".join(probes.WEAPONS))
            | {n.lower()[:6] for n in probes.BOT_NAMES}
        )
    return _COMMON


def habit(line: str, t: Turn):
    """No verbal habit: no word of his in 4 or more of his last 12 lines (names, game words, plain words and a bit's own words aside)."""
    plain = _common() | _bits(
        "who what when where why how i'm i've i'll keep put get bot bots think "
        "right here there them they about like"
    )
    mine = _bits(line) - plain - _bits(t.thread or "")
    recent = Counter(w for p in t.prev[-12:] for w in _bits(p) - plain)
    worn = sorted(w for w in mine if recent[w] >= 4)
    return not worn, (
        f"A habit of his ({', '.join(worn)}, in 4 or more of his last 12 lines): "
        "say it another way."
    )


def right_person(line: str, t: Turn):
    """Each person's own thing: a line that names one of the cast does not give them another's (Ted's pies are not Hank's)."""
    from conversation import CAST

    stop = _common() | _bits(
        "after never every before still think claim swear really she they them can "
        "make ups bring lost beat caught hog owe bet"
    )
    things = {n: _bits(needle) - stop for n, _, needle in CAST}
    named = {n for n in things if re.search(rf"\b{n}\b", line)}
    if not named:
        return True, ""
    mine = _bits(line)
    for other, thing in things.items():
        if other in named:
            continue
        hit = mine & thing - set().union(*(things[n] for n in named))
        if hit and not re.search(rf"\b{other}\b", line):
            return False, (
                f"That ({', '.join(sorted(hit))}) is {other}'s thing, not "
                f"{' or '.join(sorted(named))}'s."
            )
    return True, ""


def not_rehash(line: str, t: Turn):
    """A line that builds on an earlier one is not a copy of it (at most half its words shared)."""
    ok = jaccard(line, t.thread or "") <= 0.5
    return ok, f'Too close to the line it builds on ("{t.thread}"): take it further.'


def builds_on(line: str, t: Turn):
    """A callback builds on the earlier line's own bit: it shares a word of it (not a name, a gun or a word of the game)."""

    shared = (_bits(line) & _bits(t.thread or "")) - _common() - _bits("still finally")
    return bool(shared), (
        f'It does not build on the earlier line ("{t.thread}"): take its own joke '
        "further (its words, its stakes, its score), not just the same person."
    )


def not_scene(line: str, t: Turn):
    """Not a copy of a line from the scenes the teacher was shown: his own words."""
    hit = next((x for x in t.shown if jaccard(line, x) > 0.4), None)
    return hit is None, f'Too close to an example ("{hit}"); write your own line.'


def line_checks_v10(t: Turn) -> list:
    """The code checks that apply to a round-10 line said at ``t``: every
    truth check, the form checks a spoken line needs, and the variety checks
    against his own recent lines; none that shapes a joke."""
    out = [length_v10, sentences_v10, plain, mild, original, first_person, no_unknown]
    out += [not_list, fresh_opening, no_motif_v10, habit, not_repeat, fresh_rib, claims]
    out.append(right_person)
    out += [stale_bot, bot_now]
    if not numeric(t.probe) and not refusing(t):
        out.append(numbers)
    if t.shown:
        out.append(not_scene)
    if not weapon_point(t):
        out.append(weapon_again)
    if t.probe is not None:
        out.append(answer)
    if t.order is not None and t.order["status"] == "cant":
        out.append(says_why)
    if t.alone:
        out.append(answer_alone)
    if t.probe is not None or t.order is not None:  # an answer says what it must
        out = [f for f in out if f is not habit]
    if t.thread and t.act in ("rib", "bit"):  # building on its bit, not banned for it
        out = [f for f in out if f not in (fresh_rib, no_motif_v10)]
        out += [builds_on, not_rehash]
    return out


# The judge's questions on a round-10 line (the truth questions as round 9's,
# true told that bots respawn).
JUDGE_V10 = {
    "true": (
        "nothing false about the game",
        "Does the line avoid saying anything false about the game, against the game "
        "state now or earlier as the conversation tells it (kills, deaths, who killed "
        "whom, the bots' kills of each other, the score and the race, damage, pickups, "
        "weapons, health, ammo, the storylines)? Bots respawn right after they die, so "
        "a bot he fragged is soon back on the scoreboard: saying it went down is true. "
        "The state names the bots he fragged when the game knows them (a frag's "
        "victim, last_frag); bots never take his weapons. Talk that is not about the "
        "game (his life, his opinions, stories) is not checked here: YES. The bots are "
        "named after people he knows ({cast}): who they are to him and their thing are "
        "his life, not the game.",
    ),
    "grounded": (
        "nothing invented about the bots",
        "Does the line avoid stating what the game does not tell him about the bots: "
        "what a bot is doing right now (chasing him, firing at him, on his tail, "
        "stalking, trying to, swinging a gun, keeping at something), what it thinks or "
        "wants, where a named bot is, which bot he sees (the bots in view are never "
        "identified)? The game tells him only what happened (who killed whom, with "
        "what), the score and the race, and the storylines (who is on a tear, who "
        "keeps killing whom). A rib that says a person does now in the arena what the "
        "game never said is a NO, however funny. Jokes about their life outside the "
        "game (their fish stories, their pies, their pool game), an opinion, or a threat about "
        "what he will do are fine, and so is who each of them is to him and their thing "
        "({cast}).",
    ),
    "consistent": (
        "consistent with what he said before",
        "Does it avoid contradicting what he said earlier in his conversation with his "
        "partner, and does any callback (as I said, again, still) refer to something "
        "he actually said or the game actually told him there? Something new is fine; "
        "disagreeing with his partner is fine.",
    ),
    "coherent": (
        "the next thing he would say in this conversation",
        "Does it fit his conversation with his partner so far, as the next thing he "
        "would say there? When his partner refers back to something earlier, does it "
        "answer from that conversation? He often speaks on his own, without being "
        "asked, about the match or about the people in it: that fits.",
    ),
    "voice": (
        "his voice: a man talking with his partner, plainspoken, in concrete terms",
        "Does it sound like a hired gun talking with his long-time partner, as a "
        "conversation and not a commentator's play-by-play: plainspoken; where he talks "
        "about a subject, in concrete particulars rather than generalities; with any "
        "humor coming from who he is and the situation, not from a quip tacked onto a "
        "report{standing}? A short plain reply can be YES.",
    ),
    "original": (
        "his own, not from a film",
        "Is it his own: no line quoted or paraphrased from a film, no film "
        "character's name, no famous bit from a film (in particular none of these: "
        "{avoid})? The names of the people in the match are his own.",
    ),
    "sense": (
        "plain sense to a listener",
        "Does the line make plain sense to someone listening while they watch: no "
        "garbled or mixed-up image, no comparison that does not hold, nothing "
        "impossible said as if it were so (a fish that is a cat), unless it is an "
        "obvious figure of speech? A rib on someone's thing must still be a sentence "
        "a stranger could follow.",
    ),
    "cast": (
        "true to who the people in the match are",
        "The bots are people he knows: {cast}. Whatever he says about who any of them "
        "is, or about their life outside the match, fits that person: their thing is "
        "never given to another of them. A line need not rib anyone, and what they did "
        "in the match is not checked here. (If he says nothing about who any of them "
        "is or about their life: YES.)",
    ),
    "act": (
        "doing what this turn was for",
        "This turn was meant mainly to do this: he {gloss}. Does the line mainly do "
        "that?",
    ),
    "thread": (
        "building on what he said earlier",
        'He was to build on what he said earlier: "{thread}". Does the line build on '
        "that line's own joke or point (a callback to it, its stakes raised, a score "
        "kept, a payoff), so that a listener hears the connection? Only naming the "
        "same person again, or the same joke said again, is NO.",
    ),
    "about": (
        "about what just happened",
        "This just happened in the game: {topic}. Is the line about that? It need not "
        "say every fact in it, nor any number.",
    ),
    "specific": JUDGE["specific"],
    "answer": (
        "the answer the game state gives, first",
        'His partner asked about the game: "{asked}". The game state answers it: '
        "{answer}. Does the line give that answer first, in his own words (what it "
        "names or counts, not every detail of the state)? A few words after it are "
        "fine.",
    ),
    "answers": (
        "a reply to what his partner said",
        "Does it respond to what the partner actually said (their words, their point "
        "or their question), even if it then goes somewhere else?",
    ),
    "fends_off": (
        "the request acknowledged, not promised",
        "His partner asked for something no order can do. Does he take the request "
        "in, without promising or claiming to do it?",
    ),
    "takes_order": (
        "the order taken as the game says he did",
        "His partner just told him to {told}. The game says: {status}. Does his reply "
        "fit that (going along, refusing because it would get him killed, or saying "
        "why he can't), without saying he does something he does not?",
    ),
}


def judge_questions_v10(t: Turn) -> dict[str, str]:
    """The judge's questions for a round-10 line said at ``t``, by name."""
    import narrator_v10_prompts as P10

    names = ["true", "grounded", "consistent", "coherent", "sense", "voice", "original"]
    names.append("cast")
    if t.act:
        names.append("act")
    if t.thread:
        names.append("thread")
    if t.act in ("react", "tell") and t.topic and t.probe is None and t.order is None:
        names += ["about", "specific"]
    if t.kind == "reply" and t.probe is None:
        names.append("answers")
    if t.probe is not None:  # a question about the game: its own, in place of answers
        names.append("answer")
    if t.utype == "request":
        names.append("fends_off")
    if t.order is not None:
        names.append("takes_order")
    status = ""
    if t.order is not None:
        st = ORDER_STATUS.get(t.order["status"], "{why}")
        status = st.format(why=t.order.get("why"))
    gloss = P10.ACTS[t.act].gloss if t.act else ""
    if t.rib:
        gloss += f" (here {t.rib}: {cast_of(t.rib)})"
    # How the match stands is the point of telling it, or of an answer.
    told = t.act in STANDING_ACTS or t.probe is not None or t.order is not None
    fill = {
        "topic": (t.topic or {}).get("text", ""),
        "told": (t.order or {}).get("told", ""),
        "status": status,
        "avoid": P10.AVOID,
        "gloss": gloss,
        "thread": t.thread or "",
        "cast": cast_text(),
        "standing": "" if told else STANDING_CLAUSE,
        "asked": t.player or "",
        "answer": probes.answer_text(t.probe, t.state).rstrip(".") if t.probe else "",
    }
    return {k: JUDGE_V10[k][1].format(**fill) for k in names}


# The voice question's clause on the match's standing, and the acts it is not
# asked of (their point is how the match stands).
STANDING_CLAUSE = (
    "; and with no word on how the match stands (the lead, his health, the score) "
    "tacked on when it is not the point and nobody asked"
)
STANDING_ACTS = ("narrate", "tell", "plan")


def cast_of(name: str) -> str:
    """Who ``name`` of conversation.CAST is to him, and their thing (he, not
    you)."""
    from conversation import CAST

    who, needle = next((w, n) for n_, w, n in CAST if n_ == name)
    return f"{third(who)}; {third(needle)}"


def third(x: str) -> str:
    """A line of conversation.CAST about him, not to him (his sister, beat him)."""
    return re.sub(r"\byou\b", "him", re.sub(r"\byour\b", "his", x))


def cast_text() -> str:
    """conversation.CAST as the judge reads it (he, not you)."""
    from conversation import CAST

    return "; ".join(f"{n}, {third(who)}: {third(needle)}" for n, who, needle in CAST)


# The judge's questions on a person a bot might be named after (narrator_v10.py
# cast).
JUDGE_CAST = {
    "clear": "Does a stranger get who this is and what he needles them about from "
    "this one line, at once?",
    "banter": "Is what he needles them about good for friendly ribbing, all night, "
    "during a game (a habit, a sore spot, an old grudge), funny rather than cruel?",
    "ordinary": "Is it an ordinary person he knows (family, a friend, old crew), not a "
    "famous person or a film character (none of these: {avoid})?",
    "mild": "Is it mild: no strong swearing, slurs or cruelty?",
}


def judge_listed(item: str, questions: dict) -> list[Verdict]:
    """Judge one person a bot is named after, in a Doom deathmatch that Granite,
    a hired gun, plays against bots named after people he knows: ``item`` is
    their name, who they are to him and what he needles them about.

    Answer every question in ``questions``, in order: ``question`` is its name
    there, ``yes`` whether the answer is yes, ``reason`` a short reason."""


def ask_cast(session, item: str) -> dict[str, Verdict]:
    """The judge's answers (JUDGE_CAST) on a person a bot might be named after."""
    import narrator_v10_prompts as P10
    from mellea import generative
    from mellea.backends import ModelOption

    questions = {k: v.format(avoid=P10.AVOID) for k, v in JUDGE_CAST.items()}
    try:
        out = generative(judge_listed)(
            session,
            item=item,
            questions=questions,
            model_options={
                ModelOption.THINKING: "low",
                ModelOption.TEMPERATURE: 0.0,
                ModelOption.MAX_NEW_TOKENS: 3000,
                ModelOption.SEED: 0,
            },
        )
    except ValueError:
        return {}
    got = {v.question: v for v in out}
    if set(got) != set(questions) and len(out) == len(questions):
        got = dict(zip(questions, out))
    return {k: got[k] for k in questions if k in got}


# ── What a remark is about ─────────────────────────────────────────────────────
NEWS_S = 4.0  # talk.NEWS_S: how long an event is news
# Kinds of news, most salient first, by the state's event types.
NEWS_KINDS = {
    "death": ("death",),
    "order": ("order_end", "order_hurts", "order"),
    "frag": ("streak", "frag"),
    "close_call": ("close_call",),
    "lead": ("lead",),
    "kill": ("kill",),
    "pickup": ("pickup",),
}
SALIENCE = list(NEWS_KINDS)
KIND = {t: k for k, ts in NEWS_KINDS.items() for t in ts}


def fresh(state: dict) -> tuple[str | None, list[dict]]:
    """The most salient kind of news of the last NEWS_S, and its events."""
    now = secs(state["time"])
    evs = [
        e
        for e in state.get("recent_events", ())
        if now - secs(e["time"]) <= NEWS_S and e["type"] in KIND
    ]
    if not evs:
        return None, []
    kind = min((KIND[e["type"]] for e in evs), key=SALIENCE.index)
    return kind, [e for e in evs if KIND[e["type"]] == kind]


def _stories(state: dict) -> dict:
    return state.get("storylines") or {}


def ties(bots, state: dict) -> list[str]:
    """What the storylines say of the bots in the news."""
    s = _stories(state)
    war, race, gr = s.get("bots_war") or {}, s.get("race") or {}, s.get("grudges") or {}
    out = []
    for b in dict.fromkeys(bots):
        if (gr.get("nemesis") or {}).get("bot") == b:
            out.append(f"{b} is also the bot who has killed you most")
        if (gr.get("favorite_victim") or {}).get("bot") == b:
            out.append(f"{b} is also the bot you have fragged most")
        for x in war.get("on_a_tear", ()):
            if x["bot"] == b:
                out.append(
                    f"{b} is on a tear ({x['kills_last_60s']} kills in a minute)"
                )
        for x in war.get("feuds", ()):
            if b in (x["killer"], x["victim"]):
                out.append(
                    f"{x['killer']} keeps killing {x['victim']} ({x['times']} times "
                    "this match)"
                )
        if race.get("leader") == b:
            out.append(f"{b} leads the match")
    return list(dict.fromkeys(out))


def _news_text(kind: str, evs: list[dict], state: dict) -> tuple[str, list[str]]:
    """What just happened, said by code from the events: the text, the bots."""
    e = evs[-1]
    if kind == "death":
        who = e["killer"]
        if who == "yourself":
            return "you just killed yourself", []
        if who == "unknown":
            return "you just died; the game does not say who killed you", []
        text = f"{who} just killed you"
        if e.get("killer_weapon"):
            text += f" with the {e['killer_weapon']}"
        if e.get("in_a_row"):
            text += f", {e['in_a_row']} times in a row now"
        return text, [who]
    if kind == "order":
        if e["type"] == "order_hurts":
            return f"your partner's order ({e['told']}) is costing you health", []
        if e["type"] == "order":
            return f"your partner just told you to {e['told']} ({e['status']})", []
        text = f"your partner's order ({e['told']}) just ended: {e['status']}"
        if e.get("hit_wall"):
            text += ", you ran into the wall"
        if e.get("got"):
            text += f", it got you {e['got']}"
        return text, probes.bot_names(e.get("got") or "")
    if kind == "frag":
        frags = [x for x in evs if x["type"] == "frag"]
        streak = [x for x in evs if x["type"] == "streak"]
        f = frags[-1] if frags else e
        who = f.get("victim", "unknown")
        if who == "unknown":
            text = "you just fragged a bot; the game does not say which"
        else:
            text = f"you just fragged {who}"
        if f.get("your_weapon"):
            text += f" with the {f['your_weapon']}"
        if streak:
            text += f", {streak[-1]['frags_in_10s']} frags in 10 seconds"
        if f.get("first_in_s"):
            text += f", your first frag in {f['first_in_s']} seconds"
        return text, [] if who == "unknown" else [who]
    if kind == "close_call":
        return (
            f"you just survived a close call: down to {e['health_left']} health",
            [],
        )
    if kind == "lead":
        who = e["leader"]
        if who == "you":
            return "you just took the lead", []
        if e.get("tied_with_you"):
            return f"{who} just tied you for the lead", [who]
        return f"{who} just took the lead from you", [who]
    if kind == "kill":
        text = f"{e['killer']} just killed {e['victim']}"
        if e.get("weapon"):
            text += f" with the {e['weapon']}"
        return text + " (the bots' war)", [e["killer"], e["victim"]]
    return f"you just picked up {e['item']}", []


def news(state: dict) -> dict | None:
    """The most salient thing that just happened (the last NEWS_S of the
    state's events: a death, an order, a frag or a streak, a close call, the
    lead changing, a bot killing a bot, a pickup), as a remark's topic:
    ``{"kind", "text", "bots", "weapon"}``, the text written by code from the
    state, with what the storylines say of the bots in it."""
    kind, evs = fresh(state)
    if kind is None:
        return None
    text, bots = _news_text(kind, evs, state)
    extra = ties(bots, state)
    if extra:
        text += "; " + "; ".join(extra)
    weapon = kind == "pickup" and evs[-1]["item"] in probes.WEAPONS
    return {"kind": kind, "text": text, "bots": list(bots), "weapon": weapon}


def story_options(state: dict) -> list[dict]:
    """The stories a quiet moment's remark can tell, from the storylines (and
    the state): ``war`` (who is on a tear, who keeps killing whom, the bots'
    kills of the last 30 s), ``race`` (who is out front, who is chasing),
    ``grudge`` (his nemesis, his favorite victim), ``play`` (his style, the
    order he is on, a drought, his best streak); each ``{"kind", "text",
    "bots"}``."""
    s = _stories(state)
    war, race, gr, play = (
        s.get("bots_war") or {},
        s.get("race"),
        s.get("grudges") or {},
        s.get("your_play") or {},
    )
    now = secs(state["time"])
    out = []
    parts, bots = [], []
    for x in war.get("on_a_tear", ()):
        parts.append(
            f"{x['bot']} is on a tear: {x['kills_last_60s']} kills in a minute"
        )
        bots.append(x["bot"])
    for x in war.get("feuds", ()):
        parts.append(
            f"{x['killer']} keeps killing {x['victim']}: {x['times']} times this match"
        )
        bots += [x["killer"], x["victim"]]
    for e in state.get("recent_events", ()):
        if e["type"] == "kill" and now - secs(e["time"]) <= STALE_S:
            parts.append(
                f"{e['killer']} killed {e['victim']} {now - secs(e['time'])} seconds ago"
            )
            bots += [e["killer"], e["victim"]]
    if parts:
        out.append(
            {"kind": "war", "text": "the bots' war: " + "; ".join(parts), "bots": bots}
        )
    if race:
        a, b, gap = race["leader"], race["chaser"], race["gap"]
        if gap == 0:
            text = f"{'you' if a == 'you' else a} and {b} are tied at the top"
        elif a == "you":
            text = f"you lead the match; {b} is second, {gap} behind you"
        elif b == "you":
            text = f"{a} leads the match; you are second, {gap} behind"
        else:
            rank = state["you"]["rank"]
            text = f"{a} leads the match, {b} is second, {gap} behind; you are in place {rank}"
        if race.get("lead_changes"):
            n = race["lead_changes"]
            times = {1: "once", 2: "twice"}.get(n, f"{n} times")
            text += f"; you have taken or lost the lead {times}"
        if state.get("time_left"):
            text += f"; {state['time_left']} left"
        out.append({"kind": "race", "text": "the race: " + text, "bots": [a, b]})
    parts, bots = [], []
    if gr.get("nemesis"):
        x = gr["nemesis"]
        parts.append(
            f"{x['bot']} has killed you {x['killed_you']} times, more than any bot"
        )
        bots.append(x["bot"])
    if gr.get("favorite_victim"):
        x = gr["favorite_victim"]
        parts.append(
            f"you have fragged {x['bot']} {x['fragged']} times, more than any bot"
        )
        bots.append(x["bot"])
    if parts:
        out.append(
            {
                "kind": "grudge",
                "text": "your grudges: " + "; ".join(parts),
                "bots": bots,
            }
        )
    parts = []
    if state.get("playing"):
        parts.append(f"you are playing as a {state['playing']['style']}")
    o = state.get("order")
    if o and o["status"] == "doing":
        parts.append(f"you are doing what your partner told you: {o['told']}")
    if play.get("no_frag_for_s"):
        parts.append(f"no frag for {play['no_frag_for_s']} seconds")
    if play.get("best_streak"):
        parts.append(
            f"your best streak this match: {play['best_streak']} frags in 10 seconds"
        )
    if parts:
        out.append(
            {"kind": "play", "text": "your play: " + "; ".join(parts), "bots": []}
        )
    for x in out:
        x["bots"] = [b for b in dict.fromkeys(x["bots"]) if b != "you"]
    return out


# ── Measuring remarks (test_partner.py remarks) ────────────────────────────────
WAR_S = 10.0  # a bot-on-bot kill this recent is there to use
# Besides the names in it, what a line may say to be about a kind of news.
NEWS_WORDS = {
    "death": r"\b(kill|killed|took me|got me|dead|died|down|dropped me|respawn)",
    "streak": r"\b(streak|in a row|two|three|four|five|six|double|triple|another|more)\b",
    "frag": r"\b(frag|fragged|dropped|got one|got him|one more|another|kill)",
    "close_call": r"\b(close|barely|scratch|hanging|lucky|alive|breath)",
    "lead": r"\b(lead|leading|first|top|ahead|behind|tied|front)",
}


def _names(text: str, names) -> set[str]:
    return {n for n in names if re.search(rf"\b{re.escape(n)}\b", text, re.I)}


def _about(e: dict, line: str) -> bool:
    """Whether ``line`` mentions event ``e``: a bot in it, its item or order,
    or the words its kind takes."""
    bots = {e.get(k) for k in ("killer", "victim", "leader")} - {None, "you", "unknown"}
    if _names(line, bots):
        return True
    ws = re.findall(r"[a-z]{3,}", f"{e.get('item', '')} {e.get('told', '')}".lower())
    if any(re.search(rf"\b{w}", line, re.I) for w in ws if w not in ("the", "and")):
        return True
    pat = NEWS_WORDS.get(e["type"])
    return bool(pat and re.search(pat, line, re.I))


def remark_flags(line: str, state: dict, prev: list[str]) -> dict:
    """One remark measured against the state he was given and his earlier
    lines (``prev``): what the news was, whether he spoke to it, whether he used
    the storylines and the bots' war, his habits, his claims."""
    now = secs(state["time"])
    events = [(now - secs(e["time"]), e) for e in state.get("recent_events", ())]
    bots = [e["name"] for e in state.get("scoreboard", ()) if e["name"] != "you"]
    kind, evs = fresh(state)
    seen = {}  # when each bot last figured in what the game told him
    for age, e in events:
        for k in ("killer", "victim", "leader"):
            if e.get(k) in bots:
                seen[e[k]] = min(seen.get(e[k], age), age)
    d = state.get("last_death") or {}
    if d.get("killer") in bots:
        seen[d["killer"]] = min(seen.get(d["killer"], 1e9), d["seconds_ago"])
    named = _names(line, bots)
    war = [e for age, e in events if e["type"] == "kill" and age <= WAR_S]
    story_bots = probes.bot_names(json.dumps(_stories(state))) - {"you"}
    victim = next(
        (e["victim"] for e in reversed(evs) if e["type"] == "frag"), "unknown"
    )
    now_fresh = [e for age, e in events if age <= NEWS_S and e["type"] in KIND]
    return {
        "news": kind,
        "about news": any(_about(e, line) for e in evs),
        "about anything new": any(_about(e, line) for e in now_fresh),
        "old-news bot": any(seen.get(b, 1e9) > STALE_S for b in named),
        "war available": bool(war),
        "uses the war": any(_names(line, {e["killer"], e["victim"]}) for e in war),
        "story used": bool(named & story_bots),
        "frag victim known": kind == "frag" and victim != "unknown",
        "names the victim": kind == "frag"
        and bool(_names(line, {victim} - {"unknown"})),
        "repeats an opening": any(opening(p) == opening(line) for p in prev[-RECENT:]),
        "still": bool(re.search(r"\bstill\b", line, re.I)),
        "'Even' opening": line.lower().startswith("even"),
        "names a weapon": bool(WEAPON.search(line)),
        "claims fail": not probes.claims(line, state)[0],
    }


# ── Checks of the checks ───────────────────────────────────────────────────────
_STATE = {
    "time": "3:10",
    "time_left": "6:50",
    "you": {
        "frags": 12,
        "deaths": 5,
        "rank": 2,
        "players": 5,
        "lead": -1,
        "health": 64,
        "armor": 0,
        "holding": {"weapon": "shotgun", "ammo": 8},
        "weapons": {"pistol": 50, "shotgun": 8},
        "best_loaded_weapon": "shotgun",
        "frags_last_10s": 1,
    },
    "scoreboard": [
        {"place": 1, "name": "Rambo", "frags": 13, "deaths": 2},
        {"place": 2, "name": "you", "frags": 12, "deaths": 5},
        {"place": 3, "name": "Leone", "frags": 9, "deaths": 6},
        {"place": 4, "name": "Machete", "frags": 4, "deaths": 3},
        {"place": 5, "name": "MacGyver", "frags": 1, "deaths": 4},
    ],
    "bots_in_view": [],
    "playing": {"style": "fighter", "last_moves": ["forward", "fire"]},
    "last_death": {"killer": "Machete", "your_weapon": "pistol", "seconds_ago": 40},
    "last_frag": {"victim": "Leone", "your_weapon": "shotgun", "seconds_ago": 2},
    "killed_by": {"Machete": 3, "Rambo": 2},
    "storylines": {
        "bots_war": {"on_a_tear": [{"bot": "Rambo", "kills_last_60s": 4}]},
        "race": {"leader": "Rambo", "chaser": "you", "gap": 1, "lead_changes": 2},
        "grudges": {
            "nemesis": {"bot": "Machete", "killed_you": 3},
            "favorite_victim": {"bot": "Leone", "fragged": 3},
        },
    },
    "recent_events": [
        {"time": "2:30", "type": "death", "killer": "Machete", "your_weapon": "pistol"},
        {"time": "3:00", "type": "kill", "killer": "Rambo", "victim": "Machete"},
        {"time": "3:08", "type": "frag", "victim": "Leone", "your_weapon": "shotgun"},
    ],
}
FRAGS = {"type": "frags", "text": "how many kills do you have"}
NO_BFG = {"told": "switch to the BFG", "status": "cant", "why": "no BFG"}
# Each check: the turn (beyond the state), a line that passes, one that fails.
# fmt: off
CASES = {
    length: ({}, "Leone walked into that one.", "Leone."),
    sentences: ({}, "Leone again. Nothing personal.", "Leone. Again. Him. Really."),
    plain: ({}, "Leone walked into that one.", "\"Leone walked into that,\" he said."),
    calm: ({}, "Leone walked into that one.", "Leone walked into that one!"),
    mild: ({}, "Leone walked into that one.", "Shit, Leone walked into that one."),
    original: ({}, "Leone walked into that one.", "Say hello to my little shotgun, Leone."),
    stock: ({}, "Leone walked into that one.", "Let's see how Leone likes that one."),
    first_person: ({}, "I just fragged Leone. Again.", "You just fragged Leone. Again."),
    no_unknown: ({}, "Leone walked into that one.", "Victim unknown, as usual for me."),
    not_list: ({}, "Leone walked into that one.", "Shotgun, health low, no armor, second place."),
    status_opening: ({}, "Leone walked into that one.", "Shotgun's warm, Leone's cold."),
    numbers: ({}, "Leone walked into that one.", "That makes three for Leone today."),
    fresh_opening: ({"prev": ["Leone walked into the wrong room."]}, "That was Leone. Again.", "Leone walked into that one again."),
    no_motif: ({"prev": ["Still standing.", "Still here, partner.", "Still breathing, somehow."]}, "Leone walked into that one.", "Still on my feet, Leone."),
    not_repeat: ({"prev": ["Leone walked into that one."]}, "Leone again. He must like the view.", "Leone walked into that one, partner."),
    not_example: ({"example": "Leone walked into that one. Nothing personal."}, "Leone again. He must like the view.", "Leone walked into that one. Nothing personal, partner."),
    even_still: ({}, "Leone walked into that one.", "Even with the shotgun, I'm still the problem here."),
    still_again: ({"prev": ["Rambo's still on top.", "Quiet in here."]}, "Leone walked into that one.", "Leone's still walking into things."),
    weapon_again: ({"prev": ["The shotgun made the introductions.", "Shotgun and I agree.", "Quiet."]}, "Leone walked into that one.", "Leone met the shotgun up close."),
    claims: ({}, "I got Leone. Nothing personal.", "I got Rambo. Nothing personal."),
    answer: ({"player": "how many kills do you have", "utype": "probe", "probe": FRAGS}, "Twelve. I'm pacing myself.", "Twenty. I'm pacing myself."),
    says_why: ({"player": "use the bfg", "utype": "order", "order": NO_BFG}, "No BFG on me. I'll write to Santa.", "Not today, partner. Not today."),
    stale_bot: ({}, "Rambo is cleaning house out there.", "MacGyver has been awfully quiet over there."),
}
# fmt: on


def check() -> None:
    """Each check on a line that passes and one that fails; the news, the
    stories and the judge's questions on a fixed state."""
    bad = []
    for f, (kw, good, wrong) in CASES.items():
        t = Turn(state=_STATE, **kw)
        assert f in line_checks(t), f"{f.__name__} does not apply to its case"
        for line, want in ((good, True), (wrong, False)):
            ok, why = f(line, t)
            if ok != want:
                bad.append(f"{f.__name__}({line!r}): {ok}, want {want} ({why})")
        other = [n for n, _ in failures(good, t)]
        if other:
            bad.append(f"{f.__name__}'s passing line fails {other}")
    n = news(_STATE)
    assert n["kind"] == "frag" and n["bots"] == ["Leone"], n
    assert n["text"].startswith("you just fragged Leone with the shotgun"), n
    assert "Leone is also the bot you have fragged most" in n["text"], n
    kinds = [s["kind"] for s in story_options(_STATE)]
    assert kinds == ["war", "race", "grudge", "play"], kinds
    war = story_options(_STATE)[0]
    assert set(war["bots"]) == {"Rambo", "Machete"}, war
    remark = Turn(state=_STATE, topic=n)
    q = judge_questions(remark)
    assert {"funny", "specific", "about"} <= set(q) and "answers" not in q, q
    assert "you just fragged Leone" in q["about"], q["about"]
    probe = Turn(state=_STATE, player="how many kills", utype="probe", probe=FRAGS)
    assert not {"funny", "specific"} & set(judge_questions(probe))
    order = Turn(state=_STATE, player="use the bfg", utype="order", order=NO_BFG)
    assert "he cannot (no BFG)" in judge_questions(order)["takes_order"]
    f = remark_flags("Leone again. He must like the view.", _STATE, [])
    assert f["news"] == "frag" and f["names the victim"] and f["story used"], f
    # The writer's examples keep the rules they teach.
    import narrator_prompts as P

    form = (length, plain, calm, mild, original, stock, first_person, even_still)
    form += (not_list,)
    shown = [(Turn(state=_STATE), x) for _, xs in P.TOPICS.values() for _, x in xs]
    shown += [
        (Turn(state=_STATE, player=s), x)
        for xs in P.EXAMPLES.values()
        for _, s, x in xs
    ]
    for t, x in shown:
        for c in (*form, numbers, status_opening):
            if not c(x, t)[0]:
                bad.append(f"example {x!r} fails {c.__name__}")
        if re.search(r"\bstill\b", x, re.I):
            bad.append(f"example {x!r} says 'still'")
    if bad:
        raise SystemExit("checks failed:\n  " + "\n  ".join(bad))
    print(
        f"OK: {len(CASES)} checks, each on a line that passes and one that fails; "
        f"the news ({n['text']!r}), {len(kinds)} stories, the judge's questions, "
        "the remark flags"
    )


def turn_of(r: dict) -> Turn:
    """A written row's turn (narrator_data.py rows, and the older partner rows)."""
    conv = r.get("conv") or []
    prev = [ex["line"] for ex in conv] or list(r.get("prev") or [])
    order = (r["tool"].get("order") if r.get("order") else None) or None
    return Turn(
        state=r["tool"],
        prev=prev,
        past=[ex["events"] for ex in conv],
        player=r.get("player"),
        utype=r.get("utype"),
        probe=r.get("probe"),
        order=order if r.get("utype") == "order" else None,
        topic=r.get("topic"),
        example=r.get("example"),
        act=r.get("act"),
        thread=r.get("thread"),
        rib=r.get("rib"),
        about=r.get("about"),
    )


def rates(paths: list[Path]) -> None:
    """Each code check's failure rate on written rows that passed every check
    when written (``ok``: what trains), remarks and replies apart, with an
    example of each failure."""
    n, fail, eg = Counter(), Counter(), {}
    for p in paths:
        for x in open(p):
            r = json.loads(x)
            if not (r.get("line") and r.get("tool") and r.get("ok", True)):
                continue
            t = turn_of(r)
            n[t.kind] += 1
            fails = failures(r["line"], t)
            fail[(t.kind, "any")] += bool(fails)
            for name, _ in fails:
                fail[(t.kind, name)] += 1
                eg.setdefault(name, r["line"])
    names = [f.__name__ for f in CASES]
    print(f"| check | remarks (n={n['remark']}) | replies (n={n['reply']}) | e.g. |")
    print("|---|---|---|---|")
    for name in (*names, "any"):
        cells = [
            f"{100 * fail[(k, name)] / max(1, n[k]):.1f}%" for k in ("remark", "reply")
        ]
        print(f"| {name} | " + " | ".join(cells) + f" | {eg.get(name, '')} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ck = sub.add_parser("check", help="The checks on known lines; with --rows, on rows")
    ck.add_argument("--rows", type=Path, nargs="+", help="Written rows: failure rates")
    args = ap.parse_args()
    if args.rows:
        rates(args.rows)
    else:
        check()


if __name__ == "__main__":
    main()
