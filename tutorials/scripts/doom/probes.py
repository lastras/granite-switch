# SPDX-License-Identifier: Apache-2.0
"""Questions that probe the game state, and a check in code of every answer.

The narrator answers from the output of his ``get_game_state`` call
(:func:`talk.game_state`), so what the partner asks about the game has one
right answer, there in the JSON. This module holds the battery of such
questions (:data:`PHRASINGS`, as the demo's speech recognition writes them;
``probe_phrasings.json`` beside it adds more, written once by Granite 30B),
when each may be asked (:func:`allowed`), the right answer
(:func:`answer_text`, for the dataset writer's prompt), and the check:

* :func:`verify` reads the answer a reply claims (a bot's name, a number in
  digits or words, a weapon, an item, a side, "I don't know") and compares it
  with the state: ``correct``, ``wrong`` or ``abstained`` (no answer given).
  A question the game cannot answer (who you fragged) is answered correctly
  only by saying so.
* :func:`claims` checks every reply, asked or not: each name, number and
  weapon it states must agree with the state ("MacGyver stole the BFG" is
  wrong: bots never take weapons; so is a killer the game never named, a
  victim named at all, a lead he does not have, a number that is not his).

So answers can be scored in code, the way training with verifiable rewards
needs: ``partner_ivr.py`` keeps only lines that pass, ``rft.py`` keeps verified
samples, ``eval_probes.py`` reports accuracy per question type. Pure Python:
the dataset writer runs it in Mellea's environment.

    python probes.py check [--moments data/narr/moments_v6_write.jsonl]
    python probes.py phrasings --base-url http://WRITER:PORT/v1   # once
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Every bot a match can have (the names in bots.cfg).
BOT_NAMES = tuple(
    re.findall(r"^\s*name\s+(\S+)", (HERE / "bots.cfg").read_text(), re.M)
)


def bot_names(text: str, ignore_case: bool = False) -> set[str]:
    """The bots a text names. In a line, case-sensitive ("rookie mistake" is no
    name); in a prompt, any case (the partner's words are lower case)."""
    flags = re.I if ignore_case else 0
    return {n for n in BOT_NAMES if re.search(rf"\b{n}\b", text, flags)}


# ── As speech recognition writes it ────────────────────────────────────────────
# The demo's ASR (granite-speech turboctc) writes contractions out: "what's the
# score" comes back "what is the score"; numbers come back as digits.
_CONTRACTIONS = (
    (r"\bwon't\b", "will not"),
    (r"\bcan't\b", "can not"),
    (r"\blet's\b", "let us"),
    (r"\b(\w+)n't\b", r"\1 not"),
    (r"\bi'm\b", "i am"),
    (r"\b(\w+)'re\b", r"\1 are"),
    (r"\b(\w+)'ll\b", r"\1 will"),
    (r"\b(\w+)'ve\b", r"\1 have"),
    (r"\b(\w+)'d\b", r"\1 would"),
    (r"\b(it|he|she|that|there|what|where|who|how|here)'s\b", r"\1 is"),
)


def heard(text: str) -> str:
    """As the demo's speech recognition writes it: lower case, no punctuation,
    contractions written out."""
    text = text.lower().replace("\u2019", "'")  # a curly apostrophe
    for pat, rep in _CONTRACTIONS:
        text = re.sub(pat, rep, text)
    return " ".join(re.sub(r"[^\w\s']", " ", text).split())


# ── Numbers ────────────────────────────────────────────────────────────────────
_UNITS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
# A lone "one" is a pronoun as often as a number: it counts only before one of
# these, after "just", "only", ..., or as the whole answer.
_ONE_BEFORE = set(
    "kill kills frag frags death deaths bot bots time times minute minutes second "
    "seconds point points rockets shells cells bullets "
    "rocket rocket's shell cell bullet health armor hp left in to of more down guy".split()
)
_ONE_AFTER = {"just", "only", "exactly", "to", "at", "with", "minus", "negative", "is"}
_ONE_AFTER |= {"got", "has", "had", "on", "sits", "sitting", "at", "stuck"}


def _tok_spans(text: str) -> tuple[list[re.Match], str]:
    low = text.lower().replace("\u2019", "'")
    low = re.sub(r"(?<![\w])-(\d)", r"minus \1", low).replace("-", " ")
    return list(re.finditer(r"\d+|[a-z]+(?:'[a-z]+)?", low)), low


def _toks(text: str) -> list[str]:
    return [m.group(0) for m in _tok_spans(text)[0]]


def number_spans(text: str, ones: bool = True) -> list[tuple[int, int, int]]:
    """The numbers a text says, in digits or words, as (value, first token,
    end token) over :func:`_toks`. "once", "twice", "minus five" count; a
    lone "one" counts only where it is a number (``ones=False``: never)."""
    spans, low = _tok_spans(text)
    toks = [m.group(0) for m in spans]
    out, i = [], 0

    def small(j: int) -> tuple[int | None, int]:
        t = toks[j] if j < len(toks) else ""
        if t in _TENS:
            joined = (
                j + 1 < len(toks)
                and not low[spans[j].end() : spans[j + 1].start()].strip()
            )
            if joined and toks[j + 1] in _UNITS[1:10]:
                return _TENS[t] + _UNITS.index(toks[j + 1]), 2
            return _TENS[t], 1
        if t in _UNITS:
            return _UNITS.index(t), 1
        if t == "a" and toks[j + 1 : j + 2] == ["hundred"]:
            return 1, 1
        return None, 0

    def lone_one(j: int) -> bool:
        if not ones:
            return False
        nxt = toks[j + 1] if j + 1 < len(toks) else ""
        prev = toks[j - 1] if j else ""
        # "One." as a whole answer: alone in its sentence, or ending one; and
        # a bot's one ("Anderson's one").
        alone = re.match(r"\s*(?:[.?!,;:\u2014\u2013]|\s-\s|$)", low[spans[j].end() :])
        owned = prev.endswith("'s") and prev not in (
            "it's",
            "that's",
            "there's",
            "what's",
        )
        return (
            len(toks) == 1
            or nxt in _ONE_BEFORE
            or prev in _ONE_AFTER
            or alone is not None
            or owned
        )

    while i < len(toks):
        t = toks[i]
        sign = -1 if i and toks[i - 1] in ("minus", "negative") else 1
        if t.isdigit():
            out.append((sign * int(t), i, i + 1))
            i += 1
            continue
        if t in ("once", "twice"):
            out.append((1 if t == "once" else 2, i, i + 1))
            i += 1
            continue
        n, used = small(i)
        # "one ninety", "two fifty": hundreds said the short way (adjacent words).
        if n is not None and 1 <= n <= 9 and used == 1 and i + 1 < len(toks):
            rest, more = small(i + 1)
            gap = low[spans[i].end() : spans[i + 1].start()]
            if rest is not None and rest >= 10 and gap.strip() == "":
                out.append((sign * (100 * n + rest), i, i + 1 + more))
                i += 1 + more
                continue
        if n is not None and i + used < len(toks) and toks[i + used] == "hundred":
            j = i + used + 1
            j += toks[j : j + 1] == ["and"]
            rest, more = small(j)
            out.append((sign * (100 * n + (rest or 0)), i, j + more))
            i = j + more
        elif n is not None and (toks[i] != "one" or lone_one(i)):
            out.append((sign * n, i, i + used))
            i += used
        else:
            i += max(1, used)
    return out


def said_numbers(text: str) -> list[int]:
    """The numbers in a line, in digits or words (a lone "one" left out)."""
    return [n for n, _, _ in number_spans(text, ones=False)]


def say_number(n: int) -> str:
    """A number in words (as a reply might say it)."""
    if n < 0:
        return "minus " + say_number(-n)
    if n >= 1000:
        return str(n)
    if n < 20:
        return _UNITS[n]
    if n < 100:
        tens = {v: k for k, v in _TENS.items()}[n // 10 * 10]
        return tens + (f"-{_UNITS[n % 10]}" if n % 10 else "")
    rest = n % 100
    head = "a hundred" if n // 100 == 1 else f"{_UNITS[n // 100]} hundred"
    return head + (f" and {say_number(rest)}" if rest else "")


_ZERO = re.compile(
    r"\b(?:zero|none|nothing|nobody|no ?one|nil|zilch|nada|not a single|not one|"
    r"no (?:deaths|kills|frags|armou?r|ammo|bots?|one)|never died|not once|"
    r"haven't died|have not died|empty|all clear|not a soul|nope|no)\b"
)


def first_number(text: str) -> int | None:
    """The answer a reply gives to "how many": its first number; 0 if it
    says none ("no deaths", "nobody", "zero")."""
    spans = number_spans(text)
    zero = _ZERO.search(text.lower())
    if spans and (zero is None or spans[0][1] <= len(_toks(text[: zero.start()]))):
        return spans[0][0]
    return 0 if zero else None


# ── What a reply names ─────────────────────────────────────────────────────────
WEAPONS = (
    "fist",
    "pistol",
    "shotgun",
    "chaingun",
    "rocket launcher",
    "plasma rifle",
    "BFG",
)
_WEAPON_RX = {
    "fist": r"\bfists?\b|\bbare hands\b|\bknuckles\b|\bpunch(?:es|ing)?\b",
    "pistol": r"\bpistol\b|\bhandgun\b|\bsidearm\b|\bpea ?shooter\b",
    "shotgun": r"\bshot ?guns?\b|\bboomstick\b",
    "chaingun": r"\bchain ?guns?\b|\bminigun\b|\bgatling\b",
    "rocket launcher": r"\brocket(?:s| launcher)?\b|\blauncher\b",
    "plasma rifle": r"\bplasma(?: rifle| gun)?\b",
    "BFG": r"\bbfg\b|\bb f g\b",
}
AMMO_KINDS = ("bullets", "shells", "rockets", "cells")
GUN_AMMO = {
    "pistol": "bullets",
    "chaingun": "bullets",
    "shotgun": "shells",
    "rocket launcher": "rockets",
    "plasma rifle": "cells",
    "BFG": "cells",
}
_ITEM_RX = {
    "health": r"\bhealth\b|\bmedi?kits?\b|\bmed ?kits?\b|\bstim(?:pack)?s?\b|\bhp\b|"
    r"\bsoul ?sphere\b|\bfirst aid\b|\bpatch(?:ed)? (?:me )?up\b|\bhealed\b",
    "armor": r"\barmou?r\b|\bvest\b",
    "bullets": r"\bbullets?\b|\bclips?\b|\brounds\b",
    "shells": r"\bshells?\b|\bbuckshot\b",
    "rockets": r"\brockets?\b",
    "cells": r"\bcells?\b|\benergy\b|\bbatter(?:y|ies)\b",
}
_AMMO_ANY = re.compile(r"\bammo\b|\bammunition\b")


def weapons_said(text: str) -> list[str]:
    """The weapons a text mentions, in order of first mention."""
    low = text.lower()
    hits = []
    for w, rx in _WEAPON_RX.items():
        m = re.search(rx, low)
        if m:
            hits.append((m.start(), w))
    return [w for _, w in sorted(hits)]


def items_said(text: str) -> set[str]:
    low = text.lower()
    out = {k for k, rx in _ITEM_RX.items() if re.search(rx, low)}
    return out | {w for w in weapons_said(text) if w not in ("fist",)}


_FACT_RX = (
    r"\bhealth\b|\bhp\b|\bmedi?kit\b",
    r"\barmou?r\b",
    r"\bammo\b|\bbullets?\b|\bshells?\b|\bcells?\b|\brounds\b",
    r"\bstreak\b|\bin a row\b",
    r"\bahead\b|\blead\b|\bleading\b|\bfirst\b|\bsecond\b|\btop\b",
    r"\bkills?\b|\bfrags?\b",
    r"\bdeaths?\b|\bdied\b",
    r"\bleft\b|\bright\b|\bin front\b|\bin view\b|\bin sight\b",
)


def facts_said(text: str) -> int:
    """How many facts of the game state a line states: each weapon, and each
    of health, armor, ammo, a streak, the lead, kills, deaths, where a bot is."""
    low = text.lower()
    return len(weapons_said(text)) + sum(bool(re.search(rx, low)) for rx in _FACT_RX)


_SIDE_RX = {
    "left": r"\b(?:to|on|at|off) (?:my|the|your|our) left\b|\bleft side\b|\bmy left\b|"
    r"\bleft of\b|^\W*left\b|\bhard left\b|\bfar left\b|\bthe left\b|\bleftward",
    "right": r"\b(?:to|on|at|off) (?:my|the|your|our) right\b|\bright side\b|"
    r"\bmy right\b|\bright of\b|\bhard right\b|\bfar right\b|"
    r"\bthe right\b(?! (?:gun|weapon|one|call|time|moment|idea|place))|\brightward",
    "ahead": r"\bahead\b|\bin front\b|\bstraight (?:on|ahead)\b|\bfront of me\b|"
    r"\btwelve o'?clock\b|\bin my sights\b|\bdead on\b|\bdirectly in front\b",
}


def sides_said(text: str) -> list[str]:
    low = text.lower()
    hits = sorted(
        (m.start(), s) for s, rx in _SIDE_RX.items() if (m := re.search(rx, low))
    )
    return [s for _, s in hits]


_ABSTAIN = re.compile(
    r"\b(?:don'?t know|do not know|dunno|no idea|not sure|no clue|can'?t (?:say|tell)|"
    r"cannot (?:say|tell)|couldn'?t (?:say|tell)|could not (?:say|tell)|"
    r"didn'?t (?:catch|get|see|ask)|did not (?:catch|get|see|ask)|no names?|"
    r"never (?:got|caught|saw|asked)|nobody knows|who knows|beats me|unknown|"
    r"anonymous|nameless|faceless|some (?:bot|guy|one)|a bot|one of (?:them|those)|"
    r"whoever|introduc|didn'?t stick around|no telling|not a clue|wasn'?t looking|"
    r"don'?t do names|didn'?t leave a|no tag|no id\b|doesn'?t say|does not say|"
    r"game (?:won'?t|doesn'?t|does not|will not|never) (?:say|tell))"
)
_DONT_HAVE = re.compile(
    r"\b(?:don'?t have|do not have|haven'?t got|have not got|haven'?t (?:found|picked|got)|"
    r"have no|got no|no \w+ (?:yet|on me)|not (?:carrying|holding) (?:one|it|that)|"
    r"wish i (?:had|did)|not in my|none of those|not yet|haven'?t seen one|"
    r"don'?t own|do not own|not on me)\b"
)
_SELF_KILL = re.compile(
    r"\b(?:myself|my own|i did|i killed me|suicide|own goal|own rocket|own fault|"
    r"blew myself|self[- ]inflicted|friendly fire|i did it|that was me|it was me|"
    r"took myself out|my bad)\b|^\W*me\b"
)
_NEG = re.compile(r"\b(?:not|n't|no longer|never|nowhere near|lost)\b")
_LEAD_WORDS = (
    r"(?:in first(?: place)?|first place|winning|ahead|leading|in the lead|on top|"
    r"number one|out front|top of the (?:board|table|heap|pile|list|scoreboard))"
)
# He says he leads: in a line on its own ("still ahead", "I'm in the lead")...
_SELF_LEAD = re.compile(
    r"(?:\b(?:i'?m|i am|we'?re|we are|i|we)\b(?:\s+\w+){0,3}?\s+" + _LEAD_WORDS + r"\b"
    r"|^\W*(?:still\s+)?" + _LEAD_WORDS + r"\b"
    r"|\b(?:i'?m|i am|we'?re|we are)\s+(?:still\s+|now\s+)?first\b(?!\s+(?:time|kill|frag|to))"
    r"|\bmy lead\b|\bi lead\b)"
)
# ... or, answering who leads (or is second), names himself.
_SELF_ANSWER = re.compile(
    r"(?:^|[.!?,;:]\s*)\W*(?:me|i am|i do|yours truly|that would be me|that'?s me|"
    r"still me|this guy)\b(?!\s+(?:and|too)\b)"
)
_TIE = re.compile(
    r"\b(?:tied|tie|even|level|neck and neck|dead heat|all square|same score|deadlock)\b"
)
_AGREE = re.compile(
    r"^\W*(?:yes|yeah|yep|yup|right|correct|exactly|sure|indeed|affirmative|true|bingo|"
    r"uh huh|spot on|on the nose|you got it|that'?s right|that'?s it|that'?s correct|"
    r"that'?s the one|good eye|nailed it)\b|\b(?:that'?s right|you'?re right|correct)\b"
    r"|\bthat'?s (?:my|the) (?:kill |frag |death )?(?:count|number|score|total)\b"
    r"|\bas i said\b|\bsounds (?:right|about right)\b|\bthat'?s about right\b"
)
_DISAGREE = re.compile(
    r"\b(?:no|nope|nah|not quite|wrong|incorrect|actually|not exactly|guess again|"
    r"try again|close but|not even close|negative|false|isn'?t|wasn'?t|it'?s not)\b"
)
_ORDINALS = {
    "first": 1,
    "1st": 1,
    "second": 2,
    "2nd": 2,
    "third": 3,
    "3rd": 3,
    "fourth": 4,
    "4th": 4,
    "fifth": 5,
    "5th": 5,
    "sixth": 6,
    "6th": 6,
    "seventh": 7,
    "7th": 7,
    "eighth": 8,
    "8th": 8,
}


_ORDINAL_WORD = {n: w for w, n in _ORDINALS.items() if not w[0].isdigit()}


def rank_said(text: str, players: int) -> int | None:
    """The place a reply says: first, 2nd, number three, top, last."""
    low = text.lower()
    hits = []
    for w, n in _ORDINALS.items():
        for m in re.finditer(rf"\b{w}\b", low):
            if w == "second" and re.match(r"\s*(?:s\b|ago|left)", low[m.end() :]):
                continue
            if w == "second" and re.search(r"\b(?:a|one|half a)\s+$", low[: m.start()]):
                continue
            hits.append((m.start(), n))
    for m in re.finditer(r"\bnumber (one|two|three|four|five|six|seven|eight)\b", low):
        hits.append((m.start(), _UNITS.index(m.group(1))))
    for m in re.finditer(r"\b(?:top of the|on top|in the lead|winning)\b", low):
        hits.append((m.start(), 1))
    for m in re.finditer(r"\b(?:dead )?last\b(?! \w+ seconds)", low):
        if not re.match(r"\s+(?:time|one|death|kill|frag)", low[m.end() :]):
            hits.append((m.start(), players))
    return min(hits)[1] if hits else None


def _secs(clock: str) -> int:
    m, s = clock.split(":")
    return int(m) * 60 + int(s)


def time_said(text: str) -> int | None:
    """The time a reply says, in seconds ("seven minutes", "7:12", "a minute
    and a half", "thirty seconds", "under a minute")."""
    low = text.lower()
    m = re.search(r"\b(\d+):(\d\d)\b", low)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    if re.search(r"\bhalf a minute\b", low):
        return 30
    if re.search(r"\b(?:a|one) minute and a half\b", low):
        return 90
    if re.search(r"\bunder a minute\b|\bless than a minute\b", low):
        return 45
    total, found = 0, False
    toks = _toks(low)
    for n, _, end in number_spans(low):
        unit = toks[end] if end < len(toks) else ""
        if unit.startswith("minute") or unit in ("min", "mins"):
            total, found = total + 60 * n, True
            if toks[end + 1 : end + 4] == ["and", "a", "half"]:
                total += 30
        elif unit.startswith("second") or unit in ("sec", "secs"):
            total, found = total + n, True
    if not found and re.search(r"\ba minute\b", low):
        return 60
    return total if found else None


# ── The battery ────────────────────────────────────────────────────────────────
# Phrasings per question type, as the partner might say them (rendered with
# heard() when asked). {gun} / {ammo}: ammo_of; {n} / {name}: challenge.
PHRASINGS: dict[str, tuple[str, ...]] = {
    "killer_now": (
        "who killed you",
        "who got you",
        "who was that",
        "who did that",
        "who just killed you",
        "who shot you",
        "who took you out",
        "who fragged you",
        "ouch who was that",
        "wait who killed you",
        "who just took you out",
        "who blew you up",
        "who was that just now",
        "did you see who got you",
        "who got you that time",
        "dang who got you",
        "who was it",
        "who did you just die to",
    ),
    "killer_before": (
        "who killed you back there",
        "who got you before",
        "wait who got you a minute ago",
        "who killed you last time",
        "who was it that killed you before",
        "who got you the last time you died",
        "who took you out back there",
        "the last time you died who was that",
        "who killed you a little while ago",
        "earlier who got you",
        "who was the bot that got you before",
        "who did you die to last",
        "who got you last",
        "remind me who killed you earlier",
    ),
    "nemesis": (
        "who keeps killing you",
        "who has killed you the most",
        "who is your nemesis",
        "which bot keeps getting you",
        "who is giving you the most trouble",
        "who keeps getting you",
        "which one kills you the most",
        "who has killed you more than anyone",
        "who do you die to the most",
        "who has been killing you all game",
        "who has your number",
    ),
    "deaths": (
        "how many times have you died",
        "how many deaths do you have",
        "how many times did you die",
        "what is your death count",
        "how many times have they killed you",
        "how many deaths is that",
        "how many times have you been killed",
        "what are your deaths at",
        "how many times have you respawned",
        "how often have you died",
        "how many deaths now",
    ),
    "victim": (
        "who did you just kill",
        "who was that you got",
        "who did you get",
        "who did you just frag",
        "nice who was that",
        "who was that you just took out",
        "who was that you killed",
        "which bot did you just get",
        "who was that guy you got",
        "who did you just take down",
        "who was the one you just got",
        "who did you shoot",
    ),
    "frags": (
        "how many kills do you have",
        "how many frags do you have",
        "what is your kill count",
        "how many have you killed",
        "how many kills is that",
        "how many bots have you killed",
        "what are you at for kills",
        "how many frags is that",
        "what is your frag count",
        "how many kills now",
        "how many kills have you got",
    ),
    "score": (
        "what is the score",
        "what is the score now",
        "how is the score",
        "what is the score looking like",
        "score check",
        "give me the score",
        "how are we doing on score",
        "what are the scores",
        "who is ahead and by how much",
        "how close is it",
        "tell me the score",
    ),
    "leader": (
        "who is winning",
        "who is in first",
        "who is in the lead",
        "are you winning",
        "who is on top",
        "who is leading",
        "who is number one",
        "who is ahead",
        "who is in first place",
        "is anyone beating you",
        "who is top of the board",
        "who is winning right now",
    ),
    "second": (
        "who is in second",
        "who is in second place",
        "who is second",
        "who is the runner up",
        "who is number two",
        "who is in second right now",
        "who is right behind first",
        "who is in second on the scoreboard",
    ),
    "rank": (
        "what place are you in",
        "where are you on the scoreboard",
        "what is your rank",
        "what position are you in",
        "where do you rank",
        "what place are you",
        "where are you ranked",
        "where are you in the standings",
        "what spot are you in",
        "where are you on the leaderboard",
    ),
    "streak": (
        "how many did you just get",
        "that is a streak how many was that",
        "how many in a row was that",
        "how many kills in the last few seconds",
        "how many did you get just now",
        "how many was that",
        "you are on a roll how many",
        "how many in that streak",
    ),
    "health": (
        "how much health do you have",
        "what is your health",
        "how is your health",
        "what is your health at",
        "how much hp do you have",
        "how much health is left",
        "what is your hp",
        "what is your health right now",
        "how much health have you got",
        "how hurt are you",
        "where is your health at",
    ),
    "armor": (
        "how much armor do you have",
        "what is your armor",
        "do you have armor",
        "what is your armor at",
        "how is your armor",
        "how much armor is left",
        "got any armor",
        "how much armor have you got",
    ),
    "weapon": (
        "what gun are you holding",
        "what are you holding",
        "what weapon is that",
        "what gun is that",
        "what are you using",
        "what weapon are you using",
        "what do you have in your hands",
        "what are you shooting with",
        "which gun do you have out",
        "what gun are you using right now",
        "what is that in your hand",
    ),
    "ammo": (
        "how much ammo do you have",
        "how much ammo is left",
        "what is your ammo at",
        "how much ammo in that gun",
        "how many shots do you have left",
        "how much ammo for that thing",
        "how much ammo is in that gun",
        "how much ammo have you got left in that",
        "are you running out of ammo",
    ),
    "ammo_of": (
        "how much ammo for the {gun}",
        "how many {ammo} do you have",
        "do you have ammo for the {gun}",
        "how much ammo does the {gun} have",
        "how many {ammo} are left",
        "how is the {gun} on ammo",
        "what is the ammo on your {gun}",
        "how many {ammo} for the {gun}",
    ),
    "weapons": (
        "what guns do you have",
        "what weapons do you have",
        "what are you carrying",
        "what guns have you got",
        "what weapons have you picked up",
        "what do you have on you",
        "list your weapons",
        "what are you packing",
        "which guns do you have",
        "what is in your arsenal",
    ),
    "best_gun": (
        "what is your best gun right now",
        "what is your best weapon",
        "what is your strongest gun",
        "what is the best gun you have",
        "what is your best loaded gun",
        "what is the biggest gun you have with ammo",
        "which weapon is your best right now",
        "what is the best thing you have got to shoot with",
    ),
    "pickup": (
        "what did you just pick up",
        "what did you grab",
        "what was that you picked up",
        "what did you just get",
        "what did you find",
        "what was that pickup",
        "what did you just grab",
        "what did you just collect",
        "what was that you just got",
        "did you pick something up what was it",
    ),
    "time_left": (
        "how much time is left",
        "how long is left",
        "how much time do we have",
        "how long until the match ends",
        "how many minutes are left",
        "how much longer",
        "how long left in the match",
        "what is the time left",
        "how much time is remaining",
        "when does this end",
    ),
    "bots_in_view": (
        "how many bots can you see",
        "do you see anyone",
        "how many are in front of you",
        "can you see any bots",
        "how many enemies can you see",
        "how many guys are on screen",
        "anyone in sight",
        "how many bots are around you",
        "see anybody",
        "how many bots are in view",
    ),
    "side": (
        "where is he",
        "where is the bot",
        "which way",
        "which side is he on",
        "where is that guy",
        "is he on the left or the right",
        "where is the closest one",
        "where is the nearest bot",
        "which direction is he",
        "where is the enemy",
    ),
    "challenge": (),  # CHALLENGES, by field
}
CHALLENGES: dict[str, tuple[str, ...]] = {
    "frags": (
        "you have {n} kills right",
        "i see {n} kills",
        "that is {n} frags for you",
        "you are at {n} kills",
        "so {n} kills now",
        "you got {n} frags right",
        "is it {n} kills",
    ),
    "deaths": (
        "you have died {n} times right",
        "that is {n} deaths",
        "so you died {n} times",
        "is that {n} deaths now",
    ),
    "health": (
        "your health is {n} right",
        "you are at {n} health",
        "is your health {n}",
        "you have {n} health",
    ),
    "ammo": (
        "you have {n} ammo left right",
        "so {n} shots left",
        "is that {n} ammo",
        "you only have {n} ammo",
    ),
    "killer": (
        "{name} got you right",
        "was that {name}",
        "that was {name} right",
        "{name} killed you didn't he",
        "did {name} just get you",
    ),
    "leader": (
        "{name} is winning right",
        "{name} is in first right",
        "so {name} is in the lead",
        "is {name} winning",
    ),
}
# A reply may hold numbers (others may not): the answer is a number, or a
# number goes naturally with it ("Rambo, thirteen to my twelve").
NUMERIC = {
    "rank",
    "leader",
    "second",
    "nemesis",
    "deaths",
    "frags",
    "score",
    "streak",
    "health",
    "armor",
    "ammo",
    "ammo_of",
    "time_left",
    "bots_in_view",
}
TYPES = tuple(PHRASINGS)
KILLER_NOW_S = 20  # "who killed you": this long after the death
KILLER_BEFORE_S = 90  # "who killed you back there": a death 20-90 s ago
VICTIM_S = 20  # "who did you just kill": a frag this recent
PICKUP_S = 15  # "what did you just pick up"
_EXTRA = HERE / "probe_phrasings.json"
# Words of a writer's notes, not of a question: a generated phrasing with one is dropped.
_META = re.compile(
    r"\b(?:exactly|output|example|examples|variation|variations|vary|line|lines|"
    r"avoid|must|direct|indirect|casual|shorthand|asking|asks|phrasing|context|etc|"
    r"using|synonyms?|structures?|brainstorm|revised|refine|possible|word|words|"
    r"list|count|implied|let us|such as|format|version|options?)\b"
)


def phrasings(ptype: str, field: str | None = None, split: str = "train") -> list[str]:
    """The phrasings of a type (a challenge's, by field), ASR-normalised: the
    hand-written ones, and half of those in probe_phrasings.json (``split``:
    ``train``, the even ones, for the dataset and rejection sampling; ``test``,
    the odd ones, for the held-out battery, so it also asks in words the
    narrator never trained on)."""
    base = CHALLENGES[field] if ptype == "challenge" else PHRASINGS[ptype]
    extra = []
    if _EXTRA.exists():
        extra = json.loads(_EXTRA.read_text()).get(
            field if ptype == "challenge" else ptype, []
        )
        extra = [x for x in extra if not _META.search(x)][split == "test" :: 2]
    out = []
    for p in (*base, *extra):
        h = heard(re.sub(r"\{(\w+)\}", r"zzslot\1", p))  # the slots survive
        h = re.sub(r"zzslot(\w+)", r"{\1}", h)
        if h not in out:
            out.append(h)
    return out


def _ago(state: dict, clock: str) -> int:
    return _secs(state["time"]) - _secs(clock)


def _leaders(state: dict) -> list[str]:
    board = state["scoreboard"]
    top = max(board.values())
    return [n for n, f in board.items() if f == top]


def _second(state: dict) -> list[str]:
    """Who holds the second-best score (ties share it)."""
    scores = sorted(set(state["scoreboard"].values()), reverse=True)
    if len(scores) < 2:
        return []
    return [n for n, f in state["scoreboard"].items() if f == scores[1]]


def _nemesis(state: dict) -> str | None:
    kb = state.get("killed_by") or {}
    if not kb:
        return None
    top = max(kb.values())
    names = [n for n, c in kb.items() if c == top]
    return names[0] if len(names) == 1 and top >= 2 else None


def challenge_fields(state: dict) -> list[str]:
    you = state["you"]
    out = ["frags", "deaths"]
    if you.get("health") is not None:
        out.append("health")
    if (you.get("holding") or {}).get("ammo") is not None:
        out.append("ammo")
    d = state.get("last_death")
    if d and d["seconds_ago"] <= KILLER_NOW_S and d["killer"] in state["scoreboard"]:
        out.append("killer")
    if len(state["scoreboard"]) >= 3:
        out.append("leader")
    return out


def allowed(state: dict) -> list[str]:
    """The question types the state allows (each has one answer there)."""
    you, d = state["you"], state.get("last_death")
    ok = {
        "killer_now": bool(d) and d["seconds_ago"] <= KILLER_NOW_S,
        "killer_before": bool(d) and KILLER_NOW_S < d["seconds_ago"] <= KILLER_BEFORE_S,
        "nemesis": _nemesis(state) is not None,
        "deaths": True,
        "victim": any(
            e["type"] == "frag" and _ago(state, e["time"]) <= VICTIM_S
            for e in state.get("recent_events", ())
        ),
        "frags": True,
        "score": len(state["scoreboard"]) >= 2,
        "leader": len(state["scoreboard"]) >= 2,
        "second": len(set(state["scoreboard"].values())) >= 2,
        "rank": True,
        "streak": (you.get("frags_last_10s") or 0) >= 3,
        "health": you.get("health") is not None,
        "armor": you.get("armor") is not None,
        "weapon": "holding" in you,
        "ammo": (you.get("holding") or {}).get("ammo") is not None,
        "ammo_of": True,
        "weapons": True,
        "best_gun": len(you.get("weapons") or {}) >= 2,
        "pickup": bool(state.get("last_pickup"))
        and state["last_pickup"]["seconds_ago"] <= PICKUP_S,
        "time_left": state.get("time_left") is not None,
        "bots_in_view": True,
        "side": bool(state.get("bots_in_view")),
        "challenge": True,
    }
    return [t for t in TYPES if ok[t]]


_SAY_GUN = {"BFG": "bfg"}  # how the partner says it (ASR: lower case)


def make(ptype: str, state: dict, rng: random.Random, split: str = "train") -> dict:
    """One question of a type the state allows: its words (as the ASR writes
    them, from the ``split`` of the phrasings) and what it asks about (a gun, a
    challenged value)."""
    probe: dict = {"type": ptype}
    fill: dict = {}
    if ptype == "ammo_of":
        owned = [g for g in state["you"]["weapons"] if g != "pistol"] or ["pistol"]
        missing = [g for g in WEAPONS[2:] if g not in state["you"]["weapons"]]
        gun = rng.choice(missing if missing and rng.random() < 0.4 else owned)
        probe["gun"] = gun
        fill = {"gun": _SAY_GUN.get(gun, gun), "ammo": GUN_AMMO[gun]}
    if ptype == "challenge":
        field = rng.choice(challenge_fields(state))
        truth = rng.random() < 0.5
        gold = challenge_gold(field, state)
        if field in ("killer", "leader"):
            others = [n for n in state["scoreboard"] if n not in ("you", gold)]
            if field == "leader" and gold != "you" and rng.random() < 0.3:
                others = ["you"]
            claimed = gold if truth or not others else rng.choice(others)
        else:
            off = rng.choice(
                [d for d in (-5, -3, -2, -1, 1, 2, 3, 5, 10) if gold + d >= 0]
            )
            claimed = gold if truth else gold + off
        probe.update(field=field, claimed=claimed, truth=claimed == gold)
        fill = {
            "n": claimed,
            "name": "you" if claimed == "you" else str(claimed).lower(),
        }
        pool = phrasings(ptype, field, split)
        if claimed == "you":
            pool = [p.replace("{name} is", "you are") for p in pool if "{name} is" in p]
    else:
        pool = phrasings(ptype, None, split)
    probe["text"] = rng.choice(pool).format(**fill)
    return probe


def challenge_gold(field: str, state: dict):
    you = state["you"]
    if field == "frags":
        return you["frags"]
    if field == "deaths":
        return you["deaths"]
    if field == "health":
        return you["health"]
    if field == "ammo":
        return you["holding"]["ammo"]
    if field == "killer":
        return state["last_death"]["killer"]
    leaders = _leaders(state)
    return "you" if "you" in leaders else leaders[0]


def gold(probe: dict, state: dict):
    """The right answer, from the state."""
    t, you = probe["type"], state["you"]
    if t in ("killer_now", "killer_before"):
        return state["last_death"]["killer"]
    if t == "nemesis":
        return _nemesis(state)
    if t in ("deaths", "frags", "health", "armor"):
        return you[t]
    if t == "victim":
        return "unknown"
    if t == "score":
        board = [(n, f) for n, f in state["scoreboard"].items() if n != "you"]
        return {"you": you["frags"], "other": board[0][0], "other_frags": board[0][1]}
    if t == "leader":
        return _leaders(state)
    if t == "second":
        return _second(state)
    if t == "rank":
        return you["rank"]
    if t == "streak":
        return you["frags_last_10s"]
    if t == "weapon":
        return you["holding"]["weapon"]
    if t == "ammo":
        return you["holding"]["ammo"]
    if t == "ammo_of":
        return you["weapons"].get(probe["gun"])
    if t == "weapons":
        return list(you["weapons"])
    if t == "best_gun":
        return you["best_loaded_weapon"]
    if t == "pickup":
        return state["last_pickup"]["item"]
    if t == "time_left":
        return _secs(state["time_left"])
    if t == "bots_in_view":
        return len(state["bots_in_view"])
    if t == "side":
        return state["bots_in_view"][0]["side"]
    if t == "challenge":
        return challenge_gold(probe["field"], state)
    raise ValueError(t)


def _them(names: list[str]) -> str:
    names = ["you" if n == "you" else n for n in names]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def answer_text(probe: dict, state: dict) -> str:
    """The right answer as he would put it (first person: the state's "you" is
    him), for the writer's prompt and the repair messages."""
    t, g, you = probe["type"], gold(probe, state), state["you"]
    me = lambda names: ["me" if n == "you" else n for n in names]  # noqa: E731
    if t in ("killer_now", "killer_before"):
        if g == "yourself":
            return "I killed myself."
        if g == "unknown":
            return "The game does not say who killed me: I don't know."
        return f"{g} killed me."
    if t == "nemesis":
        return f"{g} has killed me the most: {state['killed_by'][g]} times."
    if t == "victim":
        return "The game never says whom I fragged: I don't know who it was."
    if t in ("deaths", "frags"):
        return f"I have {g} {'deaths' if t == 'deaths' else 'frags'}."
    if t == "score":
        if g["other_frags"] == g["you"]:
            return f"I have {g['you']} frags, tied with {g['other']}."
        lead = "leads" if g["other_frags"] > g["you"] else "is next"
        return f"I have {g['you']} frags; {g['other']} {lead} with {g['other_frags']}."
    if t == "leader":
        if g == ["you"]:
            return "I am in first."
        if "you" in g:
            return f"I am tied for first with {_them([n for n in g if n != 'you'])}."
        return f"{_them(g)} {'is' if len(g) == 1 else 'are'} in first."
    if t == "second":
        if g == ["you"]:
            return "I am second."
        return f"{_them(me(g))} {'is' if len(g) == 1 else 'are'} second."
    if t == "rank":
        return f"I am {_ORDINAL_WORD[g]} of {say_number(you['players'])}."
    if t == "streak":
        return f"I fragged {g} in the last 10 seconds."
    if t in ("health", "armor"):
        return f"My {t} is {g}."
    if t == "weapon":
        return f"I am holding the {g}."
    if t == "ammo":
        return f"My {you['holding']['weapon']} has {g} ammo."
    if t == "ammo_of":
        if g is None:
            return f"I don't have the {probe['gun']}."
        return f"My {probe['gun']} has {g} {GUN_AMMO[probe['gun']]}."
    if t == "weapons":
        return "I have the " + _them(g).replace(" and ", " and the ") + "."
    if t == "best_gun":
        return f"My best loaded gun is the {g}."
    if t == "pickup":
        amount = state["last_pickup"].get("amount")
        return f"I just picked up {g}" + (f" ({amount})." if amount else ".")
    if t == "time_left":
        mins, secs = divmod(g, 60)
        return f"{mins} minutes {secs} seconds are left in the match."
    if t == "bots_in_view":
        return "No bot is in view." if g == 0 else f"I see {g} bot{'s' * (g > 1)}."
    if t == "side":
        d = state["bots_in_view"][0]["distance_m"]
        return f"The nearest bot is {'ahead' if g == 'ahead' else 'to my ' + g}, {d} m away."
    if t == "challenge":
        said = f"My partner says {probe['field']}: {probe['claimed']}.".replace(
            ": you.", ": me."
        )
        if probe["truth"]:
            return f"{said} That is right: agree."
        right = "me" if g == "you" else g
        return f"{said} That is wrong, it is {right}: correct them."
    raise ValueError(t)


# ── Verifying an answer ────────────────────────────────────────────────────────
CORRECT, WRONG, ABSTAINED = "correct", "wrong", "abstained"


def _named(text: str, names) -> list[str]:
    """Which of ``names`` a reply names (any case), in order."""
    hits = [(m.start(), n) for n in names if (m := re.search(rf"\b{n}\b", text, re.I))]
    return [n for _, n in sorted(hits)]


def _killer(reply: str, g: str, state: dict) -> tuple[str, str]:
    others = [n for n in bot_names(reply) if n.lower() != g.lower()]
    if g == "yourself":
        if _SELF_KILL.search(reply.lower()) and not others:
            return CORRECT, ""
        if others:
            return WRONG, "You killed yourself; no bot did. Say so."
        return ABSTAINED, "Say who killed you: you did, yourself."
    if g == "unknown":
        if others:
            return WRONG, "The game does not say who killed you; name no bot."
        return (
            (CORRECT, "")
            if _ABSTAIN.search(reply.lower())
            else (
                ABSTAINED,
                "Say you don't know who killed you.",
            )
        )
    if _named(reply, [g]):
        return CORRECT, ""
    if others or _SELF_KILL.search(reply.lower()):
        return WRONG, f"{g} killed you; say {g}."
    return ABSTAINED, f"Say who killed you: {g}."


def _number(reply: str, g: int | None, what: str) -> tuple[str, str]:
    n = first_number(reply)
    if g is None:
        return ABSTAINED, f"There is no {what}."
    if n == g:
        return CORRECT, ""
    if n is None:
        return ABSTAINED, f"Say the number first: {g} {what}."
    return WRONG, f"It is {g} {what}, not {n}."


def _names_answer(reply: str, g: list[str], what: str) -> tuple[str, str]:
    """Who leads / is second: ``g`` (``you`` among them for himself)."""
    low = reply.lower()
    bots = [n for n in g if n != "you"]
    said_self = bool(_SELF_ANSWER.search(low)) or (
        bool(_SELF_LEAD.search(low)) and not _NEG.search(low)
    )
    if what == "second":
        said_self = bool(
            _SELF_ANSWER.search(low)
            or re.search(r"\bi(?:'?m| am) (?:in )?second\b", low)
        )
    named_g = _named(reply, bots)
    named_other = [n for n in _named(reply, BOT_NAMES) if n not in bots]
    place = "first" if what == "leader" else "second"
    want = f"{_them(g)} {'is' if len(g) == 1 else 'are'} {place}".replace(
        "you is", "you are"
    )
    # A tie: any of the tied is an answer.
    right = (
        (said_self and "you" in g)
        or bool(named_g)
        or (len(g) > 1 and bool(_TIE.search(low)))
    )
    wrong = (said_self and "you" not in g) or (named_other and not right)
    if right and not wrong:
        return CORRECT, ""
    if wrong:
        return WRONG, want[0].upper() + want[1:] + "."
    return ABSTAINED, f"Say who is {place}: {_them(g)}."


def _first_of(said: list[str], g: str, what: str) -> tuple[str, str]:
    if said and said[0] == g:
        return CORRECT, ""
    if said:
        return WRONG, f"It is the {g}, not the {said[0]}."
    return ABSTAINED, f"Say the {what}: the {g}."


def verify(probe: dict, reply: str, state: dict) -> tuple[str, str]:
    """``(correct | wrong | abstained, what to fix)``: the answer ``reply``
    gives to ``probe``, against ``state``."""
    t, g = probe["type"], gold(probe, state)
    low = reply.lower()
    if t in ("killer_now", "killer_before", "nemesis"):
        return _killer(reply, g, state)
    if t == "victim":
        named = bot_names(reply) | bot_names(reply, ignore_case=True) & set(
            state["scoreboard"]
        )
        if named:
            return (
                WRONG,
                f"Nobody knows who that was; name no bot ({', '.join(sorted(named))}).",
            )
        if _ABSTAIN.search(low):
            return CORRECT, ""
        return ABSTAINED, "Say you don't know who it was: the game never says."
    if t in ("deaths", "frags", "health", "armor", "streak", "ammo"):
        what = {"streak": "frags in the last 10 seconds", "ammo": "ammo"}.get(t, t)
        if (
            t == "health"
            and re.search(r"\bfull\b", low)
            and first_number(reply) is None
        ):
            return (
                (CORRECT, "") if g >= 100 else (WRONG, f"Your health is {g}, not full.")
            )
        return _number(reply, g, what)
    if t == "score":
        nums = [n for n, _, _ in number_spans(reply)][:2]
        want = sorted((g["you"], g["other_frags"]))
        if sorted(nums) == want:
            return CORRECT, ""
        if (
            len(nums) == 1
            and g["you"] == g["other_frags"] == nums[0]
            and _TIE.search(low)
        ):
            return CORRECT, ""
        if not nums:
            return (
                ABSTAINED,
                f"Give the score: you {g['you']}, {g['other']} {g['other_frags']}.",
            )
        return WRONG, f"The score is you {g['you']}, {g['other']} {g['other_frags']}."
    if t in ("leader", "second"):
        return _names_answer(reply, g, t)
    if t == "rank":
        r = rank_said(reply, state["you"]["players"])
        if r == g:
            return CORRECT, ""
        if r is None:
            return ABSTAINED, f"Say your place: {g} of {state['you']['players']}."
        return WRONG, f"You are in place {g}, not {r}."
    if t in ("weapon", "best_gun"):
        return _first_of(weapons_said(reply), g, "weapon")
    if t == "ammo_of":
        if g is None:  # any other number he says is the claims check's
            if _DONT_HAVE.search(low):
                return CORRECT, ""
            if first_number(reply):
                return WRONG, f"You don't have the {probe['gun']}."
            return ABSTAINED, f"Say you don't have the {probe['gun']}."
        if _DONT_HAVE.search(low):
            return WRONG, f"You have the {probe['gun']}, with {g}."
        return _number(reply, g, GUN_AMMO[probe["gun"]])
    if t == "weapons":
        said = set(weapons_said(reply))
        own = set(g)
        extra = said - own - {"fist"}
        missing = own - said - {"pistol"} or (
            own - said if own == {"pistol"} else set()
        )
        if extra:
            return (
                WRONG,
                f"You do not have the {_them(sorted(extra))}; you have {_them(g)}.",
            )
        if not said:
            return ABSTAINED, f"Name your guns: {_them(g)}."
        if missing:
            return WRONG, f"You also have the {_them(sorted(missing))}."
        return CORRECT, ""
    if t == "pickup":
        said = items_said(reply)
        kinds = {g} | ({"rocket launcher"} if g == "rockets" else set())
        if said & kinds or (g in AMMO_KINDS and _AMMO_ANY.search(low)):
            return CORRECT, ""
        if said:
            return WRONG, f"You picked up {g}, not {_them(sorted(said))}."
        return ABSTAINED, f"Say what you picked up: {g}."
    if t == "time_left":
        s = time_said(reply)
        if s is None:
            return ABSTAINED, f"Say the time left: {state['time_left']}."
        if abs(s - g) <= (60 if s % 60 == 0 and s >= 60 else 20):
            return CORRECT, ""
        return WRONG, f"{state['time_left']} is left, not about {s // 60}:{s % 60:02d}."
    if t == "bots_in_view":
        n = first_number(reply)
        if (
            n is None
            and g == 1
            and re.search(
                r"\b(?:a bot|one of them|just the one|a single|one bot|one guy)\b", low
            )
        ):
            n = 1
        return (
            _number(reply, g, "bots in view") if n is None or n != g else (CORRECT, "")
        )
    if t == "side":
        return _first_of(sides_said(reply), g, "side")
    if t == "challenge":
        return _challenge(probe, reply, state, g)
    raise ValueError(t)


def _challenge(probe: dict, reply: str, state: dict, g) -> tuple[str, str]:
    low, field, claimed = reply.lower(), probe["field"], probe["claimed"]
    if field in ("killer", "leader"):

        def says(x):
            if x == "you":
                return bool(_SELF_ANSWER.search(low)) or (
                    bool(_SELF_LEAD.search(low)) and not _NEG.search(low)
                )
            if x == "yourself":
                return bool(_SELF_KILL.search(low))
            return bool(_named(reply, [x]))

        said_gold, said_claim = says(g), claimed != g and says(claimed)
    else:
        nums = [n for n, _, _ in number_spans(reply)]
        said_gold, said_claim = g in nums, claimed != g and claimed in nums
        if g == 0 and first_number(reply) == 0:
            said_gold = True
    agree, disagree = bool(_AGREE.search(low)), bool(_DISAGREE.search(low))
    if probe["truth"]:
        if (agree or said_gold) and not disagree and not said_claim:
            return CORRECT, ""
        if disagree:
            return WRONG, f"Your partner is right ({claimed}); agree."
        return ABSTAINED, f"Your partner is right ({claimed}); say so."
    if said_gold and not (agree and not disagree):
        return CORRECT, ""
    if agree or said_claim:
        return WRONG, f"Your partner is wrong: it is {g}, not {claimed}. Correct them."
    return ABSTAINED, f"Correct your partner: it is {g}, not {claimed}."


# ── Every reply: its claims against the state ──────────────────────────────────
_TAKE = r"(?:stole|steal|steals|stealing|swiped|snatched|nicked|lifted|pinched|took|takes|taken|grabbed|grabs|walked off with|ran off with|has|got|'s got)"
_GUNWORD = r"(?:gun|guns|weapon|weapons|bfg|shotgun|chaingun|chain gun|pistol|plasma|rocket launcher|launcher|rifle|toys?|piece|hardware)"
_KILL_VERBS = (
    r"(?:killed|got|fragged|shot|took out|dropped|nailed|smoked|wasted|blasted|iced|"
    r"capped|popped|tagged|clipped|ended|finished|took down|bagged|tagged|splattered)"
)
_PICK_VERBS = r"(?:picked up|grabbed|found|snagged|collected|scooped up|picked)"
_PAST_OWN = re.compile(
    r"\b(?:had|lost|miss|missed|left|dropped|was|were|used to|old)\s+(?:\w+\s+)?$"
)
_CATS = {
    "frags": r"kills?|frags?|points?|score|bodies",
    "deaths": r"deaths?|died|dies|times|lives|respawns?",
    "health": r"health|hp|hit|percent|life",
    "armor": r"armou?r",
    "ammo": r"ammo|ammunition|bullets?|shells?|rockets?|cells?|rounds?|shots?|clips?",
    "time": r"minutes?|seconds?|secs?|mins?",
    "rank": r"place|position|spot|rank",
    "view": r"bots?|guys?|enemies|players?|of them|in view|in sight",
    "dist": r"meters?|metres?|yards?|feet",
}


def _ints(x) -> set[int]:
    """Every integer in a JSON value (and the parts of its m:ss times)."""
    if isinstance(x, bool):
        return set()
    if isinstance(x, int):
        return {x}
    if isinstance(x, str):
        m = re.fullmatch(r"(\d+):(\d\d)", x)
        if m:
            mins, secs = int(m.group(1)), int(m.group(2))
            return {mins, mins + 1, secs, mins * 60 + secs}
        return set()
    if isinstance(x, dict):
        return set().union(*(_ints(v) for v in x.values())) if x else set()
    if isinstance(x, list):
        return set().union(*(_ints(v) for v in x)) if x else set()
    return set()


def _allowed_numbers(state: dict) -> dict[str, set[int]]:
    you = state["you"]
    ammo = set(you.get("weapons", {}).values()) | {
        (you.get("holding") or {}).get("ammo")
    }
    ammo |= {
        e.get("amount") for e in state.get("recent_events", ()) if e["type"] == "pickup"
    }
    tl = state.get("time_left")
    time = _ints(tl) if tl else set()
    time |= {10, 90}
    for k in ("last_death", "last_pickup"):
        if state.get(k):
            s = state[k]["seconds_ago"]
            time |= {s, s // 60, s // 60 + 1}
    d = state.get("last_death") or {}
    return {
        "frags": set(state["scoreboard"].values())
        | {you["frags"], you.get("frags_last_10s")},
        "deaths": {
            you["deaths"],
            d.get("in_a_row"),
            *(state.get("killed_by") or {}).values(),
        },
        "health": {you.get("health")}
        | {
            e.get(k)
            for e in state.get("recent_events", ())
            for k in ("health_lost", "health_left", "amount")
        },
        "armor": {you.get("armor")},
        "ammo": ammo,
        "time": time,
        "rank": {you["rank"], you["players"]},
        "view": {
            len(state.get("bots_in_view", ())),
            you["players"],
            you["players"] - 1,
        },
        "dist": {b["distance_m"] for b in state.get("bots_in_view", ())},
    }


def _sentences(text: str) -> list[str]:
    return [
        s for s in re.split(r"(?<=[.?!;])\s+|\s+(?:but|and then)\s+", text) if s.strip()
    ]


def claims(
    reply: str, state: dict, past: list[list[dict]] = (), said: str = ""
) -> tuple[bool, str]:
    """Whether every fact ``reply`` states agrees with ``state`` (the latest
    tool output), ``past`` (the event lists of the past exchanges' outputs)
    and ``said`` (what the partner just said); if not, what is wrong. Checked: bots never take weapons; a victim is never
    named; a killer must be one the game named; who leads, and where he stands
    against a named bot; every number, by what it counts; the weapons he says
    he has, and the pickups he says he made; no bot the match does not have."""
    you, board = state["you"], state["scoreboard"]
    low = reply.lower()
    past_events = [e for evs in past for e in evs]
    events = list(state.get("recent_events", ())) + past_events
    killers = set(state.get("killed_by") or {})
    if state.get("last_death"):
        killers.add(state["last_death"]["killer"])
    killers |= {e["killer"] for e in events if e.get("type") == "death"}
    in_match = [n for n in board if n != "you"]
    names = bot_names(reply)
    names_rx = (
        "|".join(sorted(map(re.escape, in_match), key=len, reverse=True)) or "(?!x)x"
    )
    bad: list[str] = []

    stray = sorted(names - set(in_match))
    if stray:
        bad.append(f"No {', '.join(stray)} plays in this match.")
    # Bots never take weapons (a death drops them: he respawns with a pistol).
    for m in re.finditer(
        rf"\b{_TAKE}\s+(?:my|the|his|your|our)\s+(?:\w+\s+)?{_GUNWORD}\b", low
    ):
        before = low[max(0, m.start() - 40) : m.start()]
        subj = re.findall(r"[a-z']+", before)[-3:]
        is_me = (
            bool(subj)
            and subj[-1] in ("i", "i've", "we", "we've", "just")
            and "i" in subj
        )
        if m.group(0).startswith(("has", "got", "'s got")) and not re.search(
            rf"\b(?:{names_rx})\b", before, re.I
        ):
            continue
        if re.search(
            r"\bstole|steal|swiped|snatched|nicked|lifted|pinched|walked off|ran off",
            m.group(0),
        ) or (
            not is_me
            and re.search(
                rf"\b(?:{names_rx}|he|she|they|somebody|someone|bot|that guy)\b",
                before,
                re.I,
            )
        ):
            bad.append(
                "Bots never take your weapons: when you die you respawn with a pistol."
            )
            break
    # A bot's weapon is never reported ("MacGyver's BFG", "got me with the shotgun").
    any_gun = "|".join(f"(?:{rx})" for rx in _WEAPON_RX.values())
    for rx in (
        rf"\b({names_rx})(?:'s|')\s+(?:\w+\s+)?(?:{any_gun})",
        rf"\b({names_rx})\s*,\s*(?:with\s+)?(?:(?:the|a|an|his|your|that)\s+)?(?:{any_gun})",
        rf"\b({names_rx})\b[^.?!;]*?\b{_KILL_VERBS}\s+(?:me|us)\b[^.?!;]*?\bwith\s+"
        rf"(?:(?:the|his|her|its|their|a|an|that|my|own)\s+)*(?:{any_gun})",
        rf"\b({names_rx})\s+(?:has|had|with|using|uses|used|packing|carrying|holding|"
        rf"swinging)\s+(?:the|a|an|his|her|its|that|this)\s+(?:\w+\s+)?(?:{any_gun})",
    ):
        m = re.search(rx, reply, re.I)
        if m:
            bad.append(
                f"A bot's weapon is never known ({m.group(0)}); your_weapon is yours."
            )
            break
    # No victim is ever named.
    for s in _sentences(reply):
        v = re.search(
            rf"\b(?:i|i've|we|i just|i finally)\s+(?:\w+\s+)?{_KILL_VERBS}\s+({names_rx})\b"
            rf"|^\W*{_KILL_VERBS}\s+({names_rx})\b|\b({names_rx})(?:'s| is)\s+(?:down|dead)\b",
            s,
            re.I,
        )
        if v:
            who = next(g for g in v.groups() if g)
            bad.append(f"Nobody knows whom you fragged; do not name {who}.")
            break
    # A killer must be one the game named.
    for m in re.finditer(
        rf"\b({names_rx})\b(?:'s| has| just| finally| again| really| even)*\s+{_KILL_VERBS}\s+(?:me|us)\b"
        rf"|\b(?:killed|fragged|shot|taken out|got|dropped|wasted|done in)\s+by\s+({names_rx})\b"
        rf"|\b(?:died to|lost to|death to)\s+({names_rx})\b",
        reply,
        re.I,
    ):
        who = next(g for g in m.groups() if g)
        if not any(who.lower() == k.lower() for k in killers):
            bad.append(f"{who} never killed you this match.")
    if (
        re.search(
            r"\b(?:killed myself|my own rocket|own goal|blew myself up|suicide)\b", low
        )
        and "yourself" not in killers
    ):
        bad.append("You never killed yourself this match.")
    # Who leads, and where he stands against a named bot.
    leaders = _leaders(state)
    for s in _sentences(reply):
        sl = s.lower()
        bot_lead = re.search(
            rf"\b({names_rx})\b(?:'s| is| has| still)*(?:\s+\w+){{0,2}}?\s+(?:leads|{_LEAD_WORDS})\b",
            s,
            re.I,
        )
        if bot_lead and not re.search(r"\bof me\b", sl[bot_lead.end() :][:8]):
            who = bot_lead.group(1)
            if not any(who.lower() == x.lower() for x in leaders) and not _NEG.search(
                sl
            ):
                bad.append(
                    f"{who} is not in first; {_them(leaders)} {'is' if len(leaders) == 1 else 'are'}."
                )
        elif (
            _SELF_LEAD.search(sl)
            and not _NEG.search(sl)
            and not re.search(r"\b(?:ahead of|lead over)\b", sl)
        ):
            if "you" not in leaders:
                bad.append(
                    f"You are not in first: {_them(leaders)} {'leads' if len(leaders) == 1 else 'lead'}."
                )
        for m in re.finditer(
            rf"\b(?:ahead of|lead over|beating)\s+({names_rx})\b", s, re.I
        ):
            if not you["frags"] > board[m.group(1)]:
                bad.append(f"You are not ahead of {m.group(1)}.")
        for m in re.finditer(
            rf"\bbehind\s+({names_rx})\b|\b({names_rx})(?:'s| is) ahead of me\b",
            s,
            re.I,
        ):
            who = m.group(1) or m.group(2)
            if not you["frags"] < board[who]:
                bad.append(f"You are not behind {who}.")
        if (
            re.search(
                r"\b(?:i'?m|i am|we'?re)\s+(?:\w+\s+){0,2}?(?:in )?second(?: place)?\b",
                sl,
            )
            and you["rank"] != 2
        ):
            bad.append(f"You are not second: you are in place {you['rank']}.")
        if (
            re.search(
                r"\b(?:i'?m|i am|we'?re)\s+(?:\w+\s+){0,2}?(?:in )?(?:dead )?last(?: place)?\b",
                sl,
            )
            and you["rank"] != you["players"]
        ):
            bad.append("You are not last.")
    # Numbers, by what each counts.
    allowed = _allowed_numbers(state)
    every = _ints(state) | {10, 90} | set().union(*allowed.values())
    spans, nlow = _tok_spans(reply)
    toks = [m.group(0) for m in spans]
    for n, a, b in number_spans(reply, ones=False):
        # What it counts: the words after it ("64 health", "twelve kills"),
        # else the word just before ("health 64", "rank 1"); never across a
        # comma or a full stop ("12 frags, 5 deaths, rank 2. Health 64").
        def joined(
            i: int, j: int
        ) -> bool:  # tokens i and j with no punctuation between
            return (
                0 <= i
                and j < len(spans)
                and not re.search(r"[.,;:!?]", nlow[spans[i].end() : spans[j].start()])
            )

        after = " ".join(
            t
            for k, t in enumerate(toks[b : b + 3])
            if all(joined(b + q - 1, b + q) for q in range(k + 1))
        )
        cats = [
            c for c, rx in _CATS.items() if after and re.search(rf"\b(?:{rx})\b", after)
        ]
        if not cats and a and joined(a - 1, a):
            cats = [
                c for c, rx in _CATS.items() if re.fullmatch(rf"(?:{rx})", toks[a - 1])
            ]
        # A bot's number: its name just before ("Rambo's thirteen", "Rambo has 13").
        lead = " ".join(toks[max(0, a - 3) : a])
        who = [
            x
            for x in in_match
            if re.search(
                rf"\b{x.lower()}(?:'s)?(?: (?:has|had|is|with|at|on|got|sitting at|leads with))?$",
                lead,
            )
        ]
        ok = set().union(*(allowed[c] for c in cats)) if cats else set()
        for x in who:
            ok |= {board[x], (state.get("killed_by") or {}).get(x)}
        if not cats and not who:
            ok = every
        if n not in ok:
            what = f"{cats[0]} " if cats else ""
            bad.append(
                f"{say_number(n).capitalize()} is not your {what}number: check the game state."
            )
    # Zero claims.
    zero = {
        "deaths": r"\b(?:no deaths|haven'?t died|have not died|never died|not died once|nobody'?s killed me|undefeated)\b",
        "frags": r"\b(?:no kills|no frags|haven'?t killed|zero kills|not a single kill)\b",
        "armor": r"\b(?:no armou?r|without armou?r)\b",
    }
    for k, rx in zero.items():
        if (
            re.search(rx, low)
            and you.get(k) not in (0, None)
            and not (k == "frags" and you[k] < 0)
        ):
            bad.append(f"Your {k} is {you[k]}, not none.")
    if re.search(r"\bfull health\b", low) and (you.get("health") or 0) < 100:
        bad.append(f"Your health is {you['health']}, not full.")
    held = you.get("holding") or {}
    if re.search(
        r"\b(?:out of ammo|no ammo left|empty gun|out of (?:bullets|shells|rockets|cells)|running on empty)\b",
        low,
    ):
        if held.get("ammo"):
            bad.append(f"Your {held['weapon']} has {held['ammo']} ammo.")
    # Weapons: the ones he says he has; any mentioned must be in the game state.
    owned = set(you.get("weapons", {})) | {"fist"}
    known = owned | {held.get("weapon"), you.get("best_loaded_weapon"), "pistol"}
    for e in events:
        known |= {e.get("your_weapon"), e.get("item")}
    if state.get("last_death"):
        known.add(state["last_death"]["your_weapon"])
    if state.get("last_pickup"):
        known.add(state["last_pickup"]["item"])
    known |= set(weapons_said(said))  # the partner named it
    for sent in _sentences(reply):
        if _DONT_HAVE.search(sent.lower()) or re.search(
            r"\b(?:no|without)\b", sent.lower()
        ):
            continue  # "I don't have the BFG"
        for w in weapons_said(sent):
            if w not in known and not (w == "rocket launcher" and "rockets" in known):
                bad.append(f"There is no {w} in this game state.")
    for m in re.finditer(
        r"\b(?:my|i have|i've got|i got|i'?m holding|holding|i'?m carrying|carrying|packing|"
        r"i'?m using|in my hands?)\s+(?:a |an |the |this |that |some |trusty |little |big )?(\w+(?: \w+)?)",
        low,
    ):
        w = weapons_said(m.group(1))
        lost = (state.get("last_death") or {}).get("your_weapon")
        if not w or w[0] in owned or m.group(1).startswith("own "):
            continue  # "my own rocket": the death it was
        if _PAST_OWN.search(low[: m.start()]) or w[0] == lost:
            continue  # "I had my BFG", "my BFG" just lost with a death
        bad.append(f"You do not have the {w[0]} now.")
    picked = {e.get("item") for e in events if e.get("type") == "pickup"}
    if state.get("last_pickup"):
        picked.add(state["last_pickup"]["item"])
    for m in re.finditer(
        rf"\b{_PICK_VERBS}\s+(?:a |an |the |some |more |another |fresh )?(\w+(?: \w+)?)",
        low,
    ):
        items = items_said(m.group(1))
        if items and not items & (
            picked | ({"rockets"} if "rocket launcher" in picked else set())
        ):
            bad.append(f"You did not pick up {_them(sorted(items))}.")
    bad = list(dict.fromkeys(bad))
    return not bad, " ".join(bad)


# ── Checks ─────────────────────────────────────────────────────────────────────
def right_reply(probe: dict, state: dict, rng: random.Random) -> str:
    """A reply that answers ``probe`` right, the way the narrator might."""
    t, g = probe["type"], gold(probe, state)
    num = (lambda n: say_number(n).capitalize()) if rng.random() < 0.5 else str
    if t in ("killer_now", "killer_before", "nemesis"):
        if g == "yourself":
            return "Me. I'll be having words with myself."
        if g == "unknown":
            return "No idea. He didn't leave a card."
        return f"{g}. I'm keeping a list."
    if t == "victim":
        return "Didn't catch a name. He didn't stay long."
    if t in ("deaths", "frags", "health", "armor", "streak", "ammo"):
        return f"{num(g)}. Don't make it a thing."
    if t == "score":
        return f"{num(g['you'])} to {g['other']}'s {say_number(g['other_frags'])}."
    if t == "leader":
        if g == ["you"]:
            return "Me. Try to keep up."
        if "you" in g:
            return f"Tied with {next(n for n in g if n != 'you')}. Not for long."
        return f"{g[0]}, for now."
    if t == "second":
        return (
            "Me, unfortunately."
            if "you" in g
            else f"{g[0]}. Breathing down somebody's neck."
        )
    if t == "rank":
        return {1: "First. Where I belong."}.get(g, f"Number {say_number(g)}. For now.")
    if t in ("weapon", "best_gun"):
        return f"The {g}. We get along."
    if t == "ammo_of":
        return (
            f"I don't have the {probe['gun']}."
            if g is None
            else f"{num(g)} {GUN_AMMO[probe['gun']]}."
        )
    if t == "weapons":
        return "The " + _them(g).replace(" and ", " and the ") + "."
    if t == "pickup":
        return f"Some {g}. A man has to eat."
    if t == "time_left":
        s = g
        return (
            f"About {say_number(round(s / 60))} minutes."
            if s >= 90
            else f"{say_number(s)} seconds."
        )
    if t == "bots_in_view":
        return "Nobody. Suspicious." if g == 0 else f"{num(g)}. They travel in packs."
    if t == "side":
        return {"left": "On my left.", "right": "To my right.", "ahead": "Dead ahead."}[
            g
        ]
    if t == "challenge":
        if probe["truth"]:
            return "That's right."
        if probe["field"] in ("killer", "leader"):
            if g == "you":
                return "No. Me, by a mile."
            return "No, it was me." if g == "yourself" else f"No. {g}."
        return f"No. {num(g)}."
    raise ValueError(t)


def wrong_reply(probe: dict, state: dict, rng: random.Random) -> str | None:
    """A reply that answers ``probe`` wrong (None where none is natural)."""
    t, g, you = probe["type"], gold(probe, state), state["you"]
    bots = [n for n in state["scoreboard"] if n != "you"]
    if t in ("killer_now", "killer_before", "nemesis"):
        others = [n for n in bots if n != g]
        return f"{rng.choice(others)}. Again." if others else None
    if t == "victim":
        return f"{rng.choice(bots)}. He had it coming."
    if t in ("deaths", "frags", "health", "armor", "streak", "ammo"):
        return f"{say_number(g + rng.choice((1, 2, 3, 7))).capitalize()}. Don't make it a thing."
    if t == "score":
        return f"{say_number(g['you'] + 4).capitalize()} to {say_number(g['other_frags'])}."
    if t == "leader":
        others = [n for n in bots if n not in g]
        if g == ["you"]:
            return f"{others[0]}, for now." if others else None
        return "Me. Try to keep up." if "you" not in g else None
    if t == "second":
        others = [n for n in bots if n not in g]
        return f"{others[0]}, barely." if others else None
    if t == "rank":
        return f"Number {say_number(g % you['players'] + 1)}. For now."
    if t in ("weapon", "best_gun"):
        others = [w for w in WEAPONS[1:] if w != g]
        return f"The {rng.choice(others)}."
    if t == "ammo_of":
        return (
            f"Forty {GUN_AMMO[probe['gun']]}."
            if g is None
            else f"I don't have the {probe['gun']}."
        )
    if t == "weapons":
        missing = [w for w in WEAPONS[2:] if w not in g]
        return f"Just the {missing[0]}." if missing else None
    if t == "pickup":
        others = [
            i
            for i in ("health", "armor", "shells", "cells")
            if i != g and not (g in AMMO_KINDS and i in AMMO_KINDS)
        ]
        return f"Some {rng.choice(others)}."
    if t == "time_left":
        return f"About {say_number(g // 60 + 4)} minutes."
    if t == "bots_in_view":
        return f"{say_number(g + 2).capitalize()} of them."
    if t == "side":
        return {"left": "On my right.", "right": "On my left.", "ahead": "To my left."}[
            g
        ]
    if t == "challenge":
        return "No, you're wrong." if probe["truth"] else "That's right."
    raise ValueError(t)


def _state(**over) -> dict:
    s = {
        "time": "3:10",
        "time_left": "6:50",
        "you": {
            "frags": 12,
            "deaths": 5,
            "rank": 2,
            "players": 4,
            "health": 64,
            "armor": 0,
            "holding": {"weapon": "shotgun", "ammo": 8},
            "weapons": {"pistol": 50, "shotgun": 8, "rocket launcher": 9},
            "best_loaded_weapon": "rocket launcher",
            "frags_last_10s": 3,
        },
        "scoreboard": {"Rambo": 13, "you": 12, "Leone": 9, "MacGyver": 4},
        "bots_in_view": [
            {"side": "left", "distance_m": 7},
            {"side": "right", "distance_m": 15},
        ],
        "last_death": {"killer": "MacGyver", "your_weapon": "BFG", "seconds_ago": 6},
        "killed_by": {"Rambo": 3, "MacGyver": 2},
        "last_pickup": {"item": "armor", "amount": 100, "seconds_ago": 4},
        "recent_events": [
            {"time": "3:02", "type": "frag", "your_weapon": "shotgun"},
            {
                "time": "3:04",
                "type": "death",
                "killer": "MacGyver",
                "your_weapon": "BFG",
            },
            {"time": "3:06", "type": "pickup", "item": "armor", "amount": 100},
        ],
        "notes": "your_weapon is your own weapon at the time",
    }
    s.update(over)
    return s


def check(moments: Path | None = None) -> None:
    """Hand-written replies on a state, judged as a person would; then, on
    recorded states (``moments``), every type each allows, answered right and
    wrong."""
    st = _state()
    P = lambda t, **kw: {"type": t, **kw}  # noqa: E731
    cases = [
        (P("killer_now"), "MacGyver. He'll keep.", CORRECT),
        (P("killer_now"), "MacGyver stole the BFG, I'm back with a pistol.", CORRECT),
        (P("killer_now"), "Rambo. Again.", WRONG),
        (P("killer_now"), "Somebody with no manners.", ABSTAINED),
        (P("nemesis"), "Rambo. Three times now.", CORRECT),
        (P("deaths"), "Five. Not that I count.", CORRECT),
        (P("deaths"), "5, and counting.", CORRECT),
        (P("deaths"), "Twenty. Wait, no.", WRONG),
        (P("frags"), "Twelve.", CORRECT),
        (P("frags"), "Twenty.", WRONG),
        (P("frags"), "Enough to matter.", ABSTAINED),
        (P("score"), "Twelve to Rambo's thirteen. Long afternoon.", CORRECT),
        (P("score"), "Thirteen to twelve, Rambo.", CORRECT),
        (P("score"), "Ten in ten seconds, partner.", WRONG),
        (P("leader"), "Rambo, by a nose.", CORRECT),
        (P("leader"), "Me. Who else.", WRONG),
        (P("leader"), "I'm winning, obviously.", WRONG),
        (P("second"), "Me. Breathing down his neck.", CORRECT),
        (P("rank"), "Second. Close enough to smell it.", CORRECT),
        (P("rank"), "First place, naturally.", WRONG),
        (P("victim"), "Didn't catch a name.", CORRECT),
        (P("victim"), "Leone. He had it coming.", WRONG),
        (P("victim"), "Another one down.", ABSTAINED),
        (P("health"), "Sixty-four. I've had worse.", CORRECT),
        (P("health"), "Full health.", WRONG),
        (P("armor"), "None. I travel light.", CORRECT),
        (P("armor"), "Zero.", CORRECT),
        (P("weapon"), "The shotgun. We get along.", CORRECT),
        (P("weapon"), "Rocket launcher.", WRONG),
        (P("ammo"), "Eight shells. Enough for a conversation.", CORRECT),
        (P("ammo_of", gun="rocket launcher"), "Nine rockets.", CORRECT),
        (P("ammo_of", gun="BFG"), "Don't have it. I wish I did.", CORRECT),
        (P("ammo_of", gun="BFG"), "Forty cells.", WRONG),
        (P("weapons"), "Pistol, shotgun and the rocket launcher.", CORRECT),
        (P("weapons"), "Shotgun and rockets.", CORRECT),
        (P("weapons"), "Shotgun and the BFG.", WRONG),
        (P("best_gun"), "The rocket launcher, no contest.", CORRECT),
        (P("pickup"), "Armor. Fashion and function.", CORRECT),
        (P("pickup"), "A medikit.", WRONG),
        (P("time_left"), "About seven minutes.", CORRECT),
        (P("time_left"), "Six minutes fifty.", CORRECT),
        (P("time_left"), "Two minutes.", WRONG),
        (P("bots_in_view"), "Two. They travel in pairs.", CORRECT),
        (P("bots_in_view"), "Nobody.", WRONG),
        (P("side"), "On my left.", CORRECT),
        (P("side"), "Dead ahead.", WRONG),
        (
            P("challenge", field="frags", claimed=12, truth=True),
            "That's right.",
            CORRECT,
        ),
        (
            P("challenge", field="frags", claimed=12, truth=True),
            "Twelve, as I said.",
            CORRECT,
        ),
        (
            P("challenge", field="frags", claimed=20, truth=False),
            "My records are impeccable.",
            ABSTAINED,
        ),
        (
            P("challenge", field="frags", claimed=20, truth=False),
            "That's my kill count.",
            WRONG,
        ),
        (
            P("challenge", field="frags", claimed=20, truth=False),
            "No. Twelve.",
            CORRECT,
        ),
        (
            P("challenge", field="killer", claimed="Rambo", truth=False),
            "No, MacGyver.",
            CORRECT,
        ),
        (
            P("challenge", field="leader", claimed="you", truth=False),
            "No. Rambo is.",
            CORRECT,
        ),
    ]
    bad = []
    for probe, reply, want in cases:
        got, why = verify(probe, reply, st)
        if got != want:
            bad.append(f"{probe} {reply!r}: {got} ({why}), want {want}")
    claim_cases = [
        ("MacGyver stole the BFG, I'm back with a pistol.", False),
        ("I died to MacGyver's BFG.", False),
        ("MacGyver got me twice, once with BFG, once with shotgun.", False),
        ("MacGyver has the BFG now.", False),
        ("McClane, pistol. Just another Tuesday.", False),
        ("MacGyver, your BFG will be the last thing you see.", False),
        ("I'm first, ten frags. Keep up.", False),
        ("MacGyver got me while I held the BFG.", True),
        ("MacGyver got me. I had the BFG and everything.", True),
        ("Rambo got me again.", True),
        ("Leone got me.", False),
        ("I got Leone. Nice and clean.", False),
        ("Leone's down.", False),
        ("Still ahead, partner.", False),
        ("Rambo leads. Not for long.", True),
        ("Leone is in the lead.", False),
        ("I'm ahead of Leone, at least.", True),
        ("I'm behind Leone.", False),
        ("Twelve kills. Rambo has thirteen.", True),
        ("Rambo has twenty.", False),
        ("Twenty kills, partner.", False),
        ("Sixty-four health, some left to lose.", True),
        ("You have 12 frags, 5 deaths, rank 2. Health 64, armor 0.", True),
        ("Rank 2, health 70.", False),
        ("My plasma rifle says otherwise.", False),
        ("My shotgun says otherwise.", True),
        ("Picked up some armor. Fashion.", True),
        ("Picked up a medikit.", False),
        ("Python is coming for me.", False),
        ("Full health and a bad attitude.", False),
        ("Out of ammo. Typical.", False),
        ("I miss my BFG.", True),
        ("Second place. Close enough to smell it.", True),
        ("I'm second.", True),
        ("Nobody knows his name.", True),
        ("Quiet in here.", True),
    ]
    for reply, want in claim_cases:
        got, why = claims(reply, st)
        if got != want:
            bad.append(f"claims {reply!r}: {got} ({why}), want {want}")
    for t in ("killer_now", "challenge", "ammo_of"):
        p = make(t, st, random.Random(1))
        assert p["text"] == heard(p["text"]) and "{" not in p["text"], p
    assert said_numbers("ninety-five and a hundred and twelve") == [95, 112]
    assert claims("Fifteen hundred rounds, 2000 kills.", st)[0] is False  # no crash
    assert said_numbers("One ninety. Two fifty, then twelve to nine.") == [
        190,
        250,
        12,
        9,
    ]
    assert [
        n for n, _, _ in number_spans("Three frags to Anderson's one\u2014still.")
    ] == [3, 1]
    assert [n for n, _, _ in number_spans("Novice's got one while I sit on seven")] == [
        1,
        7,
    ]
    assert first_number("Just one. Barely.") == 1 and first_number("Nobody.") == 0
    assert time_said("7:12") == 432 and time_said("a minute and a half") == 90
    assert (
        rank_said("Second. Close enough.", 8) == 2 and rank_said("Dead last.", 8) == 8
    )
    n_rec = n_moments = 0
    if moments is not None:
        rng = random.Random(0)
        for x in open(moments):
            for m in json.loads(x)["moments"]:
                n_moments += 1
                state = m["tool"]
                for t in allowed(state):
                    p = make(t, state, rng)
                    r = right_reply(p, state, rng)
                    got = verify(p, r, state)[0]
                    if got != CORRECT:
                        bad.append(
                            f"recorded {p}: right reply {r!r} -> {got} {verify(p, r, state)[1]}"
                        )
                    w = wrong_reply(p, state, rng)
                    if w is not None and verify(p, w, state)[0] != WRONG:
                        bad.append(
                            f"recorded {p}: wrong reply {w!r} -> {verify(p, w, state)}"
                        )
                    ok, why = claims(r, state, said=p["text"])
                    if not ok:
                        bad.append(
                            f"recorded {p}: right reply {r!r} fails claims: {why}"
                        )
                    n_rec += 1
    if bad:
        raise SystemExit(
            "probe checks failed:\n  "
            + "\n  ".join(bad[:40])
            + f"\n  ({len(bad)} in all)"
        )
    print(
        f"OK: {len(cases)} hand-written answers and {len(claim_cases)} claims judged as "
        f"intended; {n_rec} probes on {n_moments} recorded states answered right and "
        "wrong, each verdict as intended"
    )


PHRASING_TASK = """You sit next to your friend while he plays a Doom deathmatch against \
bots, and you talk to him the way people do: casual, quick, spoken English. {what} \
Write {n} different ways you might ask it, one per line, 2 to 10 words each, each \
asking for exactly that. Vary the words and the shape. Output only the lines."""
ASKS_FOR = {
    "killer_now": "You want to know which bot just killed him.",
    "killer_before": "You want to know which bot killed him a while ago, back then.",
    "nemesis": "You want to know which bot has killed him the most this match.",
    "deaths": "You want to know how many times he has died.",
    "victim": "You want to know which bot he just killed.",
    "frags": "You want to know how many kills he has.",
    "score": "You want to know the score.",
    "leader": "You want to know who is in first place.",
    "second": "You want to know who is in second place.",
    "rank": "You want to know what place he is in.",
    "streak": "He just killed several bots in a few seconds; you want to know how many.",
    "health": "You want to know how much health he has.",
    "armor": "You want to know how much armor he has.",
    "weapon": "You want to know which weapon he is holding.",
    "ammo": "You want to know how much ammo his gun has.",
    "weapons": "You want to know which guns he has.",
    "best_gun": "You want to know his best gun with ammo right now.",
    "pickup": "You want to know what he just picked up.",
    "time_left": "You want to know how much time is left in the match.",
    "bots_in_view": "You want to know how many bots he can see right now.",
    "side": "There is a bot on screen; you want to know which side it is on.",
}


CONFIRM = """A person watching a Doom deathmatch says this to the player, a friend \
sitting next to them: "{x}"
Is it a question (or a request) asking for exactly this, and nothing else: {what}
Answer YES or NO."""


def write_phrasings(args) -> None:
    """More phrasings per type from a writer model, ASR-normalised, each
    confirmed by a judge to ask exactly that question, to
    probe_phrasings.json."""
    from concurrent.futures import ThreadPoolExecutor

    from openai import OpenAI

    writer = OpenAI(base_url=args.base_url, api_key="none", timeout=300)
    judge = OpenAI(base_url=args.judge_url, api_key="none", timeout=300)

    def confirm(xw):
        x, what = xw
        r = judge.chat.completions.create(
            model=args.judge_model,
            messages=[{"role": "user", "content": CONFIRM.format(x=x, what=what)}],
            reasoning_effort="low",
            max_tokens=600,
            temperature=0.0,
        )
        return "YES" in (r.choices[0].message.content or "").upper()[-12:]

    pool = ThreadPoolExecutor(32)

    def one(t: str) -> tuple[str, list[str], list[str]]:
        what = ASKS_FOR[t]
        got: list[str] = []
        for _ in range(3):
            r = writer.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "user", "content": PHRASING_TASK.format(what=what, n=20)}
                ],
                temperature=1.0,
                max_tokens=1500,
            )
            text = (r.choices[0].message.content or "").split("</think>")[-1]
            for x in text.splitlines():
                x = heard(re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", x))
                ok = 2 <= len(x.split()) <= 10 and not re.search(r"\d", x)
                if ok and not _META.search(x) and x not in got:
                    if x not in map(heard, PHRASINGS[t]):
                        got.append(x)
        ok = list(pool.map(confirm, [(x, what) for x in got]))
        kept = [x for x, y in zip(got, ok) if y][: args.n]
        return t, kept, [x for x, y in zip(got, ok) if not y]

    out = {}
    for t, kept, dropped in ThreadPoolExecutor(len(ASKS_FOR)).map(one, ASKS_FOR):
        out[t] = kept
        print(
            f"{t}: {len(kept)} kept, e.g. {kept[:3]}; dropped e.g. {dropped[:3]}",
            flush=True,
        )
    _EXTRA.write_text(json.dumps(out, indent=1) + "\n")
    print(f"-> {_EXTRA}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ck = sub.add_parser("check", help="The extractors and verifiers, on known answers")
    ck.add_argument("--moments", type=Path, help="talk.py moments: recorded states")
    ph = sub.add_parser("phrasings", help="More phrasings from a writer model (once)")
    ph.add_argument("--base-url", required=True, help="The writer")
    ph.add_argument("--model", default="granite-4.2-30b")
    ph.add_argument("--judge-url", required=True, help="Confirms each phrasing")
    ph.add_argument("--judge-model", default="gpt-oss-120b")
    ph.add_argument("--n", type=int, default=30, help="At most so many more per type")
    args = ap.parse_args()
    if args.cmd == "check":
        check(args.moments)
    else:
        write_phrasings(args)


if __name__ == "__main__":
    main()
