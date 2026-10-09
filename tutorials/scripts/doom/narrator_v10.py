# SPDX-License-Identifier: Apache-2.0
"""Write the v10 narrator's data: Granite's turns as a compilable function, on the Mellea fork.

Granite's line at a moment of a match is one call of :meth:`Narrator.line`, a
*compilable function* (``mellea.stdlib.compile``, the fork's). Its signature is
what the student (the narrator adapter) reads, and its body is the teacher:

* **The boundary.** ``line(chat, *, turn)``: ``chat`` is the narrator's own
  conversation (a ``ChatContext``: the chat template's messages and the
  ``get_game_state`` tool, as :mod:`conversation` builds them with the ``v10``
  system prompt); its result is his next line. The student is trained and
  served on exactly that conversation (the compiled view). ``turn``
  (:class:`TurnFacts`) is privileged: what the driver knows about the moment
  and the student is never told (the question about the game as data, what
  kind of thing the partner said, their words before speech recognition, the
  talk clock). It steers the teacher and checks its line; every fact the line
  states is in the chat (the game state answers a question about the game).
  ``self``, one :class:`Narrator` per match, is the teacher's state between
  calls: its sessions, his ribs and his running bit, which the student reads
  back from the conversation.
* **The body.** It reads the turn from the chat (:func:`conversation.read`),
  plans how to take it (:func:`plan_turn`: the answer or the order first, if
  his partner asked or ordered; else an act of
  ``narrator_v10_prompts.ACT_WEIGHTS``, a rib among them), and has the teacher
  (gpt-oss-120b) write it: :func:`narrator_turn`, a chat-turn generative
  function whose docstring is who he is and how to read the conversation
  (:mod:`narrator_v10_prompts`), with the turn's requirements
  (:func:`requirements`: how to take it, every code check of
  :func:`checks.line_checks_v10`, and the judge's questions) under
  ``MultiTurnStrategy``, which repairs up to ``--loop-budget`` tries. The judge
  (gpt-oss-120b, its own servers) is asked once per try, every question at
  once (:func:`checks.ask_questions`); ``true`` and ``grounded`` are asked again
  of a second judge server, and pass only if both say yes. The body notes the
  row's fields (``annotate``) and returns the line, passed or not.

Tonight's bots are people Granite knows (``conversation.CAST``, named in his
system prompt): a laser tag night. Round 9's recorded matches are played back
with their bots renamed after the cast (:func:`cast_names`, in code, as the
rows are read). He talks in short bursts: how the match is going, what just
happened, and, when one of the cast is in the news, a rib about their thing,
tied to what they did and building on what he said about them before.

A match is an *episode* (:func:`episode`), plain driver code over the match's
live talk clock (:func:`talk.live_moments`, ``--idle-s`` of silence before he
goes on): at each moment the partner may speak first (:func:`partner_turn`:
the order the match recorded, a question about the game (``--probe-rate``),
something else written by the partner's voice in this match's temperament
(``--reply-rate``; of those, ``--follow-rate`` react to what he just said), or
nothing); then he speaks, and the next moment follows the line he said and
how long it took to say.

Those rates are the live partner's: they shape the conversations, and so his
unprompted turns, which are the persona. How well he answers needs more
answers than a quiet partner gives, so at ``--branch-rate`` of the moments the
driver also puts a *prompted branch* to him (:func:`branch_turn`): a question
about the game, of the types asked least so far, or (``--branch-talk`` of
them) other partner talk, off the same conversation. His answer is recorded
and trained on, but never said: the conversation and what he carries on do
not take it in (``TurnFacts.branch``). Orders are never branched: an order
changes the game, which the recording did not play out.

Mellea's ``emulate`` runs the episodes, records every call, scores it at the
episode's end (:func:`score`: kept if the teacher's own loop passed it) and
writes each episode whole; a rerun resumes. Each row is
``train_alora.py``'s narrator row (:func:`row_of`: the match, the moment, the
turn, the line, every try) with the record's standard fields beside it (the
``prompt`` the student reads, the ``completion``, the ``score``). Every random
draw is seeded from the match and the moment.

Runs in the fork's environment (``DATA_PY10``: the fork with its hooks extra,
and ViZDoom), one process per shard (``recipe.sh narrator10_text``)::

    python narrator_v10.py write --moments data/r9/moments_write.jsonl \\
        --out data/r10/narrator/shard_0.jsonl --shard 0/32 \\
        --writer-url http://WRITER:PORT/v1 --judge-url http://JUDGE:PORT/v1
    python narrator_v10.py report --rows data/r10/narrator/shard_*.jsonl
    python narrator_v10.py check --rows runs/r10/narrator/narrator/heldout_gen.jsonl \\
        --keys adapter --judge-url http://JUDGE:PORT/v1
    python narrator_v10.py parity --rows data/r10/narrator/shard_0.jsonl --n 500
    python narrator_v10.py selfcheck
"""

import argparse
import dataclasses
import json
import logging
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import checks
import conversation
import narrator_v10_prompts as P
import probes
from conversation import (
    GET_GAME_STATE,
    NARRATOR_SYSTEM_PROMPTS,
    Conversation,
    Exchange,
    messages,
    new_stretch,
    tool_text,
)
from mellea import generative
from mellea.backends import ModelOption
from mellea.stdlib.compile import (
    Completion,
    Record,
    Score,
    TraceRecorder,
    annotate,
    compilable,
    compiled_view,
)
from mellea.stdlib.components.genstub import openai_messages
from mellea.stdlib.context import ChatContext
from mellea.stdlib.sampling import MultiTurnStrategy
from probes import heard

SYSTEM = "v10"  # the narrator system prompt these rows are written for
TIC_HZ = 35


def seed_of(*key) -> int:
    import zlib

    return zlib.crc32(repr(key).encode())


def one_word_off(a: str, b: str) -> bool:
    """``b`` is ``a`` with exactly one word swapped (a mishearing)."""
    x, y = a.split(), b.split()
    return len(x) == len(y) and sum(p != q for p, q in zip(x, y)) == 1


# ── The teacher's generative functions ────────────────────────────────────────
def narrator_turn(chat: ChatContext) -> str: ...


narrator_turn.__doc__ = f"{P.PERSONA}\n\n{P.READING}\n\n{P.TURN}"
narrator_turn = generative(narrator_turn)


@generative
def partner_says(
    kind: str, how: str, temper: str, game_state: str, conversation: str
) -> list[str]:
    """Three different things the partner might say out loud now to Granite,
    who is playing a Doom deathmatch against bots as a job while the partner,
    his partner of many years, sits next to him watching the screen.

    The partner talks the way people talk sitting side by side for hours:
    about the game and about anything else, 2 to 20 words of plain spoken
    English each, in their own temperament (``temper``). They only watch, so
    they never talk as if they were in the game themselves, and they never
    read numbers or stats off the screen. ``kind`` names what they say and
    ``how`` says it; ``game_state`` is the game's own state ("you" in it is
    Granite); ``conversation`` is what has been said so far, oldest first
    (Partner: is them, Granite: is him). The three differ in meaning, and
    from everything the partner said before."""


@generative
def mishear(sentence: str) -> str:
    """``sentence`` as a speech recognizer heard it, with exactly one word
    wrong: swapped for a similar-sounding real word ("go get the rocket
    launcher" heard as "go get the rocket lunch"). It has the same number of
    words."""


def chat_of(conv: Conversation, state: dict, player: str | None) -> ChatContext:
    """The narrator's conversation now, as :meth:`Narrator.line` takes it."""
    return ChatContext.from_openai(
        messages(conv, state, player, NARRATOR_SYSTEM_PROMPTS[SYSTEM]),
        tools=[GET_GAME_STATE],
    )


# ── What the partner says ──────────────────────────────────────────────────────
def utterance_kinds(m: dict, conv: Conversation) -> list[str]:
    """The partner's utterances (not a question, not an order) the moment
    allows: ``clarify`` only if the partner spoke in the last exchange."""
    events = m.get("events") or []
    ok = {None: True, "event": m.get("cue") == "event"}
    ok["good"] = any(k in events for k in P.GOOD)
    ok["asked"] = bool(len(conv)) and conv.exchanges[-1].player is not None
    return [k for k, (_, _, when) in P.UTTERANCES.items() if ok[when]]


def partner_turn(m, conv, rng, voice, args, weights, seed, temper) -> dict:
    """What the partner says at moment ``m``: ``{"utype", "player", "said",
    "probe", "move"}``; ``utype`` None when the partner says nothing."""
    state = m["tool"]
    if m.get("order"):  # the partner gave an order: these were its words
        return {"utype": "order", "player": heard(m["order"]["said"])}
    if {"order_hurts", "order_done"} & set(m.get("events") or ()) and state.get(
        "order"
    ):
        return {"utype": None}  # his own word on what the order did
    roll = rng.random()
    if roll < args.probe_rate:
        types = probes.allowed(state)
        return ask(state, rng.choices(types, [weights[t] for t in types])[0], rng)
    if roll >= args.probe_rate + args.reply_rate:
        return {"utype": None}
    return talk(m, conv, rng, voice, args, seed, temper)


def ask(state: dict, ptype: str, rng) -> dict:
    """The partner asks about the game: a question of type ``ptype``."""
    probe = probes.make(ptype, state, rng)
    return {"utype": "probe", "player": probe["text"], "probe": probe}


def talk(m, conv, rng, voice, args, seed, temper, kind: str | None = None) -> dict:
    """The partner says something other than a question or an order, written
    by the partner's voice (``partner_says``): of ``kind``, or a follow-up
    (``--follow-rate``) or a kind of :data:`narrator_v10_prompts.UTTERANCES`
    the moment allows; a word of it misheard ``--mishear`` of the time.
    ``utype`` None if nothing usable came back; ``about``: the one of the
    cast an ``about_cast`` asks about (:func:`about_whom`)."""
    state = m["tool"]
    about = None
    if kind is not None:
        utype, how = kind, P.UTTERANCES[kind][1]
    elif len(conv) and rng.random() < args.follow_rate:
        utype, how = "followup", P.FOLLOW_UP
    else:
        kinds = utterance_kinds(m, conv)
        utype = rng.choices(kinds, [P.UTTERANCES[k][0] for k in kinds])[0]
        how = P.UTTERANCES[utype][1]
    if utype == "about_cast":
        about = about_whom(conv, rng)
        how = how.format(name=about)
    # The writer's server parses its reasoning out, so the JSON comes after it:
    # room for both, and brief reasoning.
    options = {
        ModelOption.THINKING: "low",
        ModelOption.TEMPERATURE: 1.0,
        ModelOption.MAX_NEW_TOKENS: 1600,
        ModelOption.SEED: seed,
    }
    try:
        lines = partner_says(
            voice,
            kind=utype,
            how=how,
            temper=temper,
            game_state=tool_text(state),
            conversation=checks.conversation_text(conv),
            model_options=options,
        )
    except ValueError:  # its answer was not the JSON asked for: say nothing
        return {"utype": None}
    before = [heard(ex.player) for ex in conv if ex.player]
    keep = [
        x
        for x in dict.fromkeys(str(x).strip().strip('"') for x in lines)
        if 2 <= len(x.split()) <= 20
        and not checks.NUMBER.search(x)
        and not any(checks.jaccard(heard(x), b) > 0.5 for b in before)
    ]
    if not keep:
        return {"utype": None}
    said = rng.choice(keep)
    out = {"utype": utype, "player": heard(said), "said": said, "about": about}
    if rng.random() < args.mishear:
        try:
            x = heard(mishear(voice, sentence=out["player"], model_options=options))
        except ValueError:
            x = ""
        if one_word_off(out["player"], x):
            out.update(player=x, move="misheard")
    return out


def about_whom(conv: Conversation, rng) -> str:
    """Whom the partner asks about: half the time one of the cast he named in
    his last three lines, if any; else any of them."""
    named = [
        n
        for n in CAST
        if any(re.search(rf"\b{n}\b", ex.line) for ex in conv.exchanges[-3:])
    ]
    return rng.choice(named) if named and rng.random() < 0.5 else rng.choice(list(CAST))


def branch_turn(m, conv, rng, voice, args, seed, temper, asked: Counter) -> dict:
    """What the partner says in a prompted branch (:class:`TurnFacts`
    ``branch``): a question about the game, of the types asked least so far
    in this match (``asked``, by type, branches and the conversation alike, as
    ``eval_probes.py`` balances its side questions), or, ``--branch-talk`` of
    the time, other partner talk, mostly of the kinds conversations rarely
    hold (:data:`narrator_v10_prompts.BRANCH_TALK`: about one of the cast,
    asking again)."""
    if rng.random() < args.branch_talk:
        kinds = [
            k
            for k in P.BRANCH_TALK
            if k != "clarify" or "clarify" in utterance_kinds(m, conv)
        ]
        kind = rng.choices(kinds, [P.BRANCH_TALK[k] for k in kinds])[0]
        return talk(m, conv, rng, voice, args, seed, temper, kind)
    types = probes.allowed(m["tool"])
    least = min(asked[t] for t in types)
    ptype = rng.choice([t for t in types if asked[t] == least])
    asked[ptype] += 1
    return ask(m["tool"], ptype, rng)


# ── How to take the turn ───────────────────────────────────────────────────────
MAJOR = ("died", "took_lead", "lost_lead", "close_call", "streak", "drought_ended")
MAJOR += ("order_hurts", "order_done")
# The cast as the teacher is told it (his sister, not your sister).
CAST = {
    n: (checks.third(who), checks.third(needle)) for n, who, needle in conversation.CAST
}


def bucket(turn: checks.Turn, events: list[str]) -> str:
    """What the moment is: his partner gave an order (``order``), asked about
    the game (``probe``) or said something else (``partner``); else major news
    (``major``), a frag (``frag``) or nothing new (``idle``), by the moment's
    ``events``."""
    if turn.order is not None:
        return "order"
    if turn.probe is not None:
        return "probe"
    if turn.utype:
        return "partner"
    events = set(events)
    if events & set(MAJOR):
        return "major"
    return "frag" if "frag" in events else "idle"


RIB_REST_S = 60.0  # the same person is ribbed again only this long after


def rib_target(
    turn: checks.Turn, quiet: bool, rng, resting: set[str] = frozenset()
) -> tuple[str, str] | None:
    """Whom he can rib now, and on what: one of the cast the game just named
    (the news), or, on a quiet turn, one on the scoreboard (where they stand);
    never one he ribbed in the last RIB_REST_S (``resting``)."""
    n = checks.news(turn.state)
    named = [b for b in (n["bots"] if n else ()) if b in CAST and b not in resting]
    if named:
        return named[0], P.NEWS_HINT.format(news=n["text"])
    if not quiet:
        return None
    board = [
        e
        for e in turn.state.get("scoreboard", ())
        if e["name"] in CAST and e["name"] not in resting
    ]
    if not board:
        return None
    e = rng.choice(board)
    standing = f"{ordinal(e['rank'])} with {e['frags']} frags" if "rank" in e else ""
    return e["name"], P.STANDING_HINT.format(
        name=e["name"], standing=standing or "in it"
    )


def ledger(state: dict, name: str) -> str:
    """``name``'s record against him tonight (the state's ``killed_by``), as
    the teacher is told it, in words; "" if none."""
    times = (state.get("killed_by") or {}).get(name, 0)
    if not times:
        return ""
    most = max((state.get("killed_by") or {}).values())
    return " " + P.LEDGER.format(
        name=name,
        times=P.TIMES.get(times, "again and again"),
        most=", more than anyone" if times == most and times > 1 else "",
    )


def ordinal(k: int) -> str:
    return {1: "first", 2: "second", 3: "third"}.get(k, f"{k}th")


def draw_act(b: str, turn: checks.Turn, rng, resting) -> tuple[str, tuple | None]:
    """The turn's act, from its bucket's weights, among the acts the moment
    allows (one of the cast to rib, words to echo); and the rib's target."""
    weights = P.ACT_WEIGHTS[b if b in P.ACT_WEIGHTS else "idle"]
    target = rib_target(turn, b == "idle", rng, resting)
    ok = {"rib": target is not None, "echo": bool(turn.player)}
    acts = [a for a in weights if ok.get(a, True)]
    return rng.choices(acts, [weights[a] for a in acts])[0], target


def act_hint(act: str, turn: checks.Turn, target, rng) -> str:
    """An act's hint, with what it needs: the person and what they did for a
    rib, the news for a reaction, the match's story for telling it."""
    if act in ("rib", "setup"):
        name, context = target
        who, needle = CAST[name]
        out = P.ACTS[act].hint.format(name=name, who=who, needle=needle) + " " + context
        return out + ledger(turn.state, name)
    out = P.ACTS[act].hint
    if act in ("react", "brief") and turn.topic:
        out += " " + P.NEWS_HINT.format(news=turn.topic["text"])
    if act in ("narrate", "tell", "plan"):
        parts = [turn.topic["text"]] if turn.topic else []
        stories = checks.story_options(turn.state)
        if stories:
            parts.append(rng.choice(stories)["text"])
        if parts:
            out += " " + P.GAME_STORY_HINT.format(story="; ".join(parts))
    return out


def thread_for(
    turn: checks.Turn, name: str | None, said_about: dict, rng
) -> str | None:
    """An earlier line of his about ``name`` (two or more exchanges back) to
    pick up this turn, THREAD_SHARE of the time; else None."""
    earlier = said_about.get(name or "", [])
    back = [x for x in earlier if x not in turn.prev[-1:]]
    if back and rng.random() < P.THREAD_SHARE:
        return back[-1]
    return None


def bit_turn(turn: checks.Turn, bit: dict) -> str:
    """The next step of his bit about ``bit["name"]``: ``turn.act`` is ``bit``,
    its thread the bit's last line; the teacher's hint."""
    name = bit["name"]
    who, needle = CAST[name]
    last = bit["done"] + 1 >= bit["steps"]
    n = checks.news(turn.state)
    news = bool(n and name in n["bots"])
    anchor = (P.BIT_NEWS if news else P.BIT_QUIET).format(name=name)
    stage = P.BIT_LAND if last else P.BIT_RAISE.format(anchor=anchor)
    turn.act, turn.thread = "bit", bit["last"]
    context = " " + P.NEWS_HINT.format(news=n["text"]) if news else ""
    hint = P.ACTS["bit"].hint.format(
        name=name, who=who, needle=needle, last=bit["last"], stage=stage
    )
    return hint + context + ledger(turn.state, name)


def plan_turn(
    b: str,
    turn: checks.Turn,
    rng,
    resting: set[str] = frozenset(),
    said_about: dict | None = None,
    bit: dict | None = None,
) -> tuple[str, str | None]:
    """How to take this turn: ``(the teacher's hint, the person ribbed or
    None)``; sets ``turn.act``. An order, a question or a remark of his partner
    first (then a few words more, at most, or a quick rib); else an act of the
    moment's bucket. ``resting``: the cast he ribbed too recently to again."""
    first = None
    if turn.order is not None:
        st = turn.order["status"] if turn.order["status"] in P.ORDER_HINT else "done"
        first = P.ORDER_HINT[st].format(
            told=turn.order["told"], why=turn.order.get("why")
        )
    elif turn.probe is not None:
        first = (
            P.PROBE_HINT
            + " The game state answers it: "
            + probes.answer_text(turn.probe, turn.state)
        )
    elif turn.utype == "about_cast" and turn.about in CAST:
        who, needle = CAST[turn.about]
        first = P.ABOUT_HINT.format(name=turn.about, who=who, needle=needle)
    elif turn.utype == "clarify":
        first = P.CLARIFY_HINT
    if first is not None:  # the answer or the order, then at most a quick rib
        target = rib_target(turn, False, rng, resting)
        if target and rng.random() < P.RIB_AFTER:
            name, context = target
            who, needle = CAST[name]
            turn.act = "rib"
            then = P.THEN_RIB.format(name=name, who=who, needle=needle, context=context)
            return first + then, name
        turn.alone = rng.random() < P.ALONE_SHARE
        return first + (P.ALONE if turn.alone else P.THEN), None
    if bit is not None:
        # His bit goes on, unless bigger news about someone else cuts in.
        cut = b == "major" and bit["name"] not in news_cast(turn)
        if not cut:
            hint = bit_turn(turn, bit)
            if turn.utype:
                hint = P.REPLY_HINT + " " + hint
            return hint, bit["name"]
    turn.act, target = draw_act(b, turn, rng, resting)
    if turn.act == "rib" and not (said_about or {}).get(target[0]):
        turn.act = "setup"  # the first time tonight: who they are, plainly
    hint = act_hint(turn.act, turn, target, rng)
    # Connectedness: a rib builds on his last rib of the same person (a bit).
    if turn.act == "rib":
        turn.thread = thread_for(turn, target[0], said_about or {}, rng)
        if turn.thread:
            hint += " " + P.CALLBACK.format(name=target[0], line=turn.thread)
    if turn.utype:
        hint = P.REPLY_HINT + " " + hint
    return hint, target[0] if turn.act in ("rib", "setup") else None


def news_cast(turn: checks.Turn) -> set[str]:
    """The cast the game just named (the news)."""
    n = checks.news(turn.state)
    return {b for b in (n["bots"] if n else ()) if b in CAST}


# ── The requirements of a turn ─────────────────────────────────────────────────
def requirements(turn: checks.Turn, how: str, judges, verdicts: dict) -> list:
    """The checks of a line said at ``turn``, as Mellea requirements: how to
    take the turn first (``how``, the teacher's guidance, checked by the
    judge's verdicts on it: the order taken, the answer, the act); every code
    check of :func:`checks.line_checks_v10`; and the judge's other questions
    (:func:`checks.judge_questions_v10`). The judge is asked once per line,
    every question at once (``verdicts``: each judged line's answers, so the
    requirements of one try share one call); ``true`` and ``grounded`` are
    asked again, alone, of the second judge (``judges``), and pass only if
    both say yes."""
    from mellea.stdlib.requirements import Requirement, simple_validate

    judge, second = judges
    questions = checks.judge_questions_v10(turn)
    truth = {k: questions[k] for k in ("true", "grounded")}

    def ask(line: str) -> dict:
        if line not in verdicts:
            got = checks.ask_questions(judge, turn, line, questions, v10=True)
            again = checks.ask_questions(second, turn, line, truth, v10=True)
            for k in truth:  # both must say yes
                if k in got and k in again and not again[k].yes:
                    got[k] = again[k]
            verdicts[line] = got
        return verdicts[line]

    def judged(names: list[str]):
        def fn(raw: str):
            got = ask(checks.clean(raw))
            no = [
                f"{k}: {got[k].reason}" if k in got else f"{k}: not answered"
                for k in names
                if not (k in got and got[k].yes)
            ]
            return not no, "The judge said no. " + " ".join(no)

        return simple_validate(fn)

    how_check = [
        q for q in ("takes_order", "answer", "answers", "act") if q in questions
    ]
    reqs = [Requirement(how, validation_fn=judged(how_check))]
    reqs += [
        Requirement(
            checks.rule(f),
            validation_fn=simple_validate(lambda x, f=f: f(checks.clean(x), turn)),
        )
        for f in checks.line_checks_v10(turn)
    ]
    rest = [k for k in questions if k not in how_check]
    desc = (
        "A judge must find it: " + "; ".join(checks.JUDGE_V10[k][0] for k in rest) + "."
    )
    reqs.append(Requirement(desc, validation_fn=judged(rest)))
    return reqs


# ── One match: the sessions, the narrator, the driver ──────────────────────────
def writer(args, key: tuple, **kw):
    """A session on the writer's server (gpt-oss-120b); given several
    (comma-separated), a match always uses the same one."""
    from mellea import start_session

    urls = args.writer_url.split(",")
    return start_session(
        "openai",
        model_id=args.writer_model,
        base_url=urls[seed_of(*key) % len(urls)],
        api_key="none",
        **kw,
    )


def sessions(args, key: tuple):
    """The judge and a second judge for the truth (the next judge server, if
    there are several); a match always uses the same ones."""
    from mellea import start_session

    urls = args.judge_url.split(",")
    h = seed_of(*key)
    judge, second = (
        start_session(
            "openai",
            model_id=args.judge_model,
            base_url=urls[(h + i) % len(urls)],
            api_key="none",
        )
        for i in (0, 1)
    )
    return judge, second


@dataclass
class TurnFacts:
    """What the driver knows about a turn and the student is never told:
    :meth:`Narrator.line`'s privileged parameter. They steer the teacher and
    check its line; every fact the line states is in the chat.

    The moment on the talk clock: its tick ``t``, its ``cue`` and ``events``
    (``talk.live_moments``), what happened since his last line (``moment``),
    and the order the recording's partner gave then, if any (``order``). The
    partner's turn as the driver chose it: what kind of thing they said
    (``utype``: ``probe``, ``order``, ``followup``, ...; None when they said
    nothing), the question about the game behind their words, as data
    (``probe``), the one of the cast they asked about (``about``), their
    words before speech recognition (``said``),
    ``move="misheard"`` when a word of them was misheard, and their
    temperament tonight (``temper``).

    ``branch``: the partner's words are a prompted branch, put to him off the
    same conversation for the answer alone: recorded and trained on, never
    said, so neither the conversation nor what he carries on (``self``) takes
    it in. The conversation keeps the partner's live rates, which shape his
    unprompted turns; the branches give his answers the volume accuracy
    needs."""

    t: int
    cue: str | None = None
    events: list[str] = field(default_factory=list)
    moment: list[dict] = field(default_factory=list)
    order: dict | None = None
    utype: str | None = None
    probe: dict | None = None
    about: str | None = None
    said: str | None = None
    move: str | None = None
    temper: str | None = None
    branch: bool = False


def turn_of(
    chat: ChatContext, turn: TurnFacts, plan: dict | None = None
) -> checks.Turn:
    """The turn the checks read: from the chat (as the student reads it) and
    the facts; with the plan the body noted (``act``, ``rib``, ``thread``,
    ``alone``), as its checks applied to the line."""
    _, conv, player, state = conversation.read(openai_messages(chat))
    t = checks.Turn(
        state=state,
        prev=[ex.line for ex in conv],
        past=[ex.events for ex in conv],
        player=player,
        utype=turn.utype,
        probe=turn.probe,
        order=state.get("order") if turn.utype == "order" else None,
        conversation=checks.conversation_text(conv),
        topic=checks.news(state),
        about=turn.about,
    )
    for k in ("act", "rib", "thread", "alone"):
        if plan and k in plan:
            setattr(t, k, plan[k])
    return t


class Narrator:
    """Granite in one match, as the teacher writes him: the teacher's sessions
    (the writer, in a chat; the judge and the second judge) and what he carries
    from line to line (his lines, whom he ribbed and when, his ribs of each
    person, his running bit). The student keeps none of it: it reads it back
    from the conversation (``self`` is the body's hidden state)."""

    def __init__(self, match: dict, args):
        self.match, self.args = match, args
        self.key = (match["data"], match["ep"])
        # In a chat, so a failed line stays for the repair; brief reasoning first.
        self.teacher = writer(
            args,
            self.key,
            ctx=ChatContext(),
            model_options={
                ModelOption.THINKING: "low",
                ModelOption.TEMPERATURE: args.temperature,
                ModelOption.MAX_NEW_TOKENS: args.max_tokens,
            },
        )
        self.judges = sessions(args, self.key)
        self.prev: list[str] = []  # his lines this match
        self.ribbed: Counter = Counter()  # the times he has ribbed each of the cast
        self.ribbed_at: dict[str, int] = {}  # when he last did
        self.said_about: dict[str, list[str]] = defaultdict(list)  # his passed ribs
        # His bit going: {"name", "steps", "done", "last", "t"}.
        self.bit: dict | None = None
        self.gap = 0  # turns before the next bit may start

    @compilable(privileged=("turn",))
    def line(self, chat: ChatContext, *, turn: TurnFacts) -> str:
        """Granite's next line in this conversation."""
        t0 = time.time()
        seed = seed_of(*self.key, turn.t, self.args.seed, *("branch",) * turn.branch)
        rng = random.Random(seed)
        # The turn, read from the chat as the student reads it, and the facts.
        _, conv, _, state = conversation.read(openai_messages(chat))
        t = turn_of(chat, turn)
        b = bucket(t, turn.events)
        resting = {
            n for n, at in self.ribbed_at.items() if (turn.t - at) / TIC_HZ < RIB_REST_S
        }
        if self.bit and (turn.t - self.bit["t"]) / TIC_HZ > P.BIT_MAX_S:
            self.bit = None  # too long: dropped
        bit = dict(self.bit, t=None) if self.bit else None
        how, rib = plan_turn(b, t, rng, resting, self.said_about, self.bit)
        t.rib = rib
        # The teacher writes it; the requirements judge each try.
        verdicts: dict = {}
        self.teacher.reset()
        with TraceRecorder() as rec:
            narrator_turn(
                self.teacher,
                chat=chat,
                requirements=requirements(t, how, self.judges, verdicts),
                strategy=MultiTurnStrategy(loop_budget=self.args.loop_budget),
                model_options={ModelOption.SEED: seed},
            )
        trace = rec.traces[-1]
        text = checks.clean(trace.output or "")  # as the checks read it
        line = " ".join(text.split())  # as the conversation will hold it
        ok = trace.success and bool(line)
        failed = [n for n, _ in checks.failures_v10(text, t)]
        got = verdicts.get(text, {})
        failed += [f"judge: {k}" for k, v in got.items() if not v.yes]
        annotate(
            **{k: self.match[k] for k in ("data", "style", "ep")},
            t=turn.t,
            cue=turn.cue,
            events=turn.events,
            order=turn.order,
            system=SYSTEM,
            idle_s=self.args.idle_s,
            kind=t.kind,
            utype=t.utype,
            player=t.player,
            said=turn.said,
            move=turn.move,
            probe=t.probe,
            about=turn.about,
            bucket=b,
            act=t.act,
            rib=rib,
            ribbed_before=self.ribbed[rib] if rib else None,
            thread=t.thread,
            alone=t.alone,
            bit=bit,
            temper=turn.temper,
            topic=t.topic,
            how=how,
            line=line,
            failed=failed,
            judge={k: v.yes for k, v in got.items()},
            conv=conv.to_json(),  # what the narrator reads before now
            tool=state,  # the game state he answers from
            moment=turn.moment,  # what this exchange keeps of the moment
            prev=self.prev[-checks.RECENT :],
            branch=turn.branch,
            s=round(time.time() - t0, 1),
        )
        if not turn.branch:  # a branch is never said: nothing to carry on
            self.after(t, rib, line, ok, turn.t, rng)
        print(
            f"  {self.match['ep']} {turn.t / TIC_HZ:6.1f}s {b:<7} {t.act} "
            f"{'branch ' * turn.branch}"
            f"{'ok' if ok else 'NO'} {len(trace.attempts)} {time.time() - t0:4.1f}s "
            f"{line[:120]!r}",
            flush=True,
        )
        return line

    def after(self, t: checks.Turn, rib, line: str, ok: bool, tick: int, rng) -> None:
        """What he carries on after saying ``line`` (``ok``: it passed): whom
        he ribbed and when, his ribs by person, the bit's next step, his
        lines."""
        if rib and ok:
            self.ribbed[rib] += 1
            self.ribbed_at[rib] = tick
            self.said_about[rib].append(line)  # a running bit: his ribs, in order
        if t.act == "bit" and ok:  # a step of the bit, or its landing
            self.bit["done"] += 1
            self.bit["last"] = line
            if self.bit["done"] >= self.bit["steps"]:
                self.bit, self.gap = None, rng.choice(P.BIT_GAP)
        elif t.act == "bit":  # a step that failed: after BIT_FAILS, dropped
            self.bit["fails"] = self.bit.get("fails", 0) + 1
            if self.bit["fails"] >= P.BIT_FAILS:
                self.bit, self.gap = None, rng.choice(P.BIT_GAP)
        elif t.act in ("rib", "setup") and ok and self.bit is None and self.gap <= 0:
            self.bit = {
                "name": rib,
                "steps": rng.choice(P.BIT_STEPS),
                "done": 1,
                "last": line,
                "t": tick,
            }
        elif self.bit is None and t.act not in (None, "bit"):
            self.gap -= 1
        if line:  # passed or not: the next moment follows it
            self.prev.append(line)


def score(record: Record, completion: Completion, episode: list[Record]) -> Score:
    """A completion's score. The line the call returned: kept if the teacher's
    own loop passed it (its trace, the body's IVR verdict), its parts the
    judge's verdicts and the checks it failed (the body's notes). Any other
    (:func:`alternatives`): from the call and the line alone, every code check
    of the turn (:func:`checks.failures_v10`), as the body applied them."""
    if completion is record.completions[0]:
        return Score(
            parts={"judge": record.meta["judge"], "failed": record.meta["failed"]},
            keep=record.inner[-1].success,
        )
    t = turn_of(record.args["chat"], record.privileged["turn"], record.meta)
    failed = [n for n, _ in checks.failures_v10(completion.text(), t)]
    return Score(parts={"failed": failed}, keep=not failed)


def row_of(record: Record) -> dict:
    """A record as ``train_alora.py``'s narrator row: the body's notes (the
    match, the moment, the turn, the line), whether the line was kept
    (``ok``) and every try; beside them, the record's standard fields (the
    ``prompt`` the student reads, its ``completions``: the line, then any
    :func:`alternatives`, each scored), and the line's own completion and
    score again at the top, as the readers of these rows read them."""
    trace = record.inner[-1]
    said = record.completions[0]
    return {
        **record.meta,
        "ok": said.score.keep,
        "attempts": len(trace.attempts),
        "tries": [checks.clean(a.text) for a in trace.attempts],
        "trace": {
            "function": trace.function,
            "guidance": trace.guidance,
            "attempts": [
                {
                    "text": a.text,
                    "failed": [v.requirement for v in a.verdicts if not v.passed],
                    "reasons": [v.reason for v in a.verdicts if not v.passed],
                }
                for a in trace.attempts
            ],
        },
        "function": record.function,
        "interface": record.interface,
        "view": record.view,
        "call_id": record.call_id,
        "prompt": record.prompt,
        "completion": said.text(),
        "score": dataclasses.asdict(said.score),
        "completions": [c.to_json() for c in record.completions],
    }


def episode(match: dict, entries: list[str], rows: list[dict], args, weights: dict):
    """One match, in order, over its live talk clock: at each moment the
    partner may speak (:func:`partner_turn`), a prompted branch may be put
    to him off the conversation as it stands (:func:`branch_turn`), then
    Granite's line (:meth:`Narrator.line`); the next moment follows the line
    he said, passed or not, and how long it took to say."""
    import talk

    key = (match["data"], match["ep"])
    narrator = Narrator(match, args)
    voice = writer(args, key)  # the partner's voice
    moments = talk.live_moments(entries, rows, idle_s=args.idle_s, style=match["style"])
    m = next(moments, None)
    temper = random.Random(seed_of(*key, args.seed)).choice(P.TEMPERS)
    said: list[Exchange] = []  # this stretch's exchanges
    pending: list[dict] = []  # what happened since his last line
    asked: Counter = Counter()  # questions about the game so far, by type
    last_t = None
    while m is not None:
        if new_stretch(last_t, m["t"]):
            said, pending = [], []
        last_t = m["t"]
        seed = seed_of(*key, m["t"], args.seed)
        rng = random.Random(seed_of(*key, m["t"], args.seed, "partner"))
        pending = pending + list(m.get("moment") or [])
        conv = Conversation(said)
        p = partner_turn(m, conv, rng, voice, args, weights, seed, temper)
        if p.get("probe"):
            asked[p["probe"]["type"]] += 1
        player = p.get("player")
        facts = TurnFacts(
            t=m["t"],
            cue=m.get("cue"),
            events=list(m.get("events") or ()),
            moment=pending,
            order=m.get("order"),
            utype=p["utype"],
            probe=p.get("probe"),
            about=p.get("about"),
            said=p.get("said"),
            move=p.get("move"),
            temper=temper,
        )
        # A prompted branch, put to him off the same conversation, as it
        # stands before his line: his answer is recorded, never said.
        bseed = seed_of(*key, m["t"], args.seed, "branch")
        brng = random.Random(bseed)
        if brng.random() < args.branch_rate:
            q = branch_turn(m, conv, brng, voice, args, bseed, temper, asked)
            if q["utype"]:
                bfacts = dataclasses.replace(
                    facts,
                    order=None,
                    utype=q["utype"],
                    probe=q.get("probe"),
                    about=q.get("about"),
                    said=q.get("said"),
                    move=q.get("move"),
                    branch=True,
                )
                narrator.line(chat_of(conv, m["tool"], q["player"]), turn=bfacts)
        line = narrator.line(chat_of(conv, m["tool"], player), turn=facts)
        if line:  # what he said, passed or not: the next moment follows it
            said.append(Exchange(m["t"], pending, player, line))
            pending = []
        try:
            m = moments.send((line or None, bool(player)))
        except StopIteration:
            m = None


CAST_TASK = """{persona}

Tonight Granite plays a Doom deathmatch the way a crowd plays laser tag: every
bot in the match is named after someone he knows, family and old friends and
old crew, so the whole night is him ribbing people he has known for years. List
{n} of those people. For each: a first name or nickname of one or two
syllables, who they are to him, and the one thing about them he can needle them
about all night: a habit, a sore spot, an old grudge between them, said so that
a stranger gets it at once. Every name different in sound and first letter;
every person different from the others; ordinary people, not famous ones;
mild; never a character or a famous bit from any film.

Write them in this form and nothing else:
- <Name>: <who they are to him>; <what he needles them about>
..."""


def write_cast(args) -> None:
    """``cast``: candidates for the people the bots are named after, by the
    storyteller (``--batches`` requests of ``--per-batch``), the names
    deduplicated; each judged (:data:`checks.JUDGE_CAST`). A person picks the
    seven."""
    from mellea import start_session

    writer = start_session(
        "openai",
        model_id=args.storyteller_model,
        base_url=args.storyteller_url,
        api_key="none",
        model_options={
            ModelOption.TEMPERATURE: args.temperature,
            "extra_body": {"chat_template_kwargs": {"reasoning_effort": 0.5}},
        },
    )
    judge = start_session(
        "openai", model_id=args.judge_model, base_url=args.judge_url, api_key="none"
    )
    rows, names = [], set()
    for b in range(args.batches):
        text = writer.instruct(
            CAST_TASK.format(persona=P.STORYTELLER_PERSONA, n=args.per_batch),
            model_options={
                ModelOption.SEED: seed_of("cast", b, args.seed),
                ModelOption.MAX_NEW_TOKENS: 12000,
            },
        )
        for x in parse_list(str(text)):
            name, _, rest = x.partition(":")
            name = name.strip()
            if not rest or not name or name.lower() in names or len(name.split()) > 2:
                continue
            names.add(name.lower())
            v = checks.ask_cast(judge, x)
            ok = set(v) == set(checks.JUDGE_CAST) and all(a.yes for a in v.values())
            rows.append(
                {
                    "name": name,
                    "line": rest.strip(),
                    "ok": ok,
                    "batch": b,
                    "judge": {k: [a.yes, a.reason] for k, a in v.items()},
                }
            )
            print(f"  {'ok' if ok else 'NO'} {name}: {rest.strip()}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{sum(r['ok'] for r in rows)} of {len(rows)} people passed -> {args.out}")


def parse_list(text: str) -> list[str]:
    """The items of a list in the storyteller's text (``- <item>`` lines)."""
    out = []
    for raw in re.sub(r"<\|[a-z_]+\|>", "", text or "").splitlines():
        x = raw.strip()
        if x.startswith(("- ", "* ")) and 3 <= len(x[2:].split()) <= 25:
            out.append(x[2:].strip().rstrip("."))
    return out


def load_matches(moments: Path, limit: int = 0) -> list[dict]:
    """The matches to write (the moments file's split: its data dirs and
    episodes), without their precomputed moments' bulk."""
    out = []
    for x in open(moments):
        r = json.loads(x)
        out.append(
            {k: r[k] for k in ("data", "style", "ep")} | {"moments": r["moments"]}
        )
        if limit and len(out) >= limit:
            break
    return out


def cast_names(names: list[str], board: list[str] = ()) -> dict[str, str]:
    """A recorded match's bots renamed after the cast: each to the cast member
    at its place in its own tier (doom_env.BOT_SETS), or, taken, the next one
    free. With more names than the cast has people (a partner's order naming a
    bot not in the match, beside its seven), only the match's own (``board``,
    its scoreboard) are renamed: a name heard only in the partner's words
    stays as said, someone who is not there."""
    import doom_env

    cast = list(doom_env.BOT_SETS["cast"])
    if len(names) > len(cast):
        names = [n for n in names if n in board]
    place = {n: i for tier in doom_env.BOT_SETS.values() for i, n in enumerate(tier)}
    out, free = {}, list(cast)
    for n in sorted(names, key=lambda n: (place.get(n, 0), n)):
        want = cast[place.get(n, 0) % len(cast)]
        out[n] = want if want in free else free[0]
        free.remove(out[n])
    return out


def renamed(text: str, names: dict[str, str]) -> str:
    """``text`` with each bot's name replaced, as written and in lower case
    (the partner's recorded words)."""
    if not names:
        return text
    both = {**names, **{k.lower(): v.lower() for k, v in names.items()}}
    rx = re.compile(r"\b(" + "|".join(map(re.escape, both)) + r")\b")
    return rx.sub(lambda m: both[m.group(1)], text)


ORDER_KEEP = 1 / 3  # of the orders the recording's partner gave, the ones he says


def drop_orders(rows: list[dict], keep: set[int]) -> None:
    """A recorded match with only the orders numbered ``keep`` (1 for the
    first): the others leave the facts (the state's order, its events), so
    the narrator is never told them; the game still played them out."""
    told = sorted(
        {
            (e["tick"], e["i"])
            for r in rows
            for e in r["facts"]["news"]
            if e["kind"] == "order"
        }
    )
    ticks = [t for t, _ in told]

    def number(e: dict) -> int:  # which order an order event belongs to
        return sum(t <= e["tick"] for t in ticks)

    for r in rows:
        f = r["facts"]
        f["news"] = [
            e
            for e in f["news"]
            if not e["kind"].startswith("order") or number(e) in keep
        ]
        if f.get("order") and f["order"]["n"] not in keep:
            f.pop("order")


def raw_rows(matches: list[dict], seed: int = 0) -> dict:
    """Each match's collect.py rows and log entries, read once per data dir,
    its bots renamed after the cast (:func:`cast_names`), and only
    ORDER_KEEP of its orders told (:func:`drop_orders`, seeded by the match)."""
    want = defaultdict(set)
    for m in matches:
        want[(m["data"], m["style"])].add(m["ep"])
    got: dict = {}
    old = [n for n in probes.BOT_NAMES if n not in CAST]
    for (d, style), eps in want.items():
        lines: dict = defaultdict(list)
        for x in open(Path(d) / f"{style}.jsonl"):
            ep = json.loads(x)["ep"]
            if ep in eps:
                lines[ep].append(x)
        for x in open(Path(d) / f"{style}_history.jsonl"):
            h = json.loads(x)
            if h["ep"] not in eps:
                continue
            text = "".join(lines[h["ep"]])  # the names are in its rows' facts
            board = [n for n, _ in json.loads(lines[h["ep"]][0])["facts"]["board"]]
            found = sorted(n for n in old if re.search(rf"\b{n}\b", text))
            names = cast_names(found, board)
            entries = [renamed(e, names) for e in h["entries"]]
            rows = [json.loads(renamed(x, names)) for x in lines[h["ep"]]]
            rng = random.Random(seed_of(d, h["ep"], "orders", seed))
            n = max(
                ((r["facts"].get("order") or {}).get("n", 0) for r in rows), default=0
            )
            drop_orders(rows, {k for k in range(1, n + 1) if rng.random() < ORDER_KEEP})
            got[(d, h["ep"])] = (entries, rows)
    return got


def write_file(args) -> None:
    """``write``: this shard's matches, whole, in order, by Mellea's
    ``emulate``. Resumable: a match already written is skipped; one cut short
    is dropped and written again."""
    from mellea.stdlib.compile import emulate

    matches = load_matches(args.moments, args.matches)
    weights = probes.weights([m for mt in matches for m in mt["moments"]])
    k, n = map(int, args.shard.split("/"))
    mine = matches[k::n]
    raw = raw_rows(mine, args.seed)
    print(f"{len(mine)} matches, their rows read", flush=True)

    def run(match):
        entries, rows = raw[(match["data"], match["ep"])]
        return lambda: episode(match, entries, rows, args, weights)

    settings = ("seed", "loop_budget", "idle_s", "branch_rate", "branch_talk")
    settings += ("probe_rate", "reply_rate", "follow_rate", "mishear", "temperature")
    summary = emulate(
        [(json.dumps([m["data"], m["ep"]]), run(m)) for m in mine],
        args.out,
        score=score,
        alternatives=alternatives,
        record=row_of,
        meta={"code": code_version(), **{k: getattr(args, k) for k in settings}},
        log=lambda s: print(s, flush=True),
    )
    print(f"done: {summary}")
    if summary["failed"]:
        raise SystemExit(f"{len(summary['failed'])} matches failed; run again")


# ── Other completions of a call: his mistakes, made in code ───────────────────
def named_cast(line: str) -> list[str]:
    return [n for n in CAST if re.search(rf"\b{n}\b", line)]


def alternatives(record: Record, episode: list[Record]) -> list[Completion]:
    """Other completions of a call, for preferences against the line he said
    (``emulate``'s ``alternatives``): the mistakes a live session showed, each
    made by swapping one name in his line (or by saying an earlier line
    again), and kept only if the check it targets now fails and passed
    before. Their ``meta``: ``made`` (which mistake), ``why`` (the check's
    word). Seeded from the call.

    * ``person``: a line on one of the cast's own thing, given to another of
      them (Milo's case of beer, said of Nora): :func:`checks.right_person`;
    * ``news``: a line naming a bot the game just told him about, naming
      instead one of the cast it has said nothing about lately (Hank fragged,
      when it was Lou): :func:`checks.stale_bot`;
    * ``intro``: his introduction of someone he has already introduced in this
      conversation, said again (Milo's three times in 40 seconds)."""
    said = record.completions[0]
    if not (record.meta and record.inner and record.inner[-1].success):
        return []  # only against a line his own loop passed
    line = said.text()
    t = turn_of(record.args["chat"], record.privileged["turn"], record.meta)
    rng = random.Random(seed_of(record.call_id, "alternatives"))
    swap = lambda a, b: re.sub(rf"\b{a}\b", b, line)  # noqa: E731
    out = []
    named = named_cast(line)
    if len(named) == 1 and checks.stems(line) & checks._cast_words()[named[0]]:
        bad = swap(named[0], rng.choice([m for m in CAST if m != named[0]]))
        if checks.right_person(line, t)[0] and not checks.right_person(bad, t)[0]:
            out.append(
                Completion(
                    bad, {"made": "person", "why": checks.right_person(bad, t)[1]}
                )
            )
    known = checks.known_bots(t)
    news = sorted(set(named) & known)
    if news and checks.stale_bot(line, t)[0]:
        quiet = [
            m
            for m in probes.board(t.state)
            if m in CAST and m not in known | set(named)
        ]
        if quiet:
            bad = swap(rng.choice(news), rng.choice(quiet))
            if not checks.stale_bot(bad, t)[0]:
                out.append(
                    Completion(
                        bad, {"made": "news", "why": checks.stale_bot(bad, t)[1]}
                    )
                )
    rib = record.meta.get("rib")
    if rib and record.meta.get("act") != "setup":
        before = {
            m["content"]
            for m in record.prompt["messages"]
            if m["role"] == "assistant" and m.get("content")
        }
        intro = next(
            (
                r.completions[0].text()
                for r in episode
                if r.call < record.call
                and r.meta.get("act") == "setup"
                and r.meta.get("rib") == rib
                and r.completions[0].text() in before
            ),
            None,
        )
        if intro and intro != line:
            why = f"He introduced {rib} already, earlier in this conversation."
            out.append(Completion(intro, {"made": "intro", "why": why}))
    return out


def code_version() -> dict:
    """The code that writes the data: ``VERSION`` beside this script (the
    commits it was copied from, written by whoever copied it), else the
    commits of this checkout and of the Mellea it imports, ``+dirty`` if
    either has changes not committed."""
    import subprocess

    import mellea

    here = Path(__file__).resolve().parent
    if (here / "VERSION").exists():
        return json.loads((here / "VERSION").read_text())

    def rev(where: Path) -> str | None:
        def git(*a):
            return subprocess.run(
                ["git", *a], cwd=where, capture_output=True, text=True
            ).stdout.strip()

        head = git("rev-parse", "HEAD")
        return head and head + ("+dirty" if git("status", "--porcelain") else "")

    return {
        "granite-switch": rev(here),
        "mellea": rev(Path(mellea.__file__).resolve().parent),
    }


# ── Lines written elsewhere: the same requirements, once ───────────────────────
def check_file(args) -> None:
    """``check``: lines written elsewhere (``--keys``: the columns holding them,
    e.g. a narrator's held-out samples) asked the code checks and the judge's
    questions once, without repair, in the turn each row records."""
    from mellea import start_session

    rows = [json.loads(x) for p in args.rows for x in open(p)]
    rows = [r for r in rows if r.get("tool")]
    if args.limit:
        rows = random.Random(0).sample(rows, min(args.limit, len(rows)))
    judge = start_session(
        "openai",
        model_id=args.judge_model,
        base_url=args.judge_url.split(",")[0],
        api_key="none",
    )
    keys = args.keys.split(",")
    fails = {k: Counter() for k in keys}
    out = []
    for i, r in enumerate(rows):
        t = checks.turn_of(r)
        t.conversation = checks.conversation_text(
            Conversation.from_json(r.get("conv") or [])
        )
        t.topic = checks.news(t.state)
        got = {}
        for k in keys:
            line = checks.clean(r.get(k) or "")
            qs = checks.judge_questions_v10(t)
            v = checks.ask_questions(judge, t, line, qs, v10=True)
            names = [n for n, _ in checks.failures_v10(line, t)]
            names += [f"judge: {q}" for q in qs if not (q in v and v[q].yes)]
            fails[k].update(names)
            fails[k]["(any)"] += bool(names)
            got[k] = {"failed": names, "judge": {q: a.yes for q, a in v.items()}}
        out.append({**r, "checked": got})
        if (i + 1) % 25 == 0:
            print(f"{i + 1} of {len(rows)} rows", flush=True)
    if args.out:
        args.out.write_text("".join(json.dumps(r) + "\n" for r in out))
    n = max(1, len(rows))
    print(f"| failed (of {len(rows)}) | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for x in sorted({x for c in fails.values() for x in c}):
        print(
            f"| {x} | "
            + " | ".join(f"{100 * fails[k][x] / n:.1f}%" for k in keys)
            + " |"
        )


# ── The report ─────────────────────────────────────────────────────────────────
def report(args) -> None:
    """``report``: how many passed and after how many tries, by act and by
    bucket; what the lines that did not pass failed; and style_eval.py's code
    metrics on the lines that passed."""
    import style_eval

    rows = [json.loads(x) for p in args.rows for x in open(p)]
    print(f"{len(rows)} lines, {sum(r['ok'] for r in rows)} passed")
    said = [r for r in rows if not r.get("branch")]
    print(
        f"of them, prompted branches (never said): {len(rows) - len(said)}, "
        f"{sum(r['ok'] for r in rows if r.get('branch'))} passed"
    )
    for col in ("act", "bucket", "utype"):
        by = defaultdict(list)
        for r in rows:
            by[r.get(col) or "-"].append(r)
        print(f"\n| {col} | lines | passed | 1st try | 2nd | 3rd | seconds per line |")
        print("|---|---|---|---|---|---|---|")
        for kind, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
            n = len(rs)
            tries = Counter(r["attempts"] for r in rs if r["ok"])
            print(
                f"| {kind} | {n} | {100 * sum(r['ok'] for r in rs) / n:.0f}% | "
                + " | ".join(f"{100 * tries[a] / n:.0f}%" for a in (1, 2, 3))
                + f" | {sum(r['s'] for r in rs) / n:.1f} |"
            )
    failed = Counter(x for r in rows if not r["ok"] for x in r["failed"])
    print("\nWhat the lines that did not pass failed:")
    for x, c in failed.most_common(25):
        print(f"  {x}: {c}")
    probes_ = [r for r in rows if r["utype"] == "probe"]
    print(
        f"\npassed within the tries: {100 * sum(r['ok'] for r in rows) / max(1, len(rows)):.0f}%"
        f" (gate 65%); probes {100 * sum(r['ok'] for r in probes_) / max(1, len(probes_)):.0f}%"
        " (gate 85%)"
    )
    # His persona: the lines he said, branches apart.
    own = [r for r in said if not r.get("player")]
    told = [r for r in own if r.get("thread")]
    print(
        f"his own turns (the partner silent): {100 * len(own) / max(1, len(said)):.0f}%;"
        f" of those, given a thread to pick up: {100 * len(told) / max(1, len(own)):.0f}%,"
        f" passed {100 * sum(r['ok'] for r in told) / max(1, len(told)):.0f}%"
    )
    ribs = Counter(r["rib"] for r in said if r.get("rib"))
    print(
        f"turns ribbing one of the cast: {100 * sum(ribs.values()) / max(1, len(said)):.0f}%"
        f" ({', '.join(f'{k} {v}' for k, v in ribs.most_common())})"
    )
    print()
    style_eval.print_metrics(
        {"rows": style_eval.lines_of_rows([r for r in said if r["ok"]])}
    )


# ── The student's prompt against the narrator's live prompt ────────────────────
def parity(args) -> None:
    """``parity``: on ``--n`` rows, what the student is trained on against
    what it is served (:func:`conversation.narrator_text`, the prompt
    ``doom_live.py`` builds), byte for byte.

    * A record (this script's rows): the chat template on its ``prompt`` (the
      compiled view, as written) against ``narrator_text`` from the same row's
      ``conv``, ``tool``, ``player`` and ``system``; then ``train_alora.py``'s
      record loader against its row loader on the same rows: the same prompt
      and target token ids.
    * A row without a prompt (round 9's): the compiled view of its chat, for
      both system prompts."""
    import tempfile

    import train_alora
    from conversation import narrator_text
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = [json.loads(x) for p in args.rows for x in open(p)]
    rows = [r for r in rows if r.get("tool")]
    rows = random.Random(0).sample(rows, min(args.n, len(rows)))
    bad = checked = 0
    for r in rows:
        conv = Conversation.from_json(r.get("conv") or [])
        if "prompt" in r:
            views = {r["system"]: r["prompt"]}
        else:
            views = {
                key: compiled_view(
                    Narrator.line,
                    {
                        "chat": ChatContext.from_openai(
                            messages(conv, r["tool"], r.get("player"), system),
                            tools=[GET_GAME_STATE],
                        )
                    },
                )
                for key, system in NARRATOR_SYSTEM_PROMPTS.items()
            }
        for key, view in views.items():
            got = tok.apply_chat_template(
                view["messages"],
                tools=view["tools"],
                add_generation_prompt=True,
                tokenize=False,
            )
            system = NARRATOR_SYSTEM_PROMPTS[key]
            want = narrator_text(tok, conv, r["tool"], r.get("player"), system=system)
            checked += 1
            if got != want:
                bad += 1
                print(f"MISMATCH ({key}) at {r['data']}:{r['ep']} t={r['t']}")
    print(f"{len(rows)} rows, {checked} prompts: {bad} mismatches")
    records = [r for r in rows if "prompt" in r]
    if records:
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
            f.flush()
            new = train_alora.load_record_rows([Path(f.name)], tok)
            old = train_alora.load_narration_rows([Path(f.name)], tok)
        same = [(a[2], a[3]) == (b[2], b[3]) for a, b in zip(new, old)]
        diff = len(new) != len(old) or not all(same)
        print(
            f"training ids: {len(new)} records kept, {len(old)} rows passed, "
            f"{len(same) - sum(same)} differ"
        )
        bad += diff
    if bad:
        raise SystemExit(1)


# ── Checks of the prompts ──────────────────────────────────────────────────────
def selfcheck(args) -> None:
    """``selfcheck``: every act has a hint and a gloss and every bucket's acts
    exist; the cast is the bots' cast tier, without numbers; a recorded match
    renames to it; a turn's plan and the judge's questions come out as they
    should, for a rib and for an answer."""
    bad = []
    st = checks._STATE
    for b, w in P.ACT_WEIGHTS.items():
        unknown = [a for a in w if a not in P.ACTS]
        if unknown:
            bad.append(f"bucket {b}: unknown acts {unknown}")
    import doom_env

    if tuple(CAST) != doom_env.BOT_SETS["cast"]:
        bad.append("conversation.CAST and doom_env.BOT_SETS['cast'] differ")
    for name, (who, needle) in CAST.items():
        if checks.NUMBER.search(needle):
            bad.append(f"{name}: a number in {needle!r}")
    names = cast_names(["Rambo", "Rookie", "Viper", "Leone"])
    assert len(set(names.values())) == 4 and names["Rambo"] == "Gwen", names
    nato = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf"]
    names = cast_names(sorted([*nato, "Machete"]), nato)  # "hunt Machete", not there
    assert sorted(names) == nato and len(set(names.values())) == 7, names
    assert renamed("Rambo killed you; go after rambo", {"Rambo": "Gwen"}) == (
        "Gwen killed you; go after gwen"
    )
    st = dict(
        st, recent_events=[{"type": "death", "time": st["time"], "killer": "Milo"}]
    )
    st["last_death"] = {"killer": "Milo", "seconds_ago": 0, "killer_weapon": "shotgun"}
    t = checks.Turn(state=st, topic=checks.news(st))
    rng = random.Random(0)
    for _ in range(20):
        how, rib = plan_turn("major", t, rng)
        if rib:
            break
    assert rib == "Milo" and "case of beer" in how, (how, rib)
    q = checks.judge_questions_v10(t)
    assert {"act", "voice", "original", "consistent"} <= set(q), q
    probe = {"type": "frags", "text": "how many kills do you have"}
    t = checks.Turn(state=st, player=probe["text"], utype="probe", probe=probe)
    how, rib = plan_turn("probe", t, random.Random(1))
    assert how.startswith(P.PROBE_HINT) and rib in (None, "Milo"), how
    assert rib is None or ("your" not in how and "owes him" in how), how
    bit = {
        "name": "Ted",
        "steps": 3,
        "done": 1,
        "last": "Ted, store-bought pie.",
        "t": 0,
    }
    t = checks.Turn(state=checks._STATE)
    how, rib = plan_turn("idle", t, random.Random(0), bit=bit)
    assert t.act == "bit" and t.thread == bit["last"] and rib == "Ted", (how, t)
    assert "pies" in how and "Raise it" in how, how
    names = [f.__name__ for f in checks.line_checks_v10(t)]
    assert "builds_on" in names and "fresh_rib" not in names, names
    rows = [
        {"facts": {"news": [{"kind": "order", "tick": 10, "i": 1}], "order": {"n": 1}}},
        {
            "facts": {
                "news": [
                    {"kind": "order_end", "tick": 20, "i": 2},
                    {"kind": "order", "tick": 30, "i": 3},
                ],
                "order": {"n": 2},
            }
        },
    ]
    drop_orders(rows, {2})
    assert "order" not in rows[0]["facts"] and rows[0]["facts"]["news"] == [], rows
    assert [e["i"] for e in rows[1]["facts"]["news"]] == [3], rows
    t = checks.Turn(state=st, prev=["one", "two"])
    line = thread_for(t, "Milo", {"Milo": ["Milo owes me a case."]}, random.Random(3))
    assert line in (None, "Milo owes me a case."), line
    # Asked about one of the cast: the answer first, from who they are.
    t = checks.Turn(
        state=st, player="who is ted to you", utype="about_cast", about="Ted"
    )
    how, _ = plan_turn("partner", t, random.Random(0))
    assert how.startswith("His partner asked about Ted") and "pies" in how, how
    quiet = Conversation([Exchange(70, [], None, "Milo, again.")])
    assert "clarify" not in utterance_kinds({"events": []}, quiet)
    asked = Conversation([Exchange(70, [], "who got you", "Gwen did.")])
    assert "clarify" in utterance_kinds({"events": []}, asked)
    # Another completion of a call: Milo's thing given to someone else, which
    # the scorer, reading the call and the line alone, does not keep.
    from mellea.stdlib.compile.trace import Trace

    chat = chat_of(Conversation(), checks._STATE, None)
    said = Completion("Milo still owes me that case of beer.")
    rec = Record(
        function="Narrator.line",
        interface=Narrator.line.interface,
        view=1,
        episode="e",
        call=0,
        call_id="e#0",
        parent=None,
        args={"chat": chat},
        privileged={"turn": TurnFacts(t=0)},
        prompt=compiled_view(Narrator.line, {"chat": chat}),
        completions=[said],
        inner=[Trace("narrator_turn", {}, {}, [], said.value, True)],
        meta={"act": "react"},
    )
    alts = alternatives(rec, [rec])
    assert [c.meta["made"] for c in alts] == ["person"], alts
    assert not score(rec, alts[0], [rec]).keep, alts[0]
    # The teacher reads the turn back from the chat the student reads.
    conv = Conversation(
        [
            Exchange(70, [{"type": "frag", "victim": "Milo"}], None, "Milo, again."),
            Exchange(210, [], "who got you", "Gwen did."),
        ]
    )
    msgs = messages(conv, st, "nice one", NARRATOR_SYSTEM_PROMPTS[SYSTEM])
    system, back, player, state = conversation.read(msgs)
    assert messages(back, state, player, system) == msgs, back.to_json()
    assert (player, state) == ("nice one", st) and len(back) == 2, back.to_json()
    # The student's view: the chat, never the turn's facts.
    facts = TurnFacts(t=210, utype="probe", probe={"type": "killer_now"}, said="who")
    view = compiled_view(
        Narrator.line, {"chat": chat_of(conv, st, None), "turn": facts}
    )
    assert view["tools"] == [GET_GAME_STATE] and "killer_now" not in json.dumps(view)
    assert Narrator.line.spec.privileged == ("turn",), Narrator.line.spec
    if bad:
        raise SystemExit("selfcheck failed:\n  " + "\n  ".join(bad))
    print(f"OK: {len(P.ACTS)} acts, {len(CAST)} of the cast: {', '.join(CAST)}")


def main() -> None:
    from mellea.core import MelleaLogger

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="Write his turns at every moment of the matches")
    c = sub.add_parser("check", help="Lines written elsewhere: every requirement, once")
    r = sub.add_parser("report", help="What written rows hold")
    pa = sub.add_parser("parity", help="The student's rendering against the prompt")
    sub.add_parser("selfcheck", help="The prompts keep their own rules")
    ca = sub.add_parser("cast", help="People to name the bots after, by an open model")
    ca.add_argument("--out", type=Path, required=True)
    ca.add_argument("--storyteller-url", required=True)
    ca.add_argument("--storyteller-model", default="inkling-small")
    ca.add_argument("--batches", type=int, default=3)
    ca.add_argument("--per-batch", type=int, default=10)
    ca.add_argument("--temperature", type=float, default=1.0)
    ca.add_argument("--seed", type=int, default=0)
    w.add_argument(
        "--moments", type=Path, required=True, help="talk.py moments (the split)"
    )
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--shard", default="0/1", help="K/N: every N-th match from the K-th")
    w.add_argument("--matches", type=int, default=0, help="The first so many (0: all)")
    w.add_argument("--writer-url", required=True, help="Several, comma-separated")
    w.add_argument("--writer-model", default="gpt-oss-120b")
    # Most of the time the person watching says nothing: he carries the talk.
    w.add_argument(
        "--probe-rate", type=float, default=0.12, help="Questions (probes.py)"
    )
    w.add_argument("--reply-rate", type=float, default=0.08, help="Other partner talk")
    w.add_argument(
        "--follow-rate", type=float, default=0.45, help="Of those, follow-ups"
    )
    w.add_argument(
        "--mishear", type=float, default=0.12, help="Of those, a word misheard"
    )
    w.add_argument(
        "--idle-s", type=float, default=6.0, help="Silence before he goes on talking"
    )
    w.add_argument(
        "--branch-rate",
        type=float,
        default=0.5,
        help="Moments with a prompted branch (asked off the conversation, never said)",
    )
    w.add_argument(
        "--branch-talk",
        type=float,
        default=0.5,
        help="Of the branches, other partner talk (the rest: questions)",
    )
    w.add_argument("--loop-budget", type=int, default=3, help="Tries per line")
    w.add_argument("--temperature", type=float, default=0.9)
    w.add_argument("--max-tokens", type=int, default=1600, help="Reasoning included")
    w.add_argument("--seed", type=int, default=0)
    for p in (w, c, ca):
        p.add_argument("--judge-url", required=True, help="Several, comma-separated")
        p.add_argument("--judge-model", default="gpt-oss-120b")
    c.add_argument("--rows", type=Path, nargs="+", required=True)
    c.add_argument("--keys", default="line", help="The columns holding lines")
    c.add_argument("--limit", type=int, default=0, help="A random sample of rows")
    c.add_argument("--out", type=Path, help="Each row with what it failed")
    r.add_argument("--rows", type=Path, nargs="+", required=True)
    pa.add_argument("--rows", type=Path, nargs="+", required=True)
    pa.add_argument("--n", type=int, default=500)
    pa.add_argument("--tokenizer", default="ibm-granite/granite-4.1-3b")
    args = ap.parse_args()
    MelleaLogger.get_logger().setLevel(logging.WARNING)  # no line per request
    {
        "write": write_file,
        "check": check_file,
        "report": report,
        "parity": parity,
        "selfcheck": selfcheck,
        "cast": write_cast,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
