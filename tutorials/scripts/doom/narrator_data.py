# SPDX-License-Identifier: Apache-2.0
"""Write the narrator's data with Mellea: Granite's line at every speaking moment of recorded matches.

The moments are a recorded match's, every one in order (``talk.py moments``),
each with the output of his ``get_game_state`` call (``tool``) and what had
happened since the moment before (``moment``). At each, the partner may speak
first:

* where the partner gave an order (``collect.py --orders``), the order's words:
  his reply must fit what the game did with it;
* else, at about ``--probe-rate`` of the moments, a question about the game
  state (:mod:`probes`), whose answer is checked in code;
* else, at about ``--reply-rate``, something else, written for this moment by
  the partner's voice (:func:`partner_says`, three candidates, one kept) and
  rendered as speech recognition writes it, sometimes with a word misheard
  (:func:`mishear`);
* else nothing: he speaks on his own, about a topic code picks
  (:func:`remark_topic`: an order that hurt him or ended; else the news,
  :func:`checks.news`; else a story of the match, :func:`checks.story_options`).

Then the writer writes his line in one Mellea ``instruct``: the task and how
to take this turn (:mod:`narrator_prompts`), a grounding context (the game
state, the conversation so far, the partner's words, what to talk about), one
example, the requirements (:func:`checks.requirements`: every code check, and
the judge's questions in one call) and ``MultiTurnStrategy``: a failed line
stays in the chat, and the reasons it failed come back as the next user turn,
up to ``--loop-budget`` tries. The writer and the judge see what the trained
narrator will see (:mod:`conversation`), nothing more. Every request is seeded
from the match and the moment, so a rerun asks the same requests.

Each row is one line, passed or not (``ok``); the next moment follows it
either way, as live play would. ``train_alora.py`` trains on the rows that
passed. Row fields: the match (``data``, ``style``, ``ep``, ``t``), the
moment (``cue``, ``events``, ``tool``, ``moment``), the conversation before it
(``conv``) and his earlier lines (``prev``), what the partner said
(``utype``, ``player`` as heard, ``said`` as meant, ``probe``, ``order``), how
the writer was told to take it (``move``, ``topic``, ``example``), the line,
``ok``, what it ``failed`` and the ``judge``'s answers, and every try's line
(``tries``, ``attempts``).

Runs in an environment with Mellea (``pip install mellea==0.7.0``), one
process per shard (``recipe.sh narrator_text``)::

    python narrator_data.py write --moments data/r9/moments_write.jsonl \\
        --out data/r9/narrator/shard_0.jsonl --shard 0/32 \\
        --writer-url http://WRITER:PORT/v1 --judge-url http://JUDGE:PORT/v1
    python narrator_data.py report --rows data/r9/narrator/shard_*.jsonl
    python narrator_data.py check --rows runs/r9/narrator/narrator/heldout_gen.jsonl \\
        --keys adapter,base --judge-url http://JUDGE:PORT/v1

``check`` asks lines written elsewhere (a trained narrator's held-out samples)
the same requirements, once, without repair.
"""

import argparse
import json
import logging
import random
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import checks
import narrator_prompts as P
import probes
from conversation import Conversation, Exchange, new_stretch, tool_text
from mellea import generative, start_session
from mellea.backends import ModelOption
from mellea.core import MelleaLogger
from mellea.stdlib.context import ChatContext
from mellea.stdlib.sampling import MultiTurnStrategy
from probes import heard


# ── The partner's voice: Mellea generative stubs ───────────────────────────────
@generative
def partner_says(kind: str, how: str, game_state: str, conversation: str) -> list[str]:
    """Three different things the partner might say out loud now to Granite,
    who is playing a Doom deathmatch against bots while the partner sits next
    to him watching the screen.

    The partner talks the way they always do: casual, reactive, a little
    cheeky, with feeling, 2 to 12 words of spoken English each. They only
    watch, so they never talk as if they were in the game themselves, and they
    never read numbers or stats off the screen. ``kind`` names what they say
    and ``how`` says it; ``game_state`` is the game's own state ("you" in it
    is Granite); ``conversation`` is what has been said so far, oldest first
    (Partner: is them, Granite: is him). The three differ in meaning, and from
    everything the partner said before."""


@generative
def mishear(sentence: str) -> str:
    """``sentence`` as a speech recognizer heard it, with exactly one word
    wrong: swapped for a similar-sounding real word, so that it comes out a
    little funny ("go get the rocket launcher" heard as "go get the rocket
    lunch"). It has the same number of words."""


def one_word_off(a: str, b: str) -> bool:
    """``b`` is ``a`` with exactly one word swapped (a mishearing)."""
    x, y = a.split(), b.split()
    return len(x) == len(y) and sum(p != q for p, q in zip(x, y)) == 1


def seed_of(*key) -> int:
    return zlib.crc32(repr(key).encode())


# ── What the partner says ──────────────────────────────────────────────────────
def utterance_kinds(m: dict) -> list[str]:
    """The partner's utterances (not a question, not an order) the moment allows."""
    events = m.get("events") or []
    ok = {None: True, "event": m.get("cue") == "event"}
    ok["good"] = any(k in events for k in P.GOOD)
    return [k for k, (_, _, when) in P.UTTERANCES.items() if ok[when]]


def partner_turn(m, conv, rng, voice, args, weights, seed) -> dict:
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
        ptype = rng.choices(types, [weights[t] for t in types])[0]
        probe = probes.make(ptype, state, rng)
        return {"utype": "probe", "player": probe["text"], "probe": probe}
    if roll >= args.probe_rate + args.reply_rate:
        return {"utype": None}
    last = conv.exchanges[-1] if len(conv) else None
    if last and last.player and rng.random() < args.follow_rate:
        utype, how = "followup", P.FOLLOW_UP
    else:
        kinds = utterance_kinds(m)
        utype = rng.choices(kinds, [P.UTTERANCES[k][0] for k in kinds])[0]
        how = P.UTTERANCES[utype][1]
    options = {
        ModelOption.TEMPERATURE: 1.0,
        ModelOption.MAX_NEW_TOKENS: 300,
        ModelOption.SEED: seed,
    }
    try:
        lines = partner_says(
            voice,
            kind=utype,
            how=how,
            game_state=tool_text(state),
            conversation=checks.conversation_text(conv),
            model_options=options,
        )
    except ValueError:  # its answer was not the JSON asked for: say nothing
        return {"utype": None}
    before = [heard(ex.player) for ex in conv if ex.player]
    keep = [
        x
        for x in dict.fromkeys(x.strip().strip('"') for x in lines)
        if 2 <= len(x.split()) <= 14
        and not checks.NUMBER.search(x)
        and not any(checks.jaccard(heard(x), b) > 0.5 for b in before)
    ]
    if not keep:
        return {"utype": None}
    said = rng.choice(keep)
    out = {"utype": utype, "player": heard(said), "said": said}
    if rng.random() < args.mishear:
        try:
            x = heard(mishear(voice, sentence=out["player"], model_options=options))
        except ValueError:
            x = ""
        if one_word_off(out["player"], x):
            out.update(player=x, move="misheard")
    return out


# ── What he talks about on his own ─────────────────────────────────────────────
def order_topic(state: dict, m: dict) -> dict:
    """An order remark's topic: the order told, and what it just did to him."""
    o = state["order"]
    events = set(m.get("events") or ())
    if o.get("got") and o["told"].startswith("hunt"):
        sent = o["told"][len("hunt ") :]
        what = f"you fragged {o['got']}" + (
            f", though you were sent after {sent}"
            if sent not in ("a bot", o["got"])
            else ""
        )
    elif o.get("got"):
        what = f"you got the {o['got']}"
    elif o.get("why") in ("found none", "no frag"):
        what = {"found none": "you looked and found none", "no frag": "no frag"}[
            o["why"]
        ] + f" in {o['seconds_ago']} seconds"
    elif "order_hurts" in events and o["status"] == "doing":
        what = f"you lost {o.get('health_lost', 0)} health obeying it, and it goes on"
    elif o.get("hit_wall"):
        what = "you ran into the wall, as told"
    elif o["status"] == "refused":
        what = f"you quit halfway: {o.get('why')}"
    elif o["told"] == "stop" and o["status"] == "done":
        what = f"you stood still {o['seconds_ago']} seconds and nobody called it off"
    else:
        what = f"it is {o['status']}" + (f" ({o['why']})" if o.get("why") else "")
    bots = probes.bot_names(f"{o['told']} {o.get('got') or ''}", ignore_case=True)
    return {
        "kind": "order",
        "text": f"your partner's order ({o['told']}): {what}",
        "bots": sorted(bots),
    }


def remark_topic(m: dict, rng, recent: list[str]) -> dict:
    """What his remark at ``m`` is about: an order that hurt him or just ended;
    else the news (:func:`checks.news`; a frag, at
    :data:`narrator_prompts.FRAG_STORY`, with a story of the match to tie it
    to); else a story (:func:`checks.story_options`, by
    :data:`narrator_prompts.STORY_WEIGHTS`). A story is never the one of
    either of his last two remarks (``recent``)."""
    state = m["tool"]
    if {"order_hurts", "order_done"} & set(m.get("events") or ()) and state.get(
        "order"
    ):
        return order_topic(state, m)
    news = checks.news(state)
    stories = checks.story_options(state)
    fresh = [s for s in stories if s["kind"] not in recent[-2:]] or stories

    def story() -> dict:
        return rng.choices(fresh, [P.STORY_WEIGHTS[s["kind"]] for s in fresh])[0]

    if news is not None:
        if news["kind"] == "frag" and fresh and rng.random() < P.FRAG_STORY:
            s = story()
            return {
                **news,
                "text": f"{news['text']}; and the story around it, {s['text']}",
                "bots": [*news["bots"], *s["bots"]],
                "story": s["kind"],
            }
        return news
    if not fresh:
        return {"kind": "play", "text": "a quiet stretch of the match", "bots": []}
    return story()


# ── How to take the turn ───────────────────────────────────────────────────────
def how_to(turn: checks.Turn, move: str | None, conv, rng) -> tuple:
    """The task for this turn, its move, and the example the writer is shown:
    ``(task, move, example as shown, example line)``."""
    t, state = turn, turn.state
    if t.kind == "remark":
        how, examples = P.TOPICS[t.topic["kind"]]
        sit, line = rng.choice(examples)
        if t.topic.get("story"):
            how += P.STORY_TIE
        move = rng.choice(P.REMARK_ANGLES)
        how = f"{how} {P.REMARK_HOW.format(angle=move)}"
        task = P.REMARK_TASK.format(persona=P.PERSONA, context=P.CONTEXT_NOTE, how=how)
        return task, move, P.example_text(sit, line), line
    if t.probe is not None:
        fam = P.PROBE_FAMILY[t.probe["type"]]
        if fam == "victim" and probes.gold(t.probe, state) != "unknown":
            fam = "victim_known"
        sit, said, line = rng.choice(P.PROBE_EXAMPLES[fam])
        move = rng.choice(P.PROBE_ANGLES)
        how = P.PROBE_HOW.format(
            answer=probes.answer_text(t.probe, state),
            in_words=" in words" if checks.numeric(t.probe) else "",
            angle=move,
        )
    elif t.order is not None:
        st = t.order["status"] if t.order["status"] in P.ORDER_HOW else "done"
        sit, said, line = rng.choice(P.ORDER_EXAMPLES[st])
        move = f"order_{st}"
        how = P.ORDER_HOW[st].format(told=t.order["told"], why=t.order.get("why"))
        if st in ("doing", "done"):
            angle = rng.choice(P.ORDER_ANGLES)
            move += f": {angle}"
            how += P.ORDER_ANGLE.format(angle=angle)
        how += P.ORDER_FORM
    else:
        if move is None:
            pool = P.TYPE_MOVES.get(t.utype) or [
                k for k in P.GENERAL_MOVES if k != "callback" or len(conv)
            ]
            move = rng.choice(list(pool))
        sit, said, line = rng.choice(
            P.EXAMPLES["request" if t.utype == "request" else move]
        )
        how = "Move: " + P.MOVES[move].format(topic=rng.choice(P.TANGENTS))
        if t.utype == "request":
            how = f"{P.REQUEST_HOW} {how}"
    task = P.REPLY_TASK.format(persona=P.PERSONA, context=P.CONTEXT_NOTE, how=how)
    return task, move, P.example_text(sit, line, said), line


# ── One line ───────────────────────────────────────────────────────────────────
def write_line(writer, judge, turn: checks.Turn, task: str, example: str, args, seed):
    """One line, by Mellea's instruct-validate-repair loop: ``(line, passed,
    every try's line, the judge's answers on the last)``."""
    grounding = {
        "game state": tool_text(turn.state),
        "conversation so far": turn.conversation,
    }
    if turn.player:
        grounding["your partner just said"] = turn.player
    if turn.topic:
        grounding["what to talk about"] = turn.topic["text"]
    verdicts: dict = {}
    writer.reset()
    res = writer.instruct(
        task,
        grounding_context=grounding,
        icl_examples=[example],
        requirements=checks.requirements(turn, judge, verdicts),
        strategy=MultiTurnStrategy(loop_budget=args.loop_budget),
        return_sampling_results=True,
        model_options={ModelOption.SEED: seed},
    )
    line = checks.clean(res.result.value)
    tries = [checks.clean(g.value) for g in res.sample_generations]
    return line, bool(res.success), tries, verdicts.get(line, {})


def sessions(args, key=()):
    """The writer (a chat: its failed lines stay for the repair), the partner's
    voice (the same model) and the judge. Given several servers of each
    (comma-separated URLs), a match (``key``) always uses the same one."""
    h = seed_of(*key)
    writers, judges = args.writer_url.split(","), args.judge_url.split(",")
    writer_url, judge_url = writers[h % len(writers)], judges[h % len(judges)]
    writer = start_session(
        "openai",
        model_id=args.writer_model,
        ctx=ChatContext(),
        base_url=writer_url,
        api_key="none",
        model_options={
            # Granite 4.2 at "low" effort reasons briefly and closes </think>.
            ModelOption.THINKING: "low",
            ModelOption.TEMPERATURE: args.temperature,
            ModelOption.MAX_NEW_TOKENS: args.max_tokens,
        },
    )
    voice = start_session(
        "openai", model_id=args.writer_model, base_url=writer_url, api_key="none"
    )
    judge = start_session(
        "openai", model_id=args.judge_model, base_url=judge_url, api_key="none"
    )
    return writer, voice, judge


def write_match(match: dict, args, weights: dict, write) -> None:
    """Every moment of one match, in order; ``write`` gets each row."""
    key = (match["data"], match["ep"])
    rng = random.Random(seed_of(*key, args.seed))
    writer, voice, judge = sessions(args, key)
    said: list[Exchange] = []  # this stretch's exchanges
    pending: list[dict] = []  # what happened since his last line
    prev: list[str] = []  # his lines this match
    topics: list[str] = []  # his remarks' topics
    last_t = None
    for m in match["moments"]:
        if new_stretch(last_t, m["t"]):
            said, pending = [], []  # a long gap: a new conversation
        last_t = m["t"]
        seed = seed_of(*key, m["t"], args.seed)
        state = m["tool"]
        pending = pending + list(m.get("moment") or [])
        conv = Conversation(said)
        p = partner_turn(m, conv, rng, voice, args, weights, seed)
        turn = checks.Turn(
            state=state,
            prev=[ex.line for ex in conv],
            past=[ex.events for ex in conv],
            player=p.get("player"),
            utype=p["utype"],
            probe=p.get("probe"),
            order=state.get("order") if p["utype"] == "order" else None,
            conversation=checks.conversation_text(conv),
        )
        if turn.kind == "remark":
            turn.topic = remark_topic(m, rng, topics)
            topics.append(turn.topic.get("story") or turn.topic["kind"])
        task, move, example, turn.example = how_to(turn, p.get("move"), conv, rng)
        t0 = time.time()
        line, ok, tries, verdicts = write_line(
            writer, judge, turn, task, example, args, seed
        )
        attempts = len(tries)
        failed = [n for n, _ in checks.failures(line, turn)]
        failed += [f"judge: {k}" for k, yes in verdicts.items() if not yes]
        what = turn.utype or turn.topic["kind"]
        print(
            f"  {match['ep']} {m['t'] / 35:6.1f}s {what:<12} "
            f"{'ok' if ok else 'NO'} {attempts} {time.time() - t0:4.1f}s {line!r}"
            + ("" if ok else f" failed {failed}"),
            flush=True,
        )
        write(
            {
                **{k: match[k] for k in ("data", "style", "ep")},
                **{k: m.get(k) for k in ("t", "cue", "events", "order")},
                "kind": turn.kind,
                "utype": turn.utype,
                "player": turn.player,
                "said": p.get("said"),
                "probe": turn.probe,
                "move": move,
                "topic": turn.topic,
                "example": turn.example,
                "line": line,
                "ok": ok and bool(line),
                "failed": failed,
                "judge": verdicts,
                "attempts": attempts,
                "tries": tries,  # every line the writer wrote, the last kept
                "conv": conv.to_json(),  # what the narrator reads before now
                "tool": state,  # the game state he answers from
                "moment": pending,  # what this exchange keeps of the moment
                "prev": prev[-checks.RECENT :],
                "s": round(time.time() - t0, 1),
            }
        )
        if line:  # what he said, passed or not: the next moment follows it
            prev.append(line)
            said.append(Exchange(m["t"], pending, turn.player, line))
            pending = []


def write_file(args) -> None:
    """``write``: this shard's matches, whole, in order. Resumable: a match
    already written is skipped; one cut short is dropped and written again."""
    matches = [json.loads(x) for x in open(args.moments)][: args.matches or None]
    pool = [m for mt in matches for m in mt["moments"]]
    weights = probes.weights(pool)  # over every match, so every shard agrees
    k, n = map(int, args.shard.split("/"))
    mine = matches[k::n]
    want = {(m["data"], m["ep"]): len(m["moments"]) for m in mine}
    done: set = set()
    if args.out.exists():
        rows = [json.loads(x) for x in open(args.out)]
        count = Counter((r["data"], r["ep"]) for r in rows)
        done = {key for key, c in count.items() if c >= want.get(key, 0)}
        kept = [r for r in rows if (r["data"], r["ep"]) in done]
        if len(kept) < len(rows):
            args.out.write_text("".join(json.dumps(r) + "\n" for r in kept))
            print(f"dropped {len(rows) - len(kept)} rows of an unfinished match")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_all = 0
    failed = []
    t0 = time.time()
    with open(args.out, "a") as f:
        for match in mine:
            if (match["data"], match["ep"]) in done:
                continue
            rows: list[dict] = []
            try:
                write_match(match, args, weights, rows.append)
            except Exception as e:  # a server went away: the next run writes it
                failed.append(match["ep"])
                print(
                    f"match {match['ep']} failed: {type(e).__name__}: {e}", flush=True
                )
                continue
            f.writelines(json.dumps(r) + "\n" for r in rows)  # a whole match at once
            f.flush()
            n_all += len(rows)
            n_ok += sum(r["ok"] for r in rows)
            print(
                f"match {match['ep']}: {len(rows)} lines; so far {n_all}, "
                f"{100 * n_ok / max(1, n_all):.0f}% passed, "
                f"{n_all / (time.time() - t0):.2f} lines/s",
                flush=True,
            )
    print(f"done: {n_all} lines, {n_ok} passed every check -> {args.out}")
    if failed:
        raise SystemExit(
            f"{len(failed)} matches failed ({failed}); run again to write them"
        )


# ── Lines written elsewhere: the same requirements, once ───────────────────────
def check_file(args) -> None:
    """``check``: lines written elsewhere (``--keys``: the columns holding
    them, e.g. a narrator's held-out samples) asked every requirement once,
    without repair, in the turn each row records."""
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
        if t.kind == "remark" and t.topic is None:
            t.topic = checks.news(t.state)
        got = {}
        for k in keys:
            line = checks.clean(r.get(k) or "")
            verdicts = {q: v.yes for q, v in checks.ask(judge, t, line).items()}
            names = [n for n, _ in checks.failures(line, t)]
            names += [
                f"judge: {q}" for q in checks.judge_questions(t) if not verdicts.get(q)
            ]
            fails[k].update(names)
            fails[k]["(any)"] += bool(names)
            got[k] = {"failed": names, "judge": verdicts}
        out.append({**r, "checked": got})
        if (i + 1) % 25 == 0:
            print(f"{i + 1} of {len(rows)} rows", flush=True)
    if args.out:
        args.out.write_text("".join(json.dumps(r) + "\n" for r in out))
    n = max(1, len(rows))
    names = sorted({x for c in fails.values() for x in c})
    print(f"| failed (of {len(rows)}) | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for x in names:
        print(
            f"| {x} | "
            + " | ".join(f"{100 * fails[k][x] / n:.1f}%" for k in keys)
            + " |"
        )


# ── The report ─────────────────────────────────────────────────────────────────
# The dry run's gates (docs/DOOM_RECIPE.md).
GATES = {
    "passed within the tries": 70,
    "remarks judged about their topic": 85,
    "remarks with 'still'": -12,
    "remarks naming a weapon": -40,
    "after a frag of a named bot, names it": 50,
}


def report(args) -> None:
    """``report``: what the rows hold: how many passed and after how many
    tries, by kind; what the final lines failed; the remarks' topics; and the
    remark measures against the dry run's gates."""
    rows = [json.loads(x) for p in args.rows for x in open(p)]
    by = defaultdict(list)
    for r in rows:
        kind = r["utype"] if r["utype"] in ("probe", "order") else r["kind"]
        by[kind].append(r)
    print(f"{len(rows)} lines, {sum(r['ok'] for r in rows)} passed")
    print("| kind | lines | passed | 1st try | 2nd | 3rd | seconds per line |")
    print("|---|---|---|---|---|---|---|")
    for kind, rs in sorted(by.items()):
        n = len(rs)
        tries = Counter(r["attempts"] for r in rs if r["ok"])
        print(
            f"| {kind} | {n} | {100 * sum(r['ok'] for r in rs) / n:.0f}% | "
            + " | ".join(f"{100 * tries[a] / n:.0f}%" for a in (1, 2, 3))
            + f" | {sum(r['s'] for r in rs) / n:.1f} |"
        )
    failed = Counter(x for r in rows if not r["ok"] for x in r["failed"])
    print("\nWhat the lines that did not pass failed:")
    for x, c in failed.most_common(20):
        print(f"  {x}: {c}")
    remarks = [r for r in rows if r["kind"] == "remark" and r["ok"]]
    topics = Counter(r["topic"]["kind"] for r in remarks)
    print(f"\nremark topics (passed): {dict(topics.most_common())}")
    f = Counter()
    by_match = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["data"], r["ep"], r["t"])):
        by_match[(r["data"], r["ep"])].append(r)
    for rs in by_match.values():
        for r in rs:
            if r["kind"] != "remark" or not r["ok"]:
                continue
            fl = checks.remark_flags(r["line"], r["tool"], r["prev"])
            f["remarks"] += 1
            f["about"] += bool(r["judge"].get("about"))
            f["still"] += fl["still"]
            f["weapon"] += fl["names a weapon"]
            f["opening"] += fl["repeats an opening"]
            f["story"] += fl["story used"]
            f["victim known"] += fl["frag victim known"]
            f["names victim"] += fl["frag victim known"] and fl["names the victim"]
    n = max(1, f["remarks"])
    got = {
        "passed within the tries": 100 * sum(r["ok"] for r in rows) / max(1, len(rows)),
        "remarks judged about their topic": 100 * f["about"] / n,
        "remarks with 'still'": 100 * f["still"] / n,
        "remarks naming a weapon": 100 * f["weapon"] / n,
        "after a frag of a named bot, names it": 100
        * f["names victim"]
        / max(1, f["victim known"]),
    }
    print(f"\nremarks that passed: {f['remarks']}")
    for k, v in got.items():
        gate = GATES[k]
        ok = v >= gate if gate > 0 else v <= -gate
        print(
            f"  {k}: {v:.0f}% ({'>=' if gate > 0 else '<='} {abs(gate)}%: {'ok' if ok else 'NO'})"
        )
    print(
        f"  remarks repeating an opening of his last 8 lines: {100 * f['opening'] / n:.0f}%"
    )
    print(f"  remarks naming a bot of the storylines: {100 * f['story'] / n:.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="Write lines at every moment of the matches")
    c = sub.add_parser("check", help="Lines written elsewhere: every requirement, once")
    r = sub.add_parser("report", help="What written rows hold, against the gates")
    w.add_argument("--moments", type=Path, required=True, help="talk.py moments")
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--shard", default="0/1", help="K/N: every N-th match from the K-th")
    w.add_argument("--matches", type=int, default=0, help="The first so many (0: all)")
    w.add_argument(
        "--writer-url",
        required=True,
        help="The writer's server (OpenAI API); several, comma-separated",
    )
    w.add_argument("--writer-model", default="granite-4.2-30b")
    w.add_argument(
        "--probe-rate", type=float, default=0.55, help="Questions (probes.py)"
    )
    w.add_argument("--reply-rate", type=float, default=0.15, help="Other partner talk")
    w.add_argument(
        "--follow-rate", type=float, default=0.3, help="Of those, follow-ups"
    )
    w.add_argument(
        "--mishear", type=float, default=0.12, help="Of those, a word misheard"
    )
    w.add_argument("--loop-budget", type=int, default=3, help="Tries per line")
    w.add_argument("--temperature", type=float, default=0.9)
    w.add_argument(
        "--max-tokens", type=int, default=1200, help="The writer's, reasoning in"
    )
    w.add_argument("--seed", type=int, default=0)
    for p in (w, c):
        p.add_argument(
            "--judge-url",
            required=True,
            help="The judge's server; several, comma-separated",
        )
        p.add_argument("--judge-model", default="gpt-oss-120b")
    c.add_argument("--rows", type=Path, nargs="+", required=True)
    c.add_argument("--keys", default="line", help="The columns holding lines")
    c.add_argument("--limit", type=int, default=0, help="A random sample of rows")
    c.add_argument("--out", type=Path, help="Each row with what it failed")
    r.add_argument("--rows", type=Path, nargs="+", required=True)
    args = ap.parse_args()
    MelleaLogger.get_logger().setLevel(logging.WARNING)  # no line per request
    {"write": write_file, "check": check_file, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
