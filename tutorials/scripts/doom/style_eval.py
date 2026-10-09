# SPDX-License-Identifier: Apache-2.0
"""How the narrator talks, measured: style against another run, his acts, his habits, his truth.

The unit is a *window*: 12 consecutive exchanges of one match, each what had
just happened in the game (its events since the line before, compact), what
his partner said (as heard) and his line. A run is any file
``test_partner.read_lines`` reads: a test-partner run, a live log
(``doom_live.py serve --log-dir``), narrator rows (r9's or v10's, the lines
that passed), or a ``rollout`` here (a model zero-shot over r9's moments).

* ``pairwise --runs A=... B=...``: the judge (gpt-oss-120b) reads a window of
  each, in both orders, scores the rubric (:data:`RUBRIC`, from the style
  docs/DOOM_RECIPE.md describes) and a truth guard, then picks the one that
  reads more like two hired guns talking on the job in a Tarantino-style crime
  film. A's win rate with a 95% bootstrap interval, and A's share per rubric
  item. ``--export-for-human DIR``: the same pairs, blind, for a person to
  label (``DIR/pairs.md``, ``DIR/labels.txt``); ``agree --dir DIR``: the
  judge's agreement with those labels (the style gates count only at 75%).
* ``acts --runs ...``: the judge labels every line of a sample of windows with
  its main act (``narrator_v10_prompts.ACTS``) and :class:`LineLabels`' flags;
  the act mix (entropy, top share) and each flag's share.
* ``metrics --runs ...``: in code, on every line: questions, lengths, distinct
  bigrams, how varied the openings are.
* ``truth --runs ...``: his game claims checked in code (``probes.claims``)
  and, on a sample, judged ``true`` and ``grounded``.
* ``rollout``: a model's lines over r9's moments, zero-shot, his own earlier
  lines in the conversation (the baselines: the 3B base and the 30B teacher
  with the v10 system prompt).

Runs in the data writers' environment (Mellea); ``rollout`` needs only
``openai``::

    python style_eval.py metrics --runs r9=data/r9/narrator/shard_0.jsonl live=out/live.jsonl
    python style_eval.py pairwise --runs v10=data/r10/narrator/shard_0.jsonl \\
        r9=data/r9/narrator/shard_0.jsonl --judge-url http://JUDGE:PORT/v1 --n 60
    python style_eval.py acts --runs r9=... --judge-url http://JUDGE:PORT/v1 --n 40
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import checks
import narrator_v10_prompts as P
import probes

WINDOW = 12  # exchanges per window
MAX_EVENTS = 6  # events shown per exchange, the latest

# ── Runs and windows ───────────────────────────────────────────────────────────


def event_text(e: dict) -> str | None:
    """An event of the game's ``recent_events``, as a few words."""
    k = e.get("type")
    if k == "death":
        who = e.get("killer")
        return {"yourself": "you killed yourself", "unknown": "you died"}.get(
            who, f"{who} killed you"
        )
    if k == "frag":
        v = e.get("victim", "unknown")
        return "you fragged a bot" if v == "unknown" else f"you fragged {v}"
    if k == "kill":
        return f"{e['killer']} killed {e['victim']}"
    if k == "pickup":
        return f"you picked up {e['item']}"
    if k == "lead":
        if e.get("leader") == "you":
            return "you took the lead"
        return (
            f"{e['leader']} {'tied you' if e.get('tied_with_you') else 'took the lead'}"
        )
    if k == "close_call":
        return "you barely survived"
    if k == "streak":
        return "a streak of frags"
    if k == "order":
        return f"your partner told you to {e['told']} ({e['status']})"
    if k == "order_end":
        return f"the order to {e['told']} ended ({e['status']})"
    if k == "order_hurts":
        return f"the order to {e['told']} is costing you health"
    return None


def exchanges(lines: list[dict]) -> list[dict]:
    """A match's lines (``test_partner.read_lines``) as exchanges: the events
    since the line before, the partner's words, his line."""
    out, since = [], -1
    for x in sorted(lines, key=lambda x: x["tick"]):
        st = x.get("state") or {}
        evs = [
            event_text(e)
            for e in st.get("recent_events", ())
            if checks.secs(e["time"]) > since
        ]
        evs = [e for e in evs if e][-MAX_EVENTS:]
        if st.get("time"):
            since = checks.secs(st["time"])
        out.append(
            {
                "time": st.get("time", ""),
                "events": evs,
                "partner": x.get("heard") or "",
                "line": " ".join(str(x["line"]).split()),
                "state": st,
            }
        )
    return out


def load(path: Path) -> dict[str, list[dict]]:
    """A run's matches, each a list of exchanges."""
    from test_partner import read_lines

    return {k: exchanges(v) for k, v in read_lines(path).items() if v}


def lines_of_rows(rows: list[dict]) -> dict[str, list[dict]]:
    """Written rows (narrator_data.py's or narrator_v10.py's) as a run (a
    prompted branch of narrator_v10.py's, never said, is not in it)."""
    by: dict = {}
    for r in rows:
        if r.get("line") and not r.get("branch"):
            by.setdefault(f"{r['data']}:{r['ep']}", []).append(
                {
                    "tick": r["t"],
                    "line": r["line"],
                    "heard": r.get("player"),
                    "state": r["tool"],
                }
            )
    return {k: exchanges(v) for k, v in by.items()}


def windows(run: dict[str, list[dict]], n: int, seed: int) -> list[list[dict]]:
    """``n`` windows (WINDOW consecutive exchanges, none overlapping) drawn
    from a run's matches."""
    pool = [
        xs[i : i + WINDOW]
        for xs in run.values()
        for i in range(0, len(xs) - WINDOW + 1, WINDOW)
    ]
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


def blind(text: str) -> str:
    """``text`` with any bots that are not the cast renamed after it
    (narrator_v10.cast_names), so a window from before round 10 cannot be told
    by its bots' names."""
    import narrator_v10 as N

    names = sorted(
        {
            n
            for n in probes.BOT_NAMES
            if n not in N.CAST and re.search(rf"\b{n}\b", text, re.I)
        }
    )
    return N.renamed(text, N.cast_names(names))


def window_text(w: list[dict]) -> str:
    """A window as the judge (and a person) reads it, one exchange per block,
    its bots named after the cast (:func:`blind`)."""
    return blind(_window_text(w))


def _window_text(w: list[dict]) -> str:
    out = []
    for i, x in enumerate(w, 1):
        block = [f"{i}. [{x['time']}]"]
        if x["events"]:
            block.append(f"   Game: {'; '.join(x['events'])}")
        if x["partner"]:
            block.append(f'   Partner: "{x["partner"]}"')
        block.append(f'   Granite: "{x["line"]}"')
        out.append("\n".join(block))
    return "\n".join(out)


def runs_arg(specs: list[str]) -> dict[str, Path]:
    return {k: Path(v) for k, v in (s.split("=", 1) for s in specs)}


# ── Metrics in code ────────────────────────────────────────────────────────────


def entropy(counts: Counter) -> float:
    n = sum(counts.values())
    return -sum(c / n * math.log2(c / n) for c in counts.values() if c) if n else 0.0


def pctl(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


def metrics(run: dict[str, list[dict]], n: int = 1000, seed: int = 0) -> dict:
    """Code metrics on a sample of ``n`` of a run's lines (the same size for
    every run, since entropies grow with it)."""
    lines = [x["line"] for xs in run.values() for x in xs]
    lines = random.Random(seed).sample(lines, min(n, len(lines)))
    words = [len(x.split()) for x in lines]
    bigrams = Counter()
    for x in lines:
        w = checks.words(x)
        bigrams.update(itertools.pairwise(w))
    openings = Counter(checks.opening(x) for x in lines)
    k = max(1, len(lines))
    return {
        "lines": len(lines),
        "questions": sum("?" in x for x in lines) / k,
        "words, mean": sum(words) / k,
        "words, p10": pctl(words, 0.1),
        "words, p50": pctl(words, 0.5),
        "words, p90": pctl(words, 0.9),
        "20 words or more": sum(w >= 20 for w in words) / k,
        "distinct bigrams": len(bigrams) / max(1, sum(bigrams.values())),
        "opening entropy (bits)": entropy(openings),
        "top opening": openings.most_common(1)[0][1] / k if openings else 0.0,
    }


def print_metrics(runs: dict[str, dict[str, list[dict]]], n: int = 1000) -> None:
    got = {k: metrics(v, n) for k, v in runs.items()}
    names = list(got)
    print("| code metric | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for key in next(iter(got.values()), {}):
        cells = []
        for k in names:
            v = got[k][key]
            share = key in (
                "questions",
                "20 words or more",
                "top opening",
                "distinct bigrams",
            )
            cells.append(
                f"{100 * v:.1f}%"
                if share
                else f"{v:.1f}"
                if isinstance(v, float)
                else str(v)
            )
        print(f"| {key} | " + " | ".join(cells) + " |")


# ── The judge ──────────────────────────────────────────────────────────────────
# The rubric of the style (docs/DOOM_RECIPE.md, "The style"), scored before the
# choice.
RUBRIC = {
    "connected": "lines that build on each other across the window: callbacks, "
    "running bits about the same people, a thread picked back up; not one-off "
    "remarks",
    "conversation": "talking with his partner, not a commentator's play-by-play",
    "the people": "ribbing the people in the match as people he knows, each about "
    "their own thing, tied to what they just did",
    "fresh": "no joke or phrase told twice; each rib new, or building on the last",
    "easy to follow": "something a viewer watching the game gets at once",
    "humor from character": "funny because of who he and they are, not from quips "
    "tacked onto reports",
    "rhythm": "short lines beside longer ones, questions, a word handed back",
}
# The question a pair answers, for the judge and for a person.
CHOICE = (
    "which you would rather hear while watching the match: a hired gun at a laser "
    "tag night with people he knows, ribbing them and talking with his partner; "
    "connected (each line building on what came before) and true to the game "
    "first, funny second, and easy to follow"
)


class ItemVerdict(BaseModel):
    item: str
    better: Literal["A", "B", "same"]


class Comparison(BaseModel):
    items: list[ItemVerdict]
    truth_problems: Literal["A", "B", "both", "neither"]
    choice: Literal["A", "B"]
    reason: str


def compare_windows(rubric: dict, window_a: str, window_b: str) -> Comparison:
    """Compare two windows of a hired gun's talk with his long-time partner
    while he plays a Doom deathmatch against bots as a job. Each window is a
    dozen consecutive exchanges: what the game reported (Game), what his
    partner said, as speech recognition heard it (Partner), and what he said
    (Granite).

    First score every item of ``rubric`` (name: what to look for): ``better``
    says which window does it more and better, or ``same``. Then the truth
    guard: ``truth_problems`` says which windows have Granite state game facts
    that the Game lines contradict or never report (a kill, a name, a score).
    Then ``choice``: which window you would rather hear while watching the
    match: a hired gun at a laser tag night with people he knows, ribbing them
    and talking with his partner; connected (his lines building on each other:
    callbacks, running bits, a thread picked back up) and true to the game
    first, funny second, and easy to follow; rather than one-off remarks, or a
    commentator's play-by-play with a joke attached to each line.
    The bots may be named after people he knows; a window with truth problems
    the other lacks does not win on style alone. ``reason``: one or two
    sentences."""


class LineLabel(BaseModel):
    n: int
    act: str
    report_plus_quip: bool
    thread: bool
    resumed: bool
    conviction: bool
    concrete: bool
    addresses_partner: bool
    off_event: bool


def label_lines(acts: dict, window: str) -> list[LineLabel]:
    """Label every Granite line of a window of a hired gun's talk with his
    long-time partner while he plays a Doom deathmatch against bots. The window
    is a dozen consecutive exchanges, numbered: what the game reported (Game),
    what his partner said (Partner), what he said (Granite).

    One label per Granite line, ``n`` its exchange's number: ``act``, the one
    name in ``acts`` (name: what it is) that best says what the line mainly
    does; ``report_plus_quip``, whether it is an event report with a quip
    attached; ``thread``, whether it picks up something said two or more
    exchanges back; ``resumed``, whether it picks up something the game's news
    had cut into; ``conviction``, whether it states or defends a standing
    opinion of his; ``concrete``, whether it has a specific, concrete detail
    (a named thing, a place, a how or a what, not a generality);
    ``addresses_partner``, whether it is said to his partner (a reply, a
    question, a remark aimed at them); ``off_event``, whether it is about
    something other than what just happened in the game."""


def _judge(args):
    from mellea import generative, start_session
    from mellea.core import MelleaLogger

    MelleaLogger.get_logger().setLevel("WARNING")
    m = start_session(
        "openai",
        model_id=args.judge_model,
        base_url=args.judge_url.split(",")[0],
        api_key="none",
    )
    return m, generative


def _options(seed: int):
    from mellea.backends import ModelOption

    return {
        ModelOption.THINKING: "low",
        ModelOption.TEMPERATURE: 0.0,
        ModelOption.MAX_NEW_TOKENS: 6000,
        ModelOption.SEED: seed,
    }


def judged_pairs(args, pairs: list[tuple[str, str]]) -> list[dict]:
    """The judge on each pair (x, y) of window texts, in both orders: x's
    score (1 chosen both times, 0.5 once, 0 never), x's share per rubric item
    and the truth guard's verdicts."""
    m, generative = _judge(args)
    stub = generative(compare_windows)

    def one(item):
        i, (x, y) = item
        got = []
        for a, b in ((x, y), (y, x)):
            try:
                got.append(
                    stub(
                        m,
                        rubric=RUBRIC,
                        window_a=a,
                        window_b=b,
                        model_options=_options(i),
                    )
                )
            except ValueError:
                got.append(None)
        return got

    with ThreadPoolExecutor(args.threads) as pool:
        res = list(pool.map(one, enumerate(pairs)))
    out: list[dict | None] = []
    for first, second in res:
        if first is None or second is None:  # an answer not in the JSON asked for
            out.append(None)
            continue
        # x is A in the first order, B in the second.
        score = ((first.choice == "A") + (second.choice == "B")) / 2
        items = {}
        for k in RUBRIC:
            v = []
            for c, me in ((first, "A"), (second, "B")):
                b = next((it.better for it in c.items if it.item == k), "same")
                v.append(1.0 if b == me else 0.5 if b == "same" else 0.0)
            items[k] = sum(v) / 2
        truth = [
            (c.truth_problems in (me, "both"), c.truth_problems in (other, "both"))
            for c, me, other in ((first, "A", "B"), (second, "B", "A"))
        ]
        out.append(
            {
                "score": score,
                "items": items,
                "x_truth_problems": sum(t[0] for t in truth) / 2,
                "y_truth_problems": sum(t[1] for t in truth) / 2,
                "reasons": [first.reason, second.reason],
            }
        )
    return out


def bootstrap(xs: list[float], seed: int = 0, n: int = 2000) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def pair_windows(args, a: Path, b: Path) -> list[tuple[str, str]]:
    wa = windows(load(a), args.n, args.seed)
    wb = windows(load(b), args.n, args.seed + 1)
    k = min(len(wa), len(wb))
    return [(window_text(x), window_text(y)) for x, y in zip(wa[:k], wb[:k])]


def pairwise(args) -> None:
    """``pairwise``: A's win rate against B, with its 95% interval and per
    rubric item; or, with ``--export-for-human``, the pairs for a person."""
    (na, a), (nb, b) = runs_arg(args.runs).items()
    pairs = pair_windows(args, a, b)
    if args.export_for_human:
        export(args.export_for_human, pairs, na, nb, args.seed)
        return
    got = [g for g in judged_pairs(args, pairs) if g is not None]
    scores = [g["score"] for g in got]
    lo, hi = bootstrap(scores, args.seed)
    print(f"{na} against {nb}: {len(got)} pairs, each judged in both orders")
    print(
        f"  {na} wins {100 * sum(scores) / max(1, len(scores)):.0f}% (95% CI {100 * lo:.0f}-{100 * hi:.0f}%)"
    )
    for k in RUBRIC:
        v = [g["items"][k] for g in got]
        print(f"  {k}: {na} {100 * sum(v) / max(1, len(v)):.0f}%")
    for who, key in ((na, "x_truth_problems"), (nb, "y_truth_problems")):
        v = [g[key] for g in got]
        print(
            f"  truth problems in {who}'s windows: {100 * sum(v) / max(1, len(v)):.0f}%"
        )
    if args.out:
        args.out.write_text("".join(json.dumps(g) + "\n" for g in got))


def export(d: Path, pairs, na: str, nb: str, seed: int) -> None:
    """The pairs, blind: ``pairs.md`` to read (each pair's windows as A and B,
    shuffled), ``labels.txt`` to fill in (A or B per pair), ``key.jsonl`` (which
    run is A, and the window texts)."""
    d.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    md, key = [], []
    for i, (x, y) in enumerate(pairs, 1):
        flip = rng.random() < 0.5
        a, b = (y, x) if flip else (x, y)
        key.append(
            {
                "pair": i,
                "flip": flip,
                "A": nb if flip else na,
                "B": na if flip else nb,
                "x": x,
                "y": y,
            }
        )
        md.append(f"## Pair {i}\n\n### A\n\n```\n{a}\n```\n\n### B\n\n```\n{b}\n```\n")
    head = f"# For each pair: {CHOICE}?\n\nWrite A or B on its line of labels.txt.\n\n"
    (d / "pairs.md").write_text(head + "\n".join(md))
    (d / "key.jsonl").write_text("".join(json.dumps(k) + "\n" for k in key))
    (d / "labels.txt").write_text("".join(f"{k['pair']}: \n" for k in key))
    print(f"{len(pairs)} pairs -> {d}/pairs.md; labels go in {d}/labels.txt")


def agree(args) -> None:
    """``agree``: the judge's choice on each labeled pair (both orders; a split
    counts half) against the person's label."""
    key = [json.loads(x) for x in open(args.dir / "key.jsonl")]
    labels = {}
    for x in open(args.dir / "labels.txt"):
        m = re.match(r"\s*(\d+)\s*:\s*([AB])\b", x, re.I)
        if m:
            labels[int(m.group(1))] = m.group(2).upper()
    key = [k for k in key if k["pair"] in labels]
    got = judged_pairs(args, [(k["x"], k["y"]) for k in key])
    hits = []
    for k, g in zip(key, got):
        if g is None:
            continue
        picked_x = (labels[k["pair"]] == "A") != k["flip"]  # A is x unless flipped
        hits.append(g["score"] if picked_x else 1 - g["score"])
    rate = sum(hits) / max(1, len(hits))
    print(f"the judge agrees with {len(hits)} labels: {100 * rate:.0f}% (gate 75%)")


# ── Acts ───────────────────────────────────────────────────────────────────────


def acts(args) -> None:
    """``acts``: every line of ``--n`` windows per run labeled by the judge;
    the act mix and each flag's share, one column per run."""
    m, generative = _judge(args)
    stub = generative(label_lines)
    gloss = {k: a.gloss for k, a in P.ACTS.items()}
    cols = {}
    for name, path in runs_arg(args.runs).items():
        ws = windows(load(path), args.n, args.seed)

        def one(item):
            i, w = item
            try:
                return stub(
                    m, acts=gloss, window=window_text(w), model_options=_options(i)
                )
            except ValueError:
                return []

        with ThreadPoolExecutor(args.threads) as pool:
            labels = [x for got in pool.map(one, enumerate(ws)) for x in got]
        c = Counter(lb.act if lb.act in P.ACTS else "(other)" for lb in labels)
        n = max(1, len(labels))
        long_ = [lb for lb in labels if lb.act in ("riff", "argue", "explain", "story")]
        cols[name] = {
            "lines labeled": len(labels),
            "act entropy (bits)": entropy(c),
            "top act": f"{c.most_common(1)[0][0]} {100 * c.most_common(1)[0][1] / n:.0f}%"
            if c
            else "-",
            "report + quip": sum(lb.report_plus_quip for lb in labels) / n,
            "thread pickups": sum(lb.thread for lb in labels) / n,
            "resumed after news": sum(lb.resumed for lb in labels) / n,
            "conviction": sum(lb.conviction for lb in labels) / n,
            "concrete (riff, argue, explain, story)": sum(lb.concrete for lb in long_)
            / max(1, len(long_)),
            "addresses partner": sum(lb.addresses_partner for lb in labels) / n,
            "off the event": sum(lb.off_event for lb in labels) / n,
            **{f"act: {k}": c[k] / n for k in P.ACTS},
        }
        if args.out:
            with open(args.out, "a") as f:
                f.writelines(
                    json.dumps({"run": name, **lb.model_dump()}) + "\n" for lb in labels
                )
    names = list(cols)
    print("| acts | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for key in next(iter(cols.values())):
        cells = []
        for k in names:
            v = cols[k][key]
            cells.append(
                f"{100 * v:.0f}%"
                if isinstance(v, float) and "bits" not in key
                else f"{v:.2f}"
                if isinstance(v, float)
                else str(v)
            )
        print(f"| {key} | " + " | ".join(cells) + " |")


# ── Truth ──────────────────────────────────────────────────────────────────────


def truth(args) -> None:
    """``truth``: per run, the share of lines whose game claims fail in code,
    and of a ``--n`` sample, the share the judge finds true and grounded."""
    import probes

    m, _ = _judge(args)
    cols = {}
    for name, path in runs_arg(args.runs).items():
        run = load(path)
        items = []
        for xs in run.values():
            prev = []
            for x in xs:
                items.append((x, list(prev)))
                prev.append(x["line"])
        fails = sum(
            not probes.claims(x["line"], x["state"], [], x["partner"])[0]
            for x, _ in items
        )
        sample = random.Random(args.seed).sample(items, min(args.n, len(items)))
        qs = {k: checks.JUDGE_V10[k][1] for k in ("true", "grounded")}

        def one(item):
            x, prev = item
            t = checks.Turn(
                state=x["state"],
                prev=prev,
                player=x["partner"] or None,
                conversation="\n".join(f'Granite: "{p}"' for p in prev[-8:])
                or "(nothing yet)",
            )
            return checks.ask_questions(m, t, x["line"], qs, v10=True)

        with ThreadPoolExecutor(args.threads) as pool:
            got = list(pool.map(one, sample))
        if args.out:
            with open(args.out, "a") as f:
                for (x, _), g in zip(sample, got):
                    v = {q: [a.yes, a.reason] for q, a in g.items()}
                    f.write(
                        json.dumps({"run": name, "line": x["line"], "judge": v}) + "\n"
                    )
        k = max(1, len(got))
        cols[name] = {
            "lines": len(items),
            "claims fail (code)": fails / max(1, len(items)),
            "judged": len(got),
            "true": sum(bool(g.get("true") and g["true"].yes) for g in got) / k,
            "grounded": sum(bool(g.get("grounded") and g["grounded"].yes) for g in got)
            / k,
        }
    names = list(cols)
    print("| truth | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for key in next(iter(cols.values())):
        print(
            f"| {key} | "
            + " | ".join(
                f"{100 * cols[k][key]:.1f}%"
                if isinstance(cols[k][key], float)
                else str(cols[k][key])
                for k in names
            )
            + " |"
        )


# ── Baselines: a model zero-shot over r9's moments ─────────────────────────────


def rollout(args) -> None:
    """``rollout``: ``--model`` at ``--url`` writes his line at every moment of
    ``--matches`` matches of r9's rows (the same partner's words, game states
    and events), zero-shot, on the narrator's chat with the ``--system`` prompt
    and the tool, his own earlier lines in the conversation. Rows as the
    narrator rows (``ok`` true), so every subcommand here reads them."""
    from conversation import (
        GET_GAME_STATE,
        NARRATOR_SYSTEM_PROMPTS,
        Conversation,
        Exchange,
        messages,
    )
    from openai import OpenAI

    by: dict = {}
    for p in args.rows:
        for x in open(p):
            r = json.loads(x)
            by.setdefault((r["data"], r["ep"]), []).append(r)
    keys = sorted(by)[: args.matches]
    client = OpenAI(base_url=args.url, api_key="none")
    system = NARRATOR_SYSTEM_PROMPTS[args.system]

    def wire(msgs: list[dict]) -> list[dict]:
        """The OpenAI API's form: a tool call's arguments as JSON text, and an
        id that the tool's output answers."""
        n = 0
        for x in msgs:
            for c in x.get("tool_calls") or ():
                n += 1
                c["id"] = f"call{n}"
                c["function"]["arguments"] = json.dumps(c["function"]["arguments"])
            if x["role"] == "tool":
                x["tool_call_id"] = f"call{n}"
        return msgs

    def one(key):
        conv, out = Conversation(), []
        for r in sorted(by[key], key=lambda r: r["t"]):
            msgs = wire(messages(conv, r["tool"], r.get("player"), system))
            kw = {"reasoning_effort": args.reasoning} if args.reasoning else {}
            res = client.chat.completions.create(
                model=args.model,
                messages=msgs,
                tools=[GET_GAME_STATE],
                tool_choice="none",
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=r["t"],
                **kw,
            )
            line = checks.clean(res.choices[0].message.content or "")
            out.append(
                {
                    **{
                        k: r[k]
                        for k in (
                            "data",
                            "style",
                            "ep",
                            "t",
                            "player",
                            "tool",
                            "moment",
                        )
                    },
                    "utype": r.get("utype"),
                    "conv": conv.to_json(),
                    "line": line,
                    "ok": bool(line),
                    "system": args.system,
                }
            )
            if line:
                conv.add(Exchange(r["t"], r.get("moment") or [], r.get("player"), line))
        return out

    with ThreadPoolExecutor(args.threads) as pool:
        rows = [r for got in pool.map(one, keys) for r in got]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(keys)} matches, {len(rows)} lines -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pw = sub.add_parser("pairwise", help="A's style against B's, by the judge")
    ag = sub.add_parser("agree", help="The judge against a person's labels")
    ac = sub.add_parser("acts", help="The judge's labels of every line")
    me = sub.add_parser("metrics", help="Code metrics")
    tr = sub.add_parser("truth", help="Claims in code; judged true, grounded")
    ro = sub.add_parser("rollout", help="A model zero-shot over r9's moments")
    for p in (pw, ac, me, tr):
        p.add_argument("--runs", nargs="+", required=True, help="NAME=PATH ...")
        p.add_argument("--seed", type=int, default=0)
    for p in (pw, ag, ac, tr):
        p.add_argument("--judge-url", required=False, default="")
        p.add_argument("--judge-model", default="gpt-oss-120b")
        p.add_argument("--threads", type=int, default=8)
    pw.add_argument("--n", type=int, default=60, help="Pairs")
    pw.add_argument("--export-for-human", type=Path, help="A dir: the pairs, blind")
    pw.add_argument("--out", type=Path, help="Each pair's verdicts")
    ag.add_argument(
        "--dir", type=Path, required=True, help="pairwise --export-for-human's"
    )
    ag.add_argument("--seed", type=int, default=0)
    ac.add_argument("--n", type=int, default=40, help="Windows per run")
    ac.add_argument("--out", type=Path, help="Every label")
    me.add_argument("--n", type=int, default=1000, help="Lines per run")
    tr.add_argument("--n", type=int, default=300, help="Lines judged per run")
    tr.add_argument("--out", type=Path, help="Each judged line and its verdicts")
    ro.add_argument(
        "--rows", type=Path, nargs="+", required=True, help="r9's narrator rows"
    )
    ro.add_argument("--matches", type=int, default=8)
    ro.add_argument("--url", required=True)
    ro.add_argument("--model", required=True)
    ro.add_argument("--system", default="v10")
    ro.add_argument("--reasoning", default="", help="e.g. low (the teacher)")
    ro.add_argument("--max-tokens", type=int, default=96)
    ro.add_argument("--temperature", type=float, default=0.8)
    ro.add_argument("--threads", type=int, default=8)
    ro.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "metrics":
        print_metrics({k: load(v) for k, v in runs_arg(args.runs).items()}, args.n)
        return
    {
        "pairwise": pairwise,
        "agree": agree,
        "acts": acts,
        "truth": truth,
        "rollout": rollout,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
