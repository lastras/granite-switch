# SPDX-License-Identifier: Apache-2.0
"""The narrator on held-out matches: the question battery, and the closed loop.

Each held-out match (``talk.py moments ... --split``) is played in order, as
the demo serves it: the composed checkpoint, the live prompt
(:func:`conversation.narrator_ids`) and the live sampling, each talker's own
lines fed back into its conversation. A scripted partner (:class:`Partner`)
speaks at the same moments for every talker: a probe of the game state
(:mod:`probes`) at about 55% of them, a line of small talk at about 15%; at
the rest he speaks on his own. At every moment, side probes of the types asked
least so far are answered off the same conversation but not fed back, so the
battery comes out balanced across types.

Reported per talker (the narrator adapter, the base model on the same prompt):

* accuracy per question type: correct, wrong, abstained (:func:`probes.verify`);
* the claims of every line (:func:`probes.claims`), remarks apart;
* the closed loop, first half of the match against the second: near-repeats of
  a recent line, the motif share (a line repeating a content word from 3 or
  more of his last 8 lines), "thanks / got it" lines, words per line.

    python eval_probes.py --model models/doom-narr6-r1 \\
        --moments data/narr/moments_v6_heldout.jsonl --out out/narr6_eval_r1.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import probes
from conversation import Conversation, Exchange, narrator_ids

PROBE_RATE, OTHER_RATE = 0.55, 0.15
# What the partner says when not asking about the game (as the ASR writes it).
SMALL_TALK = tuple(
    probes.heard(x)
    for x in (
        "hey can you hear me",
        "who are you",
        "nice",
        "you are on fire",
        "that was close",
        "you suck",
        "play it safe for a while",
        "go get some armor",
        "what are you thinking about",
        "this is crazy",
        "are you even trying",
        "what is your name",
        "be careful",
        "get him",
        "what just happened",
        "that was awesome",
    )
)
THANKS = re.compile(
    r"\b(?:thanks|thank you|got it|noted|copy that|understood|roger)\b", re.I
)
_STOP = set(
    "that this with have from they them what when your just like there their about "
    "been were will would could should into than then over some only also still "
    "here where which while more most much very really right back down even ever "
    "it's i'm that's don't can't won't you're he's let's".split()
)


class Partner:
    """The scripted partner of one match, the same for every talker: at each
    moment, what he says (a probe, small talk, or nothing), drawn from a seed
    of the match and the moment."""

    def __init__(self, key, weights: dict[str, float], seed: int = 0, split="train"):
        self.key, self.weights, self.seed, self.split = key, weights, seed, split

    def say(self, m: dict) -> tuple[dict | None, str | None]:
        rng = random.Random(zlib.crc32(repr((self.key, m["t"], self.seed)).encode()))
        roll = rng.random()
        if roll < PROBE_RATE:
            types = probes.allowed(m["tool"])
            t = rng.choices(types, [self.weights[x] for x in types])[0]
            p = probes.make(t, m["tool"], rng, self.split)
            return p, p["text"]
        if roll < PROBE_RATE + OTHER_RATE:
            return None, rng.choice(SMALL_TALK)
        return None, None


def jaccard(a: str, b: str) -> float:
    x, y = (
        set(re.findall(r"[a-z']+", a.lower())),
        set(re.findall(r"[a-z']+", b.lower())),
    )
    return len(x & y) / max(1, len(x | y))


def content_words(line: str) -> set[str]:
    return {
        w
        for w in re.findall(r"[a-z']+", line.lower())
        if len(w) >= 4 and w not in _STOP
    }


def loop_stats(lines: list[str]) -> dict:
    """Closed-loop measures over one talker's lines of one match, in order."""
    out = []
    for i, line in enumerate(lines):
        before = lines[max(0, i - 10) : i]
        last8 = lines[max(0, i - 8) : i]
        counts = Counter(w for b in last8 for w in content_words(b))
        out.append(
            {
                "near": any(jaccard(line, b) >= 0.6 for b in before),
                "motif": any(counts[w] >= 3 for w in content_words(line)),
                "thanks": bool(THANKS.search(line)),
                "words": len(line.split()),
                "half": int(i >= len(lines) // 2),
            }
        )
    return out


def clean_line(text: str) -> str:
    return re.sub(r"\[[^\]]*\]\s*", "", text).strip().split("\n")[0].strip().strip('"')


def run(args) -> dict:
    from policy import NARRATOR, VLLMPolicy

    matches = [json.loads(x) for x in open(args.moments)][: args.matches or None]
    weights = probes.weights([m for mt in matches for m in mt["moments"]])
    pol = VLLMPolicy(
        args.model,
        layout="chat",
        max_model_len=16384,  # as live (doom_live.py serve): whole matches run long
        max_num_seqs=256,
        warmup=0,
        gpu_memory_utilization=args.gpu_mem,
    )
    talkers = {"narrator": NARRATOR, "base": None}
    talkers = {k: v for k, v in talkers.items() if k in args.talkers.split(",")}
    sp = pol.talk_params(max_tokens=32, temperature=args.temperature)
    runs = [
        {
            "mi": mi,
            "who": who,
            "said": [],
            "pending": [],
            "partner": Partner((mt["data"], mt["ep"]), weights, args.seed, "test"),
        }
        for mi, mt in enumerate(matches)
        for who in talkers
    ]
    asked: dict[str, Counter] = {who: Counter() for who in talkers}
    log, battery = [], []
    for k in range(max(len(mt["moments"]) for mt in matches)):
        prompts, meta = [], []
        for run in runs:
            ms = matches[run["mi"]]["moments"]
            if k >= len(ms):
                continue
            m = ms[k]
            run["pending"] = run["pending"] + list(m.get("moment") or [])
            conv = Conversation(run["said"])
            probe, player = run["partner"].say(m)
            if probe is not None:
                asked[run["who"]][probe["type"]] += 1
            adapter = talkers[run["who"]]
            prompts.append(narrator_ids(pol.tok, conv, m["tool"], player, adapter))
            meta.append(("main", run, m, conv, probe, player))
            # Side probes: the types this talker was asked least, not fed back.
            types = sorted(
                probes.allowed(m["tool"]), key=lambda t: asked[run["who"]][t]
            )
            rng = random.Random(zlib.crc32(repr((run["mi"], k, "side")).encode()))
            for t in types[: args.side]:
                p = probes.make(t, m["tool"], rng, "test")
                prompts.append(
                    narrator_ids(pol.tok, conv, m["tool"], p["text"], adapter)
                )
                meta.append(("side", run, m, conv, p, p["text"]))
                asked[run["who"]][t] += 1
        if not prompts:
            break
        outs = pol.run(prompts, [sp] * len(prompts))
        for (role, run, m, conv, probe, player), o in zip(meta, outs):
            line = clean_line(o.outputs[0].text)
            past = [ex.events for ex in conv]
            ok, why = probes.claims(line, m["tool"], past, player or "")
            row = {
                "who": run["who"],
                "role": role,
                "data": matches[run["mi"]]["data"],
                "ep": matches[run["mi"]]["ep"],
                "k": k,
                "n": len(matches[run["mi"]]["moments"]),
                "t": m["t"],
                "player": player,
                "probe": probe,
                "state": m["tool"],
                "past": past,
                "line": line,
                "claims": ok,
                "claims_why": why,
            }
            if probe is not None:
                row["verdict"], row["why"] = probes.verify(probe, line, m["tool"])
                row["answer"] = probes.answer_text(probe, m["tool"])
                battery.append(row)
            if role == "main":
                log.append(row)
                if line:
                    run["said"].append(Exchange(m["t"], run["pending"], player, line))
                    run["pending"] = []
        print(f"moment {k}: {len(battery)} probes so far", flush=True)
    return {"log": log, "battery": battery, "talkers": list(talkers)}


def add_clean(res: dict) -> None:
    """Mark every line ``clean``: it passes the dataset's code checks
    (:func:`checks.line_checks`: its claims, a probe's answer, its form, its
    variety against his recent lines). A reply that recites the game state
    holds the right fact but is not clean."""
    import checks

    prev: dict = defaultdict(list)  # (who, match) -> his lines, by moment
    for r in sorted(res["log"], key=lambda r: r["k"]):
        prev[(r["who"], r["data"], r["ep"])].append((r["k"], r["line"]))
    seen = set()
    for r in [*res["log"], *res["battery"]]:
        if id(r) in seen:
            continue
        seen.add(id(r))
        before = [
            x for k, x in prev[(r["who"], r["data"], r["ep"])] if k < r["k"] and x
        ]
        turn = checks.Turn(
            state=r["state"],
            prev=before,
            past=r["past"],
            player=r["player"],
            utype="probe" if r["probe"] else None,
            probe=r["probe"],
        )
        fails = [name for name, _ in checks.failures(r["line"], turn)]
        r["line_fails"], r["clean"] = fails, bool(r["line"]) and not fails


def report(res: dict) -> dict:
    """Accuracy per type, claims, and the closed loop, per talker. Each
    accuracy twice: correct (the answer holds the right fact) and clean (right,
    and a line he would say: :func:`add_clean`)."""
    add_clean(res)
    out = {}
    for who in res["talkers"]:
        rows = [r for r in res["battery"] if r["who"] == who]
        per = defaultdict(Counter)
        for r in rows:
            per[r["probe"]["type"]][r["verdict"]] += 1
            per[r["probe"]["type"]]["clean"] += r["clean"]
        types = {
            t: {
                "n": sum(c[v] for v in ("correct", "wrong", "abstained")),
                **{
                    v: round(
                        100
                        * c[v]
                        / max(1, sum(c[x] for x in ("correct", "wrong", "abstained"))),
                        1,
                    )
                    for v in ("correct", "clean", "wrong", "abstained")
                },
            }
            for t, c in sorted(per.items())
        }
        total = Counter(r["verdict"] for r in rows)
        lines = [r for r in res["log"] if r["who"] == who]
        remarks = [r for r in lines if r["player"] is None]
        loop = defaultdict(list)
        by_match = defaultdict(list)
        for r in lines:
            by_match[(r["data"], r["ep"])].append(r["line"])
        for ls in by_match.values():
            for s in loop_stats(ls):
                loop[s["half"]].append(s)
        halves = {
            ("first", "second")[h]: {
                "n": len(v),
                "near_repeat": round(
                    100 * sum(s["near"] for s in v) / max(1, len(v)), 1
                ),
                "motif": round(100 * sum(s["motif"] for s in v) / max(1, len(v)), 1),
                "thanks": round(100 * sum(s["thanks"] for s in v) / max(1, len(v)), 1),
                "words": round(sum(s["words"] for s in v) / max(1, len(v)), 1),
            }
            for h, v in sorted(loop.items())
        }
        out[who] = {
            "battery": {
                "n": len(rows),
                "correct": round(100 * total["correct"] / max(1, len(rows)), 1),
                "wrong": round(100 * total["wrong"] / max(1, len(rows)), 1),
                "abstained": round(100 * total["abstained"] / max(1, len(rows)), 1),
                "clean": round(
                    100 * sum(r["clean"] for r in rows) / max(1, len(rows)), 1
                ),
                # The battery's scores, each type weighted alike: correct, and
                # clean (right, and in a line he would say: the one that counts).
                "macro_correct": round(
                    sum(t["correct"] for t in types.values()) / max(1, len(types)), 1
                ),
                "macro_clean": round(
                    sum(t["clean"] for t in types.values()) / max(1, len(types)), 1
                ),
            },
            "lines_clean": round(
                100 * sum(r["clean"] for r in lines) / max(1, len(lines)), 1
            ),
            "remarks_clean": round(
                100 * sum(r["clean"] for r in remarks) / max(1, len(remarks)), 1
            ),
            "types": types,
            "claims_all": round(
                100 * sum(r["claims"] for r in lines) / max(1, len(lines)), 1
            ),
            "claims_remarks": round(
                100 * sum(r["claims"] for r in remarks) / max(1, len(remarks)), 1
            ),
            "loop": halves,
        }
    return out


def table(rep: dict) -> str:
    """The report as markdown: one column per talker."""
    who = list(rep)
    types = sorted({t for w in who for t in rep[w]["types"]})
    head = (
        "| type | "
        + " | ".join(f"{w}: correct (clean) / wrong / abstained (n)" for w in who)
        + " |"
    )
    rows = [head, "|---|" + "---|" * len(who)]
    for t in types:
        cells = []
        for w in who:
            x = rep[w]["types"].get(t)
            cells.append(
                f"{x['correct']:.0f} ({x['clean']:.0f}) / {x['wrong']:.0f} / "
                f"{x['abstained']:.0f} ({x['n']})"
                if x
                else "-"
            )
        rows.append(f"| {t} | " + " | ".join(cells) + " |")
    b = [rep[w]["battery"] for w in who]
    rows.append(
        "| **all** | "
        + " | ".join(
            f"{x['correct']:.1f} (**{x['clean']:.1f}**) / {x['wrong']:.1f} / "
            f"{x['abstained']:.1f} ({x['n']})"
            for x in b
        )
        + " |"
    )
    rows.append(
        "| **macro** (types alike): correct (clean) | "
        + " | ".join(
            f"{x['macro_correct']:.1f} (**{x['macro_clean']:.1f}**)" for x in b
        )
        + " |"
    )
    rows.append(
        "| clean, all lines / remarks | "
        + " | ".join(
            f"{rep[w]['lines_clean']:.1f} / {rep[w]['remarks_clean']:.1f}" for w in who
        )
        + " |"
    )
    rows.append(
        "| claims, all lines | "
        + " | ".join(f"{rep[w]['claims_all']:.1f}" for w in who)
        + " |"
    )
    rows.append(
        "| claims, remarks | "
        + " | ".join(f"{rep[w]['claims_remarks']:.1f}" for w in who)
        + " |"
    )
    for h in ("first", "second"):
        cells = []
        for w in who:
            x = rep[w]["loop"].get(h)
            cells.append(
                f"near {x['near_repeat']:.0f}%, motif {x['motif']:.0f}%, thanks {x['thanks']:.0f}%, {x['words']:.1f} words"
                if x
                else "-"
            )
        rows.append(f"| closed loop, {h} half | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--model", required=True, help="A composed checkpoint with the narrator"
    )
    ap.add_argument("--moments", type=Path, required=True, help="Held-out matches")
    ap.add_argument("--matches", type=int, default=0, help="0: all")
    ap.add_argument("--talkers", default="narrator,base")
    ap.add_argument("--side", type=int, default=1, help="Side probes per moment")
    ap.add_argument("--temperature", type=float, default=0.8, help="As live")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True, help="The report (JSON)")
    ap.add_argument(
        "--rescore",
        type=Path,
        help="Score the rows of an earlier run (its .rows.jsonl) instead of running",
    )
    args = ap.parse_args()
    if args.rescore:
        rows = [json.loads(x) for x in open(args.rescore)]
        res = {
            "log": [r for r in rows if r["role"] == "main"],
            "battery": [r for r in rows if r["probe"] is not None],
            "talkers": list(dict.fromkeys(r["who"] for r in rows)),
        }
    else:
        res = run(args)
    rep = report(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=1) + "\n")
    with open(args.out.with_suffix(".rows.jsonl"), "w") as f:
        for r in res["battery"] + [r for r in res["log"] if r["probe"] is None]:
            f.write(json.dumps(r) + "\n")
    args.out.with_suffix(".md").write_text(table(rep) + "\n")
    print(table(rep))
    for who in res["talkers"]:
        wrong = [r for r in res["battery"] if r["who"] == who and not r["clean"]]
        print(f"== {who}: some answers that were not clean")
        for r in random.Random(0).sample(wrong, min(12, len(wrong))):
            print(
                f"  [{r['probe']['type']}] {r['player']!r} -> {r['line']!r} "
                f"({', '.join(r['line_fails'])}; {r['answer']})"
            )
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
