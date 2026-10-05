# SPDX-License-Identifier: Apache-2.0
"""One round of rejection sampling for the narrator: sample, verify, keep.

On matches the narrator did not train on (``talk.py moments ... --split``,
the pool), each match is played in order as the demo serves it (the composed
checkpoint, the live prompt and sampling), with the scripted partner of
:mod:`eval_probes` (a probe of the game state at about 55% of moments, small
talk at 15%). At every moment the narrator samples ``--n`` replies; each is
checked by the dataset's code checks (:func:`checks.failures`: a probe's
answer, every line's claims, its form and its variety against his recent
lines). Up to ``--keep`` verified replies per prompt are kept, the most
distinct from his recent lines first (the judge's "voice" verdict breaks ties,
with ``--judge-url``).

The kept replies are partner rows (``train_alora.py --extra-rows``). With
``--pairs``, each prompt with a passing and a failing sample gives a DPO pair, by
this criterion (:data:`REJECT_ORDER`):

* chosen: a sample that passes every check, the least like his recent
  lines, then the judge's voice;
* rejected: the failing sample whose failure ranks first: repetition (an
  opening he used, a word in 3+ of his last 8 lines, a near-copy, "Even with
  X, I'm still Y", a weapon again), then an invention (a claim the state does
  not have, a bot nothing happened to), then a wrong answer, then form.

The conversation goes on with the first sample, as live play would say it, so
the prompts hold his own repetitions (the exposure DPO corrects).

The report gives each question type's accuracy over every sample: correct,
wrong, abstained.

    python rft.py --model models/doom-narr6-sft --moments data/narr/moments_v6_pool.jsonl \\
        --out runs/narr6-r1/rft.jsonl --judge-url http://JUDGE:PORT/v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import checks
import probes
from conversation import Conversation, Exchange, narrator_ids
from eval_probes import Partner, clean_line

# What a rejected sample failed, most important first (checks.py's checks).
REJECT_ORDER = (
    (
        "repetition",
        (
            "fresh_opening",
            "no_motif",
            "not_repeat",
            "even_still",
            "still_again",
            "weapon_again",
        ),
    ),
    ("invention", ("claims", "stale_bot")),
    ("wrong answer", ("answer",)),
    ("form", tuple(f.__name__ for f in checks.CASES)),  # any other check
)


def reject_rank(fails: list[str]) -> tuple[int, str] | None:
    """(rank, category) of a failing sample's most important failure."""
    for i, (cat, names) in enumerate(REJECT_ORDER):
        if any(f in names for f in fails):
            return i, cat
    return None


VOICE_Q = (
    "A calm, dry player in a Doom deathmatch says this to his partner, who sits next "
    'to him:\n"{line}"\n\n' + checks.JUDGE["voice"][1] + " Answer YES or NO."
)


def check(
    line: str, m: dict, probe, player, past, prev
) -> tuple[str | None, bool, list[str]]:
    """A probe's verdict (None if none), whether the line passes every code
    check, and the names of those it fails."""
    verdict = probes.verify(probe, line, m["tool"])[0] if probe else None
    turn = checks.Turn(
        state=m["tool"],
        prev=prev,
        past=past,
        player=player,
        utype="probe" if probe else None,
        probe=probe,
    )
    fails = [name for name, _ in checks.failures(line, turn)]
    return verdict, bool(line) and not fails, fails


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--model", required=True, help="A composed checkpoint with the narrator"
    )
    ap.add_argument("--moments", type=Path, required=True, help="The pool's matches")
    ap.add_argument("--matches", type=int, default=0, help="0: all")
    ap.add_argument("--n", type=int, default=8, help="Samples per prompt")
    ap.add_argument(
        "--keep", type=int, default=2, help="Verified replies kept per prompt"
    )
    ap.add_argument("--temperature", type=float, default=0.8, help="As live")
    ap.add_argument("--judge-url", help="The judge, for the voice tiebreak (optional)")
    ap.add_argument("--judge-model", default="gpt-oss-120b")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument(
        "--out", type=Path, required=True, help="Kept replies (partner rows)"
    )
    ap.add_argument(
        "--pairs", type=Path, help="DPO pairs (chosen, rejected) by REJECT_ORDER"
    )
    args = ap.parse_args()

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
    sp = pol.talk_params(max_tokens=32, temperature=args.temperature, n=args.n)
    judge = None
    if args.judge_url:
        from openai import OpenAI

        judge = OpenAI(base_url=args.judge_url, api_key="none", timeout=300)

    def voice(line: str) -> int:
        if judge is None:
            return 0
        r = judge.chat.completions.create(
            model=args.judge_model,
            messages=[{"role": "user", "content": VOICE_Q.format(line=line)}],
            reasoning_effort="low",
            max_tokens=800,
            temperature=0.0,
        )
        return int("YES" in (r.choices[0].message.content or "").upper()[-20:])

    runs = [
        {
            "mt": mt,
            "said": [],
            "pending": [],
            "prev": [],
            "partner": Partner((mt["data"], mt["ep"]), weights, args.seed),
        }
        for mt in matches
    ]

    def kept_row(run, m, conv, probe, player, line) -> dict:
        """A partner row (train_alora.py --extra-rows) for one reply."""
        return {
            "data": run["mt"]["data"],
            "style": run["mt"]["style"],
            "ep": run["mt"]["ep"],
            "t": m["t"],
            "kind": "reply" if player else "remark",
            "utype": "probe" if probe else ("small_talk" if player else None),
            "probe": probe,
            "player": player,
            "line": line,
            "ok": True,
            "conv": conv.to_json(),
            "tool": m["tool"],
            "moment": run["pending"],
            "prev": run["prev"][-5:],
            "rft_round": args.round,
        }

    pairs: list[dict] = []
    pair_kinds: Counter = Counter()
    per_type: dict[str, Counter] = defaultdict(Counter)
    per_kind: dict[str, Counter] = defaultdict(Counter)
    kept_rows = []
    pool = ThreadPoolExecutor(32)
    for k in range(max(len(mt["moments"]) for mt in matches)):
        batch = []
        for run in runs:
            ms = run["mt"]["moments"]
            if k >= len(ms):
                continue
            m = ms[k]
            run["pending"] = run["pending"] + list(m.get("moment") or [])
            conv = Conversation(run["said"])
            probe, player = run["partner"].say(m)
            batch.append((run, m, conv, probe, player))
        if not batch:
            break
        prompts = [
            narrator_ids(pol.tok, c, m["tool"], p, NARRATOR) for _, m, c, _, p in batch
        ]
        outs = pol.run(prompts, [sp] * len(prompts))
        for (run, m, conv, probe, player), o in zip(batch, outs):
            past = [ex.events for ex in conv]
            cands = list(dict.fromkeys(clean_line(c.text) for c in o.outputs))
            judged, failed = [], []
            for line in cands:
                verdict, ok, fails = check(line, m, probe, player, past, run["prev"])
                if not ok and line and (rk := reject_rank(fails)):
                    failed.append((rk, line))
                if probe is not None:
                    per_type[probe["type"]][verdict] += 1
                kind = "probe" if probe else ("reply" if player else "remark")
                per_kind[kind]["passed" if ok else "failed"] += 1
                if ok:
                    near = max(
                        (checks.jaccard(line, b) for b in run["prev"][-8:]),
                        default=0.0,
                    )
                    judged.append((line, near))
            voices = list(pool.map(voice, [x for x, _ in judged])) if judged else []
            best = sorted(zip(judged, voices), key=lambda jv: (jv[0][1], -jv[1]))
            keep = [line for (line, _), _ in best[: args.keep]]
            for line in keep:
                kept_rows.append(kept_row(run, m, conv, probe, player, line))
            if args.pairs and best and failed:
                chosen = best[0][0][0]
                (_, why), rejected = min(failed, key=lambda f: f[0][0])
                if rejected:
                    pairs.append(
                        {
                            **kept_row(run, m, conv, probe, player, chosen),
                            "chosen": chosen,
                            "rejected": rejected,
                            "why": why,
                        }
                    )
                    pair_kinds[why] += 1
            # The conversation goes on as live play would: with the first sample.
            said = next((c for c in cands if c), "")
            if said:
                run["prev"].append(said)
                run["said"].append(Exchange(m["t"], run["pending"], player, said))
                run["pending"] = []
        print(f"moment {k}: {len(kept_rows)} replies kept", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in kept_rows)
    if args.pairs:
        with open(args.pairs, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in pairs)
        print(
            f"{len(pairs)} DPO pairs, by what the rejected sample did: {dict(pair_kinds)} -> {args.pairs}"
        )
    report = {
        "types": {
            t: {
                "n": sum(c.values()),
                **{
                    v: round(100 * c[v] / max(1, sum(c.values())), 1)
                    for v in ("correct", "wrong", "abstained")
                },
            }
            for t, c in sorted(per_type.items())
        },
        "kinds": {k: dict(c) for k, c in per_kind.items()},
        "kept": len(kept_rows),
        "prompts_with_one_kept": len({(r["data"], r["ep"], r["t"]) for r in kept_rows}),
    }
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(
        f"{'type':<14} {'n':>5} {'correct':>8} {'wrong':>7} {'abstained':>10}  (over every sample)"
    )
    for t, x in report["types"].items():
        print(
            f"{t:<14} {x['n']:>5} {x['correct']:>7.1f}% {x['wrong']:>6.1f}% {x['abstained']:>9.1f}%"
        )
    for kind, c in report["kinds"].items():
        n = sum(c.values())
        print(
            f"{kind}: {100 * c.get('passed', 0) / max(1, n):.1f}% of {n} samples pass every check"
        )
    print(
        f"{len(kept_rows)} replies kept, for {report['prompts_with_one_kept']} prompts -> {args.out}"
    )


if __name__ == "__main__":
    main()
