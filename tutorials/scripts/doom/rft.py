# SPDX-License-Identifier: Apache-2.0
"""One round of rejection sampling for the narrator: sample, verify, keep.

On matches the narrator did not train on (``talk.py moments ... --split``,
the pool), each match is played in order as the demo serves it (the composed
checkpoint, the live prompt and sampling), with the scripted partner of
:mod:`eval_probes` (a probe of the game state at about 55% of moments, small
talk at 15%). At every moment the narrator samples ``--n`` replies; each is
checked in code: a probe's answer (:func:`probes.verify`), every line's claims
(:func:`probes.claims`), and the dataset's code checks (length, numbers only
where asked, no stock phrase, calm, mild, not a film line, not a repeat:
:func:`partner_ivr.code_fns`). Up to ``--keep`` verified replies per prompt
are kept, the most distinct from his recent lines first (the judge's "voice"
verdict breaks ties, with ``--judge-url``). The conversation goes on with the
first one kept (or, if none passed, the first sample, as live would say it),
so the prompts are the narrator's own.

The kept replies are partner rows (``train_alora.py --extra-rows``). The report
gives each question type's accuracy over every sample: correct, wrong,
abstained.

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

import probes
from conversation import Conversation, Exchange, narrator_ids
from eval_probes import Partner, clean_line, jaccard
from partner_ivr import JUDGE_ALL, code_fns, probe_weights

VOICE_Q = (
    "A calm, dry player in a Doom deathmatch says this to his partner, who sits next "
    'to him:\n"{line}"\n\n' + JUDGE_ALL["voice"] + " Answer YES or NO."
)


def checks(
    line: str, m: dict, probe, player, past, prev
) -> tuple[str | None, bool, list[str]]:
    """A probe's verdict (None if none), whether the line passes every check,
    and what failed."""
    fails = []
    verdict = None
    if probe is not None:
        verdict, why = probes.verify(probe, line, m["tool"])
        if verdict != probes.CORRECT:
            fails.append(f"verify: {verdict}")
    ok, why = probes.claims(line, m["tool"], past, player or "")
    if not ok:
        fails.append("claims")
    kind = "reply" if player else "remark"
    for d, f in code_fns(kind, prev, player, (), probe):
        if not f(line)[0]:
            fails.append(d)
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
    args = ap.parse_args()

    from policy import NARRATOR, VLLMPolicy

    matches = [json.loads(x) for x in open(args.moments)][: args.matches or None]
    weights = probe_weights([m for mt in matches for m in mt["moments"]])
    pol = VLLMPolicy(
        args.model,
        layout="chat",
        max_model_len=8192,
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
            judged = []
            for line in cands:
                verdict, ok, fails = checks(line, m, probe, player, past, run["prev"])
                if probe is not None:
                    per_type[probe["type"]][verdict] += 1
                kind = "probe" if probe else ("reply" if player else "remark")
                per_kind[kind]["passed" if ok else "failed"] += 1
                if ok:
                    near = max(
                        (jaccard(line, b) for b in run["prev"][-8:]), default=0.0
                    )
                    judged.append((line, near))
            voices = list(pool.map(voice, [x for x, _ in judged])) if judged else []
            best = sorted(zip(judged, voices), key=lambda jv: (jv[0][1], -jv[1]))
            keep = [line for (line, _), _ in best[: args.keep]]
            for line in keep:
                kept_rows.append(
                    {
                        "data": run["mt"]["data"],
                        "style": run["mt"]["style"],
                        "ep": run["mt"]["ep"],
                        "t": m["t"],
                        "kind": "reply" if player else "remark",
                        "utype": "probe"
                        if probe
                        else ("small_talk" if player else None),
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
                )
            said = keep[0] if keep else next((c for c in cands if c), "")
            if said:
                run["prev"].append(said)
                run["said"].append(Exchange(m["t"], run["pending"], player, said))
                run["pending"] = []
        print(f"moment {k}: {len(kept_rows)} replies kept", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in kept_rows)
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
