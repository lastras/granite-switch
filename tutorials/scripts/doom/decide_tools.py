# SPDX-License-Identifier: Apache-2.0
"""What decide.py's records hold, and the checks around them.

``decide.py`` is the tutorial's piece: the teacher at a compilable boundary.
These are the tools around it, in the same environment (the Mellea fork):

* ``report``: the rows by function and match, the prompts' length in tokens,
  the conversation's pushes and events, the teacher's entropy, bytes per row.
* ``parity``: on a sample of records, the chat template on the record's prompt
  against the ids live play serves (:meth:`game_chat.GameChat.prompt_ids`)
  and the ids ``train_alora.py`` trains on: the same, all three.
* ``floor``: a small MLP from the look alone to the teacher's distribution,
  measured on held-out matches: what a student that reads only the present
  can reach (``--history K``: with the previous K looks too).
* ``selfcheck``: the boundary holds nothing the student must not see, each
  call holds the teacher's whole distribution, and a scripted student can
  drive the same loop.

::

    python decide_tools.py report --rows data/s1/decide/shard_*.jsonl.gz
    python decide_tools.py parity --rows data/s1/decide/shard_0.jsonl.gz --n 500
    python decide_tools.py floor --rows data/s1/decide/shard_*.jsonl.gz
    python decide_tools.py selfcheck --teacher runs/teacher_rl0_706M.pt
"""

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import game_chat
import numpy as np
from doom_env import TIC_HZ

MOVE_FN = "Player.move"  # decide.py's move calls, as their records name them


def teacher_id(path: str) -> dict:
    """The teacher checkpoint: its name and the hash of its bytes."""
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    return {"name": Path(path).name, "sha256": h}


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


# ── What the rows hold ─────────────────────────────────────────────────────────
def entropy(dist: dict[str, float]) -> float:
    return -sum(p * math.log(p) for p in dist.values() if p > 0)


def pct(xs, q) -> float:
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def sample_rows(paths, n: int, seed: int = 0) -> list[dict]:
    """``n`` rows drawn evenly from the files (a reservoir: they are read one
    at a time)."""
    rng, out = random.Random(seed), []
    for i, r in enumerate(game_chat.iter_rows(paths)):
        if i < n:
            out.append(r)
        elif (j := rng.randrange(i + 1)) < n:
            out[j] = r
    return out


def report(args) -> None:
    """``report``: the rows by function and match; the prompts' length in
    tokens (a sample), the conversation's pushes per minute and events by
    type, the look's tokens; the teacher's entropy and top-1 probability; and
    bytes per row, on disk and as JSON."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    pieces = game_chat.Pieces(tok)
    by_fn, eps, n_rows = Counter(), set(), 0
    last: dict = {}  # each match's last record: its whole conversation
    teach = defaultdict(lambda: ([], [], Counter()))  # entropy, top-1, played
    for r in game_chat.iter_rows(args.rows):
        n_rows += 1
        by_fn[r["function"]] += 1
        eps.add(r["episode"])
        if r["call"] >= last.get(r["episode"], {}).get("call", -1):
            last[r["episode"]] = r
        h, top, played = teach[r["function"]]
        h.append(entropy(r["meta"]["dist"]))
        top.append(max(r["meta"]["dist"].values()))
        played[r["completions"][0]["completion"]] += 1
    print(f"{n_rows} rows from {len(eps)} matches: {dict(by_fn)}")
    sample = sample_rows(args.rows, args.n)
    disk = sum(Path(p).stat().st_size for p in args.rows)
    raw = np.mean([len(json.dumps(r)) for r in sample])
    print(
        f"bytes per row: {disk / max(1, n_rows):.0f} on disk, {raw:.0f} as JSON "
        f"({raw * n_rows / max(1, disk):.1f}x)"
    )
    lens = [len(record_ids(tok, r)) for r in sample]
    looks = [len(pieces.result(r["prompt"]["messages"][-1]["content"])) for r in sample]
    print(
        f"prompt tokens (of {len(sample)}): p50 {pct(lens, 50):.0f} "
        f"p99 {pct(lens, 99):.0f} max {max(lens)}; a look's turn: p50 "
        f"{pct(looks, 50):.0f} max {max(looks)}"
    )
    pushes, kinds, ends = [], Counter(), []
    for r in last.values():
        msgs = r["prompt"]["messages"][1:-2]  # the system turn; the suffix
        outs = [json.loads(m["content"]) for m in msgs if m["role"] == "tool"]
        minutes = max(1e-9, r["meta"]["t"] / TIC_HZ / 60)
        pushes.append(len(outs) / minutes)
        kinds.update(e["type"] for o in outs for e in o["events"])
        ends.append(len(record_ids(tok, r)))
    print(
        f"pushes per minute: {np.mean(pushes):.1f} (matches {min(pushes):.1f}-"
        f"{max(pushes):.1f}); events {dict(kinds.most_common())}; the prompt at a "
        f"match's last call: {min(ends)}-{max(ends)} tokens"
    )
    for fn in sorted(by_fn):
        h, top, played = teach[fn]
        print(
            f"{fn}: the teacher's entropy {np.mean(h):.3f} nats (p50 {pct(h, 50):.3f}), "
            f"top-1 probability {np.mean(top):.3f}; played {dict(played.most_common(8))}"
        )


def record_ids(tok, r: dict) -> list[int]:
    """The chat template on a record's prompt (with the generation prompt):
    the base-model ids the student trains on."""
    text = tok.apply_chat_template(
        r["prompt"]["messages"],
        tools=r["prompt"]["tools"],
        add_generation_prompt=True,
        tokenize=False,
    )
    return tok(text, add_special_tokens=False).input_ids


# ── The student's prompt against the live one ──────────────────────────────────
def parity(args) -> None:
    """``parity``: on ``--n`` records, the chat template on the record's
    prompt (what was recorded: the compiled view) against
    :meth:`GameChat.prompt_ids`, the ids live play serves (the conversation
    rebuilt from the record's own messages), and against
    ``train_alora.py``'s loader on the same rows: the same ids, all three."""
    import tempfile

    import train_alora
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    pieces = game_chat.Pieces(tok)
    rows = sample_rows(args.rows, args.n)
    bad = 0
    for r in rows:
        want = record_ids(tok, r)
        chat, lk = game_chat.GameChat.from_messages(r["prompt"]["messages"], pieces)
        if chat.prompt_ids(lk) != want:
            bad += 1
            print(f"MISMATCH (GameChat) at {r['call_id']}")
    print(f"{len(rows)} records: {bad} differ from GameChat.prompt_ids")
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
        f.flush()
        loaded = []
        for a in game_chat.OUTPUTS:
            loaded += train_alora.load_record_dists([Path(f.name)], tok, a)
    by_call = {x[5]["call_id"]: x for x in loaded}
    differ = 0
    for r in rows:
        x = by_call[r["call_id"]]
        target = {c["completion"]: c["score"]["value"] for c in r["completions"]}
        if train_alora.record_prompt(x) != record_ids(tok, r) or x[3] != target:
            differ += 1
    print(f"training rows: {len(loaded)} loaded, {differ} differ")
    if bad or differ or len(loaded) != len(rows):
        raise SystemExit(1)


# ── How much the teacher's memory adds ─────────────────────────────────────────
def floor(args) -> None:
    """``floor``: a small MLP from the look alone (the actor's input,
    :func:`game_chat.vector_of_look`) to the teacher's distribution, trained on
    some matches and measured on others: its KL to the teacher estimates what
    a student that reads only the present can reach, the rest being the
    teacher's recurrent memory. Beside it, the KL of the teacher's mean
    distribution (knowing nothing).

    ``--history K``: the same MLP, given the match's previous K looks too
    (the move calls' looks, one every ``--every`` tics: a 5 Hz stream, each
    with a flag for whether it is there; none while dead). How far the KL
    falls below the look alone's is what the recent past says about the
    teacher's choices: an upper bound on what history in the prompt could
    give a student. Each K is a new MLP, on the same split and seed."""
    from train_alora import teacher_ece

    rows = []  # (function, match, tick, the actor's input, the teacher's)
    # Each match's move calls' looks, by tick: its 5 Hz stream.
    stream: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for r in game_chat.iter_rows(args.rows):
        lk = json.loads(r["prompt"]["messages"][-1]["content"])
        v = game_chat.vector_of_look(lk)
        t = r["meta"]["t"]
        rows.append((r["function"], r["episode"], t, v, r["meta"]["dist"]))
        if r["function"] == MOVE_FN:
            stream[r["episode"]][t] = v
    eps = sorted({e for _, e, _, _, _ in rows})
    random.Random(0).shuffle(eps)
    held = set(eps[: max(1, len(eps) // 5)])
    if len(held) == len(eps):
        raise SystemExit("floor needs at least two matches")
    dim = len(rows[0][3])
    blank = np.zeros(dim + 1, np.float32)

    def past(e: str, t: int, k: int) -> np.ndarray:
        """The match's k looks before tick t, newest first, each with its flag."""
        last = (t - 1) // args.every * args.every  # the stream's tick before t
        out = []
        for j in range(k):
            v = stream[e].get(last - j * args.every)
            out.append(blank if v is None else np.append(v, np.float32(1)))
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    for k in args.history:
        span = f"the look and the {k} before it ({k * args.every / TIC_HZ:.1f} s)"
        for fn in sorted({f for f, *_ in rows}):
            rs = [r for r in rows if r[0] == fn]
            words = list(rs[0][4])
            x = np.stack([np.concatenate([v, past(e, t, k)]) for _, e, t, v, _ in rs])
            y = np.array([[d[w] for w in words] for *_, d in rs], np.float32)
            y /= y.sum(1, keepdims=True)
            te = np.array([e in held for _, e, *_ in rs])
            kl, q = fit_floor(x, y, te, args.epochs, args.device)
            mean = y[~te].mean(0)
            kl_mean = float(
                np.mean(
                    np.sum(y[te] * (np.log(np.clip(y[te], 1e-9, 1)) - np.log(mean)), 1)
                )
            )
            top1 = float(np.mean(q.argmax(1) == y[te].argmax(1)))
            print(
                f"{fn}: {int((~te).sum())} train / {int(te.sum())} held-out rows "
                f"({len(held)} matches): {span if k else 'the look alone'}, KL to the "
                f"teacher {kl:.4f} nats, top-1 agreement {top1:.3f}, ECE "
                f"{teacher_ece(q, y[te]):.4f}; the teacher's mean distribution, KL "
                f"{kl_mean:.4f}",
                flush=True,
            )


def fit_floor(x, y, te, epochs: int, device: str) -> tuple[float, np.ndarray]:
    """An MLP (two hidden layers of 256) from ``x`` to the distributions ``y``,
    by soft cross-entropy on the rows not held out (``te``); its best held-out
    KL over the epochs, and its held-out distributions then."""
    import torch

    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, y.shape[1]),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    xt, yt = torch.tensor(x[~te], device=device), torch.tensor(y[~te], device=device)
    xv, yv = torch.tensor(x[te], device=device), torch.tensor(y[te], device=device)
    best = (math.inf, None)
    for _ in range(epochs):
        perm = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), 256):
            j = perm[i : i + 256]
            loss = -(yt[j] * torch.log_softmax(net(xt[j]), -1)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            q = torch.cat(
                [
                    torch.softmax(net(xv[i : i + 8192]), -1)
                    for i in range(0, len(xv), 8192)
                ]
            )
            kl = (yv * (torch.log(yv.clamp(min=1e-9)) - torch.log(q))).sum(-1).mean()
        if kl.item() < best[0]:
            best = (kl.item(), q.cpu().numpy())
    return best


# ── Checks ─────────────────────────────────────────────────────────────────────
def selfcheck(args) -> None:
    """``selfcheck``: decide.py's boundary and records, on a few seconds of play:

    * the compiled view is the conversation and the tools: no distribution, and
      the chat template renders it as it renders
      :meth:`GameChat.prompt_messages` (the trip through ``ChatContext``
      changes nothing the model reads);
    * with :func:`decide.alternatives` and :func:`decide.score`, each move call
      holds 20 completions and each weapon call 7, the first the one returned,
      their values summing to 1;
    * a scripted student drives the same loop for 5 s through
      ``implemented_by``, and its calls are recorded as the teacher's were."""
    import decide
    from mellea.stdlib.compile import Recorder
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    Player = decide.Player
    for fn in (Player.move, Player.weapon):
        assert fn.spec.context == "chat", fn.spec
        assert fn.spec.structured == () and fn.spec.privileged == (), fn.spec
    with Recorder(episode="check") as rec:
        decide.episode(11, args.teacher, args.seconds)
    moves = [r for r in rec.records if r.function == Player.move.name]
    weapons = [r for r in rec.records if r.function == Player.weapon.name]
    assert moves and weapons, (len(moves), len(weapons))

    def render(messages):
        return tok.apply_chat_template(
            messages, tools=game_chat.TOOLS, add_generation_prompt=True, tokenize=False
        )

    for r in rec.records:
        view = r.prompt
        assert set(view) == {"messages", "tools"} and view["tools"] == game_chat.TOOLS
        assert '"dist"' not in json.dumps(view), r.call_id
        look_call = view["messages"][-2]["tool_calls"][0]["function"]["name"]
        assert look_call == game_chat.LOOK_TOOL
        chat, lk = game_chat.GameChat.from_messages(view["messages"])
        assert render(view["messages"]) == render(chat.prompt_messages(lk))
        r.completions += decide.alternatives(r, rec.records)
        for c in r.completions:
            c.score = decide.score(r, c, rec.records)
        n = len(r.meta["dist"])
        assert n == (20 if r in moves else 7) == len({c.value for c in r.completions})
        assert abs(sum(c.score.value for c in r.completions) - 1) < 1e-4
        assert r.completions[0].score.value > 0, "it returned what it could"
    row = moves[0].to_json()
    assert row["privileged"] == {} and json.loads(json.dumps(row)) == row
    print(
        f"teacher: {len(moves)} move and {len(weapons)} weapon calls in "
        f"{args.seconds:.0f} s; the views hold the conversation only; a row is "
        f"{len(json.dumps(row))} bytes"
    )
    # A scripted student drives: the driver cannot tell who answers.
    seen: list[dict] = []

    def student(word):
        def answer(view):
            seen.append(view)
            return word

        return answer

    with (
        Player.move.implemented_by(student("fire")),
        Player.weapon.implemented_by(student("2")),
        Recorder(episode="student") as rec,
    ):
        decide.episode(11, args.teacher, 5.0)
    assert rec.records and all(r.inner == [] and r.meta == {} for r in rec.records)
    assert [r.prompt for r in rec.records] == seen
    said = Counter(r.completions[0].value for r in rec.records)
    assert set(said) == {"fire", "2"}, said
    print(f"student: {len(rec.records)} calls answered by a script, recorded alike")
    print("OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="What written rows hold")
    pa = sub.add_parser("parity", help="Recorded prompts against the served ids")
    fl = sub.add_parser(
        "floor", help="A memoryless MLP on the look, against the teacher"
    )
    sc = sub.add_parser("selfcheck", help="The boundary and the records")
    for p in (r, pa, fl):
        p.add_argument("--rows", type=Path, nargs="+", required=True)
    for p in (r, pa, sc):
        p.add_argument("--tokenizer", default="ibm-granite/granite-4.1-3b")
    r.add_argument("--n", type=int, default=2000, help="Rows to measure in tokens")
    pa.add_argument("--n", type=int, default=500)
    fl.add_argument("--epochs", type=int, default=30)
    fl.add_argument(
        "--history",
        type=int,
        nargs="+",
        default=[0],
        help="Previous looks given too (0: the look alone); several: one MLP each",
    )
    fl.add_argument(
        "--every", type=int, default=7, help="The move calls' cadence, tics"
    )
    fl.add_argument("--device", default="cpu")
    sc.add_argument("--teacher", default="runs/teacher_rl0_706M.pt")
    sc.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()
    {"report": report, "parity": parity, "floor": floor, "selfcheck": selfcheck}[
        args.cmd
    ](args)


if __name__ == "__main__":
    main()
