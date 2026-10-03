# SPDX-License-Identifier: Apache-2.0
"""Orders data: what the partner might say, labeled with the order it gives.

The orders adapter (``policy.ORDERS``) reads the partner's words, as the demo's
speech recognition writes them, and answers one token (:data:`orders.ORDERS`).
Its rows ``{"text", "label", "ep"}``:

* each order: hand-written seeds (:data:`SEEDS`) and paraphrases written by
  Granite 4.2 30B on an OpenAI-compatible server (``--base-url``), told what
  the order means and shown a few seeds; each paraphrase is confirmed by a
  judge (gpt-oss-120b, ``--judge-url``), asked which order the line gives with
  no hint, and kept only if it agrees;
* ``none``: everything else the partner says, so a question never stops him.
  The battery's questions (``probes.PHRASINGS`` and probe_phrasings.json,
  their train split), the partner's other lines in the narrator's data
  (``--partner-rows``: praise, teasing, small talk, ...; not the backseat
  driving or the requests, which are orders now or close to them), things he
  cannot be told to do (:data:`IMPOSSIBLE`: jump, go after Rambo, fly) and talk
  about an order that gives none ("why did you stop", "nice turn"),
  :data:`NONE_SEEDS` and their paraphrases;
* every line in ASR form (``probes.heard``), and ``--mishear`` of the written
  ones with one word misheard (the order stays: the adapter should get it
  anyway).

The hand-written held-out set (:data:`HELDOUT`, never shown to the writer) is
the eval, ``heldout.jsonl``; ``heldout_none.jsonl`` holds the battery's test
phrasings and :data:`HELDOUT_NONE`, for the rate at which talk is read as no
order (an order the partner did not give is worse than a missed one)::

    python orders_data.py write --base-url http://WRITER:PORT/v1 --per-order 300 --out data/orders
    python orders_data.py write --offline --out data/orders_offline   # seeds only, no model
    python orders_data.py eval --model models/doom26-narr7-orders --data data/orders

``eval``: the composed checkpoint's orders adapter on both held-out sets, by
order, with what each miss was read as.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import probes
from orders import ORDER_WORDS
from probes import heard

MEANING = {
    "stop": "stop moving and stand still right now",
    "go": "carry on, he can move again (calling off the stop or whatever you told him)",
    "left": "turn left",
    "right": "turn right",
    "around": "turn around and face the other way",
    "back": "back up, move backwards",
    "ram": "run straight into the wall in front of him",
    "fire": "shoot, right now, wherever he is facing",
    "weapon": "switch weapons (to a gun you name, or just to another gun)",
    "fighter": "play aggressively from now on: hunt the bots, attack",
    "cautious": "play it safe from now on: avoid damage, stay alive, heal",
    "collector": "go collect everything from now on (all the items, ammo, armor, "
    "weapons: a way of playing, not one thing)",
    "fetch": "go get one particular thing he does not have yet: a gun (any, or one "
    "you name), health, armor or ammo",
    "hunt": "go after the bots and kill one (any, or one you name)",
    "explore": "go look around somewhere, wander off and explore",
}
SEEDS = {
    "stop": (
        "stop",
        "stop right there",
        "freeze",
        "hold it",
        "halt",
        "stand still",
        "don't move",
        "wait",
        "stop moving",
        "hold up",
        "stay right there",
        "stop stop stop",
        "whoa, stop",
        "hold your position",
        "hang on, stop",
    ),
    "go": (
        "go",
        "ok go",
        "go go go",
        "carry on",
        "keep going",
        "as you were",
        "move",
        "you can move now",
        "go ahead",
        "resume",
        "alright, get moving",
        "back to it",
        "never mind, go",
        "ok you can go now",
        "unfreeze",
    ),
    "left": (
        "turn left",
        "go left",
        "look left",
        "left",
        "hard left",
        "turn to your left",
        "face left",
        "swing left",
        "spin left",
        "to the left",
        "look to your left",
        "left, left",
        "check your left",
        "rotate left",
        "left now",
    ),
    "right": (
        "turn right",
        "go right",
        "look right",
        "right",
        "hard right",
        "turn to your right",
        "face right",
        "swing right",
        "spin right",
        "to the right",
        "look to your right",
        "right, right",
        "check your right",
        "rotate right",
        "right now, turn right",
    ),
    "around": (
        "turn around",
        "behind you",
        "look behind you",
        "about face",
        "spin around",
        "one eighty",
        "turn around now",
        "check behind you",
        "face the other way",
        "do a 180",
        "look back",
        "flip around",
        "turn all the way around",
        "he's behind you",
        "other way",
    ),
    "back": (
        "back up",
        "back off",
        "go backwards",
        "reverse",
        "step back",
        "back away",
        "move back",
        "walk backwards",
        "back it up",
        "backpedal",
        "take a step back",
        "back back back",
        "scoot back",
        "ease back",
        "get back",
        "reverse now",
        "back back",
        "reverse it",
    ),
    "ram": (
        "ram the wall",
        "run into the wall",
        "hit the wall",
        "charge the wall",
        "headbutt the wall",
        "walk into the wall",
        "go hug that wall",
        "smash into the wall",
        "run straight into that wall",
        "faceplant the wall",
        "bonk the wall",
        "kiss the wall",
        "crash into the wall",
        "slam into the wall",
        "go touch the wall",
    ),
    "fire": (
        "fire",
        "shoot",
        "open fire",
        "shoot now",
        "fire fire fire",
        "pull the trigger",
        "light it up",
        "start shooting",
        "blast away",
        "let them have it",
        "shoot shoot",
        "fire at will",
        "spray",
        "unload",
        "shoot, now",
    ),
    "weapon": (
        "switch weapons",
        "switch to the shotgun",
        "use the rocket launcher",
        "pull out the chaingun",
        "get the bfg out",
        "change guns",
        "use the pistol",
        "plasma rifle now",
        "switch to the plasma",
        "use your fists",
        "try the shotgun",
        "go rockets",
        "swap guns",
        "chaingun please",
        "different gun",
        "fists",
        "go fists",
        "bare knuckles now",
        "punch it out",
    ),
    "fighter": (
        "go get them",
        "be aggressive",
        "attack",
        "hunt them down",
        "go on the offensive",
        "kill everything",
        "go fight",
        "rip and tear",
        "frag them all",
        "take the fight to them",
        "no mercy",
        "stop hiding and fight",
        "get aggressive",
        "go hunting",
        "attack mode",
        "get violent",
        "violence from now on",
        "time to be brutal",
    ),
    "cautious": (
        "play it safe",
        "be careful",
        "stay alive",
        "heal up",
        "avoid them",
        "keep your distance",
        "don't die",
        "play defensive",
        "play safer",
        "lay low",
        "careful now",
        "take it easy",
        "just survive",
        "hang back",
        "stay out of trouble",
    ),
    "collector": (
        "grab the loot",
        "collect stuff",
        "pick up everything",
        "get the items",
        "loot the place",
        "gather supplies",
        "get the goodies",
        "scavenge",
        "go shopping",
        "pick stuff up",
        "hoard everything",
        "collect everything you see",
        "be a collector",
        "sweep the place for items",
        "grab whatever is lying around",
    ),
    "fetch": (
        "grab a gun",
        "get a weapon",
        "go get the rocket launcher",
        "find a shotgun",
        "get the bfg",
        "go grab the plasma rifle",
        "pick up that chaingun",
        "get some health",
        "find a medikit",
        "go get armor",
        "grab the armor",
        "get some ammo",
        "find some shells",
        "grab rockets",
        "get yourself a better gun",
    ),
    "hunt": (
        "go after rambo",
        "go kill someone",
        "hunt them down",
        "get a frag",
        "go after leone",
        "chase that guy",
        "find machete and kill him",
        "go get mcclane",
        "kill somebody",
        "go hunting for anderson",
        "chase them",
        "go after the leader",
        "take somebody out",
        "get me a kill",
        "track down plissken",
        "go get leone",
        "get machete",
        "go get anderson",
    ),
    "explore": (
        "go look around",
        "explore",
        "go somewhere else",
        "get out of here",
        "go that way",
        "wander around",
        "check out the other side",
        "go have a look",
        "move around a bit",
        "go explore the map",
        "scout around",
        "see what is over there",
        "go for a walk",
        "roam around",
        "change scenery",
        "get going somewhere",
        "move it somewhere new",
    ),
}
# What he cannot be told to do (no maneuver for it): read as no order.
IMPOSSIBLE = (
    "jump",
    "jump over it",
    "crouch",
    "duck",
    "fly",
    "go through the door",  # places: the map has no doors, the state no map
    "open the door",
    "press the button",
    "look up",
    "look down",
    "reload",
    "dance",
    "wave at them",
    "pause the game",
    "quit the game",
    "save the game",
    "go to the red room",
    "climb the stairs",
    "take the elevator",
    "hide behind the crate",
    "go upstairs",
    "use the teleporter",
    "type gg",
    "zoom in",
    "drop your gun",
    "throw a grenade",
)
# Talk about an order that gives none, and other chatter.
NONE_SEEDS = (
    "why did you stop",
    "nice turn",
    "you ran into a wall",
    "that was a good shot",
    "you are backing up a lot",
    "what gun is that",
    "do you have the shotgun",
    "are you going left",
    "is that a wall",
    "i said stop earlier",
    "you never listen",
    "that wall again",
    "you stopped",
    "fine whatever",
    "good job",
    "you are on fire",
    "wow",
    "hey",
    "can you hear me",
    "lol",
    "ouch",
    "that was close",
    "you are doing great",
    "i am getting a snack",
    "this game is old",
    "are you an ai",
    "you turned the wrong way",
    "the shotgun is my favorite",
    "you went right past him",
    "i love the rocket launcher",
    # "left" as what remains, "right" as correct: no turn.
    "how much time is left",
    "how many seconds left",
    "how much ammo is left",
    "any rockets left",
    "how many shells left in the shotgun",
    "time left",
    "is there any health left",
    "what is left on the clock",
    "that is right",
    "right on",
    "you are right",
    "alright then",
    "right you are",
    "that was the right call",
    # Score fragments, not hunts or fetches.
    "kill count",
    "frag tally",
    "kills so far",
    "armor amount",
    "points of armor",
    "health points",
)
HELDOUT = {
    "stop": (
        "stop it right now",
        "freeze right there buddy",
        "do not take another step",
        "stay put",
        "hold still a second",
        "quit moving",
        "pause right there",
        "whoa whoa hold on",
    ),
    "go": (
        "ok you are free to go",
        "go on then",
        "alright move along",
        "you can carry on now",
        "get going",
        "ok unpause",
        "fine, go",
        "keep moving",
    ),
    "left": (
        "veer left",
        "swing to your left now",
        "look over to the left",
        "turn left quick",
        "rotate to the left",
        "lefty",
        "face your left side",
        "go to your left",
    ),
    "right": (
        "veer right",
        "swing to your right now",
        "look over to the right",
        "turn right quick",
        "rotate to the right",
        "go to your right",
        "face your right side",
        "hang a right",
    ),
    "around": (
        "turn yourself around",
        "spin a one eighty",
        "look the other way",
        "he is right behind you",
        "face backwards",
        "whip around",
        "turn back around",
        "about turn",
    ),
    "back": (
        "go back a bit",
        "move backwards now",
        "back up slowly",
        "step back from there",
        "reverse reverse",
        "walk back",
        "backwards please",
        "give yourself some room, back up",
    ),
    "ram": (
        "run headfirst into the wall",
        "go bump the wall",
        "walk straight into that wall",
        "charge into the wall",
        "smack into the wall",
        "go face first into the wall",
        "ram into it",
        "boop the wall",
    ),
    "fire": (
        "shoot it",
        "open up on them",
        "pull that trigger",
        "start firing",
        "blast",
        "fire now",
        "pew pew",
        "shoot already",
    ),
    "weapon": (
        "get the shotgun out",
        "go to the rocket launcher",
        "switch to your bfg",
        "change your weapon",
        "pull out the pistol",
        "use the chain gun",
        "fists now",
        "try a different gun",
    ),
    "fighter": (
        "go wreck them",
        "time to get violent",
        "play rough from now on",
        "be the hunter",
        "get in there and fight",
        "go berserk",
        "attack attack",
        "stop playing nice",
    ),
    "cautious": (
        "be safe out there",
        "do not get yourself killed",
        "play it carefully",
        "keep out of fights",
        "go heal",
        "stay safe for a bit",
        "play like a coward",
        "survive first",
    ),
    "collector": (
        "grab everything you see",
        "go pick up the goodies",
        "load up on supplies",
        "go loot",
        "pick up the armor and health",
        "go collect",
        "be a pack rat",
        "take everything that is not nailed down",
    ),
    "fetch": (
        "get a gun already",
        "go find the rocket launcher",
        "grab that shotgun",
        "you need health go get some",
        "pick up some armor",
        "grab some ammo",
        "find a better weapon",
        "get the chain gun",
    ),
    "hunt": (
        "go get rambo",
        "kill that guy",
        "go after macgyver",
        "find someone to shoot",
        "go frag somebody",
        "chase leone down",
        "go hunt",
        "take out the leader",
    ),
    "explore": (
        "go see what is out there",
        "take a look around",
        "go somewhere new",
        "wander off",
        "check the other rooms",
        "get moving somewhere",
        "have a look over there",
        "explore a bit",
    ),
}
HELDOUT_NONE = (
    "why are you not moving",
    "nice shot",
    "that turn was smooth",
    "you hit the wall again",
    "do you have the bfg",
    "what are you doing",
    "i told you to stop",
    "which way are you going",
    "did you see that",
    "you are so slow",
    "you can not jump in this game",
    "go through that doorway",
    "open the door",
    "go up the stairs",
    "look up there",
    "how much health do you have",
    "this is fun",
    "you are terrible at this",
    "good one",
    "what was that",
)

GEN_PROMPT = """Your friend is playing a Doom deathmatch against bots, and you sit \
next to him and tell him what to do, out loud. Write {n} different things you might \
say to him that all mean: {meaning}. Vary the wording, the length (1 to 12 words), \
the tone and the slang: some short and urgent, some bossy, some polite, some joking. \
Spoken English, as you would say it. One per line, nothing else.
Examples:
{examples}"""
GEN_NONE = """Your friend is playing a Doom deathmatch against bots, and you sit next \
to him, talking. Write {n} different things you might say to him that do not tell him \
to do anything right now: {what}. Vary the wording, the length (1 to 12 words) and the \
tone. Spoken English, as you would say it. One per line, nothing else.
Examples:
{examples}"""
NONE_KINDS = (
    "comments on what he just did (a turn, a stop, a shot, running into a wall, "
    "switching guns), without telling him to do it again",
    "questions about the game (the score, who killed him, his health, his guns, "
    "where he is going)",
    "small talk, teasing and reactions (wow, ouch, nice)",
    "asking for things he cannot just be told to do: jump, crouch, open doors, "
    "reload, look up or down, go to a particular place or room",
)
CONFIRM = """Your friend is playing a Doom deathmatch against bots; you sit next to \
him. You say: "{x}"
Which of these are you telling him to do right now? stop (stand still), go (carry on, \
he can move again), left (turn left), right (turn right), around (turn around), back \
(move backwards), ram (run into the wall), fire (shoot), weapon (switch guns), \
fighter (play aggressively from now on), cautious (play it safe from now on), \
collector (collect everything from now on), fetch (go get one thing he lacks: a gun, \
health, armor, ammo), hunt (go after the bots, or one of them, to kill one), explore \
(go look around somewhere), or none (not one of these: a question, a comment, going to \
a particular place, or something else). Answer with the one word only."""
MISHEAR = """A speech recognizer heard this sentence and got exactly one word wrong: \
it swapped it for a similar-sounding real word, and the sentence came out a little \
funny. For example, "go get the rocket launcher" heard as "go get the rocket lunch".
"{x}"
Write the sentence as it was heard, with the same number of words. Output only the \
sentence."""


def clean(line: str) -> str | None:
    t = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"').strip()
    n = len(t.split())
    if not t or n > 15 or len(t) > 100 or ":" in t or probes._META.search(t.lower()):
        return None
    return t


class Writer:
    """The writer (Granite) and the judge (gpt-oss), OpenAI-compatible servers:
    the paraphrases and mishearings, and their confirmation."""

    def __init__(self, url: str, model: str, judge_url: str, judge_model: str):
        from openai import OpenAI

        self.client = OpenAI(base_url=url, api_key="none", timeout=300)
        self.judge = OpenAI(base_url=judge_url, api_key="none", timeout=300)
        self.model, self.judge_model = model, judge_model

    def ask(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """The writer's answer: at "low" effort Granite 4.2 reasons briefly and
        closes </think>; an answer without it is all reasoning (cut short)."""
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort="low",
        )
        text = r.choices[0].message.content or ""
        return text.split("</think>", 1)[1] if "</think>" in text else ""

    def lines(self, prompt: str) -> list[str]:
        return [t for x in self.ask(prompt, 1.0, 1500).splitlines() if (t := clean(x))]

    def label(self, x: str) -> str:
        r = self.judge.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": CONFIRM.format(x=x)}],
            reasoning_effort="low",
            max_tokens=600,
            temperature=0.0,
        )
        words = re.findall(r"[a-z]+", (r.choices[0].message.content or "").lower())
        return words[-1] if words else ""

    def mishear(self, x: str) -> str | None:
        lines = self.ask(MISHEAR.format(x=x), 0.8, 800).strip().splitlines()
        a = heard(lines[-1].strip('"')) if lines else ""
        p, q = x.split(), a.split()
        return (
            a if len(p) == len(q) and sum(u != v for u, v in zip(p, q)) == 1 else None
        )


def write(w: Writer, rng: random.Random, per: int, rounds: int, workers: int):
    """Paraphrases per order and of the none talk, confirmed."""
    held = {heard(t) for ts in HELDOUT.values() for t in ts} | set(
        map(heard, HELDOUT_NONE)
    )
    jobs: dict[str, set[str]] = {k: set() for k in (*SEEDS, "none")}
    with ThreadPoolExecutor(workers) as ex:
        for r in range(rounds):
            prompts, which = [], []
            for k in jobs:
                if len(jobs[k]) >= per:
                    continue
                for _ in range(4):
                    if k == "none":
                        what = rng.choice(NONE_KINDS)
                        ex_ = rng.sample((*NONE_SEEDS, *IMPOSSIBLE), 5)
                        prompts.append(
                            GEN_NONE.format(n=25, what=what, examples="\n".join(ex_))
                        )
                    else:
                        ex_ = rng.sample(SEEDS[k], 5)
                        prompts.append(
                            GEN_PROMPT.format(
                                n=25, meaning=MEANING[k], examples="\n".join(ex_)
                            )
                        )
                    which.append(k)
            if not prompts:
                break
            got = list(ex.map(w.lines, prompts))
            cands = [(k, heard(t)) for k, ts in zip(which, got) for t in ts]
            cands = [(k, t) for k, t in dict.fromkeys(cands) if t and t not in held]
            labels = list(ex.map(w.label, [t for _, t in cands]))
            for (k, t), lab in zip(cands, labels):
                if lab == k:  # the writer, asked blind, gives the same order
                    jobs[k].add(t)
            kept = sum(lab == k for (k, _), lab in zip(cands, labels))
            print(
                f"round {r}: {kept} of {len(cands)} confirmed; "
                + ", ".join(f"{k} {len(v)}" for k, v in jobs.items()),
                flush=True,
            )
            if r == 0:  # what the judge read differently, for a look
                off = [(k, t, lab) for (k, t), lab in zip(cands, labels) if lab != k]
                print(f"  e.g. not confirmed: {off[:12]}", flush=True)
    return {k: sorted(v) for k, v in jobs.items()}


def fill(phrasing: str, rng: random.Random) -> str:
    """A battery phrasing with its slots filled as the partner would say them."""
    gun = rng.choice(probes.WEAPONS[1:])
    slots = {
        "name": lambda: rng.choice(probes.BOT_NAMES[:7]).lower(),
        "n": lambda: str(rng.randint(1, 30)),
        "nth": lambda: rng.choice(("third", "fourth", "fifth")),
        "gun": lambda: gun.lower(),
        "ammo": lambda: probes.GUN_AMMO.get(gun, "bullets"),
    }
    return re.sub(r"\{(\w+)\}", lambda m: slots[m.group(1)](), phrasing)


def partner_none(paths: list[Path]) -> list[str]:
    """The partner's other lines in the narrator's data (not the backseat
    driving or the requests, which give orders or come close)."""
    out = []
    for p in paths:
        for x in open(p):
            r = json.loads(x)
            if r.get("player") and r.get("utype") not in (
                None,
                "probe",
                "backseat",
                "request",
            ):
                out.append(heard(r["player"]))
    return list(dict.fromkeys(out))


def evaluate(args) -> None:
    """The composed checkpoint's orders adapter on heldout.jsonl (each order,
    hand-written) and heldout_none.jsonl (questions and talk: no order)."""
    from policy import VLLMPolicy

    pol = VLLMPolicy(args.model, warmup=2, max_model_len=4096)
    report = {}
    for name in ("heldout", "heldout_none"):
        rows = [json.loads(x) for x in open(args.data / f"{name}.jsonl")]
        got = [pol.order(r["text"]) for r in rows]
        per: dict = {}
        for r, o in zip(rows, got):
            c = per.setdefault(r["label"], Counter())
            c["n"] += 1
            c["right"] += o.kind == r["label"]
            if o.kind != r["label"]:
                c[f"as {o.kind}"] += 1
        right = sum(c["right"] for c in per.values())
        ms = sorted(o.ms for o in got)
        report[name] = {
            "n": len(rows),
            "acc": round(right / len(rows), 4),
            "ms_p50": round(ms[len(ms) // 2], 2),
            "per": {k: dict(c) for k, c in per.items()},
            "misses": [
                (r["text"], r["label"], o.kind)
                for r, o in zip(rows, got)
                if o.kind != r["label"]
            ][:40],
        }
        print(
            f"{name}: {right}/{len(rows)} right ({100 * right / len(rows):.1f}%), "
            f"p50 {report[name]['ms_p50']} ms",
            flush=True,
        )
        for k, c in sorted(per.items()):
            miss = {m: v for m, v in c.items() if m.startswith("as ")}
            print(f"  {k:<10} {c['right']}/{c['n']} {miss or ''}")
        for t, want, kind in report[name]["misses"][:15]:
            print(f"    {t!r}: {want}, read {kind}")
    (args.data / f"eval_{Path(args.model).name}.json").write_text(
        json.dumps(report, indent=1)
    )


def main() -> None:
    top = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = top.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("eval", help="The composed checkpoint on the held-out sets")
    ev.add_argument("--model", required=True)
    ev.add_argument("--data", type=Path, required=True)
    ap = sub.add_parser("write", help="Write the dataset")
    ap.add_argument("--base-url", help="The writer: an OpenAI-compatible server")
    ap.add_argument("--model", default="granite-4.2-30b")
    ap.add_argument("--judge-url", help="The judge: an OpenAI-compatible server")
    ap.add_argument("--judge-model", default="gpt-oss-120b")
    ap.add_argument("--per-order", type=int, default=300)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--partner-rows", type=Path, nargs="*", default=[], help="partner_ivr.py rows"
    )
    ap.add_argument("--max-partner", type=int, default=800)
    ap.add_argument("--mishear", type=float, default=0.1, help="Share misheard")
    ap.add_argument("--offline", action="store_true", help="Seeds only, no model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = top.parse_args()
    if args.cmd == "eval":
        evaluate(args)
        return

    rng = random.Random(args.seed)
    w = (
        None
        if args.offline
        else Writer(args.base_url, args.model, args.judge_url, args.judge_model)
    )
    gen = (
        {k: [] for k in (*SEEDS, "none")}
        if w is None
        else write(w, rng, args.per_order, args.rounds, args.workers)
    )
    rows = [(heard(t), k) for k, ts in SEEDS.items() for t in ts]
    rows += [(t, k) for k, ts in gen.items() for t in ts[: args.per_order]]
    if w is not None and args.mishear:
        picks = [r for r in rows if rng.random() < args.mishear]
        with ThreadPoolExecutor(args.workers) as ex:
            mis = list(ex.map(w.mishear, [t for t, _ in picks]))
        rows += [(m, k) for m, (_, k) in zip(mis, picks) if m]
    # No order: the battery's questions (train split), the partner's other
    # lines, the impossible, the talk about orders.
    asks = [p for t in probes.TYPES if t != "challenge" for p in probes.phrasings(t)]
    asks += [p for f in probes.CHALLENGES for p in probes.phrasings("challenge", f)]
    asks = [fill(p, rng) for p in asks]
    mates = partner_none(args.partner_rows)
    rng.shuffle(mates)
    rows += [(t, "none") for t in (*asks, *mates[: args.max_partner])]
    rows += [(heard(t), "none") for t in (*IMPOSSIBLE, *NONE_SEEDS)]
    test_asks = [heard(p) for ts in HELDOUT.values() for p in ts]
    test_none = [heard(t) for t in HELDOUT_NONE]
    for t in probes.TYPES:
        if t == "challenge":
            continue
        for p in probes.phrasings(t, split="test"):
            test_none.append(fill(p, rng))
    test_none = list(dict.fromkeys(test_none))
    held = set(test_asks) | set(test_none)
    rows = [(t, k) for t, k in dict.fromkeys(rows) if t not in held]
    rng.shuffle(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "train.jsonl", "w") as f:
        for i, (t, k) in enumerate(rows):
            f.write(json.dumps({"text": t, "label": k, "ep": i}) + "\n")
    with open(args.out / "heldout.jsonl", "w") as f:
        i = 0
        for k, ts in HELDOUT.items():
            for t in ts:
                f.write(
                    json.dumps({"text": heard(t), "label": k, "ep": 10_000 + i}) + "\n"
                )
                i += 1
        for t in HELDOUT_NONE:
            f.write(
                json.dumps({"text": heard(t), "label": "none", "ep": 10_000 + i}) + "\n"
            )
            i += 1
    with open(args.out / "heldout_none.jsonl", "w") as f:
        for j, t in enumerate(test_none):
            f.write(json.dumps({"text": t, "label": "none", "ep": 20_000 + j}) + "\n")
    counts = Counter(k for _, k in rows)
    assert set(counts) <= set(ORDER_WORDS), counts
    print(
        f"train {len(rows)} {dict(counts)}; held-out {sum(map(len, HELDOUT.values())) + len(HELDOUT_NONE)} "
        f"hand-written, {len(test_none)} questions and talk (none) -> {args.out}"
    )


if __name__ == "__main__":
    main()
