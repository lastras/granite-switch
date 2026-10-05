# SPDX-License-Identifier: Apache-2.0
"""Orders data: what the partner might say, labeled with the order it gives.

The orders adapter (``policy.ORDERS``) reads the partner's words, as the demo's
speech recognition writes them, and answers one token (:data:`orders.ORDERS`);
its probability should mean what it says (an order read at p = 0.9 right about
90% of the time). Its rows ``{"text", "label", "ep", "source"}``, all in ASR form
(``probes.heard``):

* **each order:** the hand-written :data:`SEEDS`, and paraphrases the writer
  (Granite 4.2 30B) gives for what the order means (:func:`paraphrase_order`,
  shown a few seeds), each confirmed blind by the judge (gpt-oss-120b,
  :func:`read_order`: which order do these words give?) and kept only if it
  agrees; ``--mishear`` of them again with one word misheard
  (``narrator_data.mishear``: the order stays);
* **none:** everything else the partner says, so talk never moves him: the
  battery's questions (``probes.phrasings``, the train split), the partner's
  chatter (``narrator_data.partner_says``: praise, teasing, worry, small talk,
  ...; not the backseat driving or the requests, which are orders or close),
  :data:`NONE_SEEDS` and their paraphrases, :data:`IMPOSSIBLE` (what no order
  can do), and the hard negatives, about a quarter of the none rows:
  :data:`FILLERS`, :data:`FRAGMENTS`, :data:`ASR_GARBAGE`, the
  :data:`LIVE_MISFIRES`;
* **negations** (:data:`NEGATIONS`: "don't stop", "never mind"): labeled by
  the judge, blind, whatever it reads.

Held out, hand-written and never shown to the writer: ``heldout.jsonl`` (each
order, :data:`HELDOUT`, and :data:`HELDOUT_NONE`), ``heldout_none.jsonl`` (the
battery's test phrasings and :data:`HELDOUT_NONE`) and ``heldout_hard.jsonl``
(:data:`HELDOUT_HARD`: fillers, fragments, ASR noise, misheard orders)::

    python orders_data.py write --writer-url http://WRITER:PORT/v1 \\
        --judge-url http://JUDGE:PORT/v1 --per-order 300 --out data/r9/orders
    python orders_data.py write --offline --out data/orders_offline   # no model
    python orders_data.py eval --model models/doom26-r9 --data data/r9/orders

``eval``: the composed checkpoint's orders adapter on the held-out sets, by
order, with what each miss was read as; its calibration (reliability by
confidence, ECE, Brier, none read as an order at p >= 0.9); and its read of
each live misfire. ``write`` needs Mellea (``pip install mellea==0.7.0``),
``eval`` the demo's environment.
"""

import argparse
import functools
import json
import os
import random
import re
import sys
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

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

# ── Hard negatives: none of them gives an order ────────────────────────────────
# Fillers and reactions.
FILLERS = (
    *("no", "yes", "yeah", "yep", "nope", "okay", "ok", "okay okay", "thanks"),
    *("thank you", "thank", "awesome", "cool", "nice", "wow", "great", "sure"),
    *("fine", "uh huh", "hmm", "um", "uh", "oh", "ah", "huh", "really", "seriously"),
    *("lol", "haha", "oh no", "oh wow", "yikes", "dang", "sweet", "perfect", "good"),
    *("whatever", "alright", "got it", "i see", "oh okay", "no no", "yes yes"),
    *("ok cool", "no way", "yeah yeah", "sure thing", "of course", "makes sense"),
)
# Pieces of a sentence, as speech recognition cuts them off.
FRAGMENTS = (
    *("it", "one", "0", "for it", "y it", "the", "a", "to", "and", "so", "that"),
    *("this one", "that one", "for", "at", "in", "of", "1", "2", "3", "7", "10"),
    *("100", "zero", "two", "it is", "is it", "to it", "the the", "and then"),
    *("so the", "but the", "with the", "for the", "on the", "in the"),
)
# What speech recognition makes of noise, a cough, the game's sound.
ASR_GARBAGE = (
    *("we get the solar community", "the bus is the", "and the then", "so the the"),
    *("uh the a", "is it the", "what the one", "mm hmm yeah", "the war of"),
    *("at the at the", "by the way the", "so so so", "it was the", "i i i"),
    *("okay so the", "the sun of", "and a half", "when the a", "it is a the"),
    *("you know the", "that is the the", "for a for a", "the other the"),
)
# Heard live and read as an order when none was given (the live logs of the
# narr7-orders3 sessions): in the training data, and eval reads each again.
LIVE_MISFIRES = (
    *("no", "0", "it", "one", "for it", "y it", "thank"),
    *("we get the solar community", "doing nothing"),
)
# Talk with a "not" or a "never mind" in it: the judge reads each, blind.
NEGATIONS = (
    *("do not stop", "don't stop", "do not turn", "don't shoot", "do not shoot"),
    *("don't go", "do not go", "no don't", "never mind", "do not do that"),
    *("do not ram the wall", "don't back up", "no not that way", "don't switch"),
    *("do not pick that up", "don't go after him", "never mind that", "forget it"),
    *("cancel that", "no wait", "not now", "don't do it", "do not move yet"),
    *("never mind the gun", "no do not turn around", "stop stopping"),
)
# Hard cases held out (never written for training): fillers, fragments and
# noise unlike the lists above, and orders with a misleading or misheard word.
HELDOUT_HARD = {
    "none": (
        *("nah", "yes please", "uh huh sure", "okie dokie", "thanks man"),
        *("that is awesome", "cool cool", "oh nice one", "oops", "aw man"),
        *("it it", "the one", "and it", "4", "twelve", "a one", "so it is"),
        *("the solar is", "so then we had the", "and the uh", "it is the thing"),
        *("hello hello", "is this on", "testing testing", "my bad"),
        *("you are doing nothing", "right on time", "left it there", "go figure"),
    ),
    "stop": ("stop right now", "wait wait wait", "hold it right there"),
    "left": ("turn left right now", "go left go left"),
    "right": ("right right right", "turn right now"),
    "fetch": ("go get the rocket lunch", "get some health right now"),
    "ram": ("ram the mall", "go ram it"),
    "weapon": ("use the shot gun", "switch to your fist"),
    "around": ("turn a round", "look behind you right now"),
    "back": ("back it up back it up", "back up right now"),
    "go": ("ok ok go", "go go"),
}

NONE_KINDS = (
    "comments on what he just did (a turn, a stop, a shot, running into a wall, "
    "switching guns), without telling him to do it again",
    "questions about the game (the score, who killed him, his health, his guns, "
    "where he is going)",
    "small talk, teasing and reactions (wow, ouch, nice)",
    "asking for things he cannot just be told to do: jump, crouch, open doors, "
    "reload, look up or down, go to a particular place or room",
)

# What the judge is told each order means, to read the partner's words blind.
READ = {
    **MEANING,
    "none": "not one of these: a question, a comment, going to a "
    "particular place, something he cannot do, or anything else",
}
# The partner's chatter (narrator_data.partner_says): not the backseat driving
# or the requests, which are orders or close to them.
CHATTER_KINDS = (
    "praise",
    "tease",
    "worry",
    "what_happened",
    "greeting",
    "identity",
    "odd",
    "smalltalk",
)
HARD_SHARE = 0.25  # the hard negatives' share of the none rows


# ── The writer and the judge: Mellea generative stubs ──────────────────────────
def paraphrase_order(order: str, meaning: str, examples: list[str]) -> list[str]:
    """Twenty-five different things a person might say out loud to a friend
    who is playing a Doom deathmatch against bots, sitting next to him, that
    all mean: ``meaning``. Like ``examples``: spoken English, 1 to 12 words
    each, varied in wording, length, tone and slang (some short and urgent,
    some bossy, some polite, some joking)."""


def read_order(words: str, orders: dict[str, str]) -> Literal[ORDER_WORDS]:
    """What a friend playing a Doom deathmatch against bots would understand
    he was told to do, when the person sitting next to him says ``words`` (as
    speech recognition wrote them).

    ``orders`` maps each order's name to what it means. Read the words for
    their meaning, as a person would, whatever the wording: synonyms, slang,
    a joke, a word misheard ("freeze" and "halt" mean stop, "light it up"
    means fire). Return the name of the order they give, or "none" when they
    give none of them (a question, a comment, going to a particular place,
    something he cannot do)."""


@functools.cache
def stubs():
    """:func:`paraphrase_order` and :func:`read_order` as Mellea generative
    stubs (Mellea imported here: ``eval`` runs without it)."""
    from mellea import generative

    return generative(paraphrase_order), generative(read_order)


def ask(fn, session, seed: int, **kw):
    """One stub call (its answer, or None if it was not the JSON asked for)."""
    from mellea.backends import ModelOption

    opts = {ModelOption.TEMPERATURE: 1.0, ModelOption.MAX_NEW_TOKENS: 1500}
    if fn.__name__ == "read_order":  # the judge: reasons, then answers
        opts = {
            ModelOption.THINKING: "low",
            ModelOption.TEMPERATURE: 0.0,
            ModelOption.MAX_NEW_TOKENS: 2000,
        }
    try:
        return fn(session, model_options={**opts, ModelOption.SEED: seed}, **kw)
    except ValueError:
        return None


def zlib_seed(key) -> int:
    """A request's seed, from what it asks."""
    return zlib.crc32(repr(key).encode())


def ok_text(t: str) -> bool:
    """A paraphrase worth keeping: short, one utterance, no writer's notes."""
    return (
        0 < len(t.split()) <= 15
        and len(t) <= 100
        and ":" not in t
        and not probes._META.search(t)
    )


def write(args, rng: random.Random, held: set[str]) -> list[tuple[str, str, str]]:
    """The written rows, as (text, label, source): paraphrases per order and
    of the none talk, each confirmed blind by the judge; the partner's
    chatter; the negations, as the judge reads them; mishearings."""
    import narrator_prompts as P
    from mellea import start_session
    from narrator_data import mishear, one_word_off, partner_says

    paraphrase, read = stubs()
    first = lambda urls: urls.split(",")[0]  # noqa: E731 (one server of each is enough)
    writer = start_session(
        "openai",
        model_id=args.writer_model,
        base_url=first(args.writer_url),
        api_key="none",
    )
    judge = start_session(
        "openai",
        model_id=args.judge_model,
        base_url=first(args.judge_url),
        api_key="none",
    )
    pool = ThreadPoolExecutor(args.workers)
    kept: dict[str, set[str]] = {k: set() for k in (*SEEDS, "none")}
    for r in range(args.rounds):
        jobs = []
        for k in kept:
            if len(kept[k]) >= args.per_order:
                continue
            for j in range(4):
                if k == "none":
                    meaning = "nothing he is told to do right now: " + rng.choice(
                        NONE_KINDS
                    )
                    shown = rng.sample((*NONE_SEEDS, *IMPOSSIBLE), 5)
                else:
                    meaning, shown = MEANING[k], rng.sample(SEEDS[k], 5)
                jobs.append(
                    (
                        k,
                        dict(order=k, meaning=meaning, examples=shown),
                        (r, k, j, args.seed),
                    )
                )
        if not jobs:
            break
        got = pool.map(
            lambda jb: ask(paraphrase, writer, zlib_seed(jb[2]), **jb[1]) or [], jobs
        )
        cands = [
            (k, heard(t)) for (k, _, _), ts in zip(jobs, got) for t in ts if ok_text(t)
        ]
        cands = [(k, t) for k, t in dict.fromkeys(cands) if t and t not in held]
        labels = list(
            pool.map(lambda kt: ask(read, judge, 0, words=kt[1], orders=READ), cands)
        )
        for (k, t), lab in zip(cands, labels):
            if lab == k:  # the judge, reading blind, gives the same order
                kept[k].add(t)
        print(
            f"round {r}: {sum(lab == k for (k, _), lab in zip(cands, labels))} of "
            f"{len(cands)} confirmed; "
            + ", ".join(f"{k} {len(v)}" for k, v in kept.items()),
            flush=True,
        )
        if r == 0:
            off = [(k, t, lab) for (k, t), lab in zip(cands, labels) if lab != k]
            print(f"  e.g. read otherwise: {off[:12]}", flush=True)
    rows = [
        (t, k, "paraphrase")
        for k, ts in kept.items()
        for t in sorted(ts)[: args.per_order]
    ]
    # The partner's chatter: none.
    jobs = [(kind, i) for kind in CHATTER_KINDS for i in range(args.chatter)]
    said = pool.map(
        lambda ki: ask(
            partner_says,
            writer,
            zlib_seed((*ki, args.seed)),
            kind=ki[0],
            how=P.UTTERANCES[ki[0]][1],
            game_state="(the match on his screen)",
            conversation="(nothing yet)",
        )
        or [],
        jobs,
    )
    rows += [(heard(t), "none", "chatter") for ts in said for t in ts if ok_text(t)]
    # The negations: as the judge reads them.
    labels = pool.map(
        lambda t: ask(read, judge, 0, words=heard(t), orders=READ), NEGATIONS
    )
    rows += [(heard(t), lab, "negation") for t, lab in zip(NEGATIONS, labels) if lab]
    # Mishearings of the orders' words: the order stays.
    orders_ = [(t, k) for t, k, _ in rows if k != "none"] + [
        (heard(t), k) for k, ts in SEEDS.items() for t in ts
    ]
    picks = [(t, k) for t, k in orders_ if rng.random() < args.mishear]
    mis = pool.map(
        lambda tk: ask(mishear, writer, zlib_seed((tk[0], args.seed)), sentence=tk[0]),
        picks,
    )
    rows += [
        (heard(m), k, "misheard")
        for m, (t, k) in zip(mis, picks)
        if m and one_word_off(t, heard(m))
    ]
    return rows


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


# ── Eval: the composed checkpoint on the held-out sets ─────────────────────────
HELD = ("heldout", "heldout_none", "heldout_hard")
BINS = 10  # reliability bins, by the confidence of the read


def calibration(reads: list[tuple[str, object]]) -> dict:
    """Reliability by confidence, ECE and Brier over (label, read) pairs; and
    how often talk (label none) is read as an order at p >= 0.9."""
    bins = [[0, 0, 0.0] for _ in range(BINS)]  # n, right, sum of p
    brier = 0.0
    for label, o in reads:
        b = bins[min(BINS - 1, int(o.prob * BINS))]
        b[0] += 1
        b[1] += o.kind == label
        b[2] += o.prob
        brier += sum((o.probs.get(c, 0.0) - (c == label)) ** 2 for c in ORDER_WORDS)
    n = max(1, len(reads))
    ece = sum(abs(b[1] - b[2]) for b in bins) / n
    talk = [o for label, o in reads if label == "none"]
    loud = sum(o.kind != "none" and o.prob >= 0.9 for o in talk)
    return {
        "n": len(reads),
        "ece": round(ece, 4),
        "brier": round(brier / n, 4),
        "none_as_order_p90": round(loud / max(1, len(talk)), 4),
        "bins": [
            {
                "p": f"{i / BINS:.1f}-{(i + 1) / BINS:.1f}",
                "n": b[0],
                "acc": round(b[1] / b[0], 3) if b[0] else None,
                "mean_p": round(b[2] / b[0], 3) if b[0] else None,
            }
            for i, b in enumerate(bins)
        ],
    }


def evaluate(args) -> None:
    """The orders adapter of a composed checkpoint on heldout.jsonl,
    heldout_none.jsonl and heldout_hard.jsonl: accuracy by order, what each
    miss was read as, the calibration, and its read of each live misfire."""
    from policy import VLLMPolicy

    pol = VLLMPolicy(args.model, warmup=2, max_model_len=4096)
    report, reads = {}, []
    for name in HELD:
        path = args.data / f"{name}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(x) for x in open(path)]
        got = [pol.order(r["text"]) for r in rows]
        reads += [(r["label"], o) for r, o in zip(rows, got)]
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
                (r["text"], r["label"], o.kind, round(o.prob, 3))
                for r, o in zip(rows, got)
                if o.kind != r["label"]
            ][:40],
            "calibration": calibration([(r["label"], o) for r, o in zip(rows, got)]),
        }
        print(
            f"{name}: {right}/{len(rows)} right ({100 * right / len(rows):.1f}%), "
            f"p50 {report[name]['ms_p50']} ms",
            flush=True,
        )
        for k, c in sorted(per.items()):
            miss = {m: v for m, v in c.items() if m.startswith("as ")}
            print(f"  {k:<10} {c['right']}/{c['n']} {miss or ''}")
        for t, want, kind, p in report[name]["misses"][:15]:
            print(f"    {t!r}: {want}, read {kind} (p {p})")
    cal = report["all"] = calibration(reads)
    print(
        f"\ncalibration over every held-out row (n={cal['n']}): ECE {cal['ece']}, "
        f"Brier {cal['brier']}, talk read as an order at p >= 0.9: "
        f"{100 * cal['none_as_order_p90']:.1f}%"
    )
    print(
        "  "
        + "  ".join(
            f"{b['p']}: {b['n']} at {b['acc']} (p {b['mean_p']})"
            for b in cal["bins"]
            if b["n"]
        )
    )
    live = [(t, pol.order(heard(t))) for t in LIVE_MISFIRES]
    report["live_misfires"] = [(t, o.kind, round(o.prob, 3)) for t, o in live]
    print(
        "live misfires, read again: "
        + "; ".join(f"{t!r} {o.kind} p {o.prob:.2f}" for t, o in live)
    )
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
    ap.add_argument("--writer-url", help="The writer: an OpenAI-compatible server")
    ap.add_argument("--writer-model", default="granite-4.2-30b")
    ap.add_argument("--judge-url", help="The judge: an OpenAI-compatible server")
    ap.add_argument("--judge-model", default="gpt-oss-120b")
    ap.add_argument(
        "--per-order", type=int, default=300, help="Paraphrases kept per order"
    )
    ap.add_argument(
        "--rounds", type=int, default=8, help="At most so many rounds of them"
    )
    ap.add_argument("--chatter", type=int, default=30, help="Chatter requests per kind")
    ap.add_argument(
        "--mishear", type=float, default=0.1, help="Share of order lines misheard"
    )
    ap.add_argument("--workers", type=int, default=32, help="Requests in flight")
    ap.add_argument(
        "--offline", action="store_true", help="Hand-written lists only, no model"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = top.parse_args()
    if args.cmd == "eval":
        evaluate(args)
        return
    if not args.offline:
        from mellea.core import MelleaLogger

        MelleaLogger.get_logger().setLevel("WARNING")
    rng = random.Random(args.seed)
    # Held out: never in training (the writer's paraphrases that match are dropped).
    test_orders = [(heard(t), k) for k, ts in HELDOUT.items() for t in ts]
    test_orders += [(heard(t), "none") for t in HELDOUT_NONE]
    test_none = [heard(t) for t in HELDOUT_NONE]
    for t in probes.TYPES:
        if t != "challenge":
            test_none += [fill(p, rng) for p in probes.phrasings(t, split="test")]
    test_none = list(dict.fromkeys(test_none))
    test_hard = [(heard(t), k) for k, ts in HELDOUT_HARD.items() for t in ts]
    held = {t for t, _ in test_orders} | set(test_none) | {t for t, _ in test_hard}
    rows = [(heard(t), k, "seed") for k, ts in SEEDS.items() for t in ts]
    if not args.offline:
        rows += write(args, rng, held)
    # None: the battery's questions (train split), the impossible, the talk.
    asks = [p for t in probes.TYPES if t != "challenge" for p in probes.phrasings(t)]
    asks += [p for f in probes.CHALLENGES for p in probes.phrasings("challenge", f)]
    rows += [(fill(p, rng), "none", "battery") for p in asks]
    rows += [(heard(t), "none", "seed") for t in (*IMPOSSIBLE, *NONE_SEEDS)]
    rows = [r for r in dict.fromkeys(rows) if r[0] not in held and r[0]]
    seen: set = set()
    rows = [
        r for r in rows if not (r[0] in seen or seen.add(r[0]))
    ]  # one label per text
    # The hard negatives, repeated to a quarter of the none rows.
    hard = list(
        dict.fromkeys(
            heard(t) for t in (*FILLERS, *FRAGMENTS, *ASR_GARBAGE, *LIVE_MISFIRES)
        )
    )
    hard = [t for t in hard if t not in held and t not in seen]
    n_none = sum(k == "none" for _, k, _ in rows)
    times = max(1, round(HARD_SHARE * n_none / ((1 - HARD_SHARE) * max(1, len(hard)))))
    rows += [(t, "none", "hard") for t in hard] * times
    rng.shuffle(rows)
    args.out.mkdir(parents=True, exist_ok=True)

    def dump(name: str, items) -> None:
        with open(args.out / f"{name}.jsonl", "w") as f:
            for i, (t, k, *src) in enumerate(items):
                row = {"text": t, "label": k, "ep": i}
                if src:
                    row["source"] = src[0]
                f.write(json.dumps(row) + "\n")

    dump("train", rows)
    dump("heldout", test_orders)
    dump("heldout_none", [(t, "none") for t in test_none])
    dump("heldout_hard", test_hard)
    counts = Counter(k for _, k, _ in rows)
    assert set(counts) <= set(ORDER_WORDS), counts
    print(
        f"train {len(rows)} {dict(counts)}, by source {dict(Counter(s for *_, s in rows))}; "
        f"held out {len(test_orders)} orders and talk, {len(test_none)} questions and talk, "
        f"{len(test_hard)} hard -> {args.out}"
    )


if __name__ == "__main__":
    main()
