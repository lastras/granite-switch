# SPDX-License-Identifier: Apache-2.0
"""Write the partner dataset: the player's replies and remarks, with Mellea IVR.

The player is a calm, dry professional out of a 1990s crime movie, and the
person watching is his partner, sitting next to him. The speaking moments are a
recorded match's (``talk.py moments``): soon after a salient event, or after a
silence, each with its brief and the tracker's facts. At about half of them the
partner says something first: lines written for that moment by a model, of a
type the moment allows (who killed you, only after a death; praise, only after
something good; the score, a greeting, who are you, backseat driving, a request
to play differently, ...), one kept and rendered the way speech recognition
writes it (sometimes with a misheard word). The player replies, and the joke is
the angle the reply takes on the partner's words: one of :data:`MOVES`, shown
with a model exchange drawn from :data:`EXAMPLES`. A factual question is
answered first (the killer's name, the score), then with his angle. At the
other moments he says a line of his own, in a form the moment calls for
(:data:`REMARKS`): after an event he reacts to it, names it or says what it
means; tangents and complaints come only in quiet stretches, and rarely.

The writer (Granite 4.2 30B or gpt-oss-120b) sees what the trained narrator
will see: the brief, the last log lines, and the exchanges still inside the
10 s history window. Checks in code: length, no numbers (but in a score), no
status-report opening, no stock phrase, calm punctuation, mild language, not a
film line, not a repeat, and the facts of a factual answer (the killer named,
the score right). Checks judged by gpt-oss-120b: true, consistent, in the
voice, funny, specific to this moment; for a reply, that it answers the
partner, and a swap test (the judge must pick the partner's line among the two
others written for that moment).

**Contrast.** A line that fits other moments as well as its own is refused.
Its margin, ``log p(line | its moment) - log mean_k p(line | moment_k)`` over
``--contrast-k`` other moments (half from the same match, at least 20 s away,
half from other matches), comes from a scoring model's prompt log-probabilities
(Granite 30B, one prefill per moment, :class:`Scorer`). A generic line ("the
bots finally learned to hide") scores about the same everywhere; a memorable
one ("Rambo, twice in a row?") far better at its own moment. A reply is scored
with the partner's words in every context, so the moment is what varies. The
repair names the moment it fit as well. ``score`` reports margins for lines
written elsewhere (calibrating ``--tau``, comparing narrators); ``swap`` asks
the judge to pick a line's moment among three.

A failed check's reason goes back to the writer, which repairs its line
(MultiTurnStrategy). Runs in an environment with Mellea (``pip install mellea``)::

    python partner_ivr.py write --moments data/narr/moments_v3.jsonl \\
        --out data/narr/partner_v3.jsonl --base-url http://WRITER:PORT/v1 \\
        --judge-url http://JUDGE:PORT/v1 --tau 2.0
    python partner_ivr.py score --rows data/narr/partner_granite_0.jsonl \\
        --moments data/narr/moments6.jsonl --score-url http://WRITER:PORT/v1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import threading
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from narrate_ivr import FILM, STRONG, clean, words

TIC_HZ = 35
MAX_ENTRIES = 50  # history.History: 10 s of 5 Hz entries, then the older half goes


def window_len(n: int) -> int:
    """How many of a conversation's first ``n`` entries the live history still
    holds (``history.History.append``: grow to MAX_ENTRIES, then keep half)."""
    if n <= MAX_ENTRIES:
        return n
    keep = MAX_ENTRIES // 2
    return keep + 1 + (n - MAX_ENTRIES - 1) % keep


PERSONA = (
    "You write the lines of a character in a Doom deathmatch against bots: a calm, "
    "dry professional out of a 1990s crime movie who goes by Granite. His partner "
    "sits next to him watching the screen, and the two of them talk like partners "
    "on a long job: they bicker, needle each other and never get flustered. He is "
    "deadpan, unbothered and quick, and the humor is in how he takes what was just "
    "said or what just happened. He never says numbers, except when asked the "
    "score. Every line is original: never quote or paraphrase any film. Mild "
    "language at most."
)
# Events after which the partner might praise him.
GOOD = ("frag", "streak", "took_lead", "close_call", "drought_ended")
DEATH_ASK_S = 20  # "who killed you" makes sense this long after a death
# What the partner says: (weight, instruction to the partner's voice, when):
# when is None (any moment), "event" (something just happened), "good" (one of
# GOOD just happened) or "death" (he died in the last DEATH_ASK_S).
UTTERANCES = {
    "backseat": (3, "Tell him what to do right now, like a backseat driver.", None),
    "praise": (2, "React to something good he just did.", "good"),
    "tease": (2, "Tease him or trash-talk his play, the way a friend would.", None),
    "worry": (2, "Get nervous about what is about to happen to him.", None),
    "question": (2, "Ask him something about what is going on in the game.", None),
    "what_happened": (2, "Ask him what just happened.", "event"),
    "who_killed": (4, "Ask him who just killed him.", "death"),
    "score": (2, "Ask him the score, or who is winning.", None),
    "greeting": (
        1,
        "Check that he can hear you, or say hi, the way you do when you sit down "
        "next to him (hey, can you hear me, you there).",
        None,
    ),
    "identity": (
        1,
        "Ask him who he is or what his name is, as if you had just sat down next to "
        "a stranger.",
        None,
    ),
    "odd": (1, "Ask him an odd, idle question about the game world or the bots.", None),
    "smalltalk": (1, "Bring up something from outside the game.", None),
    "request": (
        2,
        "Ask him to change how he plays for a while: play safer, collect stuff, "
        "be more aggressive, stop hiding.",
        None,
    ),
}
# Replies that need not be about the moment: the contrast check and the
# judge's "specific" do not apply.
SOCIAL = ("greeting", "identity", "smalltalk", "request", "followup")
FOLLOW_UP = (
    "React to what he just said to you: push back, laugh at it, needle him, or ask "
    "what he meant."
)
MOVES = {
    "echo": "Repeat one of their words and bend its meaning.",
    "correct": "Object to how they put it: quibble with their word choice, "
    "pedantic under fire.",
    "behind": "Answer the question behind what they said, not the one they asked.",
    "understate": "Understate it: treat what they said as much smaller than it is.",
    "tangent": "Drift to something ordinary in the middle of the fight: {topic}.",
    "pride": "Take it as a slight on your professional pride and defend your record.",
    "theory": "Offer a small theory nobody asked for.",
    "menace": "Stay polite, with a quiet menace toward the bots.",
    "bicker": "Bicker like an old partner who has heard it all before.",
    "misheard": "Speech recognition may have misheard a word: play off the odd "
    "word, then answer what they meant.",
    "callback": "Call back to something that was said earlier in the conversation.",
    "killer": "Say who killed you, by name, first; then your angle on it (a grudge, "
    "an excuse, a plan for them).",
    "score": "Give the score in words first (yours, and the leader's or whoever is "
    "closest), then your angle on it.",
    "tell": "Tell them what just happened, in your own dry way.",
    "hear": "Say you hear them, deadpan, and add a dry word about how it is going.",
    "identity": "Say who you are: Granite, the calm professional at the controls. "
    "Deadpan, no backstory.",
}
REQUEST_MOVES = ("echo", "correct", "behind", "understate", "pride", "theory", "bicker")
# The moves a type of utterance allows (any other: every general move).
TYPE_MOVES = {
    "who_killed": ("killer",),
    "score": ("score",),
    "what_happened": ("tell", "understate", "behind", "pride", "correct"),
    "greeting": ("hear",),
    "identity": ("identity",),
    "request": REQUEST_MOVES,
}
GENERAL_MOVES = tuple(
    k
    for k in MOVES
    if k not in ("misheard", "killer", "score", "tell", "hear", "identity")
)
# For the tangent move: something ordinary, drawn at random (left to itself the
# writer drifted to coffee every time).
TANGENTS = (
    *("a parking ticket", "his dry cleaning", "a sandwich he left in the car"),
    *("the dentist", "his landlord", "a song stuck in his head", "the weather"),
    *("a cousin's wedding", "his car's check-engine light", "pie at a diner"),
    *("the neighbor's dog", "his taxes", "a bad haircut", "a crossword clue"),
    *("the laundromat", "the price of gas", "his mother's cooking", "a lost umbrella"),
)
# One exchange per move shown to the writer as a model of the move (a situation,
# what the partner said, the reply), drawn from three; ``Not a copy`` keeps the
# reply from reusing it.
EXAMPLES = {
    "echo": (
        (
            "his health is low and he was just hit",
            "you good",
            "Good is a big word. I'm upright. Let's start there.",
        ),
        (
            "he is waiting behind a wall",
            "you're hiding",
            "Hiding. I prefer waiting with intent.",
        ),
        (
            "bots all around him",
            "this is crazy",
            "Crazy is when you don't have a plan. This is just busy.",
        ),
    ),
    "correct": (
        (
            "he just fragged a bot",
            "you murdered that guy",
            "Murder? He walked into my work area. That's a safety violation.",
        ),
        (
            "he is backing away from a fight",
            "why are you running away",
            "Running is a strong word. I'm relocating with urgency.",
        ),
        (
            "he just fragged a bot",
            "you got lucky",
            "Luck is what amateurs call preparation they didn't see.",
        ),
    ),
    "behind": (
        (
            "he just got killed",
            "what happened",
            "What happened is a rocket had an opinion, and I didn't.",
        ),
        (
            "a quiet stretch",
            "are you even trying",
            "You want to know if I care. I care in private.",
        ),
        ("his health is low", "should we be worried", "You should be. I'm busy."),
    ),
    "understate": (
        (
            "several frags in a few seconds",
            "dude you're on fire",
            "I'm having a reasonable afternoon. Don't jinx it.",
        ),
        ("he took heavy damage", "that looked bad", "It was a firm handshake. Mostly."),
        (
            "two bots shooting at him",
            "they're everywhere",
            "It's a crowd. I've worked crowds.",
        ),
    ),
    "tangent": (
        (
            "two bots shooting at him",
            "what are you thinking about",
            "Honestly? Whether anybody here ever mops these floors.",
        ),
        (
            "his health is low",
            "you're gonna die",
            "Maybe. Did I leave the stove on this morning?",
        ),
        (
            "he just fragged a bot",
            "nice shot",
            "Thanks. Reminds me, the car's due for an oil change.",
        ),
    ),
    "pride": (
        (
            "he just got killed",
            "lol you suck",
            "I don't suck. I had a disagreement with physics, and physics cheats.",
        ),
        (
            "he keeps missing",
            "you missed",
            "I was establishing a pattern. The next one's a statement.",
        ),
        ("a bot just hit him", "he got you", "He got a piece of me. He'll be billed."),
    ),
    "theory": (
        (
            "a quiet stretch",
            "why do they keep coming back",
            "Same reason as everybody. Nobody ever tells them no.",
        ),
        (
            "a bot backing away",
            "why is he running",
            "Guilty conscience. They all have one. They just don't know it.",
        ),
        (
            "he just respawned",
            "how do you keep coming back",
            "Stubbornness. It's cheaper than armor.",
        ),
    ),
    "menace": (
        (
            "a bot in view",
            "there's one right there",
            "I see him. Give him a moment to make his mistake.",
        ),
        ("a bot coming at him", "he's coming for you", "Good. Saves me the walk."),
        (
            "a bot behind a pillar",
            "he's behind the pillar",
            "Let him. Pillars are temporary.",
        ),
    ),
    "bicker": (
        (
            "his partner shouting directions",
            "left left go left",
            "I heard you. The bot heard you. Everybody heard you.",
        ),
        (
            "a medikit on screen",
            "grab the health",
            "You say that like I've never seen a medikit.",
        ),
        (
            "he just got killed",
            "you should have used the rocket launcher",
            "And you should have brought snacks. We all have regrets.",
        ),
    ),
    "misheard": (
        (
            "a rocket launcher on the floor",
            "go get the rocket lunch",
            "Rocket lunch. I'll assume the launcher, not a sandwich.",
        ),
        (
            "his health is low",
            "you need some kelp",
            "Kelp. I'm fine on seaweed. Health, maybe.",
        ),
        (
            "a bot behind him",
            "there's a bot behind the whale",
            "Behind the whale. The wall, I hope. Either way, I'm turning.",
        ),
    ),
    "callback": (
        (
            "his partner told him to be careful, then he fragged a bot",
            "nice",
            "See? Careful. Exactly like you ordered.",
        ),
        (
            "he said he would take it slow, now he is charging",
            "i thought you were taking it slow",
            "I am. This is slow for me.",
        ),
        (
            "his partner said he sucked, now he fragged a bot",
            "ok that was good",
            "So I've gone from suck to good. That's progress.",
        ),
    ),
    "request": (
        (
            "a quiet stretch",
            "play it safe for a while",
            "Safe. Noted. I'll take it under advisement after this guy.",
        ),
        (
            "his armor is low",
            "go get some armor",
            "Armor's a lovely idea. I'll put it on my list, under later.",
        ),
        (
            "he is waiting behind a wall",
            "stop hiding and fight",
            "I heard you. I'm going to keep doing it my way, but I heard you.",
        ),
    ),
    "killer": (
        (
            "Rambo just killed him, the second time in a row",
            "who got you",
            "Rambo. Twice now. I'm starting to take it personally.",
        ),
        (
            "McClane killed him with a rocket",
            "who killed you",
            "McClane. He'll be getting a thank-you note. Unsigned.",
        ),
        (
            "he killed himself with his own rocket",
            "who got you",
            "Me, apparently. I'll be having words with myself.",
        ),
    ),
    "score": (
        (
            "he has twelve frags, Rambo leads with thirteen",
            "what's the score",
            "Twelve to Rambo's thirteen. It's a long afternoon.",
        ),
        (
            "he leads with nine, the next best has six",
            "are you winning",
            "Nine to six. Winning's a strong word. Leading.",
        ),
        (
            "he has three, the leader has eight",
            "what's the score",
            "Three to eight. I'm letting them get comfortable.",
        ),
    ),
    "tell": (
        (
            "a bot shot him from behind and he died",
            "what happened",
            "Somebody shot me in the back. Rude, and effective.",
        ),
        (
            "he just fragged two bots with a rocket",
            "what just happened",
            "They stood too close together. I helped.",
        ),
        (
            "he picked up the plasma rifle",
            "what was that",
            "New plasma rifle. Things are about to get bright.",
        ),
    ),
    "hear": (
        ("a quiet stretch", "hey can you hear me", "Loud and clear. Unfortunately."),
        ("he is fighting two bots", "you there", "I'm here. Busy, but here."),
        (
            "he just respawned",
            "can you hear me",
            "I hear you. The bots hear you too. Keep it down.",
        ),
    ),
    "identity": (
        (
            "a quiet stretch",
            "who are you",
            "Granite. I play, I win, I don't do interviews.",
        ),
        (
            "he just fragged a bot",
            "what's your name",
            "Granite. That one won't remember it.",
        ),
        ("bots all around him", "who are you", "The professional. Call me Granite."),
    ),
}
# His own lines: (weight after an event, weight in a quiet stretch, instruction).
REMARKS = {
    "react": (
        3,
        0,
        "React to what just happened (the first thing under Just now), in your own "
        "dry way.",
    ),
    "name": (2, 0, "Name what just happened, the way he would put it to his partner."),
    "consequence": (
        2,
        1,
        "Say what it means for the match: the lead, the race with the leader, the "
        "bot who did it, the gun you have now.",
    ),
    "bots": (
        2,
        0,
        "Answer what a bot just did as if it had said something rude to you.",
    ),
    "understate": (1, 0, "Understate what just happened."),
    "threat": (1, 1, "Say a quiet, polite threat to a bot, by name if one is named."),
    "observe": (0, 2, "Make a dry observation about where this match stands."),
    "aside": (0, 2, "Make a dry aside to your partner about how the match is going."),
    "complain": (0, 0.3, "Complain about something, deadpan."),
    "tangent": (0, 0.3, "Drift to something ordinary while you play: {topic}."),
}
REMARK_EXAMPLES = {
    "react": (
        (
            "Rambo just killed him, twice in a row",
            "Rambo again. I'm starting to think he likes me.",
        ),
        ("he took the lead", "Top of the board. The view's better than I expected."),
        (
            "he survived on almost no health",
            "Still standing. Barely counts, but it counts.",
        ),
        (
            "he fragged several bots with one BFG shot",
            "All of them at once. The BFG believes in bulk pricing.",
        ),
    ),
    "name": (
        ("a streak of frags", "That's a streak. I'd stop, but it seems rude."),
        ("he picked up the rocket launcher", "Rocket launcher. Now we can talk."),
        ("a bot tied him for the lead", "Company at the top. I hate company."),
        (
            "Rambo killed him twice in a row",
            "Rambo, twice in a row. You're pacing yourself.",
        ),
    ),
    "consequence": (
        (
            "Rambo took the lead from him",
            "Rambo's on top. That's a temporary arrangement.",
        ),
        ("he got the BFG", "With this, the conversations get a lot shorter."),
        ("he is far behind the leader", "Leone's running away with it. Let him tire."),
        (
            "he leads by a mile near the end",
            "Big lead, little clock. Don't get sentimental.",
        ),
    ),
    "bots": (
        (
            "a bot hit him from behind",
            "Rude. I was clearly in the middle of something.",
        ),
        (
            "a bot backing off",
            "Leaving already? We were just getting to know each other.",
        ),
        ("he just fragged a bot", "You should have called first."),
        (
            "a bot killed him with a rocket",
            "Rockets. Some people have no conversation.",
        ),
    ),
    "understate": (
        ("he took heavy damage", "That stung a little."),
        ("several frags in a row", "Busy few seconds."),
        (
            "he fragged a crowd of bots in a few seconds",
            "Productive little stretch, I'd say.",
        ),
    ),
    "threat": (
        (
            "Machete killed him a minute ago",
            "Machete. I haven't forgotten. Take your time.",
        ),
        ("a bot in view", "Take your time. I'm not going anywhere."),
    ),
    "observe": (
        ("a quiet stretch, he is second", "Second place. Close enough to smell it."),
        ("a quiet stretch", "Every room in this place is the same room."),
    ),
    "aside": (
        ("a quiet stretch, he leads", "Don't say it. Saying it jinxes it."),
        ("no frag for a while", "You notice they all went somewhere quiet together?"),
    ),
    "complain": (
        ("he just got killed", "Nobody here has any manners."),
        ("a quiet stretch", "All this walking, and not one decent chair."),
    ),
    "tangent": (("a quiet stretch", "Reminds me, the car's due for an oil change."),),
}
BOT_EVENTS = ("killed you", "you fragged", "you took", "in view, nearest")

CONTEXT = """What is happening right now:
{brief}
The last moments of the game log (most recent last):
{recent}
What was said in the last few seconds (oldest first):
{conv}"""
REPLY_TASK = """{persona}

{context}

Your partner just said to you: "{player}"
{how}
Write what you say back: 4 to 16 words, one to three short sentences. Output only \
the line."""
REMARK_TASK = """{persona}

{context}

Say one line out loud now. {how}
It is his take on this moment, the thing he would only say now, in natural spoken \
sentences: not a list of what is on the screen. Say it in 4 to 14 words. Output only \
the line."""
UTTER_TASK = """You and your partner are on a long job together. He is playing a Doom \
deathmatch against bots and you sit next to him watching the screen. You talk to him \
the way you always do: casual, reactive, a little cheeky, with feeling. He is the one \
playing; you only watch, so never talk as if you were in the game yourself. You never \
read numbers or stats off the screen. {how}

What is happening right now:
{brief}
What was said in the last few seconds (oldest first):
{conv}

Write four different things you might say to him now, one per line, each 2 to 12 \
words of casual spoken English. They must differ in meaning. Output only the four \
lines."""
ASK_TASK = """You sit next to your partner while he plays a Doom deathmatch against \
bots, and you talk to him the way you always do: casual, a little cheeky. {how}

What is happening right now:
{brief}

Write four different ways you might say it now, one per line, each 2 to 10 words of \
casual spoken English. Output only the four lines."""
# Utterances that are one plain question, written with ASK_TASK.
ASKS = ("who_killed", "score", "greeting", "identity", "what_happened")
MISHEAR = """A speech recognizer heard this sentence and got exactly one word wrong: \
it swapped it for a similar-sounding real word, and the sentence came out a little \
funny. For example, "go get the rocket launcher" heard as "go get the rocket lunch".
"{x}"
Write the sentence as it was heard, with the same number of words. Output only the \
sentence."""

JUDGE_ALL = {
    "true": "Does the line avoid saying anything false about the game at this moment "
    "(kills, deaths, who killed whom, the score, damage, pickups, weapons, health, "
    "ammo, enemies in view)? A line with no game facts at all is YES; jokes, "
    "opinions, plans and flavor details (a sticky floor, the coffee) are fine.",
    "consistent": "Is it consistent with what he said earlier (no contradicting his "
    "own earlier lines; a callback only to something actually said)? Disagreeing "
    "with his partner is fine.",
    "voice": "Does it sound like a calm, dry, deadpan professional from a 1990s crime "
    "movie talking to his partner (not a soldier, a sports announcer or a "
    "cheerleader)?",
    "funny": "Is it genuinely funny or sharp (a real turn of phrase or angle), not "
    "a generic quip?",
}
JUDGE_SPECIFIC = {
    "specific": "Is it about this moment: the first thing under Just now, or, in a "
    "quiet stretch, something particular to this match (the race with a named bot, "
    "a streak, a drought, who killed him)? A line that would fit almost any moment "
    "of any match is NO."
}
JUDGE_REPLY = {
    "answers": "Does it respond to what the partner actually said, playing off their "
    "words or their point?"
}
JUDGE_REQUEST = {
    "fends_off": "Does he acknowledge the partner's request but turn it down or put "
    "it off? Agreeing to do it is a NO."
}

NUMBER = re.compile(
    r"\d|\b(zero|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|dozen)\b",
    re.I,
)
_UNITS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60}
# A status report's first word ("Health's low...", "BFG's warm..."): the habit of
# the first dataset.
STATUS = {
    *("health", "health's", "armor", "armor's", "ammo", "bfg", "bfg's", "pistol"),
    *("pistol's", "chaingun", "chaingun's", "shotgun", "shotgun's", "plasma"),
    *("rocket", "rockets", "another", "time", "situation"),
}
# Phrases the first dataset repeated hundreds of times.
STOCK = (
    "let's see",
    "about to learn",
    "feels like",
    "quiet now",
    "humming",
    "meters",
    "job well done",
    "just the way i like",
)


# The demo's ASR (granite-speech turboctc) writes contractions out: "what's the
# score" comes back "what is the score".
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


def one_word_off(a: str, b: str) -> bool:
    """``b`` is ``a`` with exactly one word swapped (a mishearing)."""
    x, y = a.split(), b.split()
    return len(x) == len(y) and sum(p != q for p, q in zip(x, y)) == 1


def said_numbers(text: str) -> list[int]:
    """The numbers in a line, in digits or words (up to nine hundred and
    sixty-nine). A lone "one" is left out: it is a pronoun as often as a
    number."""
    toks = re.findall(r"\d+|[a-z]+", text.lower().replace("-", " "))
    out, i = [], 0

    def small(j: int) -> tuple[int | None, int]:
        """A number under a hundred at token j, and the tokens it took."""
        t = toks[j] if j < len(toks) else ""
        if t in _TENS:
            if j + 1 < len(toks) and toks[j + 1] in _UNITS[1:10]:
                return _TENS[t] + _UNITS.index(toks[j + 1]), 2
            return _TENS[t], 1
        if t in _UNITS:
            return _UNITS.index(t), 1
        if t == "a" and toks[j + 1 : j + 2] == ["hundred"]:
            return 1, 1
        return None, 0

    while i < len(toks):
        if toks[i].isdigit():
            out.append(int(toks[i]))
            i += 1
            continue
        n, used = small(i)
        if n is not None and i + used < len(toks) and toks[i + used] == "hundred":
            j = i + used + 1
            j += toks[j : j + 1] == ["and"]
            rest, more = small(j)
            out.append(100 * n + (rest or 0))
            i = j + more
        elif n is not None and toks[i] != "one":
            out.append(n)
            i += used
        else:
            i += max(1, used)
    return out


def fact_fns(utype: str | None, facts: dict | None):
    """A factual answer checked against the tracker's facts: who killed him
    (named), the score (his frags said, and no number that is not in the
    match)."""
    if not facts:
        return []
    if utype == "who_killed" and facts.get("last_death"):
        by = facts["last_death"]["by"]
        if by is None:
            return []

        def killer(x):
            c = clean(x).lower()
            if by == "yourself":
                ok = re.search(r"\b(me|myself|my own|i did)\b", c) is not None
                return ok, "He killed himself; say so (it was me, my own rocket)."
            return by.lower() in c, f"Say who killed him: {by}."

        return [("Names the killer", killer)]
    if utype == "score":
        me = facts["frags"]
        board = [f for _, f in facts["board"]]
        ok_nums = {me, facts["deaths"], *board, len(board) + 1}
        ok_nums.add(1 + sum(f > me for f in board))  # his rank

        def score(x):
            nums = said_numbers(clean(x))
            said_me = me in nums or (
                me == 1 and re.search(r"\bone\b", clean(x).lower())
            )
            wrong = [n for n in nums if n not in ok_nums]
            top = facts["board"][0] if board else None
            right = f"you {me}" + (f", {top[0]} {top[1]}" if top else "")
            return bool(said_me) and not wrong, (
                f"Get the score right ({right}), in words, before your angle."
            )

        return [("The score, right", score)]
    return []


def code_fns(
    kind: str, prev: list[str], player: str | None = None, shown=(), utype=None
):
    """Requirements checked in code, as (description, fn -> (ok, reason)).
    ``player``: the partner's words (a reply may open by echoing one);
    ``shown``: the example lines the writer was shown, not to be copied;
    ``utype``: what the partner asked (a score answer may hold numbers)."""
    lo, hi = (4, 16) if kind == "reply" else (4, 14)
    echo = set(words(player or ""))

    def length(x):
        n = len(clean(x).split())
        return lo <= n <= hi, f"The line has {n} words; it must have {lo} to {hi}."

    def sentences(x):
        n = len([s for s in re.split(r"[.?!]+", clean(x)) if s.strip()])
        return n <= 3, "At most three short sentences."

    def numbers(x):
        m = NUMBER.search(clean(x))
        said = m.group(0) if m else ""
        return (
            m is None,
            f'No numbers in digits or words ("{said}"): he never says them.',
        )

    def opening(x):
        w = words(clean(x))
        first = w[0] if w else ""
        ok = bool(w) and (first not in STATUS or first.split("'")[0] in echo)
        return ok, (
            f'Do not open with "{first}": no status report; open with his angle on it.'
        )

    def not_copy(x):
        w = set(words(clean(x)))
        for e in shown:
            q = set(words(e))
            if w and q and len(w & q) / len(w | q) > 0.4:
                return False, "Too close to the example; write your own line."
        return True, ""

    def stock(x):
        hit = [s for s in STOCK if s in clean(x).lower()]
        said = hit[0] if hit else ""
        return (
            not hit,
            f'"{said}" is a stock phrase in these lines; put it another way.',
        )

    def calm(x):
        c = clean(x)
        dashes = "\u2014\u2013"  # em and en dash
        dash = any(d in c for d in dashes) and any(
            d in p for p in prev[-2:] for d in dashes
        )
        return "!" not in c and not dash, (
            "Calm punctuation: no exclamation marks, and no dash (a recent line had "
            "one); use a period or a comma."
        )

    def plain(x):
        ok = not re.search(r"[\[\]*#\"]|reload", clean(x), re.I)
        return ok, "No brackets, asterisks or quote marks, and Doom has no reloading."

    def mild(x):
        bad = [s for s in STRONG if s in clean(x).lower()]
        return not bad, "Keep the language mild: no strong swearing or slurs."

    def original(x):
        hit = [s for s in FILM if s in clean(x).lower()]
        return not hit, "That echoes a famous film line; write an original one."

    def fresh(x):
        w = set(words(clean(x)))
        for p in prev[-5:]:
            q = set(words(p))
            if w and q and len(w & q) / len(w | q) > 0.5:
                return (
                    False,
                    f'Too close to an earlier line ("{p}"); say something new.',
                )
        return True, ""

    fns = [
        (f"{lo} to {hi} words", length),
        ("No status-report opening", opening),
        ("No stock phrase", stock),
        ("Calm punctuation", calm),
        ("Plain spoken text", plain),
        ("Mild language", mild),
        ("Original, not a film quote", original),
        ("Not a repeat of recent lines", fresh),
        ("Not a copy of the example", not_copy),
    ]
    if utype != "score":
        fns.insert(1, ("No numbers", numbers))
    if kind == "reply":
        fns.insert(1, ("At most three sentences", sentences))
    return fns


# ── Contrast: is the line about its own moment? ─────────────────────────────────
SCORE_HEAD = (
    "A Doom deathmatch against bots. The player is a calm, dry professional out of "
    "a 1990s crime movie; his partner sits next to him, watching the screen.\n"
)
CONTRAST_GAP_S = 20  # same-match negatives at least this far from the moment


def score_prefix(brief: str, player: str | None) -> str:
    """What the scoring model reads before the line: the moment, and what the
    partner said."""
    said = f'His partner says: "{player}"\n' if player else ""
    return f'{SCORE_HEAD}{brief}\n{said}He says: "'


class Scorer:
    """Log-probabilities of a line after several contexts, from an
    OpenAI-compatible vLLM server: one batched completions request, the
    prompts echoed with their token log-probabilities (one prefill each)."""

    def __init__(self, urls: list[str], model: str):
        from openai import OpenAI

        self.clients = [OpenAI(base_url=u, api_key="none", timeout=300) for u in urls]
        self.model = model
        self._n = 0

    def logps(self, prefixes: list[str], line: str) -> list[float]:
        client = self.clients[self._n % len(self.clients)]
        self._n += 1
        r = client.completions.create(
            model=self.model,
            prompt=[p + line + '"' for p in prefixes],
            max_tokens=1,
            echo=True,
            logprobs=1,
            temperature=0.0,
        )
        out = [0.0] * len(prefixes)
        for ch in r.choices:
            lo = len(prefixes[ch.index])
            hi = lo + len(line)
            lp = ch.logprobs
            out[ch.index] = sum(
                v
                for off, v in zip(lp.text_offset, lp.token_logprobs)
                if lo <= off < hi and v is not None
            )
        return out


def margin(own: float, others: list[float]) -> float:
    """``own - log mean exp(others)``: how much likelier the line is at its own
    moment than at a typical other one, in nats."""
    top = max(others)
    return own - (top + math.log(sum(math.exp(o - top) for o in others) / len(others)))


def signature(m: dict) -> tuple:
    """What kind of moment it is: its cue, its events, who killed him."""
    d = (m.get("facts") or {}).get("last_death") or {}
    by = d.get("by") if "died" in m.get("events", ()) else None
    return m.get("cue"), tuple(m.get("events", ())), by


def negatives(
    moment: dict, same: list[dict], pool: list[dict], k: int, rng
) -> list[dict]:
    """``k`` other moments of another kind (:func:`signature`; a line need not
    tell two near-identical moments apart): half from the same match, at least
    CONTRAST_GAP_S away, the rest from other matches (``pool``)."""
    sig = signature(moment)
    far = [
        m
        for m in same
        if abs(m["t"] - moment["t"]) >= CONTRAST_GAP_S * TIC_HZ and signature(m) != sig
    ]
    mine = rng.sample(far, min(len(far), k // 2))
    want = k - len(mine)  # from other matches, of another kind where possible
    out = [o for o in (rng.choice(pool) for _ in range(20 * k)) if signature(o) != sig]
    out = out[:want]
    while len(out) < want:
        out.append(rng.choice(pool))
    return mine + out


def contrast_fn(scorer: Scorer, brief: str, others: list[dict], player, tau, seen):
    """The contrast check; ``seen`` keeps each scored line's margin."""
    prefixes = [score_prefix(brief, player)] + [
        score_prefix(o["brief"], player) for o in others
    ]

    def fn(x):
        line = clean(x)
        if not line:
            return False, "Write a line."
        lps = scorer.logps(prefixes, line)
        m = margin(lps[0], lps[1:])
        seen[line] = round(m, 3)
        if m >= tau:
            return True, ""
        j = max(range(1, len(lps)), key=lps.__getitem__)
        other = others[j - 1]["brief"].split(" Right now:")[0]
        here = brief.split(" Right now:")[0]
        return False, (
            f'This line would fit another moment just as well ("{other}"). Make it '
            f'turn on what is particular to this one ("{here}"), as his own take, in '
            "a natural sentence: not a list of facts."
        )

    return fn


# ── The judge ──────────────────────────────────────────────────────────────────
def judge_opts(args) -> dict:
    from mellea.backends import ModelOption

    return {
        ModelOption.THINKING: args.judge_effort,
        ModelOption.TEMPERATURE: 0.0,
        ModelOption.MAX_NEW_TOKENS: 2000,  # reasoning included
    }


def judge_fn(judge, context: str, questions: dict, args):
    """The judged requirements in one call: the line passes only if every
    verdict is YES; the verdicts are the repair feedback."""
    listing = "\n".join(f"{k}: {q}" for k, q in questions.items())

    def fn(x):
        q = (
            f"Judge one spoken line from a Doom deathmatch.\n\n{context}\n\n"
            f'The line he says: "{clean(x)}"\n\nQuestions:\n{listing}\n\n'
            f"Answer with exactly {len(questions)} lines, one per question, each as "
            "`name: YES` or `name: NO - short reason`."
        )
        judge.reset()
        a = str(judge.instruct(q, strategy=None, model_options=judge_opts(args)))
        verdicts = {k: re.search(rf"{k}\W*?:\W*(YES|NO)", a, re.I) for k in questions}
        bad = [k for k, v in verdicts.items() if not v or v.group(1).upper() != "YES"]
        return not bad, (a.strip() if bad else "")

    return fn


def pick_fn(judge, question: str, opts: list[str], args):
    """Ask the judge to pick one of ``opts`` (lettered); returns the index."""
    letters = "ABCDEF"[: len(opts)]
    q = (
        question
        + "\n"
        + "\n".join(f"{c}) {o}" for c, o in zip(letters, opts))
        + "\n\nAnswer with one letter."
    )
    judge.reset()
    a = str(judge.instruct(q, strategy=None, model_options=judge_opts(args)))
    m = re.search(rf"\b([{letters}])\b", a.strip())
    return letters.index(m.group(1)) if m else -1


def swap_fn(judge, player: str, others: list[str], rng: random.Random, args):
    """The swap test: shown the partner's line among the two others written for
    the same moment, the judge must pick the one the reply answers."""
    opts = [player, *others]
    rng.shuffle(opts)

    def fn(x):
        q = (
            "In a Doom deathmatch, a player said this to his partner, in reply to "
            f'something the partner had just said:\n"{clean(x)}"\n\nWhich of these had '
            "the partner just said?"
        )
        ok = pick_fn(judge, q, opts, args) == opts.index(player)
        return ok, (
            "The reply could follow many things the partner might say; make it turn "
            f'on what they actually said: "{player}".'
        )

    return fn


def render_conv(turns: list[dict], now: int) -> str:
    out = []
    for tr in turns:
        when = f"{max(0, round((now - tr['t']) / TIC_HZ))} s ago"
        if tr.get("player"):
            out.append(f'{when}, your partner: "{tr["player"]}"')
        out.append(f'{when}, you: "{tr["line"]}"')
    return "\n".join(out) or "(nothing yet)"


def partner_lines(
    voice, how: str, brief: str, conv: str, ask: bool = False
) -> list[str]:
    """Three things the partner might say at this moment (``ask``: three ways
    of asking one plain question)."""
    from mellea.backends import ModelOption

    task = ASK_TASK if ask else UTTER_TASK
    a = str(
        voice.instruct(
            task.format(how=how, brief=brief, conv=conv),
            strategy=None,
            model_options={
                ModelOption.THINKING: False,
                ModelOption.TEMPERATURE: 1.0,
                ModelOption.MAX_NEW_TOKENS: 160,
            },
        )
    )
    out = []
    for x in clean_lines(a):
        # Numbers read off the screen made the partner sound like a HUD.
        if (
            2 <= len(x.split()) <= 14
            and not NUMBER.search(x)
            and heard(x) not in map(heard, out)
        ):
            out.append(x)
    return out[:3]


def clean_lines(text: str) -> list[str]:
    text = re.sub(r"(?s)<think>.*?</think>", "", str(text)).split("</think>")[-1]
    out = []
    for x in text.splitlines():
        x = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", x).strip().strip('"“”').strip()
        if x:
            out.append(x)
    return out


def allowed(m: dict) -> list[str]:
    """The partner's utterance types this moment allows."""
    facts, events = m.get("facts") or {}, m.get("events", [])
    d = facts.get("last_death")
    died = (
        bool(d) and d["by"] is not None and m["t"] - d["tick"] <= DEATH_ASK_S * TIC_HZ
    )
    ok = {None: True, "event": m.get("cue") == "event", "death": died}
    ok["good"] = any(k in events for k in GOOD)
    return [k for k, (_, _, when) in UTTERANCES.items() if ok[when]]


def remark_form(m: dict, rng) -> str:
    """His own line's form, by the moment's cue."""
    col = 0 if m.get("cue", "event") == "event" else 1
    forms = [
        k
        for k in REMARKS
        if REMARKS[k][col] > 0
        and (k != "bots" or any(e in m["brief"] for e in BOT_EVENTS))
    ]
    return rng.choices(forms, [REMARKS[k][col] for k in forms])[0]


def run_match(match: dict, args, write, pool: list[dict]) -> int:
    from mellea import start_session
    from mellea.backends import ModelOption
    from mellea.stdlib.context import ChatContext
    from mellea.stdlib.requirements import req, simple_validate
    from mellea.stdlib.sampling import MultiTurnStrategy

    key = (match["data"], match["ep"])
    rng = random.Random(zlib.crc32(repr((*key, args.seed)).encode()))
    wurls, jurls = args.base_url.split(","), args.judge_url.split(",")
    h = zlib.crc32(repr(key).encode())
    wkw = {"base_url": wurls[h % len(wurls)], "api_key": "none"}
    jkw = {"base_url": jurls[h % len(jurls)], "api_key": "none"}
    writer = start_session(
        "openai",
        model_id=args.model,
        ctx=ChatContext(),
        model_options={
            # Granite 4.2 at "low" effort reasons briefly and closes </think>.
            ModelOption.THINKING: args.effort,
            ModelOption.TEMPERATURE: args.temperature,
            ModelOption.MAX_NEW_TOKENS: args.max_tokens,
        },
        **wkw,
    )
    vurls = (args.voice_url or args.base_url).split(",")
    voice = start_session(  # the partner
        "openai",
        model_id=args.voice_model or args.model,
        base_url=vurls[h % len(vurls)],
        api_key="none",
    )
    judge = start_session("openai", model_id=args.judge_model, **jkw)
    scorer = Scorer(args.score_url.split(","), args.score_model) if args.tau else None
    others_pool = [p for p in pool if (p["data"], p["ep"]) != key]
    turns: list[dict] = []  # what was said: conversation position, tick, words
    prev: list[str] = []
    for m in match["moments"]:
        n = m["hist_n"] + len(turns)  # conversation entries before this turn
        win = [tr for tr in turns if tr["pos"] >= n - window_len(n)]
        conv = render_conv(win, m["t"])
        kind, utype, player, said, others, moves = "remark", None, None, None, [], []
        last = win[-1] if win else None
        if last and last.get("player") and rng.random() < args.follow_rate:
            utype = "followup"
        elif rng.random() < args.reply_rate:
            names = allowed(m)
            utype = rng.choices(names, [UTTERANCES[k][0] for k in names])[0]
        if utype:
            how = FOLLOW_UP if utype == "followup" else UTTERANCES[utype][1]
            cands = partner_lines(voice, how, m["brief"], conv, utype in ASKS)
            if len(cands) == 3:
                kind = "reply"
                i = rng.randrange(3)
                said = cands[i]
                player = heard(said)
                others = [heard(c) for j, c in enumerate(cands) if j != i]
                if rng.random() < args.misheard_rate and utype not in (
                    "who_killed",
                    "score",
                ):
                    mis = clean_lines(
                        voice.instruct(
                            MISHEAR.format(x=player),
                            strategy=None,
                            model_options={
                                ModelOption.THINKING: False,
                                ModelOption.TEMPERATURE: 0.8,
                                ModelOption.MAX_NEW_TOKENS: 60,
                            },
                        )
                    )
                    if mis and one_word_off(player, heard(mis[0])):
                        player, moves = heard(mis[0]), ["misheard"]
            else:
                utype = None
        context = CONTEXT.format(
            brief=m["brief"], recent="\n".join(m["recent"]), conv=conv
        )
        grounded = kind == "remark" or utype not in SOCIAL
        if kind == "reply":
            if not moves:
                pool_moves = TYPE_MOVES.get(utype) or [
                    k for k in GENERAL_MOVES if k != "callback" or win
                ]
                moves = [rng.choice(list(pool_moves))]
            desc = MOVES[moves[0]].format(topic=rng.choice(TANGENTS))
            sit, ex_said, ex_line = rng.choice(
                EXAMPLES["request" if utype == "request" else moves[0]]
            )
            how = f"Move: {desc}"
            if utype == "request":
                how = (
                    "It is a request to change how you play. Acknowledge it, then fend "
                    f"it off: you keep playing your way. {how}"
                )
            how += (
                f' For example, when {sit} and the partner said "{ex_said}", he '
                f'said: "{ex_line}" Write your own line; do not reuse that one.'
            )
            task = REPLY_TASK.format(
                persona=PERSONA, context=context, player=player, how=how
            )
            questions = {**JUDGE_ALL, **JUDGE_REPLY}
            if utype == "request":
                questions.update(JUDGE_REQUEST)
            jctx = f'{context}\nHis partner just said: "{player}"'
        else:
            moves = [remark_form(m, rng)]
            sit, ex_line = rng.choice(REMARK_EXAMPLES[moves[0]])
            desc = REMARKS[moves[0]][2].format(topic=rng.choice(TANGENTS))
            how = (
                f'{desc} For example, when {sit}: "{ex_line}" Write your own line; do '
                "not reuse that one."
            )
            task = REMARK_TASK.format(persona=PERSONA, context=context, how=how)
            questions = dict(JUDGE_ALL)
            jctx = f"{context}\nHis partner said nothing."
        if grounded:
            questions.update(JUDGE_SPECIFIC)
        reqs = [
            req(d, validation_fn=simple_validate(f))
            for d, f in code_fns(kind, prev, player, (ex_line,), utype)
            + fact_fns(utype, m.get("facts"))
        ]
        reqs.append(
            req(
                "Judged: " + ", ".join(questions),
                validation_fn=simple_validate(judge_fn(judge, jctx, questions, args)),
            )
        )
        if kind == "reply":
            reqs.append(
                req(
                    "Answers the partner's words (swap test)",
                    validation_fn=simple_validate(
                        swap_fn(judge, player, others, rng, args)
                    ),
                )
            )
        margins: dict[str, float] = {}
        if scorer is not None and grounded:
            negs = negatives(m, match["moments"], others_pool, args.contrast_k, rng)
            reqs.append(
                req(
                    "About this moment (contrast)",
                    validation_fn=simple_validate(
                        contrast_fn(scorer, m["brief"], negs, player, args.tau, margins)
                    ),
                )
            )
        t0 = time.time()
        writer.reset()
        res = writer.instruct(
            task,
            requirements=reqs,
            strategy=MultiTurnStrategy(loop_budget=args.loop_budget),
            return_sampling_results=True,
        )
        line = clean(res.result.value if hasattr(res.result, "value") else res.result)
        fails = [
            (r.description, str(v.reason or ""))
            for r, v in (res.result_validations or [])
            if not v.as_bool()
        ]
        write(
            {
                "data": match["data"],
                "style": match["style"],
                "ep": match["ep"],
                "t": m["t"],
                "hist_n": m["hist_n"],
                "cue": m.get("cue"),
                "events": m.get("events"),
                "kind": kind,
                "utype": utype,
                "player": player,
                "said": said,
                "others": others,
                "moves": moves,
                "line": line,
                "ok": bool(getattr(res, "success", not fails)),
                "margin": margins.get(line),
                "attempts": len(getattr(res, "sample_generations", None) or []) or 1,
                "fails": fails,
                "window": [[tr.get("player"), tr["line"]] for tr in win],
                "brief": m["brief"],
                "facts": m.get("facts"),
                "recent": m["recent"],
                "prev": prev[-5:],
                "s": round(time.time() - t0, 1),
            }
        )
        if line:  # what he said, pass or not: the next moment follows it
            prev.append(line)
            turns.append({"pos": n, "t": m["t"], "player": player, "line": line})
    return len(match["moments"])


# ── Lines written elsewhere: margins and the moment-swap test ──────────────────
def load_moments(path: Path) -> tuple[dict, list[dict]]:
    """Moments by match key, and all of them (each with its match key)."""
    by, pool = {}, []
    for x in open(path):
        mt = json.loads(x)
        ms = [{**mm, "data": mt["data"], "ep": mt["ep"]} for mm in mt["moments"]]
        by[(mt["data"], mt["ep"])] = ms
        pool += ms
    return by, pool


def load_rows(paths: list[Path], keys: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        for x in open(p):
            r = json.loads(x)
            if all(r.get(k) for k in keys) and r.get("brief"):
                rows.append(r)
    return rows


def score_file(args) -> None:
    """The contrast margin of lines written elsewhere: a dataset's (``--keys
    line``), or a narrator's held-out samples (``adapter,base``, from
    train_alora.py's heldout_gen.jsonl)."""
    by, pool = load_moments(args.moments)
    keys = args.keys.split(",")
    rows = load_rows(args.rows, keys)
    if args.limit:
        rows = random.Random(0).sample(rows, min(args.limit, len(rows)))
    scorer = Scorer(args.score_url.split(","), args.score_model)

    def one(ir):
        i, r = ir
        rng = random.Random(i)
        key = (r["data"], r["ep"])
        negs = negatives(
            r,
            by.get(key, []),
            [p for p in pool if (p["data"], p["ep"]) != key],
            args.contrast_k,
            rng,
        )
        player = r.get("player")
        prefixes = [score_prefix(r["brief"], player)] + [
            score_prefix(o["brief"], player) for o in negs
        ]
        out = {}
        for k in keys:
            lps = scorer.logps(prefixes, clean(r[k]))
            out[k] = round(margin(lps[0], lps[1:]), 3)
        return {**r, "margins": out}

    with ThreadPoolExecutor(args.concurrency) as ex:
        scored = list(ex.map(one, enumerate(rows)))
    with open(args.out, "w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")
    for k in keys:
        ms = sorted(r["margins"][k] for r in scored)
        q = [ms[int(p * (len(ms) - 1))] for p in (0.1, 0.25, 0.5, 0.75, 0.9)]
        print(
            f"{k}: n={len(ms)} mean {sum(ms) / len(ms):.2f}; quantiles 10/25/50/75/90: "
            + " ".join(f"{x:.2f}" for x in q)
        )
        for tau in (0.5, 1.0, 1.5, 2.0, 3.0):
            print(
                f"  margin >= {tau}: {100 * sum(m >= tau for m in ms) / len(ms):.0f}%"
            )
    k = keys[0]
    scored.sort(key=lambda r: r["margins"][k])
    for title, part in (("lowest", scored[:15]), ("highest", scored[-15:])):
        print(f"-- {title} ({k})")
        for r in part:
            said = f"[{r['player']}] " if r.get("player") else ""
            print(f"  {r['margins'][k]:6.2f}  {said}{r[k]}")
    if args.probe:
        for word in args.probe.split(","):
            hits = [r for r in scored if word in r[k].lower()]
            if hits:
                rank = [scored.index(r) / len(scored) for r in hits]
                print(
                    f"-- lines with {word!r}: n={len(hits)}, mean margin "
                    f"{sum(r['margins'][k] for r in hits) / len(hits):.2f}, mean "
                    f"percentile {100 * sum(rank) / len(rank):.0f}"
                )
    print(f"-> {args.out}")


def swap_file(args) -> None:
    """The judge's moment-swap test: shown a line (and the partner's words, if
    any) with its moment's brief and two others' (one from the same match, one
    from another), it must pick the moment; chance is a third."""
    from mellea import start_session

    by, pool = load_moments(args.moments)
    keys = args.keys.split(",")
    rows = load_rows(args.rows, keys)
    if args.limit:
        rows = random.Random(0).sample(rows, min(args.limit, len(rows)))
    urls = args.judge_url.split(",")

    def one(ir):
        i, r = ir
        rng = random.Random(i)
        judge = start_session(
            "openai",
            model_id=args.judge_model,
            base_url=urls[i % len(urls)],
            api_key="none",
        )
        key = (r["data"], r["ep"])
        negs = negatives(
            r, by.get(key, []), [p for p in pool if (p["data"], p["ep"]) != key], 2, rng
        )
        opts = [r["brief"], *(o["brief"] for o in negs)]
        order = list(range(len(opts)))
        rng.shuffle(order)
        out = {}
        for k in keys:
            said = f'His partner had said: "{r["player"]}"\n' if r.get("player") else ""
            q = (
                "In a Doom deathmatch, a calm, dry player said this out loud:\n"
                f'"{clean(r[k])}"\n{said}\nAt which of these moments of the match did '
                "he say it?"
            )
            got = pick_fn(judge, q, [opts[j] for j in order], args)
            out[k] = got >= 0 and order[got] == 0
        return out

    with ThreadPoolExecutor(args.concurrency) as ex:
        res = list(ex.map(one, enumerate(rows)))
    for k in keys:
        acc = sum(r[k] for r in res) / max(1, len(res))
        print(f"{k}: moment-swap accuracy {100 * acc:.1f}% (n={len(res)}, chance 33%)")


def write_file(args) -> None:
    matches = [json.loads(x) for x in open(args.moments)]
    pool = [
        {**mm, "data": mt["data"], "ep": mt["ep"]}
        for mt in matches
        for mm in mt["moments"]
    ]
    if args.matches:
        matches = matches[: args.matches]
    k, n = map(int, args.shard.split("/"))
    matches = matches[k::n]
    if args.per_match:
        for m in matches:
            m["moments"] = m["moments"][: args.per_match]
    done = set()
    if args.out.exists():
        # Resumable: a whole match already written is skipped; one cut short (a
        # server went away) is dropped and written again from its start.
        want = {(m["data"], m["ep"]): len(m["moments"]) for m in matches}
        rows = [json.loads(x) for x in open(args.out)]
        count = Counter((r["data"], r["ep"]) for r in rows)
        done = {k for k, c in count.items() if c >= want.get(k, 0)}
        kept = [r for r in rows if (r["data"], r["ep"]) in done]
        if len(kept) < len(rows):
            with open(args.out, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in kept)
            print(f"dropped {len(rows) - len(kept)} rows of unfinished matches")
    todo = [m for m in matches if (m["data"], m["ep"]) not in done]
    lock, n_ok, n_all = threading.Lock(), [0], [0]
    t0 = time.time()
    f = open(args.out, "a")

    def write(row):
        with lock:
            f.write(json.dumps(row) + "\n")
            f.flush()
            n_all[0] += 1
            n_ok[0] += row["ok"]
            if n_all[0] % 25 == 0:
                print(
                    f"{n_all[0]} lines, {100 * n_ok[0] / n_all[0]:.0f}% pass, "
                    f"{n_all[0] / (time.time() - t0):.2f}/s  last: {row['line']!r}",
                    flush=True,
                )

    failed = []

    def one(m):
        try:
            return run_match(m, args, write, pool)
        except Exception as e:  # a server went away: the others carry on
            failed.append((m["data"], m["ep"]))
            print(f"match {m['ep']} failed: {type(e).__name__}: {e}", flush=True)
            return 0

    with ThreadPoolExecutor(args.concurrency) as ex:
        list(ex.map(one, todo))
    f.close()
    print(f"done: {n_all[0]} lines, {n_ok[0]} pass every check -> {args.out}")
    if failed:
        raise SystemExit(f"{len(failed)} matches failed; run again to write them")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="Write the dataset")
    sc = sub.add_parser("score", help="Contrast margins of lines written elsewhere")
    sw = sub.add_parser("swap", help="The judge's moment-swap test on lines")
    for p in (w, sc, sw):
        p.add_argument("--moments", type=Path, required=True)
        p.add_argument("--concurrency", type=int, default=32)
        p.add_argument("--contrast-k", type=int, default=6, help="Other moments")
    for p in (w, sc):
        p.add_argument(
            "--score-url",
            help="Scoring server(s) (default: --voice-url, then --base-url)",
        )
        p.add_argument("--score-model", default="granite-4.2-30b")
    for p in (w, sw):
        p.add_argument(
            "--judge-url", required=True, help="Judge server(s), comma-separated"
        )
        p.add_argument("--judge-model", default="gpt-oss-120b")
        p.add_argument("--judge-effort", default="low", help="Judge reasoning effort")
    for p in (sc, sw):
        p.add_argument("--rows", type=Path, nargs="+", required=True)
        p.add_argument("--keys", default="line", help="Columns holding lines")
        p.add_argument("--limit", type=int, default=0, help="A random sample of rows")
    sc.add_argument("--out", type=Path, required=True)
    sc.add_argument(
        "--probe", default="hide,coffee", help="Words to locate in the ranking"
    )
    w.add_argument("--out", type=Path, required=True)
    w.add_argument(
        "--base-url", required=True, help="Writer server(s), comma-separated"
    )
    w.add_argument("--model", default="granite-4.2-30b")
    w.add_argument(
        "--shard",
        default="0/1",
        help="K/N: every N-th match from the K-th (e.g. one writer per shard)",
    )
    w.add_argument("--voice-model", help="The partner's words (default: --model)")
    w.add_argument("--voice-url", help="Its server(s) (default: --base-url)")
    w.add_argument(
        "--tau", type=float, default=0.0, help="Contrast margin to pass (0: off)"
    )
    w.add_argument("--matches", type=int, default=0, help="0: all")
    w.add_argument("--per-match", type=int, default=0, help="0: all moments")
    w.add_argument(
        "--reply-rate", type=float, default=0.5, help="Moments the partner speaks at"
    )
    w.add_argument(
        "--follow-rate", type=float, default=0.3, help="Follow-ups to a reply"
    )
    w.add_argument("--misheard-rate", type=float, default=0.12)
    w.add_argument("--loop-budget", type=int, default=3)
    w.add_argument("--temperature", type=float, default=0.9)
    w.add_argument("--max-tokens", type=int, default=800)
    w.add_argument("--effort", default="low", help="Writer reasoning effort")
    w.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd != "swap" and not args.score_url:
        args.score_url = getattr(args, "voice_url", None) or getattr(
            args, "base_url", None
        )
        if not args.score_url:
            raise SystemExit("--score-url is required")
    {"write": write_file, "score": score_file, "swap": swap_file}[args.cmd](args)


if __name__ == "__main__":
    main()
