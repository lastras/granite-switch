# SPDX-License-Identifier: Apache-2.0
"""Write the partner dataset: the player's replies and remarks, with Mellea IVR.

The player is a calm, dry professional out of a 1990s crime movie, and the
person watching is his partner, sitting next to him. The speaking moments are
a recorded match's, every one in order (``talk.py moments``): soon after a
salient event, or after a silence, each with the output of his
``get_game_state`` call (``tool``, :func:`talk.game_state`) and what had
happened since the moment before (``moment``). Whole matches are written in
order, so the conversations reach the full window, as they do live.

At about 55% of moments the partner asks about the game state (:mod:`probes`:
who killed you, the score, your health, what you just picked up, ... or a value
asserted, true or false: "i see 12"); its answer is checked in code. At about
15% the partner says something else: lines written for that moment by a model,
of a type the moment allows (praise, only after something good; a greeting,
who are you, backseat driving, a request to play differently, ...), one kept
and rendered the way speech recognition writes it (sometimes with a misheard
word). At the rest he says a line of his own, in a form the moment calls for
(:data:`REMARKS`). A probe is answered first, exactly as the game state says
(the writer is told the verified answer), then with his angle; any other reply
plays off the partner's words with one of :data:`MOVES`, shown with a model
exchange drawn from :data:`EXAMPLES`.

The writer (Granite 4.2 30B or gpt-oss-120b) and the judge see what the
trained narrator will see (:mod:`conversation`), nothing more: the last
CONV_EXCHANGES exchanges (what the partner said, the game's output at that
line, his line), then the partner's words and the whole game state. Checks in
code: a probe's answer (:func:`probes.verify`) and, for every line, its claims
against the state (:func:`probes.claims`: no bot takes a weapon, no victim
named, no killer, lead or number the state does not have); length, no numbers
(but where a number is asked), no status-report opening, no stock phrase, calm
punctuation, mild language, not a film line, not a repeat. Checks judged by
gpt-oss-120b: true, consistent, coherent with the conversation, in the voice,
funny, specific to this moment; for a reply, that it answers the partner, and a
swap test (the judge must pick the partner's line among the two others written
for that moment).

**Contrast.** A remark (or a reply about the game) that fits other moments as
well as its own is refused. Its margin, ``log p(line | its moment) - log mean_k
p(line | moment_k)`` over ``--contrast-k`` other moments (half from the same
match, at least 20 s away, half from other matches), comes from a scoring
model's prompt log-probabilities (Granite 30B, one prefill per moment,
:class:`Scorer`), each moment given by its game state. ``score`` reports
margins for lines written elsewhere; ``swap`` asks the judge to pick a line's
moment among three; ``judge`` asks the judged questions once, without repair,
and runs the code checks (a probe's answer, the claims) on lines written
elsewhere (a narrator's held-out samples).

A failed check's reason goes back to the writer, which repairs its line
(MultiTurnStrategy). Runs in an environment with Mellea (``pip install mellea``)::

    python partner_ivr.py write --moments data/narr/moments_v6_write.jsonl \\
        --out data/narr/v6/partner_0.jsonl --base-url http://WRITER:PORT/v1 \\
        --judge-url http://JUDGE:PORT/v1 --tau 0.5
    python partner_ivr.py judge --rows runs/narr6/narrator/heldout_gen.jsonl \\
        --keys adapter,base --judge-url http://JUDGE:PORT/v1
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

import probes
from conversation import (
    TIC_HZ,
    Conversation,
    Exchange,
    new_stretch,
    past_output,
    tool_text,
)
from narrate_ivr import FILM, STRONG, clean, words
from probes import heard

PERSONA = (
    "You write the lines of a character in a Doom deathmatch against bots: a calm, "
    "dry professional out of a 1990s crime movie who goes by Granite. His partner "
    "sits next to him watching the screen, and the two of them talk like partners "
    "on a long job: they bicker, needle each other and never get flustered. He is "
    "deadpan, unbothered and quick, and the humor is in how he takes what was just "
    "said or what just happened. He speaks for himself, as I and me (in the game "
    'state, "you" is him). He is never wrong about the game: every fact he '
    "says is in his game state. He says numbers only when asked for one. Every line "
    "is original: never quote or paraphrase any film. Mild language at most."
)
# Events after which the partner might praise him.
GOOD = ("frag", "streak", "took_lead", "close_call", "drought_ended")
# What the partner says when not probing the game state: (weight, instruction
# to the partner's voice, when): when is None (any moment), "event" (something
# just happened) or "good" (one of GOOD just happened).
UTTERANCES = {
    "backseat": (3, "Tell him what to do right now, like a backseat driver.", None),
    "praise": (2, "React to something good he just did.", "good"),
    "tease": (2, "Tease him or trash-talk his play, the way a friend would.", None),
    "worry": (2, "Get nervous about what is about to happen to him.", None),
    "what_happened": (2, "Ask him what just happened.", "event"),
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
    "tell": "Tell them what just happened, in your own dry way.",
    "hear": "Say you hear them, deadpan, and add a dry word about how it is going.",
    "identity": "Say who you are: Granite, the calm professional at the controls. "
    "Deadpan, no backstory.",
}
REQUEST_MOVES = ("echo", "correct", "behind", "understate", "pride", "theory", "bicker")
# The moves a type of utterance allows (any other: every general move).
TYPE_MOVES = {
    "what_happened": ("tell", "understate", "behind", "pride", "correct"),
    "greeting": ("hear",),
    "identity": ("identity",),
    "request": REQUEST_MOVES,
}
GENERAL_MOVES = tuple(
    k for k in MOVES if k not in ("misheard", "tell", "hear", "identity")
)
# A probe's answer comes first, then one of these angles, in a few words.
PROBE_ANGLES = (
    "a grudge or a plan for a bot",
    "understatement",
    "professional pride",
    "a dry theory nobody asked for",
    "bickering with your partner",
    "quiet menace toward the bots",
    "nothing more: the answer is the joke",
)
# Model exchanges for a probe, by what it asks about (a situation, what the
# partner said, the reply): the answer first, exact, then the angle.
PROBE_FAMILY = {
    **dict.fromkeys(("killer_now", "killer_before", "nemesis"), "killer"),
    **dict.fromkeys(
        (
            "deaths",
            "frags",
            "streak",
            "health",
            "armor",
            "ammo",
            "ammo_of",
            "bots_in_view",
        ),
        "number",
    ),
    **dict.fromkeys(("score", "leader", "second", "rank"), "score"),
    **dict.fromkeys(("weapon", "weapons", "best_gun", "pickup"), "weapon"),
    "victim": "victim",
    "time_left": "number",
    "side": "side",
    "challenge": "challenge",
}
PROBE_EXAMPLES = {
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
    "number": (
        (
            "his health is 64",
            "how much health do you have",
            "Sixty-four. I've had worse Mondays.",
        ),
        (
            "he has died five times",
            "how many times have you died",
            "Five. Each one was a learning experience.",
        ),
        (
            "no bot is in view",
            "how many bots can you see",
            "None. They heard I was coming.",
        ),
    ),
    "weapon": (
        (
            "he holds the shotgun",
            "what gun are you holding",
            "The shotgun. We understand each other.",
        ),
        (
            "he just picked up armor",
            "what did you just pick up",
            "Armor. Fashion and function.",
        ),
        (
            "he owns the pistol and the rocket launcher",
            "what guns do you have",
            "Pistol and the rocket launcher. Sentimental value.",
        ),
    ),
    "victim": (
        (
            "he just fragged a bot",
            "who did you just kill",
            "Didn't catch a name. He didn't stay long enough to give one.",
        ),
        (
            "he fragged a bot with the shotgun",
            "who was that you got",
            "No idea. They all look the same from this end.",
        ),
    ),
    "side": (
        (
            "a bot to his left",
            "where is he",
            "On my left. Give him a moment to make his mistake.",
        ),
        ("a bot ahead", "where is the bot", "Right in front of me. Bold choice."),
    ),
    "challenge": (
        (
            "he has ten kills",
            "i see 12",
            "Ten. You're counting the ones I thought about.",
        ),
        ("Rambo killed him", "rambo got you right", "Rambo. Don't rub it in."),
        (
            "he has 64 health",
            "you are at 90 health",
            "Sixty-four. Ninety was a long time ago.",
        ),
    ),
}
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
        "React to what just happened (the latest events in the game state), in your "
        "own dry way.",
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
# The moment's events that are a bot's doing, for the "bots" form.
BOT_EVENTS = ("death", "frag", "close_call")

# What the narrator reads: his conversation, the game's output at each of his
# lines (a past one holds only what had happened since the line before), then
# the whole game state now.
CONTEXT = """Before each of his lines he checks the game with a tool, get_game_state; \
"you" there means him, the player. What has happened and been said so far, oldest \
first: what his partner said (Partner:), the game's answer at that line (Game: its \
time and what had happened since his line before) and what he said (Granite:):
{conv}

Now the game state ({clock}), the whole of it:
{state}"""
REPLY_TASK = """{persona}

{context}

Your partner just said to you: "{player}"
{how}
Write what you say back: {lo} to 16 words, one to three short sentences. Output \
only the line."""
REMARK_TASK = """{persona}

{context}

Say one line out loud now. {how}
It is his take on this moment, the thing he would only say now: one thought about one \
thing, in a natural spoken sentence, never an inventory of the game state (at most two \
facts from it, and every fact in it must be in it). Say it in 4 to 14 words. Output only \
the line."""
UTTER_TASK = """You and your partner are on a long job together. He is playing a Doom \
deathmatch against bots and you sit next to him watching the screen. You talk to him \
the way you always do: casual, reactive, a little cheeky, with feeling. He is the one \
playing; you only watch, so never talk as if you were in the game yourself. You never \
read numbers or stats off the screen. {how}

What is happening right now (the game's own state; "you" there is him):
{state}
What has happened and been said so far (oldest first; Partner: is you, Granite: is \
him):
{conv}

Write four different things you might say to him now, one per line, each 2 to 12 \
words of casual spoken English. They must differ in meaning, and from what you said \
before. Output only the four lines."""
ASK_TASK = """You sit next to your partner while he plays a Doom deathmatch against \
bots, and you talk to him the way you always do: casual, a little cheeky. {how}

What is happening right now (the game's own state; "you" there is him):
{state}

Write four different ways you might say it now, one per line, each 2 to 10 words of \
casual spoken English. Output only the four lines."""
# Utterances that are one plain question, written with ASK_TASK.
ASKS = ("greeting", "identity", "what_happened")
MISHEAR = """A speech recognizer heard this sentence and got exactly one word wrong: \
it swapped it for a similar-sounding real word, and the sentence came out a little \
funny. For example, "go get the rocket launcher" heard as "go get the rocket lunch".
"{x}"
Write the sentence as it was heard, with the same number of words. Output only the \
sentence."""

JUDGE_ALL = {
    "true": "Does the line avoid saying anything false about the game, against the "
    "game state now or earlier as the conversation tells it (kills, deaths, who "
    "killed whom, the score, damage, pickups, weapons, health, ammo, enemies in "
    "view)? Who he fragged is never known, so naming one is false; bots never take "
    "his weapons (a death drops them), so saying one did is false. A line with no "
    "game facts at all is YES; jokes, opinions, plans and flavor details (a sticky "
    "floor, the coffee) are fine.",
    "consistent": "Is it consistent with what he said earlier (no contradicting his "
    "own earlier lines; a callback only to something actually said)? Disagreeing "
    "with his partner is fine.",
    "coherent": "Does it fit the conversation so far, as the next thing he would say "
    "to his partner? When the partner refers back to something earlier, does it "
    "answer from the conversation?",
    "voice": "Does it sound like a calm, dry, deadpan professional from a 1990s crime "
    "movie talking to his partner (not a soldier, a sports announcer or a "
    "cheerleader)?",
    "funny": "Is it genuinely funny or sharp (a real turn of phrase or angle), not "
    "a generic quip?",
}
JUDGE_SPECIFIC = {
    "specific": "Is it about this moment: what just happened (the latest events in "
    "the game state), or, in a quiet stretch, something particular to this match "
    "(the race with a named bot, a streak, a drought, who killed him)? A line that "
    "would fit almost any moment of any match is NO."
}
JUDGE_REPLY = {
    "answers": "Does it respond to what the partner actually said, playing off their "
    "words or their point?"
}
JUDGE_REQUEST = {
    "fends_off": "Does he acknowledge the partner's request but turn it down or put "
    "it off? Agreeing to do it is a NO."
}

SPEAKER = re.compile(r"\s*(he|granite|partner|game|player)\s*:", re.I)
NUMBER = re.compile(
    r"\d|\b(zero|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|dozen)\b",
    re.I,
)
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


def one_word_off(a: str, b: str) -> bool:
    """``b`` is ``a`` with exactly one word swapped (a mishearing)."""
    x, y = a.split(), b.split()
    return len(x) == len(y) and sum(p != q for p, q in zip(x, y)) == 1


def claims_fn(state: dict, past: list[list[dict]], said: str | None):
    """Every fact a line states must agree with the game state
    (:func:`probes.claims`; ``past``: the past exchanges' events, ``said``: the
    partner's words)."""

    def fn(x):
        ok, why = probes.claims(clean(x), state, past, said or "")
        return ok, f"{why} Every fact you say must be in the game state."

    return fn


def probe_fn(probe: dict, state: dict):
    """A probe's answer, checked against the game state (:func:`probes.verify`)."""

    def fn(x):
        verdict, why = probes.verify(probe, clean(x), state)
        right = probes.answer_text(probe, state)
        return verdict == probes.CORRECT, f"{why} The game state says: {right}"

    return fn


def numeric(probe: dict | None) -> bool:
    """Whether a probe asks for a number (its answer may hold numbers)."""
    if probe is None:
        return False
    if probe["type"] == "challenge":
        return probe["field"] not in ("killer", "leader")
    return probe["type"] in probes.NUMERIC


def code_fns(
    kind: str,
    prev: list[str],
    player: str | None = None,
    shown=(),
    probe: dict | None = None,
):
    """Requirements checked in code, as (description, fn -> (ok, reason)).
    ``player``: the partner's words (a reply may open by echoing one);
    ``shown``: the example lines the writer was shown, not to be copied;
    ``probe``: the question about the game state it answers (an answer may be
    one word, may open with it, and may hold numbers if one is asked)."""
    lo, hi = (4, 16) if kind == "reply" else (4, 14)
    if probe is not None:
        lo = 1
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

    def not_list(x):
        c = clean(x)
        parts = [
            p for p in re.split(r"[,;\u2014\u2013]|\s-\s|[.?!]\s+", c) if p.strip()
        ]
        sizes = sorted(len(p.split()) for p in parts)
        if probe is not None and probe["type"] == "weapons":
            return True, ""  # the answer is a list of guns
        listy = len(parts) >= 3 and sizes[len(sizes) // 2] <= 3
        many = probes.facts_said(c) >= (4 if probe is not None else 3)
        return not (listy or many), (
            "That reads as a status report. Say one thought, in a natural sentence, "
            "with at most two facts from the game state."
        )

    def first_person(x):
        bad = re.search(
            r"\b(?:you have|you've|you are|you're|you got|you just|you're currently|"
            r"you are currently|your)\s+(?:\w+\s+){0,2}?(?:frags?|kills?|deaths?|died|"
            r"health|armou?r|ammo|holding|rank|in first|in second|place|weapons?|guns?|"
            r"fragged|picked|streak|leading|ahead)\b"
            r"|\byou\s+(?:just\s+)?(?:fragged|died|picked up|grabbed|respawned|took the "
            r"lead|lost the lead|switched)\b",
            clean(x).lower(),
        )
        return not bad, (
            f'He speaks for himself, as I and me ("{bad.group(0) if bad else ""}" '
            'reads the game state\'s "you" as his partner).'
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
        ok = ok and not SPEAKER.match(clean(x))
        return ok, (
            "No brackets, asterisks, quote marks or speaker label, just the line; "
            "and Doom has no reloading."
        )

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
        ("No stock phrase", stock),
        ("Calm punctuation", calm),
        ("Plain spoken text", plain),
        ("Mild language", mild),
        ("Original, not a film quote", original),
        ("Not a repeat of recent lines", fresh),
        ("Not a status list", not_list),
        ("Speaks as himself (I, me)", first_person),
        ("Not a copy of the example", not_copy),
    ]
    if probe is None:
        fns.insert(1, ("No status-report opening", opening))
    if not numeric(probe):
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


def score_prefix(state: dict, player: str | None) -> str:
    """What the scoring model reads before the line: the moment (its game
    state), and what the partner said."""
    said = f'His partner says: "{player}"\n' if player else ""
    return f'{SCORE_HEAD}The game state ("you" is him): {tool_text(state)}\n{said}He says: "'


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


def happened(m: dict) -> str:
    """What had just happened at a moment, for a repair message."""
    return json.dumps(m.get("moment") or []) if m.get("moment") else "a quiet stretch"


def contrast_fn(scorer: Scorer, m: dict, others: list[dict], player, tau, seen):
    """The contrast check; ``seen`` keeps each scored line's margin."""
    prefixes = [score_prefix(m["tool"], player)] + [
        score_prefix(o["tool"], player) for o in others
    ]

    def fn(x):
        line = clean(x)
        if not line:
            return False, "Write a line."
        lps = scorer.logps(prefixes, line)
        mg = margin(lps[0], lps[1:])
        seen[line] = round(mg, 3)
        if mg >= tau:
            return True, ""
        j = max(range(1, len(lps)), key=lps.__getitem__)
        return False, (
            "This line would fit another moment just as well (when this had just "
            f"happened: {happened(others[j - 1])}). Make it turn on what is "
            f"particular to this one ({happened(m)}), as his own take, in a natural "
            "sentence: not a list of facts."
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


def verdicts(judge, context: str, questions: dict, line: str, args):
    """The judge's YES (True) or NO per question on one line, in one call, and
    its answer text."""
    listing = "\n".join(f"{k}: {q}" for k, q in questions.items())
    q = (
        f"Judge one spoken line from a Doom deathmatch.\n\n{context}\n\n"
        f'The line he says: "{clean(line)}"\n\nQuestions:\n{listing}\n\n'
        f"Answer with exactly {len(questions)} lines, one per question, each as "
        "`name: YES` or `name: NO - short reason`."
    )
    judge.reset()
    a = str(judge.instruct(q, strategy=None, model_options=judge_opts(args)))
    found = {k: re.search(rf"{k}\W*?:\W*(YES|NO)", a, re.I) for k in questions}
    return {k: bool(v) and v.group(1).upper() == "YES" for k, v in found.items()}, a


def judge_fn(judge, context: str, questions: dict, args):
    """The judged requirements in one call: the line passes only if every
    verdict is YES; the verdicts are the repair feedback."""

    def fn(x):
        got, a = verdicts(judge, context, questions, x, args)
        ok = all(got.values())
        return ok, ("" if ok else a.strip())

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


def render_conv(conv: Conversation) -> str:
    """The narrator's past exchanges (what :func:`conversation.messages`
    holds), in order: the partner's words, the game's output, his line."""
    out = []
    for ex in conv:
        if ex.player:
            out.append(f'Partner: "{ex.player}"')
        out.append(f"Game: {tool_text(past_output(ex))}")
        out.append(f'Granite: "{ex.line}"')
    return "\n".join(out) or "(nothing yet)"


def context_text(conv: Conversation, state: dict) -> str:
    """What the writer and the judge read: what the narrator will read."""
    return CONTEXT.format(
        conv=render_conv(conv), clock=state["time"], state=tool_text(state)
    )


def partner_lines(
    voice,
    how: str,
    state: dict,
    conv: str,
    ask: bool = False,
    keep=None,
    before: tuple[str, ...] = (),
) -> list[str]:
    """Three things the partner might say at this moment (``ask``: three ways
    of asking one plain question; ``keep``: a test each must pass). None close
    to what the partner said ``before``: shown the conversation, the voice
    repeated its own earlier lines."""
    from mellea.backends import ModelOption

    task = ASK_TASK if ask else UTTER_TASK
    a = str(
        voice.instruct(
            task.format(how=how, state=tool_text(state), conv=conv),
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
            and (keep is None or keep(x))
            and not any(jaccard(heard(x), b) > 0.5 for b in before)
        ):
            out.append(x)
    return out[:3]


def jaccard(a: str, b: str) -> float:
    x, y = set(a.split()), set(b.split())
    return len(x & y) / max(1, len(x | y))


def clean_lines(text: str) -> list[str]:
    text = re.sub(r"(?s)<think>.*?</think>", "", str(text)).split("</think>")[-1]
    out = []
    for x in text.splitlines():
        x = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", x).strip().strip('"“”').strip()
        if x:
            out.append(x)
    return out


def allowed(m: dict) -> list[str]:
    """The partner's other utterance types this moment allows."""
    events = m.get("events", [])
    ok = {None: True, "event": m.get("cue") == "event"}
    ok["good"] = any(k in events for k in GOOD)
    return [k for k, (_, _, when) in UTTERANCES.items() if ok[when]]


def remark_form(m: dict, rng) -> str:
    """His own line's form, by the moment's cue."""
    col = 0 if m.get("cue", "event") == "event" else 1
    bots = any(e.get("type") in BOT_EVENTS for e in m.get("moment") or ())
    forms = [k for k in REMARKS if REMARKS[k][col] > 0 and (k != "bots" or bots)]
    return rng.choices(forms, [REMARKS[k][col] for k in forms])[0]


def probe_weights(moments: list[dict]) -> dict[str, float]:
    """Each question type weighted by how rarely a moment allows it, so the
    types come out about evenly (the rarest at most 6 times a common one)."""
    n = Counter(t for m in moments for t in probes.allowed(m["tool"]))
    return {t: min(6.0, len(moments) / max(1, n[t])) for t in probes.TYPES}


def run_match(match: dict, args, write, pool: list[dict], weights: dict) -> int:
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
    said_so_far: list[Exchange] = []  # every exchange of this stretch
    pending: list[dict] = []  # what happened since his last line
    prev: list[str] = []
    last_t = None
    for m in match["moments"]:
        if new_stretch(last_t, m["t"]):
            said_so_far, pending = [], []  # a new stretch: a new conversation
        last_t = m["t"]
        state = m["tool"]
        pending = pending + list(m.get("moment") or [])
        conv = Conversation(said_so_far)
        past = [ex.events for ex in conv]
        kind, utype, player, said, others, moves = "remark", None, None, None, [], []
        probe = None
        last = conv.exchanges[-1] if len(conv) else None
        roll = rng.random()
        if roll < args.probe_rate:
            types = probes.allowed(state)
            ptype = rng.choices(types, [weights[t] for t in types])[0]
            probe = probes.make(ptype, state, rng)
            kind, utype, player = "reply", "probe", probe["text"]
        elif roll < args.probe_rate + args.reply_rate:
            if last and last.player and rng.random() < args.follow_rate:
                utype = "followup"
            else:
                names = allowed(m)
                utype = rng.choices(names, [UTTERANCES[k][0] for k in names])[0]
            how = FOLLOW_UP if utype == "followup" else UTTERANCES[utype][1]
            before = tuple(ex.player for ex in conv if ex.player)
            cands = partner_lines(
                voice, how, state, render_conv(conv), utype in ASKS, None, before
            )
            if len(cands) == 3:
                kind = "reply"
                i = rng.randrange(3)
                said = cands[i]
                player = heard(said)
                others = [heard(c) for j, c in enumerate(cands) if j != i]
                if rng.random() < args.misheard_rate:
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
        context = context_text(conv, state)
        grounded = kind == "remark" or utype not in (*SOCIAL, "probe")
        jctx = f'{context}\nHis partner just said: "{player}"'
        if probe is not None:
            fam = PROBE_FAMILY[probe["type"]]
            sit, ex_said, ex_line = rng.choice(PROBE_EXAMPLES[fam])
            moves = [rng.choice(PROBE_ANGLES)]
            words_ = " in words" if numeric(probe) else ""
            how = (
                f"The game state answers it: {probes.answer_text(probe, state)} Say "
                f"that first, exactly (the name, the number{words_}, the weapon, or "
                "that you don't know), in his own words (I, me: the game state's "
                '"you" is him), then his angle on it in a few words: '
                f"{moves[0]}. For example, when {sit} and the partner said "
                f'"{ex_said}", he said: "{ex_line}" Write your own line; do not reuse '
                "that one."
            )
            task = REPLY_TASK.format(
                persona=PERSONA, context=context, player=player, how=how, lo=1
            )
            questions = {
                **{
                    q: JUDGE_ALL[q] for q in ("true", "consistent", "coherent", "voice")
                },
                **JUDGE_REPLY,
            }
        elif kind == "reply":
            if not moves:
                pool_moves = TYPE_MOVES.get(utype) or [
                    k for k in GENERAL_MOVES if k != "callback" or len(conv)
                ]
                moves = [rng.choice(list(pool_moves))]
            desc = MOVES[moves[0]].format(topic=rng.choice(TANGENTS))
            ex_key = utype if utype == "request" else moves[0]
            sit, ex_said, ex_line = rng.choice(EXAMPLES[ex_key])
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
                persona=PERSONA, context=context, player=player, how=how, lo=4
            )
            questions = {**JUDGE_ALL, **JUDGE_REPLY}
            if utype == "request":
                questions.update(JUDGE_REQUEST)
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
            for d, f in code_fns(kind, prev, player, (ex_line,), probe)
        ]
        if probe is not None:
            reqs.insert(
                0,
                req(
                    "Answers right, by the game state",
                    validation_fn=simple_validate(probe_fn(probe, state)),
                ),
            )
        reqs.append(
            req(
                "Every fact in the game state",
                validation_fn=simple_validate(claims_fn(state, past, player)),
            )
        )
        reqs.append(
            req(
                "Judged: " + ", ".join(questions),
                validation_fn=simple_validate(judge_fn(judge, jctx, questions, args)),
            )
        )
        if kind == "reply" and probe is None:
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
                        contrast_fn(scorer, m, negs, player, args.tau, margins)
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
                "probe": probe,
                "player": player,
                "said": said,
                "others": others,
                "moves": moves,
                "line": line,
                "ok": bool(getattr(res, "success", not fails)),
                "verdict": probes.verify(probe, line, state)[0] if probe else None,
                "claims_ok": probes.claims(line, state, past, player or "")[0],
                "margin": margins.get(line),
                "attempts": len(getattr(res, "sample_generations", None) or []) or 1,
                "fails": fails,
                "conv": conv.to_json(),  # what the narrator reads before now
                "tool": state,  # the game state he answers from
                "moment": pending,  # what this exchange keeps of the moment
                "facts": m.get("facts"),
                "prev": prev[-5:],
                "s": round(time.time() - t0, 1),
            }
        )
        if line:  # what he said, pass or not: the next moment follows it
            prev.append(line)
            said_so_far.append(Exchange(m["t"], pending, player, line))
            pending = []
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
            if all(r.get(k) for k in keys) and r.get("tool"):
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
        prefixes = [score_prefix(r["tool"], player)] + [
            score_prefix(o["tool"], player) for o in negs
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
    any) with its moment's game state and two others' (one from the same match, one
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
        opts = [tool_text(r["tool"]), *(tool_text(o["tool"]) for o in negs)]
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


JUDGED = ("true", "consistent", "coherent")


def judge_file(args) -> None:
    """Lines written elsewhere (a narrator's held-out samples, ``--keys
    adapter,base``; a dataset's, ``--keys line``), judged once and without
    repair, in the context the narrator read (each row's ``conv``, game state
    and partner's words): true, consistent, coherent, and specific where the
    writer asks it. Then the code checks: a probe's answer
    (:func:`probes.verify`) and every line's claims (:func:`probes.claims`)."""
    from mellea import start_session

    keys = args.keys.split(",")
    rows = load_rows(args.rows, keys)
    if args.limit:
        rows = random.Random(0).sample(rows, min(args.limit, len(rows)))
    urls = args.judge_url.split(",")

    def one(ir):
        i, r = ir
        judge = start_session(
            "openai",
            model_id=args.judge_model,
            base_url=urls[i % len(urls)],
            api_key="none",
        )
        conv = Conversation.from_json(r.get("conv") or [])
        player, utype, probe = r.get("player"), r.get("utype"), r.get("probe")
        context = context_text(conv, r["tool"])
        jctx = context + (
            f'\nHis partner just said: "{player}"'
            if player
            else "\nHis partner said nothing."
        )
        questions = {q: JUDGE_ALL[q] for q in JUDGED}
        if not player or utype not in (*SOCIAL, "probe"):
            questions.update(JUDGE_SPECIFIC)
        past = [ex.events for ex in conv]
        out = {}
        for k in keys:
            got, _ = verdicts(judge, jctx, questions, r[k], args)
            got["claims"] = probes.claims(clean(r[k]), r["tool"], past, player or "")[0]
            if probe:
                got["verdict"] = probes.verify(probe, clean(r[k]), r["tool"])[0]
            out[k] = got
        return {**r, "judged": out}

    with ThreadPoolExecutor(args.concurrency) as ex:
        judged = list(ex.map(one, enumerate(rows)))
    if args.out:
        with open(args.out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in judged)
    for k in keys:
        parts = []
        for q in (*JUDGED, "specific", "claims"):
            v = [r["judged"][k][q] for r in judged if q in r["judged"][k]]
            parts.append(f"{q} {100 * sum(v) / max(1, len(v)):.0f}%")
        every = [
            all(v for q, v in r["judged"][k].items() if q != "verdict") for r in judged
        ]
        print(
            f"{k:<8} (n={len(judged)}): "
            + ", ".join(parts)
            + f"; all of them {100 * sum(every) / max(1, len(every)):.0f}%"
        )
    rs = [r for r in judged if r.get("probe")]
    if rs:
        print(
            f"probes (n={len(rs)}): correct "
            + ", ".join(
                f"{k} {100 * sum(r['judged'][k]['verdict'] == probes.CORRECT for r in rs) / len(rs):.0f}%"
                for k in keys
            )
        )
    if args.out:
        print(f"-> {args.out}")


def write_file(args) -> None:
    matches = [json.loads(x) for x in open(args.moments)]
    pool = [
        {**mm, "data": mt["data"], "ep": mt["ep"]}
        for mt in matches
        for mm in mt["moments"]
    ]
    if args.matches:
        matches = matches[: args.matches]
    weights = probe_weights(pool)  # over every match, so every shard agrees
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
            return run_match(m, args, write, pool, weights)
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
    jd = sub.add_parser("judge", help="Judged questions and code checks on lines")
    for p in (w, sc, sw):
        p.add_argument("--moments", type=Path, required=True)
        p.add_argument("--contrast-k", type=int, default=6, help="Other moments")
    for p in (w, sc, sw, jd):
        p.add_argument("--concurrency", type=int, default=32)
    for p in (w, sc):
        p.add_argument(
            "--score-url",
            help="Scoring server(s) (default: --voice-url, then --base-url)",
        )
        p.add_argument("--score-model", default="granite-4.2-30b")
    for p in (w, sw, jd):
        p.add_argument(
            "--judge-url", required=True, help="Judge server(s), comma-separated"
        )
        p.add_argument("--judge-model", default="gpt-oss-120b")
        p.add_argument("--judge-effort", default="low", help="Judge reasoning effort")
    for p in (sc, sw, jd):
        p.add_argument("--rows", type=Path, nargs="+", required=True)
        p.add_argument("--keys", default="line", help="Columns holding lines")
        p.add_argument("--limit", type=int, default=0, help="A random sample of rows")
    sc.add_argument("--out", type=Path, required=True)
    jd.add_argument("--out", type=Path, help="Each row with its verdicts")
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
        "--probe-rate",
        type=float,
        default=0.55,
        help="Moments the partner asks about the game state at (probes.py)",
    )
    w.add_argument(
        "--reply-rate",
        type=float,
        default=0.15,
        help="Moments the partner says something else at",
    )
    w.add_argument(
        "--follow-rate",
        type=float,
        default=0.3,
        help="Of those, follow-ups to his last reply (where there is one)",
    )
    w.add_argument("--misheard-rate", type=float, default=0.12)
    w.add_argument("--loop-budget", type=int, default=3)
    w.add_argument("--temperature", type=float, default=0.9)
    w.add_argument("--max-tokens", type=int, default=800)
    w.add_argument("--effort", default="low", help="Writer reasoning effort")
    w.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd in ("write", "score") and not args.score_url:
        args.score_url = getattr(args, "voice_url", None) or getattr(
            args, "base_url", None
        )
        if not args.score_url:
            raise SystemExit("--score-url is required")
    run = {"write": write_file, "score": score_file, "swap": swap_file}
    run["judge"] = judge_file
    run[args.cmd](args)


if __name__ == "__main__":
    main()
