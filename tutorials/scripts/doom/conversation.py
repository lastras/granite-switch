# SPDX-License-Identifier: Apache-2.0
"""The narrator's own conversation with the person watching, apart from the game log.

The game adapters read the 5 Hz game log (:mod:`history`); the narrator reads a
chat of its own, in the OpenAI message format Granite's chat template renders,
the game reaching him through a tool, ``get_game_state``, declared in the system
turn (the template adds its tools block)::

    system     {NARRATOR_SYSTEM_PROMPT} + the template's <tools>{GET_GAME_STATE}</tools>
      per past exchange (the last CONV_EXCHANGES):
    user       who got you                                 <- his partner, if they spoke
    assistant  <tool_call>{"name": "get_game_state", "arguments": {}}</tool_call>
    tool       {"time": "2:31", "events": [{"type": "death", "killer": "Rambo", ...}]}
    assistant  Rambo. Twice now.                           <- him
      now:
    user       {the partner's words, if any; live speech: <|audio|>}
    assistant  <tool_call>{"name": "get_game_state", "arguments": {}}</tool_call>
    tool       {the whole state: talk.game_state}
    assistant  -> his line

The harness writes every tool call; the model writes only his lines. A past
exchange's tool output keeps that moment's time and what had happened since the
line before (deaths and killers, frags, pickups, lead changes, close calls:
``talk.moment_events``), which stays true; none of the readings (score, health,
ammo, ...), so the only such numbers in the prompt are the current ones.

:func:`narrator_ids` renders it with the tokenizer's chat template
(``tools=[GET_GAME_STATE]``; ``adapter_name`` puts the narrator's control token
at the last assistant header only, so the tool calls and outputs run on base
weights). The dataset writer (``narrator_data.py``), training
(``train_alora.py``) and live play (``engine.py``, ``record_video.py``) all
build it here. This module imports nothing from the game: the writer runs in
Mellea's environment, which has no ViZDoom.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace

TIC_HZ = 35  # doom_env.TIC_HZ
# Training and serving keep the same window (about 2.5 minutes of talk), so
# training covers every conversation length the narrator meets live.
CONV_EXCHANGES = 30
# A recorded match's moments may come in a few unbroken stretches (talk.py
# moments --per-match); live, he never stays quiet this long, so in the dataset
# a gap this long between moments starts a new conversation (narrator_data.py
# writes it, train_alora.py reads it).
STRETCH_GAP_S = 30.0
TOOL_NAME = "get_game_state"

NARRATOR_SYSTEM_PROMPT = (
    "You are Granite, a calm, dry professional out of a 1990s crime movie, playing "
    "a Doom deathmatch against bots. Your partner sits next to you, watching your "
    "screen and talking to you. Your partner does not play, and the bots are not "
    "your partner. Before you speak you call get_game_state: its result is the game "
    "itself telling you how the match stands, and in it, you means you, the player. "
    "Take every fact you say from the latest result: names, numbers, weapons and "
    "what happened. Earlier results in this conversation tell only what had just "
    "happened then. When your partner asks about the game, answer first, exactly as "
    "the latest result says (the name, the number, the weapon), then add your angle "
    "if you like; when it does not say, say you don't know. If your partner says "
    "something about the game that the result contradicts, correct them. Your "
    "partner's words come as speech recognition writes them: lower case, no "
    "punctuation, now and then a word misheard; they are almost always about this "
    "match, so read them for what they most likely meant. Bots never take "
    "your weapons: when you die, you respawn with a pistol. Say a number only when "
    "your partner asks for one. Your partner can tell you what to do; order says "
    "what you were told and whether you are doing it: never say you are doing "
    "what you refused or could not do. Your turns are what you say out loud: one "
    "short line, in character, deadpan, mild language at most. Doom has no "
    "reloading."
)

GET_GAME_STATE = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "The Doom deathmatch right now, from the game. Returns JSON. time, "
            "time_left: the match clock (m:ss). map. you: frags, deaths, rank (1 is "
            "first; a tie shares the rank), players, lead (frags ahead of the best "
            "other player; negative: behind the leader), health, armor, holding (the "
            "weapon in your hands and its ammo), weapons (every gun you own and its "
            "ammo), best_loaded_weapon, frags_last_10s. scoreboard: every player's "
            "place, name, frags and deaths, best first (you is you). bots_in_view: "
            "side (left, ahead or right) and distance_m of each; who they are is "
            "never known. playing: your style of play and your last moves. "
            "order: the last thing your partner told you to do (told), its status "
            "(doing, done, refused: you would not, cant: you could not, cancelled), "
            "why, seconds_ago, health_lost while you did it, hit_wall, got (what a "
            "goal got you: the item you picked up, the bot you fragged). "
            "last_death: killer (a bot's name, yourself, or unknown), killer_weapon, "
            "your_weapon (yours then), seconds_ago, in_a_row (deaths to that bot in a "
            "row). last_frag: victim, your_weapon, seconds_ago. killed_by: how many "
            "times each bot has killed you. storylines: the match's stories so far, "
            "from its events: bots_war (on_a_tear: bots with 3 or more kills in the "
            "last 60 seconds; feuds: a bot that keeps killing the same bot), race "
            "(leader, chaser, the gap between them, lead_changes: how often you took "
            "or lost the lead), grudges (nemesis: the bot that has killed you most; "
            "favorite_victim: the bot you have fragged most), your_play (best_streak "
            "this match, no_frag_for_s). last_pickup: item, amount, seconds_ago. "
            "recent_events: the last 90 seconds, oldest first: death, frag (your "
            "victim), kill (a bot killing another bot), pickup, lead (who leads "
            "now), close_call, streak, order (told, and its status), order_end (it "
            "is over: done, cancelled, or refused halfway), order_hurts (it cost you "
            "health). A past call's result holds only its time and the events since "
            "the call before."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def new_stretch(prev_tick: int | None, tick: int) -> bool:
    return prev_tick is not None and tick - prev_tick > STRETCH_GAP_S * TIC_HZ


def clock(tick: int) -> str:
    """Match time as m:ss."""
    s = int(tick / TIC_HZ)
    return f"{s // 60}:{s % 60:02d}"


@dataclass(frozen=True)
class Exchange:
    """One moment he spoke at: its tick, what had happened since the line
    before (``talk.moment_events``), what the partner said (None if nothing),
    and his line."""

    tick: int
    events: list
    player: str | None
    line: str


class Conversation:
    """The last ``n`` exchanges, oldest first."""

    def __init__(self, exchanges=(), n: int = CONV_EXCHANGES):
        self.n = n
        self.exchanges: list[Exchange] = []
        for ex in exchanges:
            self.add(ex)

    def add(self, ex: Exchange) -> None:
        """Append one exchange (its line on one line), dropping the oldest."""
        ex = replace(ex, line=" ".join(ex.line.split()))
        self.exchanges = [*self.exchanges, ex][-self.n :]

    def __iter__(self):
        return iter(self.exchanges)

    def __len__(self) -> int:
        return len(self.exchanges)

    def to_json(self) -> list[dict]:
        return [asdict(e) for e in self.exchanges]

    @classmethod
    def from_json(cls, rows: list[dict], n: int | None = None) -> Conversation:
        """A saved conversation, whole (``n``: keep at most so many)."""
        return cls((Exchange(**r) for r in rows), n or max(1, len(rows)))


def past_output(ex: Exchange) -> dict:
    """A past exchange's tool output: its time and its events."""
    return {"time": clock(ex.tick), "events": ex.events}


def tool_text(output: dict) -> str:
    return json.dumps(output)


def _checks(output: dict) -> list[dict]:
    """His call to the tool (written by the harness) and its output."""
    call = {"type": "function", "function": {"name": TOOL_NAME, "arguments": {}}}
    return [
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "content": tool_text(output)},
    ]


def messages(conv: Conversation, state: dict, player: str | None = None) -> list[dict]:
    """The narrator's chat in OpenAI messages, the system turn first: per past
    exchange the partner's words (if any), the tool call and its short output,
    and his line; then the partner's words now (or ``policy.AUDIO_MARKER``,
    for speech the model transcribes itself), the call, and ``state`` (the
    whole state, ``talk.game_state``)."""
    out = [{"role": "system", "content": NARRATOR_SYSTEM_PROMPT}]
    for ex in conv:
        if ex.player:
            out.append({"role": "user", "content": ex.player})
        out += _checks(past_output(ex))
        out.append({"role": "assistant", "content": ex.line})
    if player:
        out.append({"role": "user", "content": player})
    return out + _checks(state)


def narrator_text(
    tok, conv: Conversation, state: dict, player: str | None = None, adapter=None
) -> str:
    """The narrator's prompt as the chat template renders it, up to and
    including the assistant header he writes after (``adapter``: the
    narrator's name in a composed checkpoint, for its control token; None:
    the base model, as PEFT trains it)."""
    kw = {"adapter_name": adapter} if adapter else {}
    return tok.apply_chat_template(
        messages(conv, state, player),
        tools=[GET_GAME_STATE],
        add_generation_prompt=True,
        tokenize=False,
        **kw,
    )


def narrator_ids(
    tok, conv: Conversation, state: dict, player: str | None = None, adapter=None
) -> list[int]:
    """:func:`narrator_text`, as token ids."""
    text = narrator_text(tok, conv, state, player, adapter)
    return tok(text, add_special_tokens=False).input_ids
