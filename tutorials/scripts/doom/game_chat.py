# SPDX-License-Identifier: Apache-2.0
"""The game as a conversation in Granite's own chat template, for fast decisions.

One match is one conversation, in the OpenAI message format the chat template
renders. The game reaches the model through two tools declared in the system
turn (the template adds its tools block); the harness writes every tool call::

    system     SYSTEM_PROMPTS["s1"] + the template's <tools>[get_game_events, look]</tools>
    assistant  <tool_call>{"name": "get_game_events", "arguments": {}}</tool_call>
    tool       {"time": "0:00", "events": [{"type": "start", "players": ["Gwen", ...]}]}
      ... one call and its result per push, as the match goes ...
    -- one decision's suffix, never kept --
    assistant  <tool_call>{"name": "look", "arguments": {}}</tool_call>
    tool       {look(obs)}
    <|move|>assistant<|end_of_role|>          -> one action token
                                                 (<|weapon|>: one slot digit)

**The conversation** grows for the whole match and is never windowed. It holds
the match's key events only (:data:`KEY_KINDS`: deaths and their killers,
frags, lead changes, close calls, streaks), in the narrator's own event JSON
(:func:`talk.event_json`), pushed at most once a second: on the first whole
second with any pending (:meth:`GameChat.observe`). Pickups, bots killing bots
and respawns stay out: the look holds health, armor and the guns, and a
respawn follows every death. A ten-minute match of the teacher's ends at 8.5k
to 11.4k tokens. Later steps add the partner's words and the player's lines
to the same conversation.

**The look** (:func:`look`) is the present, as JSON: what the RL actor reads
(:func:`doom_env.features`), lossless. :func:`vector_of_look` rebuilds the
actor's input vector (:func:`rl_teacher.obs_vector`, fighter) from the JSON
alone, and ``python game_chat.py check`` proves it equal on real play. The
student sees everything the teacher's actor reads; what it cannot see is the
teacher's recurrent memory.

**One builder** (:class:`GameChat`) serves the data writer (``decide.py``) and
live play (``engine.py --decider chat``), so the two cannot drift. Its prompt
ids are assembled from pre-tokenized pieces of the chat template's own
rendering (:class:`Pieces`), only the look's block tokenized fresh each tic;
the check proves them equal to ``apply_chat_template``. Under an aLoRA the
whole conversation runs on base weights, so its KV is shared by every
adapter and by every tic of the match (prefix caching).

This module imports no Mellea and no torch: serving and training read it too.

::

    python game_chat.py check                  # on CPU: 60 s of the scripted player
    python game_chat.py check --model models/decide-s1   # and the composed template
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from conversation import clock, tool_text
from doom_env import (
    ACTION_LABELS,
    ACTIONS,
    AMMO_ITEM_SLOT,
    ENEMY_LABEL,
    ITEMS,
    TIC_HZ,
    WEAPON_ITEM_SLOT,
    WEAPON_NAMES,
    WEAPON_SLOTS,
    DoomEnv,
    features,
)
from expert import PLAN_EVERY_TICS
from talk import SAY, Tracker, event_json

MOVE, WEAPON = "move", "weapon"  # the decision adapters
WEAPON_TOKENS: tuple[str, ...] = tuple(str(s) for s in WEAPON_SLOTS)
OUTPUTS: dict[str, tuple[str, ...]] = {MOVE: ACTIONS, WEAPON: WEAPON_TOKENS}
STYLE = "fighter"  # step 1 plays one style: it is constant, so not in the look
FORMAT = "s1"  # this module's conversation: the system prompt, tools, events, look
# The screen the look is read from (what is on screen depends on it, and the
# status bar hides the bottom of the view): collect.py's, and engine.py's.
RESOLUTION = "640X480"

# ── The system turn and the tools ──────────────────────────────────────────────
EVENTS_TOOL, LOOK_TOOL = "get_game_events", "look"


def _action_text(a: str) -> str:
    lab = ACTION_LABELS[a]
    return a if lab == a else (lab if lab.startswith(a) else f"{a} ({lab})")


ACTIONS_TEXT = ", ".join(_action_text(a) for a in ACTIONS)
SLOTS_TEXT = ", ".join(f"{s} {SAY[WEAPON_NAMES[s]]}" for s in WEAPON_SLOTS)
SYSTEM_PROMPTS = {
    "s1": (
        "You are Granite, playing a Doom deathmatch against bots. The game reaches "
        f"you through two tools. {EVENTS_TOOL} tells what has happened in the match "
        "since its last call; the game calls it for you when something happens. "
        f"{LOOK_TOOL} shows what you see and hold right now. Asked for a move, answer "
        f"with one action: {ACTIONS_TEXT}. Asked for a weapon, answer with one weapon "
        f"slot: {SLOTS_TEXT}."
    )
}


def _tool(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }


GET_GAME_EVENTS = _tool(
    EVENTS_TOOL,
    "What has happened in the match since the last call, from the game. Returns "
    "JSON: time (the match clock, m:ss) and events, oldest first. start: the match "
    "begins (players: the bots in it). death: you died (killer: a bot's name, "
    "yourself, or unknown; killer_weapon; your_weapon: yours then; in_a_row: deaths "
    "to that bot in a row). frag: you fragged a bot (victim, your_weapon; "
    "first_in_s: your first frag in that many seconds). lead: who leads now "
    "(leader: you, or a bot's name; tied_with_you). close_call: you lived through "
    "a big drop in health (health_lost, health_left). streak: frags_in_10s.",
)
LOOK = _tool(
    LOOK_TOOL,
    "What you see and hold right now, from the game. Returns JSON. hp, armor. "
    "weapon: the one in your hands, and its ammo. arms: every weapon slot you own "
    f"and its ammo ({SLOTS_TEXT}). at: your position, metres east and north of the "
    "map's south-west corner. face: your heading, degrees counter-clockwise from "
    "east. see: what is on screen, each as [name, bearing, metres]: bots (bot), "
    "nearest first, then an incoming missile, then items you would pick up, "
    "nearest first; bearing in degrees from where you face, negative to the left. "
    "walls: metres of room to your left, front, right and back (9 at most). hit: "
    "damage you took in the last second. enemy_last: where the last bot you saw "
    "was, [bearing, seconds ago], when none is in view. last: your last two "
    "actions, oldest first. weapon_tic: there when a weapon choice is due.",
)
TOOLS = [GET_GAME_EVENTS, LOOK]


def call(name: str) -> dict:
    """The harness's call to a tool, as an assistant turn."""
    fn = {"type": "function", "function": {"name": name, "arguments": {}}}
    return {"role": "assistant", "content": "", "tool_calls": [fn]}


def result(output: dict) -> dict:
    return {"role": "tool", "content": tool_text(output)}


# ── Key events ─────────────────────────────────────────────────────────────────
# The tracker's kinds (talk.Tracker) the conversation keeps. Not kept: supply
# pickups ("pickup") and new guns ("weapon": the look has hp, armor and arms;
# every respawn restarts the guns, so a match has some 60, and they made the
# conversation a quarter longer), bots killing bots ("kill"), "respawn" (every
# death has one), "drought_ended" (the frag says first_in_s), and the
# partner's orders (a later step).
KEY_KINDS = ("died", "frag", "took_lead", "lost_lead", "close_call", "streak")


def key_events(fired: list[dict]) -> list[dict]:
    """A tic's tracker events the conversation keeps, as the tool reports them."""
    out = [event_json(e) for e in fired if e["kind"] in KEY_KINDS]
    return [e for e in out if e is not None]  # a streak only at its levels


def players_of(facts: dict) -> list[str]:
    """The match's bots, from the tracker's facts (``Tracker.facts``), as the
    start event lists them."""
    return [name for name, _ in facts["board"]]


# ── The look ───────────────────────────────────────────────────────────────────
def look(obs) -> dict:
    """What the player sees and holds now (a live observation), as the look
    tool returns it: every field the RL actor reads (:func:`doom_env.features`),
    at its precision. ``enemy_last`` only when set; ``weapon_tic`` only on the
    weapon planner's tics (the actor's plan flag)."""
    out = {
        "hp": obs.hp,
        "armor": obs.armor,
        "weapon": obs.weapon,
        "ammo": obs.ammo,
        "arms": {str(s): a for s, a in sorted(obs.arms.items())},
        "at": list(obs.pos),
        "face": obs.face,
        "see": [[o.label, o.b, o.d] for o in obs.seen],
        "walls": dict(zip("lfrb", obs.walls)),
        "hit": obs.hit,
    }
    if obs.enemy_mem is not None:
        out["enemy_last"] = list(obs.enemy_mem)
    out["last"] = list(obs.last)
    if obs.tick % PLAN_EVERY_TICS == 0:
        out["weapon_tic"] = True
    return out


def _label_table() -> dict[str, tuple[str, int, str]]:
    """Each label the look can show -> (kind, slot, a class with that label).
    The actor reads an item's kind and the slot its class feeds; both must be
    the label's alone, or the look would lose what the class said."""
    out: dict[str, tuple[str, int, str]] = {
        ENEMY_LABEL: ("enemy", 0, "DoomPlayer"),
        "missile": ("missile", 0, "Rocket"),
    }
    for cls, (label, kind) in ITEMS.items():
        slot = WEAPON_ITEM_SLOT.get(cls, AMMO_ITEM_SLOT.get(cls, 0))
        if label in out:
            assert out[label][:2] == (kind, slot), (label, out[label], cls)
        else:
            out[label] = (kind, slot, cls)
    return out


LABEL_KIND_SLOT = _label_table()
SLOT_OF_WEAPON = {name: s for s, name in WEAPON_NAMES.items()}


def actor_view(lk: dict) -> SimpleNamespace:
    """The fields :func:`doom_env.features` reads, rebuilt from a look alone."""
    seen = [
        SimpleNamespace(
            cls=LABEL_KIND_SLOT[label][2], kind=LABEL_KIND_SLOT[label][0], b=b, d=d
        )
        for label, b, d in lk["see"]
    ]
    return SimpleNamespace(
        hp=lk["hp"],
        armor=lk["armor"],
        ammo=lk["ammo"],
        face=lk["face"],
        pos=tuple(lk["at"]),
        slot=SLOT_OF_WEAPON[lk["weapon"]],
        arms={int(s): a for s, a in lk["arms"].items()},
        seen=seen,
        walls=tuple(lk["walls"][k] for k in "lfrb"),
        hit=lk["hit"],
        enemy_mem=tuple(lk["enemy_last"]) if "enemy_last" in lk else None,
        last=tuple(lk["last"]),
    )


def vector_of_look(lk: dict, style: str = STYLE) -> np.ndarray:
    """The RL actor's input (:func:`rl_teacher.obs_vector`) from a look alone:
    the features, the style one-hot, the plan flag (``weapon_tic``)."""
    from rl_teacher import STYLES

    extra = np.zeros(len(STYLES) + 1, dtype=np.float32)
    extra[STYLES.index(style)] = 1.0
    extra[-1] = float(bool(lk.get("weapon_tic")))
    return np.concatenate([features(actor_view(lk)), extra])


# ── The conversation and its prompt ids ────────────────────────────────────────
class Pieces:
    """One tokenizer's fixed prompt pieces, cut from the chat template's own
    rendering of a probe conversation: the system turn (its tools block
    included), each tool's call turn, a tool result's turn around its JSON,
    and the generation prompt. Every piece begins with a special token, so
    tokenizing them apart gives the ids of the whole."""

    MARK = "@@JSON@@"

    def __init__(self, tok, system: str = SYSTEM_PROMPTS[FORMAT]):
        self.tok = tok
        sys = [{"role": "system", "content": system}]

        def render(msgs, gen=False):
            return tok.apply_chat_template(
                msgs, tools=TOOLS, add_generation_prompt=gen, tokenize=False
            )

        def after(longer: str, shorter: str) -> str:
            assert longer.startswith(shorter), "the template is not prefix-stable"
            return longer[len(shorter) :]

        head = render(sys)
        self.system = self.enc(head)
        self.call: dict[str, list[int]] = {}
        for name in (EVENTS_TOOL, LOOK_TOOL):
            self.call[name] = self.enc(after(render([*sys, call(name)]), head))
        with_call = render([*sys, call(LOOK_TOOL)])
        probe = [*sys, call(LOOK_TOOL), {"role": "tool", "content": self.MARK}]
        block = after(render(probe), with_call)
        self.before, self.after = block.split(self.MARK)
        self.header = self.enc(after(render(probe, gen=True), render(probe)))
        self.sor = tok.convert_tokens_to_ids("<|start_of_role|>")
        assert self.header[0] == self.sor, "the generation prompt opens a role"

    def enc(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def result(self, content: str) -> list[int]:
        """A tool result's turn around its JSON text (tokenized fresh)."""
        return self.enc(self.before + content + self.after)

    def suffix(self, lk: dict, adapter: str | None = None) -> list[int]:
        """A decision's suffix: the look's call, its result, the generation
        prompt (``adapter``: with its control token)."""
        return [*self.call[LOOK_TOOL], *self.result(tool_text(lk)), *self.gen(adapter)]

    def gen(self, adapter: str | None) -> list[int]:
        """The generation prompt; with ``adapter``, its control token in place
        of the ``<|start_of_role|>`` that opens it, as a composed checkpoint's
        template puts an aLoRA's (the switch gives it that token's embedding)."""
        if adapter is None:
            return self.header
        ctl = self.tok.convert_tokens_to_ids(f"<|{adapter}|>")
        assert ctl != self.tok.unk_token_id and ctl is not None, adapter
        return [ctl, *self.header[1:]]


class GameChat:
    """One match's conversation (:data:`FORMAT`), for writing and serving.

    Feed :meth:`observe` every tic, dead or alive, with the tracker's events of
    that tic; :meth:`prompt_messages` is the conversation and a decision's
    suffix (the writer's chat); with ``pieces``, :meth:`prompt_ids` is the same
    as token ids (serving), the conversation's ids kept as it grows.
    ``players``: the match's bots (:func:`players_of`), the first push's."""

    def __init__(self, players: list[str], pieces: Pieces | None = None):
        self.messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPTS[FORMAT]}
        ]
        self.pending: list[dict] = [{"type": "start", "players": list(players)}]
        self.pieces = pieces
        self.ids: list[int] = list(pieces.system) if pieces else []
        self.pushes: list[int] = []  # the tick of each push

    def observe(self, fired: list[dict], tick: int) -> bool:
        """This tic's events; on a whole second with any pending, they are
        pushed: the call and its result join the conversation. Returns
        whether it pushed."""
        self.pending += key_events(fired)
        if not self.pending or tick % TIC_HZ:
            return False
        self._push(tool_text({"time": clock(tick), "events": self.pending}), tick)
        self.pending = []
        return True

    def _push(self, content: str, tick: int) -> None:
        self.messages += [call(EVENTS_TOOL), {"role": "tool", "content": content}]
        if self.pieces is not None:
            self.ids += self.pieces.call[EVENTS_TOOL] + self.pieces.result(content)
        self.pushes.append(tick)

    def prompt_messages(self, lk: dict) -> list[dict]:
        """The conversation, then a decision's suffix: the look and its result."""
        return [*self.messages, call(LOOK_TOOL), result(lk)]

    def prompt_ids(self, lk: dict, adapter: str | None = None) -> list[int]:
        """:meth:`prompt_messages` as the chat template renders it with the
        generation prompt, as token ids; ``adapter``: its control token there
        (None: the base model's, as PEFT trains on it)."""
        return [*self.ids, *self.pieces.suffix(lk, adapter)]

    # A recorded prompt read back (a record's messages, through the compiled
    # view: a call's content may come back None for ""; the JSON as written).
    def follow(self, messages: list[dict]) -> dict:
        """A recorded prompt of this match, no earlier than any followed
        before: the pushes it holds beyond this conversation join it. Returns
        its look."""
        system, *conv, look_call, look_result = messages
        assert system["content"] == self.messages[0]["content"], "another system turn"
        assert look_call["tool_calls"][0]["function"]["name"] == LOOK_TOOL
        assert len(conv) >= len(self.messages) - 1, "an earlier prompt"
        new = conv[len(self.messages) - 1 :]
        for c, r in zip(new[::2], new[1::2]):
            assert c["tool_calls"][0]["function"]["name"] == EVENTS_TOOL, c
            mins, secs = json.loads(r["content"])["time"].split(":")
            self._push(r["content"], TIC_HZ * (60 * int(mins) + int(secs)))
        self.pending = []
        return json.loads(look_result["content"])

    @classmethod
    def from_messages(
        cls, messages: list[dict], pieces: Pieces | None = None
    ) -> tuple[GameChat, dict]:
        """A recorded prompt read back: its conversation, and its look."""
        chat = cls([], pieces)
        return chat, chat.follow(messages)


class Match:
    """One match as the decisions meet it: the game, in the data's settings,
    the tracker of its events, and its conversation, which hears every tic's
    events. ``obs``: the observation now; :meth:`messages`: the conversation
    with this tic's look; :meth:`step`: one tic. Use as a context manager (the
    game closes)."""

    def __init__(
        self, seed: int, seconds: float = 600.0, bots: str = "cast", pieces=None
    ):
        self.env = DoomEnv(
            seed=seed,
            resolution=RESOLUTION,
            hud=True,
            timeout_tics=int(seconds * TIC_HZ),
            bots=bots,
        )
        self.obs = self.env.reset(seed=seed)
        self.tracker = Tracker(match_s=seconds)
        fired = self.tracker.update(self.obs)
        self.chat = GameChat(players_of(self.tracker.facts()), pieces)
        self.chat.observe(fired, self.obs.tick)

    def __enter__(self) -> Match:
        return self

    def __exit__(self, *exc) -> None:
        self.env.close()

    @property
    def over(self) -> bool:
        return self.obs.done

    def step(self, action: str = "wait", slot: str | None = None) -> None:
        """One tic: ``action``, and ``slot`` (a slot digit) to switch to, if any."""
        self.obs = self.env.step(action, weapon=None if slot is None else int(slot))
        if not self.obs.done:
            self.chat.observe(self.tracker.update(self.obs), self.obs.tick)

    def messages(self) -> list[dict]:
        return self.chat.prompt_messages(look(self.obs))


def placement(tok) -> str:
    """How a composed checkpoint's template places the move adapter's token:
    ``alora`` (the generation prompt's role marker, which :class:`Pieces`
    assumes) or something else, which serving refuses."""
    ctl = f"<|{MOVE}|>"
    if ctl not in tok.get_vocab():
        return "base"
    text = tok.apply_chat_template(
        [{"role": "user", "content": "x"}],
        add_generation_prompt=True,
        tokenize=False,
        adapter_name=MOVE,
    )
    if text.startswith(ctl):
        return "lora"
    return "alora" if text.endswith(f"{ctl}assistant<|end_of_role|>") else "other"


# ── Written rows ───────────────────────────────────────────────────────────────
def iter_rows(paths, function: str | None = None):
    """Rows from ``.jsonl`` or ``.jsonl.gz`` files (``decide.py``'s), in
    order, one at a time: a row holds its whole prompt, so a shard's rows do
    not fit in memory at once. ``function``: only that compilable function's
    (``Player.move``), the others never parsed. The training side reads them
    so, without Mellea."""
    want = None if function is None else f'"function": "{function}"'
    for p in paths:
        p = Path(p)
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8") as f:
            for x in f:
                if x.strip() and (want is None or want in x[:200]):
                    yield json.loads(x)


def read_rows(paths, function: str | None = None) -> list[dict]:
    return list(iter_rows(paths, function))


# ── Check ──────────────────────────────────────────────────────────────────────
def check(seconds: float = 60.0, tokenizer: str = "", model: str = "") -> None:
    """The look against the actor, and the conversation against the template,
    on ``seconds`` of the scripted player (fighter) in the data's settings:

    * the label table is a function of the label (asserted at import), and
      covers every item;
    * on every live tic, the look, through JSON and back, gives exactly the
      actor's input (:func:`vector_of_look` == ``rl_teacher.obs_vector``);
    * on sampled tics, the chat template on the conversation with a decision's
      suffix starts with its rendering of the conversation (the cached prefix
      is stable), holds one call per push and one look, and its ids are
      :meth:`GameChat.prompt_ids`; with ``model`` (a composed checkpoint), with
      each adapter's control token too (``adapter_name``)."""
    from expert import Expert
    from rl_teacher import obs_vector
    from transformers import AutoTokenizer

    assert {lab for lab, _ in ITEMS.values()} <= set(LABEL_KIND_SLOT)
    names = [(tokenizer or "ibm-granite/granite-4.1-3b", None)]
    if model:
        names.append((model, (MOVE, WEAPON)))
    toks = [(AutoTokenizer.from_pretrained(n), ads) for n, ads in names]
    for tok, ads in toks:
        if ads:
            assert placement(tok) == "alora", f"{model}: {placement(tok)} placement"
    pieces = [Pieces(tok) for tok, _ in toks]
    env = DoomEnv(
        seed=3,
        resolution="640X480",
        hud=True,
        timeout_tics=int(seconds * TIC_HZ),
        bots="cast",
    )
    ex, tr = Expert(), Tracker(match_s=seconds)
    obs = env.reset(seed=3)
    fired = tr.update(obs)
    chats = [GameChat(players_of(tr.facts()), p) for p in pieces]
    live = checked = 0
    look_tokens: list[int] = []
    while not obs.done:
        for c in chats:
            c.observe(fired, obs.tick)
        if not obs.dead:
            lk = look(obs)
            back = json.loads(tool_text(lk))
            assert back == lk
            want = obs_vector(obs, STYLE)
            got = vector_of_look(back)
            assert np.array_equal(got, want), (
                obs.tick,
                lk,
                np.flatnonzero(got != want),
            )
            live += 1
            pushed = bool(chats[0].pushes) and chats[0].pushes[-1] == obs.tick
            if obs.tick % 70 == 0 or pushed:
                for (tok, ads), c in zip(toks, chats):
                    _check_prompt(tok, c, lk, ads)
                checked += 1
            look_tokens.append(len(pieces[0].result(tool_text(lk))))
        a = ex.act(obs, STYLE)
        w = ex.weapon(obs, STYLE) if obs.tick % PLAN_EVERY_TICS == 0 else None
        obs = env.step(a, weapon=w)
        fired = [] if obs.done else tr.update(obs)
    env.close()
    c = chats[0]
    print(
        f"OK: the look gives the actor's input exactly on {live} live tics; "
        f"{checked} prompts match the chat template"
        + (f" (base and {model}'s adapters)" if model else " (base)")
        + f"; {len(c.pushes)} pushes in {seconds:.0f} s, the conversation "
        f"{len(c.ids)} tokens, a look's turn {min(look_tokens)}-{max(look_tokens)} "
        f"tokens (mean {sum(look_tokens) / len(look_tokens):.0f})"
    )


def _check_prompt(tok, chat: GameChat, lk: dict, adapters) -> None:
    msgs = chat.prompt_messages(lk)
    conv = tok.apply_chat_template(chat.messages, tools=TOOLS, tokenize=False)
    full = tok.apply_chat_template(
        msgs, tools=TOOLS, add_generation_prompt=True, tokenize=False
    )
    assert full.startswith(conv), "the conversation's rendering is not a prefix"
    for name, n in ((EVENTS_TOOL, len(chat.pushes)), (LOOK_TOOL, 1)):
        tag = f'<tool_call>\n{{"name": "{name}", "arguments": {{}}}}\n</tool_call>'
        assert full.count(tag) == n, (name, full.count(tag), n)
    assert full.count("<tool_response>") == len(chat.pushes) + 1
    ids = tok(full, add_special_tokens=False).input_ids
    assert chat.prompt_ids(lk) == ids, "prompt ids differ from the template's"
    for a in adapters or ():
        text = tok.apply_chat_template(
            msgs,
            tools=TOOLS,
            add_generation_prompt=True,
            tokenize=False,
            adapter_name=a,
        )
        want = tok(text, add_special_tokens=False).input_ids
        assert chat.prompt_ids(lk, a) == want, f"{a}: prompt ids differ"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ck = sub.add_parser("check", help="The look against the actor, the prompt ids")
    ck.add_argument("--seconds", type=float, default=60.0)
    ck.add_argument("--tokenizer", default="ibm-granite/granite-4.1-3b")
    ck.add_argument("--model", default="", help="A composed checkpoint, too")
    args = ap.parse_args()
    check(args.seconds, args.tokenizer, args.model)


if __name__ == "__main__":
    main()
