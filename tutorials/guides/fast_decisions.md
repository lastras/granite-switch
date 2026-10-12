# Fast Decisions in the Chat Template

This tutorial shows how to make a Granite model take very fast decisions and still
do language work. The model is one Granite Switch checkpoint. Its decision adapters
choose an action in a Doom deathmatch on every tic, 35 times a second, one output
token each. They read the match the way a chat model reads anything: as a
conversation in Granite's own chat template, with the game reaching it through tool
calls. Because the decisions live in an ordinary conversation, the same conversation
can later carry what a person says and what the model says back. That is where the
next steps go.

This is step 1. The model learns the per-tic decision from a teacher, an RL policy
already trained on the game. Nothing in it is about language yet, but the prompt is:
every decision reads the whole conversation.

The code is in [`tutorials/scripts/doom/`](../scripts/doom/):

| file | what it holds |
|---|---|
| [`game_chat.py`](../scripts/doom/game_chat.py) | the conversation, its events, the look, and its prompt ids (writing, training and serving share it) |
| [`decide.py`](../scripts/doom/decide.py) | the tutorial's piece: the teacher at a compilable boundary (Mellea fork), its labels, the driver |
| [`decide_tools.py`](../scripts/doom/decide_tools.py) | around it: what the records hold, their parity with serving and training, the memoryless floor, the self-check |
| [`train_alora.py`](../scripts/doom/train_alora.py) | the training (`--adapter move` / `weapon --records`) |
| [`decide_model.py`](../scripts/doom/decide_model.py) | compose, verify against PEFT, latency, lockstep play |
| [`engine.py`](../scripts/doom/engine.py) | real-time play (`--decider chat`) |

The commands, in order, are the recipe's step 1
([DOOM_RECIPE.md](../../docs/DOOM_RECIPE.md#tutorial-step-1-fast-decisions-from-the-chat-template)).

## 1. The game as a conversation

One match is one conversation. It grows for the whole match and is never windowed.
The harness writes every tool call; the model writes only its decisions:

```
system     the system prompt (game_chat.SYSTEM_PROMPTS["s1"]) and, from the
           template, the tools block: get_game_events, look
assistant  <tool_call>{"name": "get_game_events", "arguments": {}}</tool_call>
tool       {"time": "0:00", "events": [{"type": "start", "players": [...]}]}
  ... one call and its result each time something happens ...
-- a decision's suffix, never kept --
assistant  <tool_call>{"name": "look", "arguments": {}}</tool_call>
tool       {the look}
<|move|>assistant<|end_of_role|>      -> one action token
```

The weapon adapter asks the same prompt with `<|weapon|>` in place of `<|move|>`
and answers with one slot digit.

The conversation holds only the match's key events (`game_chat.KEY_KINDS`): deaths
and their killers, frags, lead changes, close calls, streaks. They are in the same
JSON the narrator's game state uses (`talk.event_json`) and are pushed at most once
a second. What the look already says stays out: pickups, health, ammo. This is the
end of a real prompt, as the template renders it:

```
<|start_of_role|>assistant<|end_of_role|><tool_call>
{"name": "get_game_events", "arguments": {}}
</tool_call><|end_of_text|>
<|start_of_role|>user<|end_of_role|>
<tool_response>
{"time": "0:00", "events": [{"type": "start", "players": ["Gwen", "Hank", "Lou", "Milo", "Nora", "Ted", "Vick"]}]}
</tool_response><|end_of_text|>
...
<|start_of_role|>assistant<|end_of_role|><tool_call>
{"name": "look", "arguments": {}}
</tool_call><|end_of_text|>
<|start_of_role|>user<|end_of_role|>
<tool_response>
{"hp": 100, "armor": 0, "weapon": "chaingun", "ammo": 139, "arms": {"1": 0, "2": 139, "4": 139}, "at": [82, 12], "face": 133, "see": [["bullets", -4, 17]], "walls": {"l": 9, "f": 9, "r": 9, "b": 0}, "hit": 0, "last": ["fl", "forward"], "weapon_tic": true}
</tool_response><|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
```

## 2. The look: what the teacher reads, and nothing less

The look is the present: health, armor, the weapon and its ammo, the guns owned,
the position and heading, what is on screen (bots, an incoming missile, items),
the room to each side, the damage just taken, where the last bot was seen, and
the last two actions. It holds every field the RL teacher's actor reads
(`doom_env.features`), at the same precision.

That is checked, not assumed. `game_chat.vector_of_look` rebuilds the actor's
input vector from the JSON alone. `python game_chat.py check` plays a minute and
asserts, on every live tic, that the rebuilt vector equals the teacher's own. The
student therefore sees everything the teacher's actor sees.

What it cannot see is the teacher's recurrent memory, a GRU state that carries
the match so far. `decide_tools.py floor` measures what that costs. It trains a small
MLP from the look alone to the teacher's distribution and reports its held-out
KL. No student that reads only the present can do better than that MLP, apart
from what the conversation's events tell it.

## 3. The teacher at a compilable boundary

A decision is one call of a *compilable function* (`mellea.stdlib.compile`, on the
Mellea fork). This is all of it ([`decide.py`](../scripts/doom/decide.py)):

```python
class Player:
    """The RL teacher in one match. ``self`` is its memory: the student reads
    the conversation instead."""

    def __init__(self, teacher: str, seed: int):
        self.rl, self.rng = RLPolicy(teacher), random.Random(seed)

    def see(self, obs) -> None:
        """Every live tic, as the teacher was trained: its distributions now."""
        self.tick, self.dist = obs.tick, self.rl.labels(obs, STYLE)

    def draw(self, kind: str) -> str:
        d = self.dist[kind]
        return self.rng.choices(list(d), weights=list(d.values()))[0]

    @compilable
    def move(self, chat: ChatContext) -> Literal[ACTIONS]:
        """Your move this tic."""
        annotate(t=self.tick, dist=self.dist[MOVE])
        return self.draw(MOVE)
```

`weapon` is the same, over the slot digits.

- **The signature is the student's interface.** The student reads `chat`: the
  conversation and the tic's look, rendered by the chat template. That rendering,
  the *compiled view*, is exactly what it is trained and served on.
- **The body is the teacher.** It can be anything; here, a GRU policy. It notes
  its whole distribution and plays a draw from it, as it was evaluated.
- **`self` is the teacher's hidden state:** its memory, stepped by `see` on every
  live tic. The student never sees it.

**The labels** are two short functions that `emulate` applies after each match.
Every other answer becomes another completion, and each completion's score is
the teacher's probability of it:

```python
def alternatives(record, episode) -> list[Completion]:
    """Every other answer: with the one returned, the teacher's whole distribution."""
    said = record.completions[0].value
    return [Completion(w) for w in record.meta["dist"] if w != said]


def score(record, completion, episode) -> Score:
    """The teacher's probability of an answer: the student's soft label."""
    return Score(value=record.meta["dist"][completion.value])
```

**The driver** is plain code. The teacher plays a ten-minute match against seven
bots (`game_chat.Match`: the game, its events, its conversation). `move` is
called every 7 tics and `weapon` on every planner tic:

```python
def episode(seed: int, teacher: str, seconds: float = 600.0, run: int = 0) -> None:
    """One match against the cast, the teacher playing it."""
    with Match(seed, seconds) as match:
        player = Player(teacher, seed_of(seed, run))
        while not match.over:
            if match.obs.dead:  # no decision: the respawn passes
                match.step()
                continue
            player.see(match.obs)
            tick, slot = match.obs.tick, None
            if tick % EVERY == 0:
                move = player.move(chat_of(match))
            else:
                move = player.draw(MOVE)  # the teacher's own, not recorded
            if tick % PLAN_EVERY_TICS == 0:
                slot = player.weapon(chat_of(match))
            match.step(move, slot)
```

Mellea's `emulate` runs the matches, records every call, adds the completions,
scores them, and writes each match whole to a gzipped shard:

```python
matches = [(json.dumps(["cast", s]), partial(episode, s, teacher)) for s in seeds]
emulate(matches, out, score=score, alternatives=alternatives)
```

Because the driver cannot tell who answers, a student can take the teacher's
place later with `Player.move.implemented_by(student)` (DAgger).
`decide_tools.py selfcheck` drives it with a scripted one.

## 4. Recording a distribution, and calibration

A record keeps more than the action played. After a match, every other action is
added to each call as another completion (`decide.alternatives`), and each
completion is scored by the teacher's probability (`decide.score`). A move record
therefore holds 20 completions whose values sum to 1, and a weapon record holds 7.

The student is trained to match that distribution with a soft cross-entropy. That
is the KL to the teacher plus a constant, the teacher's entropy. So its
probabilities mean what the teacher's do. The teacher is stochastic (about 2
nats of entropy over the 20 moves), so the student is served sampling at
temperature 1.0, as the teacher plays.

Training reports three numbers on held-out matches:
- the KL to the teacher;
- top-1 agreement;
- the calibration error (ECE): the student's top-1 probability, in bins,
  against the teacher's probability of the same action.

## 5. Writing, training, serving

**One builder for the prompt.** `game_chat.GameChat` builds the prompt for the
data writer and for live play alike, so the two cannot drift.
- Its prompt ids come from pieces cut from the chat template's own rendering
  (`game_chat.Pieces`): the system turn, each call, a tool result around its
  JSON, the generation prompt.
- Only the look's turn is tokenized fresh each tic.
- `decide_tools.py parity` checks that, on 500 records, the template's ids, the
  served ids and the training ids are the same.

**Why aLoRA.** An aLoRA adapter activates only at its control token, at the
very end of the prompt. Everything before it runs on base weights: the whole
conversation, its tool calls and the look's call.
- Its KV is the same for every adapter and every tic, so vLLM's prefix cache
  keeps it.
- A decision computes only the look's turn and the generation prompt, about
  180 tokens, against a cached prefix that reaches 10k tokens by the end of a
  match.
- Both adapters share that prefix.
- A plain LoRA would recompute the whole conversation for each adapter on
  every tic.

**Serving.** `engine.py run --decider chat` keeps one `GameChat` per game, fed by
the game process's events and looks. Each tic it sends one request per adapter:
one output token, the allowed outputs only, the distribution returned. The game
runs on the wall clock and plays the freshest decision it has, so latency
shows up as lag in tics.

**What the data looks like** (100 of the teacher's ten-minute matches against
the cast):

| | |
|---|---|
| records | 293k `move`, 121k `weapon` |
| prompt tokens | p50 5.6k, p99 10.3k, at most 11.0k; a look's turn p50 167 |
| pushes | 11.6 a minute (the teacher frags 128 times a match) |
| teacher's entropy | `move` 2.04 nats, `weapon` 1.55 nats |
| a row on disk | 1.1 KB gzipped (25 KB as JSON) |
| the floor (MLP on the look, KL on 20 held-out matches) | `move` 0.18 nats (the teacher's mean distribution: 0.90); `weapon` 0.033 (0.20) |

## 6. Results

The full run trains `move` and `weapon` on 100 of the teacher's matches. It
composes them into one checkpoint and measures:

| measure | command | result |
|---|---|---|
| held-out KL to the teacher, top-1 agreement, ECE | `train_alora.py` (`metrics.json`) | not yet measured |
| composed checkpoint against PEFT (gate: 99% argmax agreement) | `decide_model.py verify` | not yet measured |
| decision latency, p50 and p99 (gate: one tic, 28.6 ms), push tics apart | `decide_model.py bench` | not yet measured |
| frags, deaths, margin and rank over 12 matches, against the teacher's on the same seeds | `decide_model.py play` | not yet measured |
| lag in tics, real time | `engine.py run --decider chat` | not yet measured |

## 7. Next steps

- **Step 2: words that steer the decision.** The partner's words join the same
  conversation as user turns. The teacher gets the style or the order as a
  privileged fact; the student has to read it from the words.
- **Step 3: the narrator in the same conversation.** His lines become assistant
  turns, and his `get_game_state` call becomes a never-kept suffix of his own.
  One KV prefix then serves the move, the weapon and the narrator adapters.
