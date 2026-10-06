# Writing IBM with frags: a set piece for the Doom demo

**Status:** designed and prototyped with a scripted player (October 2026). Set aside, to be
built toward the end of the project. Nothing here is in the demo yet.

## The idea

In 1989 Don Eigler and Erhard Schweizer, at IBM Almaden, wrote "IBM" with 35 xenon atoms on a
nickel surface, moving them one at a time with the tip of a scanning tunnelling microscope.

In the demo's version:
- **The stencil:** "IBM" as a set of spots ("atoms") laid on the map.
- **The set:** the harness offers the player the open atoms.
- **Writing an atom:** a frag he makes while standing on an open atom writes it.
- **Narrowing:** the set shrinks as he writes, and he chooses, from what he sees, which open
  atom to take next.
- **What the audience sees:** IBM appearing on a map of the match, drawn like the
  microscope image.

**Why it serves the demo** (CLAUDE.md, "The Doom Demo"):
- It is a goal over minutes, pursued while he still decides every tic and fights 7 bots.
- He has to choose a sequence. Where to go next depends on the bots, his health and the open
  atoms.
- It gives the narrator something to report: progress, the letter he is on, which bot
  fell where.

## What we measured

All on cig.wad MAP02, in the demo's own environment (`doom_env.py`).

- **Room:**
  - The map is 83 × 71 m, with 2,922 m² of reachable floor.
  - The biggest open rectangle is 32 × 16 m.
  - The biggest band at least 3 times wider than tall is 36 × 12 m, in the north-west room.
- **Motion (no bots):**
  - He runs at 16 units a tic (about 17 m/s) after about a second.
  - Letting go at 10.7 units a tic, he coasted 3.2 m. From full speed Doom's friction makes
    that about 5.4 m (computed, not measured).
  - Braking with `back` from 10.7 units a tic: 0.6 m, in 6 tics.
  - He turns 10° a tic in place. Turning while running (`fl`/`fr`) traces a circle of about
    6 m radius.
  - So he can stop on a point to within half a metre, by slowing in. Running through it
    doesn't work.
- **Frags:**
  - ViZDoom drops a dead player from the object list on the tic he dies. The victim of a
    frag is therefore the bot that vanished that tic (his target, if it is among them), and
    its last listed position is where it fell. The bodies stay visible in the game.
  - The RL teacher frags about 9 times a minute (the reaction sweep in
    [the guide](../tutorials/guides/doom_demo.md)).
  - The scripted fighter frags 2.6–4.4 times a minute in these matches.
- **Markers in the game world:**
  - With `+sv_cheats 1` in the game's arguments, `summon Candlestick` puts a candle 1.75 m in
    front of him. Without cheats `summon` does nothing.
  - A candle is the right marker: he can walk through it and can't pick it up. A health bonus
    he would pick up, and a torch would block him.
- **What the game adapters see:** his position in whole metres, and items' distances in whole
  metres. Too coarse for atoms 2.5–3 m apart: a model version needs the next atom's bearing
  and distance to 0.1 m.

## The stencil

Five rows, 31 atoms (the original had 35):

```
I      B         M
X      X X X .   X . . . X
X      X . . X   X X . X X
X      X X X .   X . X . X
X      X . . X   X . . . X
X      X X X .   X . . . X
```

- **Size:** the letters fill the 36 × 12 m band with a 1 m margin, so the atoms are 3.0 m
  apart across and 2.5 m apart down.
- **Writing radius:** a frag within 1.5 m of an open atom writes it.
- **Where an atom is drawn:** on its stencil site, the way each xenon atom sat on a site of
  the nickel lattice. His actual position at the frag (within 1.5 m of the site) is logged,
  and his track stays on the map. Drawn at his true positions, the letters blurred.

## What "the place" is: two readings, tested

Each row is a 10-minute match with 7 bots (seeds as noted), played by a scripted player built
on `expert.py`'s fighter.

| Reading | Rule | Atoms of 31 | Notes |
|---|---|---|---|
| Where the **bot** falls | *strict*: fire only while the bot is on an open atom, or to defend himself | 2–4 (3 seeds) | Holding fire gets him killed |
| | *free*: the plain fighter, waiting among the letters when no bot is in sight | 3–7 (6 seeds) | About 1 frag in 10 lands on the letters |
| | *finish*: wear the bot down anywhere, then save the last shots (up to 3 s) for an open atom | 1–6 (3 seeds) | |
| Where **he** stands | *stand*: walk to the nearest open atom, hold it, frag from there | mean 27.0 (12 seeds); complete in 5, at 8:12–9:54 | Without pick-ups, he returns from each respawn with the pistol |
| | *stand* + pick-ups on the way | **mean 29.8 (12 seeds); complete in 8, at 5:48–9:54** | The chosen version |

The bot-falls reading can't finish the word, whatever the placement. As the set narrows, a
body landing on one of the few remaining atoms by chance gets rare: by a coupon-collector
count, all 31 would take hundreds of frags. The remaining atoms have to be gone after
deliberately, and the player controls only his own position.

**The chosen version, in numbers:**
- **Atoms:** about 79% of his frags wrote an atom, counting frags after IBM was complete as
  not writing.
- **The other four matches** reached 22, 28, 30 and 30 atoms.
- **Deaths:** 16–44, against 21–26 for the fighter in *free* mode, because he stands still
  to shoot.

**Tried and dropped:**
- **A 0.75 m writing radius:** 10–24 atoms and up to 47 deaths. He held fire whenever he was
  slightly off the spot.
- **Dodging on the spot** (strafe and fire): a mean of 21 atoms, and no fewer deaths.
  Strafing spoils his aim.

## The scripted writer (the teacher-to-be)

Each tic:
- **Target:** the fighter's (`Expert._target`).
- **His spot:** the nearest open atom, kept while it stays open. When every atom is written
  he goes back to plain fighting.
- **Firing:** he fires (`Expert._fight`'s aiming, then `fire`) when he is within 1.5 m of his
  spot, or to defend himself (below 50 hp, or a bot within 3 m). More than 0.6 m off the spot,
  he fires while stepping back toward it (`charge`, `bf`, `cl` or `cr`, whichever moves toward
  it).
- **When not firing:**
  - Below 35 hp with no bot in sight, he fetches health.
  - Otherwise he picks up what the fighter would (weapons, ammo, armor, health) within 10 m,
    or walks to the spot.
  - Holding the spot with no bot in sight, he turns to look for one.
- **The walk:** a shortest path over a grid of 0.5 m cells.
  - A cell is free if it is at least his radius plus 4 units from every blocking line (the
    blocking rule is `doom_env._refresh_segments`), and reachable from the spawn.
  - The path is a BFS distance field from the spot. He steers (`expert.steer`) toward the
    cell 8 steps along it.
- **Ground truth he uses:** his exact position and the floor grid (`Observation.priv`).
  The model has neither.

## The map

- **The background:** the map's blocking walls and the reachable floor.
- **Atoms:** the stencil as rings. Written atoms glow on their sites.
- **Him:** a gold arrowhead pointing where he faces. His track covers the last 10 s and
  starts again at each respawn.
- **The bots:** red dots, moving every frame.
- **Frags:** a ring opening out at each frag, and a grey × for one that wrote nothing.
- **His deaths:** a skull where he fell, held for 1.5 s and then fading out over 1.5 s. His
  arrow is hidden while he is dead.
- **His shots:** one small bullet per shot, whatever the weapon.
  - A shot is a tic where his ammo drops without a weapon switch.
  - A bullet flies along his facing at 45 units a tic (about 45 m/s), slow enough to see.
    Doom's pistol, shotgun and chaingun hits are instant.
  - It ends at the first wall (`doom_env._raycast`), or earlier at a live bot within 24 units
    of the line. It bursts red on a bot, grey on a wall.
  - This is our geometry, not the game's damage record: no spread, and no projectile speeds.
    The fist draws nothing.
  - Seed 4: 800 shots in 5:48, 253 of them ending on a bot.
- **The still:** what he wrote, drawn like the microscope image. Gaussian bumps on the
  written sites, lit from the upper left, on a navy-to-white colour map with a little scan
  noise.
- **Video speed:** 2× shows the bullets well. At 4× they blur.

## On the screen

The mockup is in the demo's wide layout (1920 × 1080, `overlay.py`'s `wide` view, as the live
page lays out):

```
+--------------------------------------+------------------------------------+
|                                      | [fighter] [ORDER: write IBM doing] |
|            the game                  |  12.4 ms / decision                |
|           (1037 x 778)               +------------------------------------+
|                                      |                                    |
|                                      |        the IBM map (~776 x 650)    |
|                                      |                                    |
+--------------------------------------+                                    |
| which model runs                     +------------------------------------+
| action probabilities                 |  conversation                      |
+--------------------------------------+------------------------------------+
```

- **Left:** unchanged.
- **Right:** the IBM map takes the panel's place. Next to the panel it would be too small to
  read IBM at a glance. The panel's key facts move into a status strip above it: the behavior,
  the order and the latency. The conversation stays below.
- **The proposal:** this layout only while the "write IBM" order runs; the normal panel
  otherwise.
- **For the live page:**
  - Sent every tic: his position and heading, the bots' positions, his shots.
  - Sent once in the hello: the map's walls and the stencil.
  - The GPU side already has all of it (`Observation.priv`, ground truth). The display shows
    the audience the truth; the model still sees only its prompt.

## Toward the model version: what to decide

- **Who steers.** The scripted writer is the teacher. The model version needs:
  - **The atoms in his state:** the harness gives the open atoms, e.g. bearing and distance to
    his spot and the nearest few, to 0.1 m close up, plus how many are left.
  - **Training data:** matches with the order, on stencils and placements other than IBM, so
    he learns to choose and reach atoms rather than memorize one path.
  - **The recipe in the repo** (CLAUDE.md's restriction).
- **Navigation.** The script walks a privileged path. If the harness hands the model the
  path's next waypoint, the model is following directions. If it gets only the spot's bearing
  and distance, the walking is more its own work, but harder: the game adapters steer around
  walls today only through the wall clearances. Decide, and say which in the demo.
- **Which adapter.** Either a new behavior adapter ("writer") or the fighter with the spot in
  its prompt.
- **The sequence.** The script takes the nearest open atom. Choosing better, from where the
  bots are, his health and cover, is the model's opportunity.
- **The order and the narrator:**
  - "write IBM" becomes a new kind of order for the orders adapter.
  - The game state gives the narrator the progress (written and left, the letter, the last
    victim) so his lines can tell it.
- **Time.** The scripted writer completes IBM within a 10-minute match in 8 of 12 matches.
  The model must at least match that. Otherwise: fewer atoms, a larger radius, or a longer
  segment.
- **The letters in the game world.** Optional: a candle summoned on each atom as he writes
  it. That needs `sv_cheats` for that match.

## Open questions

- **Placement:** the north-west band (the biggest open floor), or nearer the central hall,
  where the fights happen.
- **31 atoms or the original 35.**
- **The live page or the overlay's wide view first.**

## The prototype

The prototype is `scratch/doom/ibm/write_ibm.py`. `scratch/` is gitignored, so it isn't in the
repo; when this is built, the code belongs in `tutorials/scripts/doom/`.

**Its options:**
- `--mode strict|free|finish|stand`: the firing rules in the table above.
- `--video --speed 2 --stop-when-done`: the game beside the map.
- `--screen TICK`: the full-screen mockup.

**Its outputs:** the map, the still and a JSON log (atoms, frags, deaths) for every match.

**Speed:** a 10-minute match simulates in about 18 s on a laptop, about 1 minute with video.

**The probes behind "What we measured"**, also in `scratch/doom/`:
- `ibm_map.py`: the room;
- `ibm_motion.py`: the motion;
- `ibm_summon.py`: the markers;
- `ibm_corpse_probe.py`: what happens to a bot at its death.
