# SPDX-License-Identifier: Apache-2.0
"""The frame the videos and the live demo show: the game, a telemetry panel,
which model runs on each tic, the action distribution as a heatmap, and the
spoken lines as captions over the game.

:mod:`record_video` draws one per tic into an MP4; :mod:`doom_live` draws them
in a renderer process and streams them as JPEG.
"""

from __future__ import annotations

import os
import re

import numpy as np
from doom_env import (
    ACTION_LABELS,
    DISPLAY_ORDER,
    SHORT_LABELS,
    TIC_HZ,
    TIC_MS,
    WEAPON_NAMES,
)
from PIL import Image, ImageDraw, ImageFont
from policy import ARMS, CRITIC, DANGER_LEVELS, NARRATOR, ORDERS

W_GAME, H = 640, 480
W_PANEL = 360
BG = (22, 22, 22)
LAYER = (38, 38, 38)
LAYER2 = (57, 57, 57)
TEXT = (244, 244, 244)
TEXT2 = (198, 198, 198)
HELP = (168, 168, 168)
BLUE = (15, 98, 254)
TEAL = (8, 189, 186)
AMBER = (210, 161, 6)
RED = (250, 77, 86)
GREEN = (66, 190, 101)
COLORS = {"fighter": RED, "cautious": TEAL, "collector": AMBER}
ORDER_COLORS = {"doing": GREEN, "refused": RED, "cant": AMBER, "cancelled": HELP}

HEAT_ROW = 10  # pixels per action row
H_HEAT = 82 + HEAT_ROW * len(DISPLAY_ORDER)  # heatmap strip under the game and panel
HEAT_PX = 3  # pixels per tic
HEAT_GUTTER = 100  # row labels
# Colour position is sqrt(p), so runner-up actions at a few percent stay visible.
STOPS = [(0.0, BG), (0.35, (0, 45, 156)), (0.7, BLUE), (1.0, (200, 228, 255))]


def font(size: int, mono: bool = False):
    for path in (
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf"
        if mono
        else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
        if mono
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


class Panel:
    def __init__(self) -> None:
        self.f_title = font(15)
        self.f_big = font(34)
        self.f = font(15)
        self.f_small = font(12)
        self.f_mono = font(12, mono=True)

    def draw(self, img: Image.Image, t: dict) -> None:
        d = ImageDraw.Draw(img)
        x0 = W_GAME
        d.rectangle([x0, 0, x0 + W_PANEL, H], fill=BG)
        x = x0 + 18
        y = 14
        d.text((x, y), "GRANITE SWITCH  ·  DOOM", font=self.f_title, fill=TEXT2)
        y += 26
        d.text((x, y), "one output token per tic", font=self.f_small, fill=HELP)
        y += 26

        # Behavior chip
        c = COLORS.get(t["adapter"], TEXT)
        d.rounded_rectangle([x, y, x + 150, y + 28], radius=14, fill=LAYER2)
        d.ellipse([x + 12, y + 10, x + 20, y + 18], fill=c)
        d.text((x + 28, y + 5), t["adapter"], font=self.f, fill=TEXT)
        d.text(
            (x + 162, y + 6),
            f"{t.get('kind', 'aLoRA')} adapter",
            font=self.f_small,
            fill=HELP,
        )
        y += 38
        if t.get("order"):  # the partner's latest order (the live demo)
            o = t["order"]
            col = ORDER_COLORS.get(o["status"], TEXT2)
            d.rounded_rectangle([x, y, x + W_PANEL - 36, y + 18], radius=9, fill=LAYER2)
            d.text((x + 10, y + 2), f"ORDER: {o['told']}", font=self.f_small, fill=TEXT)
            d.text((x + 236, y + 2), o["status"], font=self.f_small, fill=col)
            y += 20
            if o.get("why"):
                d.text((x + 10, y), o["why"][:46], font=self.f_small, fill=HELP)
            y += 16
        elif t.get("instruction"):
            d.text((x, y), f"“{t['instruction'][:40]}”", font=self.f_small, fill=TEXT2)
            y += 16
            d.text(
                (x, y),
                f"orders -> {t['routed']} ({100 * t['route_p']:.0f}%, {t['route_ms']:.1f} ms)",
                font=self.f_small,
                fill=HELP,
            )
            y += 20
        else:
            y += 36

        # Decision latency
        d.text((x, y), "DECISION  ·  1 s MEDIAN", font=self.f_small, fill=HELP)
        y += 16
        num = f"{t['ms']:.1f}"
        d.text((x, y), num, font=self.f_big, fill=TEXT)
        d.text(
            (x + d.textlength(num, font=self.f_big) + 6, y + 16),
            "ms",
            font=self.f,
            fill=TEXT2,
        )
        d.text(
            (x + 140, y + 4),
            f"run p50 {t['p50']:.1f} ms",
            font=self.f_small,
            fill=TEXT2,
        )
        d.text(
            (x + 140, y + 22),
            f"run p99 {t['p99']:.1f} ms",
            font=self.f_small,
            fill=TEXT2,
        )
        y += 46
        # Tic budget bar: 0..100 ms
        bw = W_PANEL - 36
        d.rectangle([x, y, x + bw, y + 10], fill=LAYER2)
        d.rectangle([x, y, x + int(bw * min(1.0, t["ms"] / 100.0)), y + 10], fill=TEAL)
        tx = x + int(bw * TIC_MS / 100.0)
        d.line([tx, y - 4, tx, y + 14], fill=TEXT, width=1)
        d.text((tx + 4, y + 13), "1 tic 28.6 ms", font=self.f_small, fill=TEXT2)
        d.text((x + bw - 50, y + 13), "100 ms", font=self.f_small, fill=HELP)
        y += 36

        # Action (the full distribution is in the heatmap below)
        d.text((x, y), "ACTION", font=self.f_small, fill=HELP)
        y += 16
        d.text(
            (x, y), ACTION_LABELS.get(t["action"], t["action"]), font=self.f, fill=TEXT
        )
        d.text((x + 200, y + 2), t["action"], font=self.f_mono, fill=HELP)
        y += 26

        # Critic: danger of being hit in the next second
        d.text((x, y), "CRITIC  ·  DANGER NEXT SECOND", font=self.f_small, fill=HELP)
        y += 16
        crit = [t["critic"].get(k, 0.0) for k in DANGER_LEVELS]
        tot = sum(crit) or 1.0
        x1 = x
        for p_, col in zip(crit, (GREEN, AMBER, RED)):
            x2 = x1 + int(bw * p_ / tot)
            if x2 > x1:
                d.rectangle([x1, y, x2, y + 10], fill=col)
            x1 = x2
        y += 14
        d.text(
            (x, y),
            "   ".join(
                f"{k} {100 * c / tot:.0f}%" for k, c in zip(DANGER_LEVELS, crit)
            ),
            font=self.f_small,
            fill=TEXT2,
        )
        y += 22

        # Weapon planner and match
        plan = t.get("plan")
        d.text((x, y), "WEAPON PLANNER  ·  EVERY 0.5 s", font=self.f_small, fill=HELP)
        y += 16
        if plan:
            p_ = plan["probs"].get(str(plan["slot"]), 0.0)
            d.text(
                (x, y),
                f"{WEAPON_NAMES[plan['slot']]} (slot {plan['slot']}, {100 * p_:.0f}%)"
                f"   holding {t['weapon']}",
                font=self.f_small,
                fill=TEXT2,
            )
        y += 22
        s = t["stats"]
        d.text((x, y), "MATCH VS 7 BOTS", font=self.f_small, fill=HELP)
        y += 16
        d.text(
            (x, y),
            f"hp {t['hp']}  frags {s['frags']}  deaths {s['deaths']}  "
            f"rank {s['rank']}  best bot {s['best_bot'][1]}",
            font=self.f_small,
            fill=TEXT2,
        )
        y += 24

        # What the model reads
        d.text((x, y), "MODEL INPUT", font=self.f_small, fill=HELP)
        y += 16
        words, line, lines = t["text"].split(" "), "", []
        for w in words:
            if len(line) + len(w) + 1 > 46:
                lines.append(line)
                line = w
            else:
                line = f"{line} {w}".strip()
        lines.append(line)
        for ln in lines:
            if y > H - 40:
                break
            d.text((x, y), ln, font=self.f_mono, fill=TEXT2)
            y += 14
        d.text(
            (x, H - 22), f"tic {t['tick']}  ·  {t['gpu']}", font=self.f_small, fill=HELP
        )


def cmap(p: np.ndarray) -> np.ndarray:
    """Probabilities (any shape) -> RGB uint8 (shape + (3,))."""
    t = np.sqrt(np.clip(p, 0.0, 1.0))
    xs = [s[0] for s in STOPS]
    return np.stack(
        [np.interp(t, xs, [s[1][k] for s in STOPS]) for k in range(3)], axis=-1
    ).astype(np.uint8)


class Heatmap:
    """Per-tic action distribution: fills from the left, then slides."""

    def __init__(self, width: int):
        self.win = (width - HEAT_GUTTER - 18) // HEAT_PX
        # rows: behavior, the actions, danger, fresh-decision mark
        self.buf = np.empty((len(DISPLAY_ORDER) + 3, self.win, 3), np.uint8)
        self.buf[:] = BG
        self.count = 0
        self.f = font(10, mono=True)
        self.f_small = font(11)

    def push(
        self, probs: dict[str, float], adapter: str, decided: bool, critic: dict
    ) -> None:
        col = np.empty((self.buf.shape[0], 3), np.uint8)
        col[0] = COLORS.get(adapter, TEXT)
        col[1:-2] = cmap(np.array([probs.get(a, 0.0) for a in DISPLAY_ORDER]))
        pm, ph = critic.get("mid", 0.0), critic.get("high", 0.0)
        bg, am, rd = np.array(BG), np.array(AMBER), np.array(RED)
        col[-2] = np.clip(bg + pm * (am - bg) + ph * (rd - bg), 0, 255)
        col[-1] = TEXT if decided else BG
        if self.count < self.win:
            self.buf[:, self.count] = col
        else:
            self.buf[:, :-1] = self.buf[:, 1:]
            self.buf[:, -1] = col
        self.count += 1

    def draw(self, img: Image.Image, y0: int) -> None:
        d = ImageDraw.Draw(img)
        n = len(DISPLAY_ORDER)
        d.text((18, y0 + 8), "ACTION PROBABILITIES", font=self.f_small, fill=HELP)
        d.text(
            (170, y0 + 8),
            "one column per tic, newest at the right  ·  top strip: active behavior  "
            "·  danger: critic P(mid) amber, P(high) red  ·  marks: fresh decisions",
            font=self.f_small,
            fill=HELP,
        )
        gap = np.empty((3, self.win, 3), np.uint8)
        gap[:] = BG
        rows = [np.repeat(self.buf[0:1], 6, axis=0), gap]
        rows += [
            np.repeat(self.buf[i : i + 1], HEAT_ROW, axis=0) for i in range(1, n + 1)
        ]
        rows += [gap, np.repeat(self.buf[n + 1 : n + 2], 8, axis=0)]
        rows += [gap, np.repeat(self.buf[n + 2 : n + 3], 4, axis=0)]
        arr = np.repeat(np.concatenate(rows, axis=0), HEAT_PX, axis=1)
        top = y0 + 28
        img.paste(Image.fromarray(arr), (HEAT_GUTTER, top))
        for i, a in enumerate(DISPLAY_ORDER):
            y = top + 9 + HEAT_ROW * i - 1
            d.text((18, y), SHORT_LABELS[a], font=self.f, fill=TEXT2)
        d.text((18, top + 9 + HEAT_ROW * n + 3), "danger", font=self.f, fill=HELP)
        yb = top + arr.shape[0] + 4
        d.text(
            (HEAT_GUTTER, yb),
            f"-{self.win / TIC_HZ:.0f} s",
            font=self.f_small,
            fill=HELP,
        )
        right = HEAT_GUTTER + self.win * HEAT_PX
        d.text((right - 22, yb), "now", font=self.f_small, fill=HELP)
        # Legend: p on a sqrt scale
        lx, lw = HEAT_GUTTER + 260, 240
        ramp = cmap((np.linspace(0, 1, lw) ** 2)[None, :].repeat(8, axis=0))
        img.paste(Image.fromarray(ramp), (lx, yb + 2))
        d.text((lx - 16, yb), "p", font=self.f_small, fill=HELP)
        for p in (0, 0.05, 0.25, 0.5, 1):
            px = lx + int(lw * p**0.5)
            d.text((px - 6, yb + 11), f"{p:g}", font=self.f_small, fill=HELP)


ACT_ROW = 12  # pixels per model row in the activity map
PURPLE = (165, 110, 255)
MODEL_COLORS = {
    "base": TEXT2,
    **COLORS,
    ARMS: BLUE,
    CRITIC: PURPLE,
    ORDERS: GREEN,
    NARRATOR: TEAL,
}
MODEL_LABELS = {
    "base": "base model",
    ARMS: "weapon plan",
    CRITIC: "critic",
    ORDERS: "orders",
    NARRATOR: "narrator",
}


class Activity:
    """Which model runs on each tic: every adapter in the checkpoint, and the
    base model when it is asked itself (not the shared prefill inside an
    adapter's request), one row each, a column per tic, newest at the right;
    the labels light up while their model runs."""

    def __init__(self, width: int, models: list[str]):
        self.models = models
        self.win = (width - HEAT_GUTTER - 18) // HEAT_PX
        self.buf = np.empty((len(models), self.win, 3), np.uint8)
        self.buf[:] = BG
        self.count = 0
        self.now: set[str] = set()
        self.f = font(10, mono=True)
        self.f_small = font(11)

    @property
    def height(self) -> int:
        h = 52 + ACT_ROW * len(self.models)
        return h + h % 2  # the video codec wants an even frame height

    def push(self, active: set[str]) -> None:
        col = np.array(
            [MODEL_COLORS[m] if m in active else LAYER for m in self.models], np.uint8
        )
        if self.count < self.win:
            self.buf[:, self.count] = col
        else:
            self.buf[:, :-1] = self.buf[:, 1:]
            self.buf[:, -1] = col
        self.count += 1
        self.now = active

    def draw(self, img: Image.Image, y0: int) -> None:
        d = ImageDraw.Draw(img)
        d.text((18, y0 + 8), "WHICH MODEL RUNS", font=self.f_small, fill=HELP)
        d.text(
            (170, y0 + 8),
            "one column per tic, newest at the right  ·  one request per adapter; "
            "their shared prefix is computed once (aLoRA)",
            font=self.f_small,
            fill=HELP,
        )
        rows = np.repeat(self.buf, ACT_ROW, axis=0)
        rows[ACT_ROW - 2 :: ACT_ROW] = BG  # a thin gap between rows
        arr = np.repeat(rows, HEAT_PX, axis=1)
        top = y0 + 28
        img.paste(Image.fromarray(arr), (HEAT_GUTTER, top))
        for i, m in enumerate(self.models):
            on = m in self.now
            y = top + ACT_ROW * i - 1
            if on:
                d.rectangle([10, y + 2, 14, y + 8], fill=MODEL_COLORS[m])
            d.text(
                (18, y),
                MODEL_LABELS.get(m, m),
                font=self.f,
                fill=MODEL_COLORS[m] if on else HELP,
            )


CAPTION_S = 5.0  # how long a spoken line stays on screen
KINDS = {"alora": "aLoRA", "lora": "LoRA", "sr": "Shadow Residual", "base": "no"}


def draw_captions(img: Image.Image, caps: list[tuple[float, str, str]], now: float):
    """The latest line from the watcher and from the player, over the game view."""
    f = font(16)
    shown = [
        (who, re.sub(r"\[[a-z ]+\]\s*", "", text))  # sound tags are voiced, not shown
        for t, who, text in caps
        if now - t < CAPTION_S
    ][-2:]
    if not shown:
        return
    rows = []
    for who, text in shown:
        words, line = f"{'YOU' if who == 'player' else 'GRANITE'}: {text}".split(), ""
        for w in words:
            if len(line) + len(w) + 1 > 66:
                rows.append((who, line))
                line = w
            else:
                line = f"{line} {w}".strip()
        rows.append((who, line))
    d = ImageDraw.Draw(img)
    y = H - 14 - 22 * len(rows)
    d.rectangle([0, y - 8, W_GAME, H], fill=(10, 12, 16))
    for who, row in rows:
        d.text((14, y), row, font=f, fill=AMBER if who == "player" else TEAL)
        y += 22


class Overlay:
    """One whole frame: the game with captions, the panel, the activity map
    and the heatmap. Push each tic's columns (:attr:`act`, :attr:`heat`), then
    :meth:`draw`."""

    def __init__(self, models: list[str]):
        self.panel = Panel()
        self.heat = Heatmap(W_GAME + W_PANEL)
        self.act = Activity(W_GAME + W_PANEL, models)

    @property
    def size(self) -> tuple[int, int]:
        return W_GAME + W_PANEL, H + self.act.height + H_HEAT

    def draw(self, frame: np.ndarray, info: dict, caps, now: float) -> Image.Image:
        img = Image.new("RGB", self.size, BG)
        img.paste(Image.fromarray(frame), (0, 0))
        self.panel.draw(img, info)
        draw_captions(img, caps, now)
        self.act.draw(img, H)
        self.heat.draw(img, H + self.act.height)
        return img
