# SPDX-License-Identifier: Apache-2.0
"""The frame the videos and the live demo show: the game, a telemetry panel,
which model runs on each tic, the action distribution as a heatmap, and the
spoken lines as captions over the game.

:mod:`record_video` draws one per tic into an MP4; :mod:`doom_live` draws them
in a renderer process and streams them as JPEG (its ``classic`` and ``wide``
views), or streams the game alone and lets the browser draw the rest from
:func:`schema` and the telemetry (its ``client`` view).
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
    """The telemetry panel, drawn at ``k`` times its first size (360 x 480) in
    a box of its own (``x0``, width ``w``, height ``h``)."""

    def __init__(self, k: float = 1.0) -> None:
        self.k = k
        self.f_title = font(round(15 * k))
        self.f_big = font(round(34 * k))
        self.f = font(round(15 * k))
        self.f_small = font(round(12 * k))
        self.f_mono = font(round(12 * k), mono=True)

    def draw(
        self, img: Image.Image, t: dict, x0: int = W_GAME, w: int = W_PANEL, h: int = H
    ) -> None:
        k = self.k

        def S(v: float) -> int:  # a size at this panel's scale
            return round(v * k)

        d = ImageDraw.Draw(img)
        d.rectangle([x0, 0, x0 + w, h], fill=BG)
        x = x0 + S(18)
        y = S(14)
        bw = w - S(36)
        d.text((x, y), "GRANITE SWITCH  ·  DOOM", font=self.f_title, fill=TEXT2)
        y += S(26)
        d.text((x, y), "one output token per tic", font=self.f_small, fill=HELP)
        y += S(26)

        # Behavior chip
        c = COLORS.get(t["adapter"], TEXT)
        d.rounded_rectangle([x, y, x + S(150), y + S(28)], radius=S(14), fill=LAYER2)
        d.ellipse([x + S(12), y + S(10), x + S(20), y + S(18)], fill=c)
        d.text((x + S(28), y + S(5)), t["adapter"], font=self.f, fill=TEXT)
        d.text(
            (x + S(162), y + S(6)),
            f"{t.get('kind', 'aLoRA')} adapter",
            font=self.f_small,
            fill=HELP,
        )
        y += S(38)
        if t.get("order"):  # the partner's latest order (the live demo)
            o = t["order"]
            col = ORDER_COLORS.get(o["status"], TEXT2)
            d.rounded_rectangle([x, y, x + bw, y + S(18)], radius=S(9), fill=LAYER2)
            d.text(
                (x + S(10), y + S(2)),
                f"ORDER: {o['told']}",
                font=self.f_small,
                fill=TEXT,
            )
            d.text((x + bw - S(88), y + S(2)), o["status"], font=self.f_small, fill=col)
            y += S(20)
            if o.get("why"):
                d.text((x + S(10), y), o["why"][:46], font=self.f_small, fill=HELP)
            y += S(16)
        elif t.get("instruction"):
            d.text((x, y), f"“{t['instruction'][:40]}”", font=self.f_small, fill=TEXT2)
            y += S(16)
            d.text(
                (x, y),
                f"orders -> {t['routed']} ({100 * t['route_p']:.0f}%, {t['route_ms']:.1f} ms)",
                font=self.f_small,
                fill=HELP,
            )
            y += S(20)
        else:
            y += S(36)

        # Decision latency
        d.text((x, y), "DECISION  ·  1 s MEDIAN", font=self.f_small, fill=HELP)
        y += S(16)
        num = f"{t['ms']:.1f}"
        d.text((x, y), num, font=self.f_big, fill=TEXT)
        d.text(
            (x + d.textlength(num, font=self.f_big) + S(6), y + S(16)),
            "ms",
            font=self.f,
            fill=TEXT2,
        )
        d.text(
            (x + S(140), y + S(4)),
            f"run p50 {t['p50']:.1f} ms",
            font=self.f_small,
            fill=TEXT2,
        )
        d.text(
            (x + S(140), y + S(22)),
            f"run p99 {t['p99']:.1f} ms",
            font=self.f_small,
            fill=TEXT2,
        )
        y += S(46)
        # Tic budget bar: 0..100 ms
        d.rectangle([x, y, x + bw, y + S(10)], fill=LAYER2)
        d.rectangle(
            [x, y, x + int(bw * min(1.0, t["ms"] / 100.0)), y + S(10)], fill=TEAL
        )
        tx = x + int(bw * TIC_MS / 100.0)
        d.line([tx, y - S(4), tx, y + S(14)], fill=TEXT, width=max(1, S(1)))
        d.text((tx + S(4), y + S(13)), "1 tic 28.6 ms", font=self.f_small, fill=TEXT2)
        d.text((x + bw - S(50), y + S(13)), "100 ms", font=self.f_small, fill=HELP)
        y += S(36)

        # Action (the full distribution is in the heatmap below)
        d.text((x, y), "ACTION", font=self.f_small, fill=HELP)
        y += S(16)
        d.text(
            (x, y), ACTION_LABELS.get(t["action"], t["action"]), font=self.f, fill=TEXT
        )
        d.text((x + S(200), y + S(2)), t["action"], font=self.f_mono, fill=HELP)
        y += S(26)

        # Critic: danger of being hit in the next second
        d.text((x, y), "CRITIC  ·  DANGER NEXT SECOND", font=self.f_small, fill=HELP)
        y += S(16)
        crit = [t["critic"].get(c_, 0.0) for c_ in DANGER_LEVELS]
        tot = sum(crit) or 1.0
        x1 = x
        for p_, col in zip(crit, (GREEN, AMBER, RED)):
            x2 = x1 + int(bw * p_ / tot)
            if x2 > x1:
                d.rectangle([x1, y, x2, y + S(10)], fill=col)
            x1 = x2
        y += S(14)
        d.text(
            (x, y),
            "   ".join(
                f"{c_} {100 * c / tot:.0f}%" for c_, c in zip(DANGER_LEVELS, crit)
            ),
            font=self.f_small,
            fill=TEXT2,
        )
        y += S(22)

        # Weapon planner and match
        plan = t.get("plan")
        d.text((x, y), "WEAPON PLANNER  ·  EVERY 0.5 s", font=self.f_small, fill=HELP)
        y += S(16)
        if plan:
            p_ = plan["probs"].get(str(plan["slot"]), 0.0)
            d.text(
                (x, y),
                f"{WEAPON_NAMES[plan['slot']]} (slot {plan['slot']}, {100 * p_:.0f}%)"
                f"   holding {t['weapon']}",
                font=self.f_small,
                fill=TEXT2,
            )
        y += S(22)
        st = t["stats"]
        d.text((x, y), "MATCH VS 7 BOTS", font=self.f_small, fill=HELP)
        y += S(16)
        d.text(
            (x, y),
            f"hp {t['hp']}  frags {st['frags']}  deaths {st['deaths']}  "
            f"rank {st['rank']}  best bot {st['best_bot'][1]}",
            font=self.f_small,
            fill=TEXT2,
        )
        y += S(24)

        # What the model reads
        d.text((x, y), "MODEL INPUT", font=self.f_small, fill=HELP)
        y += S(16)
        for ln in wrap(d, t["text"], self.f_mono, bw):
            if y > h - S(40):
                break
            d.text((x, y), ln, font=self.f_mono, fill=TEXT2)
            y += S(14)
        d.text(
            (x, h - S(22)),
            f"tic {t['tick']}  ·  {t['gpu']}",
            font=self.f_small,
            fill=HELP,
        )


_WRAPPED: dict[tuple, list[str]] = {}


def wrap(d: ImageDraw.ImageDraw, text: str, f, width: int) -> list[str]:
    """``text`` in lines no wider than ``width`` pixels in font ``f`` (the
    same lines are asked for every frame: kept)."""
    key = (text, id(f), width)
    if key in _WRAPPED:
        return _WRAPPED[key]
    lines, line = [], ""
    for w in text.split():
        cand = f"{line} {w}".strip()
        if line and f.getlength(cand) > width:
            lines.append(line)
            line = w
        else:
            line = cand
    if len(_WRAPPED) > 4000:
        _WRAPPED.clear()
    _WRAPPED[key] = out = [*lines, line] if line else lines
    return out


def cmap(p: np.ndarray) -> np.ndarray:
    """Probabilities (any shape) -> RGB uint8 (shape + (3,))."""
    t = np.sqrt(np.clip(p, 0.0, 1.0))
    xs = [s[0] for s in STOPS]
    return np.stack(
        [np.interp(t, xs, [s[1][k] for s in STOPS]) for k in range(3)], axis=-1
    ).astype(np.uint8)


class Heatmap:
    """Per-tic action distribution: fills from the left, then slides.
    ``row``: pixels per action row (``height`` follows)."""

    def __init__(self, width: int, row: int = HEAT_ROW):
        self.row = row
        self.height = 82 + row * len(DISPLAY_ORDER)
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
            np.repeat(self.buf[i : i + 1], self.row, axis=0) for i in range(1, n + 1)
        ]
        rows += [gap, np.repeat(self.buf[n + 1 : n + 2], 8, axis=0)]
        rows += [gap, np.repeat(self.buf[n + 2 : n + 3], 4, axis=0)]
        arr = np.repeat(np.concatenate(rows, axis=0), HEAT_PX, axis=1)
        top = y0 + 28
        img.paste(Image.fromarray(arr), (HEAT_GUTTER, top))
        every = -(-11 // self.row)  # rows too thin for a label each: every few
        for i, a in enumerate(DISPLAY_ORDER):
            if i % every == 0:
                y = top + 9 + self.row * i - 1
                d.text((18, y), SHORT_LABELS[a], font=self.f, fill=TEXT2)
        d.text((18, top + 9 + self.row * n + 3), "danger", font=self.f, fill=HELP)
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

    def __init__(self, width: int, models: list[str], row: int = ACT_ROW):
        self.models, self.row = models, row
        self.win = (width - HEAT_GUTTER - 18) // HEAT_PX
        self.buf = np.empty((len(models), self.win, 3), np.uint8)
        self.buf[:] = BG
        self.count = 0
        self.now: set[str] = set()
        self.f = font(min(10, row), mono=True)  # a label per row: no taller than it
        self.f_small = font(11)

    @property
    def height(self) -> int:
        h = 52 + self.row * len(self.models)
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
        rows = np.repeat(self.buf, self.row, axis=0)
        rows[self.row - 2 :: self.row] = BG  # a thin gap between rows
        arr = np.repeat(rows, HEAT_PX, axis=1)
        top = y0 + 28
        img.paste(Image.fromarray(arr), (HEAT_GUTTER, top))
        for i, m in enumerate(self.models):
            on = m in self.now
            y = top + self.row * i - 1
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


def _said(caps, now: float | None = None) -> list[tuple[str, str]]:
    """(who, text) of the caption lines (the recent ones if ``now``); sound
    tags are voiced, not shown."""
    return [
        (who, re.sub(r"\[[a-z ]+\]\s*", "", text))
        for t, who, text in caps
        if now is None or now - t < CAPTION_S
    ]


def draw_captions(
    img: Image.Image,
    caps: list[tuple[float, str, str]],
    now: float,
    w: int = W_GAME,
    h: int = H,
    size: int = 16,
):
    """The latest line from the watcher and from the player, over the game view
    (``w`` x ``h`` at the top left)."""
    f = font(size)
    shown = _said(caps, now)[-2:]
    if not shown:
        return
    d = ImageDraw.Draw(img)
    pad, lh = round(size * 0.9), round(size * 1.4)
    rows = [
        (who, ln)
        for who, text in shown
        for ln in wrap(
            d, f"{'YOU' if who == 'player' else 'GRANITE'}: {text}", f, w - 2 * pad
        )
    ]
    y = h - pad - lh * len(rows)
    d.rectangle([0, y - pad // 2, w, h], fill=(10, 12, 16))
    for who, row in rows:
        d.text((pad, y), row, font=f, fill=AMBER if who == "player" else TEAL)
        y += lh


_PANE: dict = {}


def draw_conversation(
    img: Image.Image, caps: list[tuple[float, str, str]], box, k: float = 1.5
) -> None:
    """The conversation so far (the watcher's words and his lines), newest at
    the bottom, as much as fits in ``box`` (x0, y0, x1, y1). Drawn again only
    when it changes."""
    key = (tuple(caps), tuple(box), k)
    if _PANE.get("key") != key:
        _PANE["key"], _PANE["img"] = key, _conversation(caps, box, k)
    img.paste(_PANE["img"], tuple(box[:2]))


def _conversation(caps, box, k: float) -> Image.Image:
    w, h = box[2] - box[0], box[3] - box[1]
    pane = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(pane)
    f, fh = font(round(14 * k)), font(round(12 * k))
    pad, lh = round(18 * k), round(20 * k)
    d.line([pad, 0, w - pad, 0], fill=LAYER2, width=1)
    d.text((pad, round(10 * k)), "CONVERSATION", font=fh, fill=HELP)
    top, y = round(34 * k), h - pad
    for who, text in reversed(_said(caps)):
        label = "YOU" if who == "player" else "GRANITE"
        lines = wrap(d, f"{label}: {text}", f, w - 2 * pad)
        if y - lh * len(lines) < top:
            break
        y -= lh * len(lines)
        for i, ln in enumerate(lines):
            col = AMBER if who == "player" else TEAL
            d.text((pad, y + lh * i), ln, font=f, fill=col)
        y -= round(6 * k)
    return pane


def schema(models: list[str]) -> dict:
    """What a browser needs to draw the panel, the activity map and the
    heatmap as this module does (the live demo's ``client`` view): the
    labels, the colours (``#rrggbb``) and the heatmap's colour stops."""

    def hx(c: tuple[int, int, int]) -> str:
        return "#{:02x}{:02x}{:02x}".format(*c)

    palette = {
        "bg": BG,
        "layer": LAYER,
        "layer2": LAYER2,
        "text": TEXT,
        "text2": TEXT2,
        "help": HELP,
        "blue": BLUE,
        "teal": TEAL,
        "amber": AMBER,
        "red": RED,
        "green": GREEN,
    }
    return {
        "models": models,
        "model_labels": {m: MODEL_LABELS.get(m, m) for m in models},
        "model_colors": {m: hx(MODEL_COLORS[m]) for m in models},
        "colors": {k: hx(v) for k, v in COLORS.items()},
        "order_colors": {k: hx(v) for k, v in ORDER_COLORS.items()},
        "palette": {k: hx(v) for k, v in palette.items()},
        "stops": [[p, hx(c)] for p, c in STOPS],  # over sqrt(p)
        "display_order": list(DISPLAY_ORDER),
        "short_labels": {a: SHORT_LABELS[a] for a in DISPLAY_ORDER},
        "action_labels": ACTION_LABELS,
        "danger_levels": list(DANGER_LEVELS),
        "weapon_names": {str(k): v for k, v in WEAPON_NAMES.items()},
        "tic_hz": TIC_HZ,
        "tic_ms": TIC_MS,
    }


VIEWS = ("classic", "wide")
WIDE = (1920, 1080)  # 16:9, the size of most screens a demo is shown on
WIDE_GAME_K = 1.62  # the game in the wide view: 1037 x 778
WIDE_PANEL_K, WIDE_PANEL_H = 1.5, 720


class Overlay:
    """One whole frame: the game with captions, the panel, the activity map
    and the heatmap. Push each tic's columns (:attr:`act`, :attr:`heat`), then
    :meth:`draw`.

    ``view``: ``classic`` (1000 x 898: the game at its own size, the panel
    beside it, the maps below; the videos) or ``wide`` (16:9, 1920 x 1080: the
    game 1.62x with large captions, the maps below it; on the right, the panel
    1.5x and the conversation so far; the live demo in full screen)."""

    def __init__(self, models: list[str], view: str = "classic"):
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}")
        self.view = view
        if view == "wide":
            self.gw, self.gh = round(W_GAME * WIDE_GAME_K), round(H * WIDE_GAME_K)
            self.panel = Panel(WIDE_PANEL_K)
            self.act = Activity(self.gw, models, row=10)
            self.heat = Heatmap(self.gw, row=4)
        else:
            self.gw, self.gh = W_GAME, H
            self.panel = Panel()
            self.heat = Heatmap(W_GAME + W_PANEL)
            self.act = Activity(W_GAME + W_PANEL, models)

    @property
    def size(self) -> tuple[int, int]:
        if self.view == "wide":
            return WIDE
        return W_GAME + W_PANEL, H + self.act.height + self.heat.height

    def draw(self, frame: np.ndarray, info: dict, caps, now: float) -> Image.Image:
        img = Image.new("RGB", self.size, BG)
        game = Image.fromarray(frame)
        if self.view == "wide":
            img.paste(game.resize((self.gw, self.gh), Image.BILINEAR), (0, 0))
            w, h = self.size
            self.panel.draw(img, info, x0=self.gw, w=w - self.gw, h=WIDE_PANEL_H)
            draw_captions(img, caps, now, self.gw, self.gh, size=26)
            self.act.draw(img, self.gh)
            self.heat.draw(img, self.gh + self.act.height)
            draw_conversation(img, caps, (self.gw, WIDE_PANEL_H, w, h), WIDE_PANEL_K)
            return img
        img.paste(game, (0, 0))
        self.panel.draw(img, info)
        draw_captions(img, caps, now)
        self.act.draw(img, H)
        self.heat.draw(img, H + self.act.height)
        return img
