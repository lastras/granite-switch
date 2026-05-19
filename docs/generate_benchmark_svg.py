#!/usr/bin/env python3
"""Generate docs/benchmark_animation.svg from a slot schedule.

Usage:
    python docs/generate_benchmark_svg.py > docs/benchmark_animation.svg

To add/remove/reorder slots, edit the SLOTS list below.
"""

# ── DATA ──────────────────────────────────────────────────────────────────────

MODELS = {
    "3b": {"label": "Granite 4.1 3B",  "params": "3B parameters · Apache 2.0"},
    "8b": {"label": "Granite 4.1 8B",  "params": "8B parameters · Apache 2.0"},
    "30b": {"label": "Granite 4.1 30B", "params": "30B parameters · Apache 2.0"},
}

INTRINSICS = {
    # ── RAG library ──────────────────────────────────────────────────────────
    "qr": {
        "label": "Query Rewrite", "lib": "raglib", "type": "alora",
        # Accuracy (%) — from adapter_catalog.html alora row [idx 1,2,3]
        "base": {"3b": 61, "8b": 67, "30b": 64},
        "improved": {"3b": 86, "8b": 84, "30b": 87},
    },
    "qc": {
        "label": "Query Clarification", "lib": "raglib", "type": "alora",
        # Overall Accuracy (%)
        "base": {"3b": 65, "8b": 48, "30b": 76},
        "improved": {"3b": 95, "8b": 96, "30b": 95},
    },
    "an": {
        "label": "Answerability", "lib": "raglib", "type": "alora",
        # Classification Accuracy (%)
        "base": {"3b": 59, "8b": 66, "30b": 65},
        "improved": {"3b": 91, "8b": 91, "30b": 92},
    },
    "hd": {
        "label": "Hallucination Det.", "lib": "raglib", "type": "lora",
        # F1 (%) — alora not available for this adapter
        "base": {"3b": 42, "8b": 45, "30b": 54},
        "improved": {"3b": 71, "8b": 74, "30b": 83},
    },
    # ── Core library ─────────────────────────────────────────────────────────
    "ca": {
        "label": "Context Attr.", "lib": "corelib", "type": "lora",
        # WAUPC ×100 — alora not available for this adapter
        "base": {"3b": 34, "8b": 27, "30b": 65},
        "improved": {"3b": 92, "8b": 96, "30b": 90},
    },
    "rc": {
        "label": "Req. Check", "lib": "corelib", "type": "alora",
        # Balanced Accuracy ×100
        "base": {"3b": 51, "8b": 54, "30b": 57},
        "improved": {"3b": 77, "8b": 77, "30b": 78},
    },
    # ── Guardian library ─────────────────────────────────────────────────────
    "gc": {
        "label": "Guardian Core", "lib": "guardianlib", "type": "alora",
        # Safety Avg F1 ×100
        "base": {"3b": 1, "8b": 67, "30b": 69},
        "improved": {"3b": 78, "8b": 80, "30b": 81},
    },
    "fd": {
        "label": "Factuality Det.", "lib": "guardianlib", "type": "alora",
        # F1 ×100
        "base": {"3b": 8, "8b": 25, "30b": 51},
        "improved": {"3b": 81, "8b": 83, "30b": 83},
    },
}

LIBS = {
    "corelib":     {"fill": "#8A3FFC", "text": "white",   "sub": "#d4bbff", "pill_bg": "#EDE7FF", "pill_text": "#8A3FFC", "bar": "#8A3FFC", "pct": "#8A3FFC"},
    "raglib":      {"fill": "#009D9A", "text": "white",   "sub": "#9ef0f0", "pill_bg": "#E0F7F6", "pill_text": "#009D9A", "bar": "#009D9A", "pct": "#009D9A"},
    "guardianlib": {"fill": "#F1C21B", "text": "#161616", "sub": "#7a5f00", "pill_bg": "#FEF3C7", "pill_text": "#B45309", "bar": "#F1C21B", "pct": "#B45309"},
}

# Each slot specifies a model and 4 adapters [row0, row1, row2, row3].
# Any adapter can appear in any row — mix and match freely.
SLOTS = [
    # 3B
    {"model": "3b", "adapters": ["rc", "qr", "gc", "an"]},
    {"model": "3b", "adapters": ["hd", "ca", "fd", "qc"]},
    {"model": "3b", "adapters": ["gc", "an", "rc", "hd"]},
    {"model": "3b", "adapters": ["qc", "fd", "ca", "qr"]},
    {"model": "3b", "adapters": ["an", "gc", "hd", "rc"]},
    {"model": "3b", "adapters": ["fd", "rc", "qr", "ca"]},
    # 8B
    {"model": "8b", "adapters": ["ca", "hd", "an", "fd"]},
    {"model": "8b", "adapters": ["qr", "qc", "fd", "gc"]},
    {"model": "8b", "adapters": ["hd", "gc", "qc", "rc"]},
    {"model": "8b", "adapters": ["rc", "an", "ca", "fd"]},
    {"model": "8b", "adapters": ["gc", "qr", "fd", "an"]},
    {"model": "8b", "adapters": ["fd", "ca", "rc", "qc"]},
    # 30B
    {"model": "30b", "adapters": ["an", "fd", "gc", "ca"]},
    {"model": "30b", "adapters": ["ca", "rc", "qc", "hd"]},
    {"model": "30b", "adapters": ["qr", "hd", "an", "gc"]},
    {"model": "30b", "adapters": ["gc", "qc", "hd", "qr"]},
    {"model": "30b", "adapters": ["hd", "an", "rc", "fd"]},
    {"model": "30b", "adapters": ["rc", "gc", "fd", "an"]},
]

# ── TIMING CONSTANTS (within a 10s slot, expressed as fraction 0–1) ───────────

SLOT_SECONDS = 10
CONTENT_IN = 0.04        # content labels + baseline bars fade in
CONTENT_OUT = 0.94       # content starts fading out
SLOT_END = 1.0

# Adapter box slide-up (fraction of slot)
ADAPTER_DELAY = [0.12, 0.36, 0.48, 0.64]   # hidden→start transition
ADAPTER_SHOW  = [0.18, 0.42, 0.54, 0.70]   # fully visible

# Bar growth
BAR_DELAY = [0.16, 0.40, 0.52, 0.68]   # bar appears (scaleX=0)
BAR_START = [0.18, 0.42, 0.54, 0.70]   # bar starts growing
BAR_DONE  = [0.28, 0.52, 0.64, 0.80]   # bar fully grown

# Percentage text
BASE_PCT_IN   = 0.04                         # baseline pct shows (same as content)
BASE_PCT_HOLD = [0.26, 0.50, 0.62, 0.78]    # baseline holds until
BASE_PCT_OUT  = [0.30, 0.54, 0.66, 0.82]    # baseline hidden
IMP_PCT_IN    = [0.32, 0.56, 0.68, 0.84]    # improved pct shows
IMP_PCT_HOLD  = 0.94                         # improved holds until (same as content out)

# ── HELPERS ──────────────────────────────────────────────────────────────────

def slots_for(iid, row_idx):
    """Return slot indices where intrinsic appears in given row (any model)."""
    return [i for i, s in enumerate(SLOTS) if s["adapters"][row_idx] == iid]


def slots_for_model(iid, row_idx, model):
    """Return slot indices where intrinsic appears in given row AND model matches."""
    return [i for i, s in enumerate(SLOTS)
            if s["adapters"][row_idx] == iid and s["model"] == model]


def slots_for_model_only(model):
    """Return slot indices where model matches."""
    return [i for i, s in enumerate(SLOTS) if s["model"] == model]


# ── GENERATOR ─────────────────────────────────────────────────────────────────

def pct(frac):
    """Convert fraction (0-1 within total animation) to percentage string."""
    val = frac * 100
    if val == int(val):
        return f"{int(val)}%"
    s = f"{val:.4f}".rstrip("0").rstrip(".")
    return f"{s}%"


def slot_frac(slot_idx, local_frac):
    """Convert local fraction (0-1 within a slot) to global fraction (0-1 within full animation)."""
    n = len(SLOTS)
    return slot_idx / n + local_frac / n


def make_visibility_entries(slot_indices):
    """Generate opacity 0→1→1→0 entries for given slot indices."""
    entries = []
    for si in slot_indices:
        s_start = slot_frac(si, 0)
        s_in = slot_frac(si, CONTENT_IN)
        s_out = slot_frac(si, CONTENT_OUT)
        s_end = slot_frac(si, SLOT_END)
        entries.append((s_start, "opacity:0"))
        entries.append((s_in, "opacity:1"))
        entries.append((s_out, "opacity:1"))
        entries.append((s_end, "opacity:0"))
    return ensure_bounds(entries, "opacity:0")


def make_adapter_entries(slot_indices, row_idx):
    """Generate adapter box slide-up entries for given slot indices."""
    entries = []
    for si in slot_indices:
        delay = slot_frac(si, ADAPTER_DELAY[row_idx])
        show = slot_frac(si, ADAPTER_SHOW[row_idx])
        hold = slot_frac(si, CONTENT_OUT)
        end = slot_frac(si, SLOT_END)
        entries.append((delay, "opacity:0;transform:translateY(14px)"))
        entries.append((show, "opacity:1;transform:translateY(0)"))
        entries.append((hold, "opacity:1;transform:translateY(0)"))
        entries.append((end, "opacity:0;transform:translateY(14px)"))
    return ensure_bounds(entries, "opacity:0;transform:translateY(14px)")


def make_bar_entries(slot_indices, row_idx):
    """Generate bar growth entries for given slot indices."""
    entries = []
    for si in slot_indices:
        delay = slot_frac(si, BAR_DELAY[row_idx])
        start = slot_frac(si, BAR_START[row_idx])
        done = slot_frac(si, BAR_DONE[row_idx])
        hold = slot_frac(si, CONTENT_OUT)
        end = slot_frac(si, SLOT_END)
        entries.append((delay, "opacity:0;transform:scaleX(0)"))
        entries.append((start, "opacity:1;transform:scaleX(0)"))
        entries.append((done, "opacity:1;transform:scaleX(1)"))
        entries.append((hold, "opacity:1;transform:scaleX(1)"))
        entries.append((end, "opacity:0;transform:scaleX(0)"))
    return ensure_bounds(entries, "opacity:0;transform:scaleX(0)")


def make_improved_pct_entries(slot_indices, row_idx):
    """Generate improved percentage text entries for given slot indices."""
    entries = []
    for si in slot_indices:
        before = slot_frac(si, IMP_PCT_IN[row_idx] - 0.04)
        show = slot_frac(si, IMP_PCT_IN[row_idx])
        hold = slot_frac(si, IMP_PCT_HOLD)
        end = slot_frac(si, SLOT_END)
        entries.append((before, "opacity:0"))
        entries.append((show, "opacity:1"))
        entries.append((hold, "opacity:1"))
        entries.append((end, "opacity:0"))
    return ensure_bounds(entries, "opacity:0")


def ensure_bounds(entries, hidden_val):
    """Ensure keyframes start at 0% and end at 100% with hidden state."""
    if not entries:
        return [(0.0, hidden_val), (1.0, hidden_val)]
    if entries[0][0] > 0.0001:
        entries.insert(0, (0.0, hidden_val))
    if entries[-1][0] < 0.9999:
        entries.append((1.0, hidden_val))
    return entries


def fmt_keyframes(name, entries):
    """Format keyframe entries into a CSS @keyframes block (compact, one line)."""
    seen = {}
    for t, v in entries:
        seen[t] = v
    deduped = sorted(seen.items())
    parts = [f"{pct(t)}{{{v}}}" for t, v in deduped]
    return f"  @keyframes {name}{{{' '.join(parts)}}}"


def bar_width(base_pct):
    """Convert percentage (0-100) to bar pixel width (430px track)."""
    return round(base_pct / 100 * 430)


# ── SVG STRUCTURE ─────────────────────────────────────────────────────────────

BAR_H = 16                          # bar height (px)
BAR_GAP = 4                         # gap between stacked bars

ROW_Y = [274, 200, 126, 52]        # baseline bar y — row 0 at bottom (matches adapter stack direction)
BASE_TEXT_Y = [286, 212, 138, 64]  # pct text centered on baseline bar (ROW_Y + 12)
IMP_TEXT_Y = [306, 232, 158, 84]   # pct text centered on improved bar (ROW_Y + BAR_H + BAR_GAP + 12)

ADAPTER_Y = [196, 144, 92, 40]     # adapter box positions (bottom to top)
ADAPTER_TEXT1_Y = [216, 164, 112, 60]
ADAPTER_TEXT2_Y = [234, 182, 130, 78]


def generate_schedule_comment():
    """Generate the schedule comment block."""
    lines = []
    lines.append("<!--")
    n = len(SLOTS)
    dur = n * SLOT_SECONDS
    lines.append(f"  SCHEDULE — {SLOT_SECONDS}s per slot · {dur}s total ({n} slots)")
    lines.append(f"  Generated by: python docs/generate_benchmark_svg.py")
    lines.append("")
    for i, slot in enumerate(SLOTS):
        model = slot["model"]
        labels = [f"{INTRINSICS[sid]['label']}" for sid in slot["adapters"]]
        lines.append(f"  Slot {i+1:2d} [{model:>3s}]: {labels[0]:20s} {labels[1]:20s} {labels[2]:20s} {labels[3]}")
    lines.append("-->")
    return "\n".join(lines)


def generate_css():
    """Generate the full CSS block.

    Class naming convention (all row-specific to avoid collisions):
      Adapter box:    .{iid}a{row}        e.g. .rca0
      Content vis:    .{iid}v{row}        e.g. .rcv0     (labels, pills, tracks)
      Baseline bar:   .{iid}x{row}{model} e.g. .rcx03b   (model-specific width)
      Baseline pct:   .{iid}p{row}{model} e.g. .rcp03b   (model-specific value)
      Improved bar:   .{iid}b{row}{model} e.g. .rcb03b   (model-specific width)
      Improved pct:   .{iid}i{row}{model} e.g. .rci03b   (model-specific value)
      Base model:     .bm{model}          e.g. .bm3b
      Header:         .hd{model}          e.g. .hd3b
    """
    n = len(SLOTS)
    dur = n * SLOT_SECONDS
    lines = []
    lines.append("<defs><style>")
    lines.append("  text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif}")
    lines.append("")

    # Base model boxes — one per model size
    lines.append("  /* Base model boxes */")
    for model in MODELS:
        cls = f"bm{model}"
        entries = make_visibility_entries(slots_for_model_only(model))
        lines.append(f"  .{cls}{{animation:{cls} {dur}s linear infinite}}")
        lines.append(fmt_keyframes(cls, entries))
    lines.append("")

    # Right panel headers — one per model size
    lines.append("  /* Right panel headers */")
    for model in MODELS:
        cls = f"hd{model}"
        entries = make_visibility_entries(slots_for_model_only(model))
        lines.append(f"  .{cls}{{animation:{cls} {dur}s linear infinite}}")
        lines.append(fmt_keyframes(cls, entries))
    lines.append("")

    # Adapter boxes
    lines.append("  /* Adapter boxes */")
    for row_idx in range(4):
        ids_in_row = set(s["adapters"][row_idx] for s in SLOTS)
        for iid in sorted(ids_in_row):
            cls = f"{iid}a{row_idx}"
            entries = make_adapter_entries(slots_for(iid, row_idx), row_idx)
            lines.append(f"  .{cls}{{animation:{cls} {dur}s linear infinite}}")
            lines.append(fmt_keyframes(cls, entries))
    lines.append("")

    # Content visibility (labels, pills, track backgrounds — model-independent)
    lines.append("  /* Content visibility */")
    for row_idx in range(4):
        ids_in_row = set(s["adapters"][row_idx] for s in SLOTS)
        for iid in sorted(ids_in_row):
            cls = f"{iid}v{row_idx}"
            entries = make_visibility_entries(slots_for(iid, row_idx))
            lines.append(f"  .{cls}{{animation:{cls} {dur}s linear infinite}}")
            lines.append(fmt_keyframes(cls, entries))
    lines.append("")

    # Baseline bars (model-specific — different widths per model)
    lines.append("  /* Baseline bars */")
    for row_idx in range(4):
        ids_in_row = set(s["adapters"][row_idx] for s in SLOTS)
        for iid in sorted(ids_in_row):
            for model in MODELS:
                si_list = slots_for_model(iid, row_idx, model)
                if not si_list:
                    continue
                cls = f"{iid}x{row_idx}{model}"
                entries = make_visibility_entries(si_list)
                lines.append(f"  .{cls}{{animation:{cls} {dur}s linear infinite}}")
                lines.append(fmt_keyframes(cls, entries))
    lines.append("")

    # Baseline pct texts (model-specific)
    lines.append("  /* Baseline pct texts */")
    for row_idx in range(4):
        ids_in_row = set(s["adapters"][row_idx] for s in SLOTS)
        for iid in sorted(ids_in_row):
            for model in MODELS:
                si_list = slots_for_model(iid, row_idx, model)
                if not si_list:
                    continue
                cls = f"{iid}p{row_idx}{model}"
                entries = make_visibility_entries(si_list)
                lines.append(f"  .{cls}{{animation:{cls} {dur}s linear infinite}}")
                lines.append(fmt_keyframes(cls, entries))
    lines.append("")

    # Improved bars (model-specific)
    lines.append("  /* Improved bars */")
    lines.append("  .bar{transform-box:fill-box;transform-origin:left center}")
    for row_idx in range(4):
        ids_in_row = set(s["adapters"][row_idx] for s in SLOTS)
        for iid in sorted(ids_in_row):
            for model in MODELS:
                si_list = slots_for_model(iid, row_idx, model)
                if not si_list:
                    continue
                cls = f"{iid}b{row_idx}{model}"
                entries = make_bar_entries(si_list, row_idx)
                lines.append(f"  .{cls}{{animation:{cls} {dur}s linear infinite}}")
                lines.append(fmt_keyframes(cls, entries))
    lines.append("")

    # Improved percentage texts (model-specific)
    lines.append("  /* Improved pct texts */")
    for row_idx in range(4):
        ids_in_row = set(s["adapters"][row_idx] for s in SLOTS)
        for iid in sorted(ids_in_row):
            for model in MODELS:
                si_list = slots_for_model(iid, row_idx, model)
                if not si_list:
                    continue
                cls = f"{iid}i{row_idx}{model}"
                entries = make_improved_pct_entries(si_list, row_idx)
                lines.append(f"  .{cls}{{animation:{cls} {dur}s linear infinite}}")
                lines.append(fmt_keyframes(cls, entries))

    lines.append("</style></defs>")
    return "\n".join(lines)


def generate_left_panel():
    """Generate left panel SVG elements (base model + adapter boxes)."""
    lines = []
    lines.append("")
    lines.append('<!-- LEFT PANEL -->')
    lines.append('<text x="137" y="28" text-anchor="middle" font-size="10" fill="#8d8d8d" letter-spacing="1">YOUR CUSTOM MODEL</text>')
    lines.append("")

    # Base model — one box per model size, each with its own visibility animation
    for model, minfo in MODELS.items():
        cls = f"bm{model}"
        lines.append(f'<g class="{cls}">')
        lines.append(f'  <rect x="22" y="248" width="230" height="106" rx="12" fill="#0F62FE"/>')
        lines.append(f'  <text x="137" y="277" text-anchor="middle" font-size="16" font-weight="700" fill="white">{minfo["label"]}</text>')
        lines.append(f'  <text x="137" y="297" text-anchor="middle" font-size="10" fill="#a8c4ff">base model</text>')
        lines.append(f'  <text x="137" y="315" text-anchor="middle" font-size="10" fill="#6ea6ff">{minfo["params"]}</text>')
        lines.append(f'  <text x="137" y="336" text-anchor="middle" font-size="9" fill="#4f83cc">ibm-granite \u00b7 HuggingFace</text>')
        lines.append(f'</g>')
    lines.append("")

    # Adapter boxes — one per (intrinsic, row) combination
    for row_idx in range(4):
        ids_in_row = set(s["adapters"][row_idx] for s in SLOTS)
        for iid in sorted(ids_in_row):
            info = INTRINSICS[iid]
            lib = LIBS[info["lib"]]
            y = ADAPTER_Y[row_idx]
            t1y = ADAPTER_TEXT1_Y[row_idx]
            t2y = ADAPTER_TEXT2_Y[row_idx]
            cls = f"{iid}a{row_idx}"
            lines.append(f'<g class="{cls}">')
            lines.append(f'  <rect x="22" y="{y}" width="230" height="48" rx="8" fill="{lib["fill"]}"/>')
            lines.append(f'  <text x="137" y="{t1y}" text-anchor="middle" font-size="12" font-weight="700" fill="{lib["text"]}">{info["label"]}</text>')
            lines.append(f'  <text x="137" y="{t2y}" text-anchor="middle" font-size="9" fill="{lib["sub"]}">{info["lib"]}</text>')
            lines.append('</g>')

    return "\n".join(lines)


def generate_right_panel():
    """Generate right panel SVG elements (bars, labels, percentages)."""
    lines = []
    lines.append("")
    lines.append('<line x1="280" y1="15" x2="280" y2="365" stroke="#e0e0e0" stroke-width="1"/>')
    lines.append("")
    lines.append('<!-- RIGHT PANEL -->')

    # Model-specific headers
    for model, minfo in MODELS.items():
        cls = f"hd{model}"
        lines.append(f'<text class="{cls}" x="530" y="28" text-anchor="middle" font-size="10" fill="#8d8d8d" letter-spacing="1">PERFORMANCE \u00b7 {minfo["label"].upper()}</text>')
    lines.append("")

    bar_x = 310        # left edge of bars
    track_width = 430  # bar track width
    pct_x = 750        # right-aligned percentage text

    for row_idx in range(4):
        ids_in_row = set(s["adapters"][row_idx] for s in SLOTS)

        for iid in sorted(ids_in_row):
            info = INTRINSICS[iid]
            lib = LIBS[info["lib"]]
            by = ROW_Y[row_idx]
            pill_w = 64 if info["lib"] == "guardianlib" else (46 if info["lib"] == "corelib" else 40)

            # Row-specific class for labels/tracks (model-independent)
            vcls = f"{iid}v{row_idx}"

            # Stacked bars: baseline on top, improved below
            base_y = by
            imp_y = by + BAR_H + BAR_GAP

            # Label + pill below bars
            label_y = imp_y + BAR_H + 14

            lines.append(f'<!-- {info["label"]} ({info["lib"]}) row {row_idx} -->')
            # Track backgrounds (model-independent)
            lines.append(f'<rect class="{vcls}" x="{bar_x}" y="{base_y}" width="{track_width}" height="{BAR_H}" rx="4" fill="#f4f4f4"/>')
            lines.append(f'<rect class="{vcls}" x="{bar_x}" y="{imp_y}" width="{track_width}" height="{BAR_H}" rx="4" fill="#f4f4f4"/>')

            # Model-specific baseline bars, improved bars, and pct texts
            for model in MODELS:
                si_list = slots_for_model(iid, row_idx, model)
                if not si_list:
                    continue
                base_val = info["base"][model]
                imp_val = info["improved"][model]
                base_w = bar_width(base_val)
                imp_w = bar_width(imp_val)

                xcls = f"{iid}x{row_idx}{model}"   # baseline bar
                pcls = f"{iid}p{row_idx}{model}"   # baseline pct
                bcls = f"{iid}b{row_idx}{model}"   # improved bar
                icls = f"{iid}i{row_idx}{model}"   # improved pct

                # Baseline bar
                lines.append(f'<rect class="{xcls}" x="{bar_x}" y="{base_y}" width="{base_w}" height="{BAR_H}" rx="4" fill="#c6c6c6"/>')
                # Improved bar
                lines.append(f'<rect class="{bcls} bar" x="{bar_x}" y="{imp_y}" width="{imp_w}" height="{BAR_H}" rx="4" fill="{lib["bar"]}"/>')
                # Percentage texts
                base_ty = BASE_TEXT_Y[row_idx]
                imp_ty = IMP_TEXT_Y[row_idx]
                lines.append(f'<text class="{pcls}" x="{pct_x}" y="{base_ty}" text-anchor="end" font-size="10" font-weight="600" fill="#8d8d8d">{base_val}%</text>')
                lines.append(f'<text class="{icls}" x="{pct_x}" y="{imp_ty}" text-anchor="end" font-size="10" font-weight="700" fill="{lib["pct"]}">{imp_val}%</text>')

            # Label + type tag + library pill (model-independent)
            type_label = "aLoRA" if info["type"] == "alora" else "LoRA"
            type_x = bar_x + len(info["label"]) * 7 + 10
            type_w = 34 if info["type"] == "alora" else 28
            pill_x = type_x + type_w + 6
            lines.append(f'<text class="{vcls}" x="{bar_x}" y="{label_y}" font-size="11" font-weight="600" fill="#161616">{info["label"]}</text>')
            lines.append(f'<rect class="{vcls}" x="{type_x}" y="{label_y - 11}" width="{type_w}" height="14" rx="4" fill="{lib["pill_bg"]}"/>')
            lines.append(f'<text class="{vcls}" x="{type_x + type_w // 2}" y="{label_y - 1}" text-anchor="middle" font-size="8" font-weight="600" fill="{lib["pill_text"]}">{type_label}</text>')
            lines.append(f'<rect class="{vcls}" x="{pill_x}" y="{label_y - 11}" width="{pill_w}" height="14" rx="4" fill="{lib["pill_bg"]}"/>')
            lines.append(f'<text class="{vcls}" x="{pill_x + pill_w // 2}" y="{label_y - 1}" text-anchor="middle" font-size="8" font-weight="600" fill="{lib["pill_text"]}">{info["lib"]}</text>')

    lines.append("")
    lines.append('<text x="310" y="372" font-size="12" fill="#a8a8a8">prompted baseline \u2192 with aLoRA/LoRA adapter</text>')
    return "\n".join(lines)


def generate_svg():
    """Generate the complete SVG."""
    parts = []
    parts.append('<svg width="820" height="380" viewBox="0 0 820 380" xmlns="http://www.w3.org/2000/svg">')
    parts.append(generate_schedule_comment())
    parts.append(generate_css())
    parts.append("")
    parts.append('<rect width="820" height="380" rx="10" fill="#ffffff"/>')
    parts.append('<rect width="820" height="380" rx="10" fill="none" stroke="#e0e0e0" stroke-width="1"/>')
    parts.append(generate_left_panel())
    parts.append(generate_right_panel())
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    print(generate_svg())
