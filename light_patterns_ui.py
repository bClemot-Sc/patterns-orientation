"""
Light Circle Editor
===================
Interactive tool for designing green-light intensity patterns on two circular
rings (left and right). Each ring has N equally-spaced slots; clicking a slot
sets its green intensity (0–255). The combined distribution is visualised as a
von Mises band around each ring, and three summary arrows are drawn per circle.

Arrows
------
White  – points to the brightest slot; length encodes salience vs. the mean.
Amber  – intensity-weighted vector sum (centre of mass of the light field).
Pink   – best rotation to align the left distribution onto the right one,
         computed via circular cross-correlation (FFT). Arrow length encodes
         the quality of that alignment (Cauchy-Schwarz normalised score).
         Multiple pink arrows appear when the alignment is perfectly degenerate.

Usage
-----
    python light_circle_editor.py

Requirements: Python 3.9+, numpy (pip install numpy). tkinter is included with
most Python distributions; on Linux you may need: sudo apt install python3-tk.

Import / Export
---------------
Patterns are saved as JSON files containing n_slots, kappa, and per-slot
intensities. Use the IMPORT / EXPORT buttons in the toolbar.

See equations.md for the full mathematical reference.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import math
import json
import numpy as np

# ---------------------------------------------------------------------------
# Colours and fonts
# ---------------------------------------------------------------------------
BG         = "#0a0e14"
PANEL_BG   = "#0f1520"
RING_BG    = "#111827"
ACCENT     = "#00ff88"
ACCENT_DIM = "#00994d"
TEXT_MAIN  = "#e0f7ef"
TEXT_SUB   = "#4a7c6a"
SLOT_OFF   = "#1a2535"
SLOT_RIM   = "#1f3040"
DIST_FILL  = "#003322"
DIST_LINE  = "#00cc66"
ARR_BRIGHT = "#ffffff"
ARR_GRAD   = "#ffcc00"
ARR_OVERLAP = "#ff66cc"

FONT_TITLE = ("Courier New", 13, "bold")
FONT_MONO  = ("Courier New", 10)
FONT_SMALL = ("Courier New", 8)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
DOT_R        = 14     # slot dot radius in pixels
DIST_H       = 70     # max height of the von Mises band outside the ring
DIST_PTS     = 360    # number of angular samples for the distribution curve
ARR_MAX_FRAC = 0.72   # max arrow length as a fraction of the circle radius


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def slot_screen_angle(i, n):
    """Standard math angle for slot i of n. Slot 0 is at the top, clockwise."""
    return 2 * math.pi * i / n - math.pi / 2


def von_mises_pdf(theta, mu, kappa):
    """Unnormalised von Mises kernel centred at mu. Peak = 1 at theta == mu."""
    return math.exp(kappa * (math.cos(theta - mu) - 1.0))


def draw_arrow(canvas, cx, cy, angle, length, color, tag, width=2):
    """Draw an arrow from (cx, cy) in the direction of a standard screen angle."""
    tip_x = cx + length * math.cos(angle)
    tip_y = cy + length * math.sin(angle)
    canvas.create_line(cx, cy, tip_x, tip_y,
                       fill=color, width=width, arrow=tk.LAST,
                       arrowshape=(10, 13, 5), tags=(tag,))


def screen_to_bearing(a):
    """Convert a screen angle (rad) to a clockwise-from-top bearing (0–360°)."""
    return math.degrees(a + math.pi / 2) % 360


def angle_diff_deg(a_l, a_r):
    """Signed difference angle_r - angle_l in degrees, wrapped to (−180, +180]."""
    return ((math.degrees(a_r - a_l) + 180) % 360) - 180


# ---------------------------------------------------------------------------
# Widget helpers
# ---------------------------------------------------------------------------
def mk_entry(parent, width=5, textvariable=None):
    return tk.Entry(parent, width=width, font=FONT_MONO,
                    fg=TEXT_MAIN, bg=RING_BG, insertbackground=ACCENT,
                    relief="flat", bd=4, highlightthickness=1,
                    highlightcolor=ACCENT, highlightbackground=SLOT_RIM,
                    textvariable=textvariable)


def mk_button(parent, text, command, solid=False):
    if solid:
        return tk.Button(parent, text=text, font=FONT_SMALL,
                         fg=BG, bg=ACCENT, activebackground=ACCENT_DIM,
                         activeforeground=BG, bd=0, padx=10, pady=3,
                         command=command)
    return tk.Button(parent, text=text, font=FONT_SMALL,
                     fg=ACCENT, bg=PANEL_BG, activebackground=RING_BG,
                     activeforeground=ACCENT, bd=0, padx=10, pady=3,
                     command=command)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class LightCircleEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LIGHT CIRCLE EDITOR")
        self.configure(bg=BG)

        self.circles: list[dict] = []
        self.popup_open = False
        self._canvas_w = 0
        self._canvas_h = 0

        self._slot_str  = tk.StringVar(value="12")
        self._kappa_str = tk.StringVar(value="8")

        self._build_ui()
        self.after(50, self._go_fullscreen)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def n_slots(self):
        try:
            return max(2, min(64, int(self._slot_str.get())))
        except ValueError:
            return 12

    @property
    def kappa(self):
        try:
            return max(0.5, min(200.0, float(self._kappa_str.get())))
        except ValueError:
            return 8.0

    @property
    def circle_r(self):
        return max(120, int(self._canvas_h * 0.38))

    # ------------------------------------------------------------------
    # Fullscreen
    # ------------------------------------------------------------------
    def _go_fullscreen(self):
        try:
            self.attributes("-zoomed", True)   # Linux / X11
        except Exception:
            try:
                self.state("zoomed")            # Windows
            except Exception:
                pass
        self.update_idletasks()
        self._on_resize()

    def _toggle_fullscreen(self):
        self.attributes("-fullscreen", not self.attributes("-fullscreen"))
        self.after(80, self._on_resize)

    def _on_resize(self, _event=None):
        cw = max(800, self.winfo_width() - 56)
        ch = max(400, self.winfo_height() - 100)
        if cw == self._canvas_w and ch == self._canvas_h:
            return
        self._canvas_w, self._canvas_h = cw, ch
        self.canvas.config(width=cw, height=ch)
        self._render()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=28, pady=(14, 4))

        tk.Label(top, text="LIGHT CIRCLE EDITOR", font=FONT_TITLE,
                 fg=ACCENT, bg=BG).pack(side="left")
        mk_button(top, "[ ]  FULLSCREEN",
                  self._toggle_fullscreen).pack(side="left", padx=(18, 0))

        ctrl = tk.Frame(top, bg=BG)
        ctrl.pack(side="right")

        tk.Label(ctrl, text="SLOTS", font=FONT_SMALL,
                 fg=TEXT_SUB, bg=BG).pack(side="left", padx=(0, 4))
        se = mk_entry(ctrl, width=4, textvariable=self._slot_str)
        se.pack(side="left")
        se.bind("<Return>", lambda _e: self._render())
        mk_button(ctrl, "APPLY", self._render,
                  solid=True).pack(side="left", padx=(6, 24))

        tk.Label(ctrl, text="PEAK WIDTH", font=FONT_SMALL,
                 fg=TEXT_SUB, bg=BG).pack(side="left", padx=(0, 4))
        ke = mk_entry(ctrl, width=6, textvariable=self._kappa_str)
        ke.pack(side="left")
        ke.bind("<Return>", lambda _e: self._redraw_distributions())
        mk_button(ctrl, "APPLY", self._redraw_distributions,
                  solid=False).pack(side="left", padx=(6, 0))

        tb = tk.Frame(self, bg=PANEL_BG)
        tb.pack(fill="x", padx=28, pady=(0, 2))

        tk.Label(tb, text=" IO", font=FONT_SMALL,
                 fg=TEXT_SUB, bg=PANEL_BG).pack(side="left", pady=4, padx=(6, 2))
        mk_button(tb, "IMPORT", self._import_state).pack(side="left", padx=2, pady=3)
        mk_button(tb, "EXPORT", self._export_state).pack(side="left", padx=2, pady=3)

        legend = tk.Frame(tb, bg=PANEL_BG)
        legend.pack(side="left", padx=20)
        for color, label in [
            (ARR_BRIGHT,  "brightest slot  (salience)"),
            (ARR_GRAD,    "intensity gradient  (vector sum)"),
            (ARR_OVERLAP, "best overlap rotation  (right ring)"),
        ]:
            tk.Canvas(legend, width=28, height=3, bg=color,
                      highlightthickness=0).pack(side="left", padx=(0, 4))
            tk.Label(legend, text=label, font=FONT_SMALL,
                     fg=color, bg=PANEL_BG).pack(side="left", padx=(0, 16))

        self.status_lbl = tk.Label(tb, text="", font=FONT_SMALL,
                                   fg=ACCENT, bg=PANEL_BG)
        self.status_lbl.pack(side="right", pady=4, padx=10)
        tk.Label(tb, text="LEFT-CLICK: set intensity   RIGHT-CLICK: clear",
                 font=FONT_SMALL, fg=TEXT_SUB, bg=PANEL_BG
                 ).pack(side="right", pady=4, padx=8)

        self.canvas = tk.Canvas(self, width=1200, height=700,
                                bg=BG, highlightthickness=0)
        self.canvas.pack(padx=28, pady=(4, 16), fill="both", expand=True)
        self.bind("<Configure>", self._on_resize)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render(self, _event=None):
        self.canvas.delete("all")
        self.circles.clear()

        cw = self._canvas_w or self.canvas.winfo_width()
        ch = self._canvas_h or self.canvas.winfo_height()

        self._draw_circle_ring(cw // 4,     ch // 2, self.n_slots, "L")
        self._draw_circle_ring(3 * cw // 4, ch // 2, self.n_slots, "R")
        self._redraw_distributions()
        self._update_status()

    def _draw_circle_ring(self, cx, cy, n, side):
        cr = self.circle_r

        for r_extra, color in [(22, "#0a1f14"), (14, "#0d2e1c"), (7, "#134d2e")]:
            self.canvas.create_oval(cx - cr - r_extra, cy - cr - r_extra,
                                    cx + cr + r_extra, cy + cr + r_extra,
                                    outline=color, width=1)

        self.canvas.create_oval(cx - cr, cy - cr, cx + cr, cy + cr,
                                outline=SLOT_RIM, width=1)
        self.canvas.create_line(cx - 10, cy, cx + 10, cy, fill=SLOT_RIM, width=1)
        self.canvas.create_line(cx, cy - 10, cx, cy + 10, fill=SLOT_RIM, width=1)

        for i in range(n):
            sa = slot_screen_angle(i, n)
            sx = cx + cr * math.cos(sa)
            sy = cy + cr * math.sin(sa)

            slot_id = self.canvas.create_oval(
                sx - DOT_R, sy - DOT_R, sx + DOT_R, sy + DOT_R,
                fill=SLOT_OFF, outline=SLOT_RIM, width=1,
                tags=(f"slot_{side}_{i}",))

            entry = {"id": slot_id, "x": sx, "y": sy, "cx": cx, "cy": cy,
                     "intensity": 0, "side": side, "index": i, "screen_angle": sa}
            self.circles.append(entry)

            self.canvas.tag_bind(slot_id, "<Button-1>",
                                 lambda e, ent=entry: self._on_left_click(e, ent))
            self.canvas.tag_bind(slot_id, "<Button-3>",
                                 lambda e, ent=entry: self._clear_slot(ent))

    def _redraw_distributions(self, _event=None):
        for tag in ("dist_L", "dist_R", "arrow_bright", "arrow_grad",
                    "arrow_overlap", "stats_label"):
            self.canvas.delete(tag)

        sides: dict[str, list] = {}
        for ent in self.circles:
            sides.setdefault(ent["side"], []).append(ent)

        kappa       = self.kappa
        thetas      = [2 * math.pi * k / DIST_PTS - math.pi / 2 for k in range(DIST_PTS)]
        thetas_draw = thetas + [thetas[0]]

        # Evaluate von Mises distributions and find the shared global maximum
        computed: dict[str, dict] = {}
        global_max = 0.0
        for side, entries in sides.items():
            active = [e for e in entries if e["intensity"] > 0]
            values = [
                sum((e["intensity"] / 255.0) * von_mises_pdf(theta, e["screen_angle"], kappa)
                    for e in active)
                for theta in thetas
            ]
            computed[side] = {"entries": entries, "active": active, "values": values,
                               "cx": entries[0]["cx"], "cy": entries[0]["cy"]}
            global_max = max(global_max, max(values) if values else 0.0)

        # Compute arrow vectors for each side
        arrow_data: dict[str, dict] = {}
        for side, data in computed.items():
            active = data["active"]
            ad = {"bright_angle": None, "bright_salience": 0.0,
                  "grad_angle": None, "grad_len": 0.0}
            if active:
                brightest  = max(active, key=lambda e: e["intensity"])
                I_max      = brightest["intensity"]
                I_mean     = sum(e["intensity"] for e in active) / len(active)
                ad["bright_angle"]    = brightest["screen_angle"]
                ad["bright_salience"] = (I_max - I_mean) / I_max if I_max > 0 else 0.0
                rx = sum((e["intensity"] / 255.0) * math.cos(e["screen_angle"]) for e in active)
                ry = sum((e["intensity"] / 255.0) * math.sin(e["screen_angle"]) for e in active)
                mag = math.sqrt(rx**2 + ry**2)
                ad["grad_angle"] = math.atan2(ry, rx) if mag > 1e-6 else None
                ad["grad_len"]   = mag / len(active)
            arrow_data[side] = ad

        # Circular cross-correlation to find best-overlap rotation angles
        # C[k] = IFFT( FFT(f_R) * conj(FFT(f_L)) )
        # overlap_score = C_max / (||f_L|| * ||f_R||)  — Cauchy-Schwarz normalised
        overlap_peaks: list[float] = []
        overlap_score = 0.0
        if "L" in computed and "R" in computed:
            f_L = np.array(computed["L"]["values"], dtype=float)
            f_R = np.array(computed["R"]["values"], dtype=float)
            if f_L.max() > 0 and f_R.max() > 0:
                xcorr     = np.fft.irfft(np.fft.rfft(f_R) * np.conj(np.fft.rfft(f_L)), n=DIST_PTS)
                C_max     = float(xcorr.max())
                C_perfect = float(np.linalg.norm(f_L) * np.linalg.norm(f_R))
                overlap_score = max(0.0, min(1.0, C_max / C_perfect if C_perfect > 1e-12 else 0.0))
                thresh    = C_max * 0.99
                xc_ext    = np.concatenate([[xcorr[-1]], xcorr, [xcorr[0]]])
                for k in range(DIST_PTS):
                    if (xcorr[k] >= thresh
                            and xcorr[k] >= xc_ext[k]
                            and xcorr[k] >= xc_ext[k + 2]
                            and xcorr[k] == xcorr.max()):
                        overlap_peaks.append(k * (2 * math.pi / DIST_PTS) - math.pi / 2)
                if not overlap_peaks:
                    for k in range(DIST_PTS):
                        if (xcorr[k] >= thresh
                                and xcorr[k] > xc_ext[k]
                                and xcorr[k] > xc_ext[k + 2]):
                            overlap_peaks.append(k * (2 * math.pi / DIST_PTS) - math.pi / 2)

        # Draw distribution bands, arrows, and stats labels
        cr = self.circle_r
        for side, data in computed.items():
            entries = data["entries"]
            active  = data["active"]
            values  = data["values"]
            cx, cy  = data["cx"], data["cy"]
            r_base  = cr + DOT_R + 4

            outer = [(cx + (r_base + ((v / global_max) * DIST_H if global_max > 0 else 0)) * math.cos(t),
                      cy + (r_base + ((v / global_max) * DIST_H if global_max > 0 else 0)) * math.sin(t))
                     for t, v in zip(thetas_draw, values + [values[0]])]
            inner = [(cx + r_base * math.cos(t), cy + r_base * math.sin(t)) for t in thetas_draw]

            if active:
                flat = [c for pt in outer for c in pt] + [c for pt in reversed(inner) for c in pt]
                self.canvas.create_polygon(flat, fill=DIST_FILL, outline="",
                                           tags=(f"dist_{side}",))
            flat_outer = [c for pt in outer for c in pt]
            if len(flat_outer) >= 4:
                self.canvas.create_line(flat_outer, fill=DIST_LINE, width=1,
                                        smooth=True, tags=(f"dist_{side}",))

            ad = arrow_data[side]
            if ad["bright_angle"] is not None:
                arr_len = ARR_MAX_FRAC * cr * ad["bright_salience"]
                if arr_len > 4:
                    draw_arrow(self.canvas, cx, cy, ad["bright_angle"], arr_len,
                               ARR_BRIGHT, "arrow_bright", width=2)

            if ad["grad_angle"] is not None and ad["grad_len"] > 1e-6:
                arr_len = ARR_MAX_FRAC * cr * ad["grad_len"]
                if arr_len > 4:
                    draw_arrow(self.canvas, cx, cy, ad["grad_angle"], arr_len,
                               ARR_GRAD, "arrow_grad", width=2)

            if side == "R":
                for peak_angle in overlap_peaks:
                    arr_len = ARR_MAX_FRAC * cr * overlap_score
                    if arr_len > 4:
                        draw_arrow(self.canvas, cx, cy, peak_angle, arr_len,
                                   ARR_OVERLAP, "arrow_overlap", width=2)

            for ent in entries:
                self.canvas.tag_raise(ent["id"])
            for ent in entries:
                try:
                    self.canvas.tag_raise(f"glow_{ent['side']}_{ent['index']}")
                except Exception:
                    pass

        # Stats labels below each circle (one coloured line per arrow type)
        LINE_H = 13

        def put_line(x, y, text, color):
            self.canvas.create_text(x, y, text=text, font=("Courier New", 8),
                                    fill=color, anchor="nw", tags=("stats_label",))

        for side, data in computed.items():
            cx, cy = data["cx"], data["cy"]
            ad     = arrow_data[side]
            lx     = cx - 90
            ly     = cy + cr + DOT_R + DIST_H + 18

            b_txt = (f"brightest  {screen_to_bearing(ad['bright_angle']):6.1f}°"
                     if ad["bright_angle"] is not None else "brightest  —")
            put_line(lx, ly, b_txt, ARR_BRIGHT)
            ly += LINE_H

            g_txt = (f"intensity  {screen_to_bearing(ad['grad_angle']):6.1f}°"
                     if ad["grad_angle"] is not None else "intensity  —")
            put_line(lx, ly, g_txt, ARR_GRAD)
            ly += LINE_H

            if ad["bright_angle"] is not None and ad["grad_angle"] is not None:
                diff = angle_diff_deg(ad["bright_angle"], ad["grad_angle"])
                put_line(lx, ly, f"Δ bri-int  {diff:+.1f}°", TEXT_SUB)
                ly += LINE_H

            if "L" in arrow_data and "R" in arrow_data:
                ad_L, ad_R = arrow_data["L"], arrow_data["R"]
                if ad_L["bright_angle"] is not None and ad_R["bright_angle"] is not None:
                    diff = angle_diff_deg(ad_L["bright_angle"], ad_R["bright_angle"])
                    put_line(lx, ly, f"Δ brightest L→R  {diff:+.1f}°", ARR_BRIGHT)
                    ly += LINE_H
                if ad_L["grad_angle"] is not None and ad_R["grad_angle"] is not None:
                    diff = angle_diff_deg(ad_L["grad_angle"], ad_R["grad_angle"])
                    put_line(lx, ly, f"Δ intensity L→R  {diff:+.1f}°", ARR_GRAD)
                    ly += LINE_H

            if side == "R":
                if overlap_peaks:
                    for idx, pa in enumerate(overlap_peaks):
                        put_line(lx, ly,
                                 f"overlap #{idx+1}  {screen_to_bearing(pa):6.1f}°"
                                 f"  ({overlap_score * 100:.0f}% match)",
                                 ARR_OVERLAP)
                        ly += LINE_H
                else:
                    put_line(lx, ly, "overlap  —", ARR_OVERLAP)

        self._update_status()

    # ------------------------------------------------------------------
    # Slot interaction
    # ------------------------------------------------------------------
    def _on_left_click(self, event, entry):
        if not self.popup_open:
            self._open_intensity_popup(event.x_root, event.y_root, entry)

    def _clear_slot(self, entry):
        entry["intensity"] = 0
        self._paint_slot(entry)
        self._redraw_distributions()

    # ------------------------------------------------------------------
    # Intensity popup
    # ------------------------------------------------------------------
    def _open_intensity_popup(self, rx, ry, entry):
        self.popup_open = True
        original = entry["intensity"]

        pop = tk.Toplevel(self)
        pop.configure(bg=PANEL_BG)
        pop.overrideredirect(True)
        pop.geometry(f"240x210+{rx+12}+{ry-28}")

        hdr = tk.Frame(pop, bg=RING_BG)
        hdr.pack(fill="x")
        tk.Label(hdr, text=f"  {entry['side']}  -  SLOT {entry['index']:02d}",
                 font=FONT_SMALL, fg=TEXT_SUB, bg=RING_BG
                 ).pack(side="left", pady=5, padx=6)
        tk.Button(hdr, text="X", font=FONT_SMALL, fg=TEXT_SUB, bg=RING_BG,
                  activebackground=RING_BG, activeforeground=ACCENT, bd=0,
                  command=lambda: _close(False)).pack(side="right", padx=6)

        val_var  = tk.IntVar(value=original)
        text_var = tk.StringVar(value=str(original))

        tk.Label(pop, textvariable=val_var, font=("Courier New", 24, "bold"),
                 fg=ACCENT, bg=PANEL_BG).pack(pady=(10, 2))

        slider = tk.Scale(pop, from_=0, to=255, orient="horizontal",
                          variable=val_var, showvalue=False,
                          bg=PANEL_BG, fg=ACCENT, troughcolor=SLOT_OFF,
                          highlightthickness=0, bd=0, sliderlength=14,
                          sliderrelief="flat", activebackground=ACCENT)
        slider.pack(fill="x", padx=14, pady=2)

        row = tk.Frame(pop, bg=PANEL_BG)
        row.pack(fill="x", padx=14, pady=(4, 0))
        tk.Label(row, text="VALUE", font=FONT_SMALL, fg=TEXT_SUB,
                 bg=PANEL_BG).pack(side="left", padx=(0, 6))
        te = mk_entry(row, width=5, textvariable=text_var)
        te.pack(side="left")

        tk.Button(pop, text="SET", font=FONT_SMALL, fg=BG, bg=ACCENT,
                  activebackground=ACCENT_DIM, activeforeground=BG,
                  bd=0, pady=5, command=lambda: _close(True)
                  ).pack(fill="x", padx=14, pady=(8, 0))

        def _slider_moved(*_):
            v = val_var.get()
            text_var.set(str(v))
            entry["intensity"] = v
            self._paint_slot(entry)
            self._redraw_distributions()

        def _text_changed(*_):
            raw = text_var.get().strip()
            if not raw:
                return
            try:
                v = max(0, min(255, int(raw)))
                val_var.set(v)
                entry["intensity"] = v
                self._paint_slot(entry)
                self._redraw_distributions()
            except ValueError:
                pass

        def _close(confirm):
            entry["intensity"] = val_var.get() if confirm else original
            self._paint_slot(entry)
            self._redraw_distributions()
            self.popup_open = False
            pop.destroy()

        slider.config(command=_slider_moved)
        text_var.trace_add("write", lambda *_: _text_changed())
        te.bind("<Return>", lambda _e: _close(True))
        pop.bind("<Escape>", lambda _e: _close(False))
        pop.focus_force()
        te.focus_set()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------
    def _paint_slot(self, entry):
        iv  = entry["intensity"]
        tag = f"glow_{entry['side']}_{entry['index']}"
        self.canvas.delete(tag)
        if iv == 0:
            self.canvas.itemconfig(entry["id"], fill=SLOT_OFF, outline=SLOT_RIM, width=1)
        else:
            brightness = iv / 255.0
            g = int(80 + 175 * brightness)
            b = int(20 + 50  * brightness)
            self.canvas.itemconfig(entry["id"],
                                   fill=f"#00{g:02x}{b:02x}",
                                   outline=ACCENT if brightness > 0.7 else ACCENT_DIM,
                                   width=2)
            self.canvas.create_text(entry["x"], entry["y"] - DOT_R - 7,
                                    text=str(iv), font=("Courier New", 7),
                                    fill=ACCENT, tags=(tag,))

    def _update_status(self):
        active = sum(1 for c in self.circles if c["intensity"] > 0)
        self.status_lbl.config(
            text=f"ACTIVE {active}/{len(self.circles)}   "
                 f"SLOTS {self.n_slots}   k {self.kappa:.1f}")

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------
    def _export_state(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export pattern")
        if not path:
            return
        data = {
            "n_slots": self.n_slots,
            "kappa": self.kappa,
            "slots": [{"side": e["side"], "index": e["index"], "intensity": e["intensity"]}
                      for e in self.circles],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self.status_lbl.config(text=f"EXPORTED  {path.split('/')[-1]}")

    def _import_state(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Import pattern")
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._slot_str.set(str(data.get("n_slots", 12)))
            self._kappa_str.set(str(data.get("kappa", 8)))
            self._render()
            slot_map = {(s["side"], s["index"]): s["intensity"]
                        for s in data.get("slots", [])}
            for ent in self.circles:
                ent["intensity"] = slot_map.get((ent["side"], ent["index"]), 0)
                self._paint_slot(ent)
            self._redraw_distributions()
            self._update_status()
            self.status_lbl.config(text=f"IMPORTED  {path.split('/')[-1]}")
        except Exception as ex:
            messagebox.showerror("Import error", str(ex))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = LightCircleEditor()
    app.mainloop()