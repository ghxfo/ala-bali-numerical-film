#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"على بالي" — a live generative numerical-typography film
=========================================================

Every image in this film — the woman, her thoughts, the moon, the nights,
the calendar, the words, the man — is built from thousands of individual
digits, symbols and punctuation marks rendered directly in a Pygame window.
Nothing is pre-rendered and nothing is exported.

Install:   pip install pygame numpy pillow
Run:       python ala_bali.py

Keys:      ESC / window close  quit
           SPACE               pause / resume
           R                   restart from the beginning
           H                   timing HUD (scene, time, fps) for fine-tuning
           L                   toggle Arabic lyric captions (optional, see CFG)

All timings, colours, densities and speeds are in the CFG class below.
"""

import math
import os
import sys
import time

import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# =============================================================================
# 1. CONFIGURATION — everything you may want to fine-tune lives here
# =============================================================================
class CFG:
    # ---- window ------------------------------------------------------------
    WIDTH, HEIGHT = 1000, 1000
    FPS = 60
    WINDOW_TITLE = "Ala Bali  —  numerical film"

    # ---- timeline (seconds) -------------------------------------------------
    INTRO = 2.2                        # darkness + drifting digits before lyric 1
    SEGMENTS = [                       # (lyric, duration) — the primary timeline
        ("على بالي ولا انت داري", 3.9),
        ("باللي جرا لي والليالي", 4.2),
        ("سنين طويلة سبتها لي", 4.8),
        ("يا انشغالي بكل كلمة قلتها", 6.7),
    ]
    OUTRO = 6.5                        # breathing room after the last lyric
    END_HOLD = 1.5                     # black before restarting
    LOOP = True                        # False = hold the final (dark) frame
    SPEED = 1.0                        # global playback speed (1.0 = real time)

    # ---- density (particle counts; DENSITY scales all of them) -------------
    DENSITY = 1.0
    N_WOMAN = 3400
    N_THOUGHT = 2600
    N_WORLD = 2800                     # moon / ground / calendar / echoes
    N_MAN = 1000

    # ---- motion -------------------------------------------------------------
    MOTION = 1.0                       # multiplies all spring (assembly) speeds
    FLOW_SPEED = 36.0                  # px/s of the drifting flow field
    FLICKER = 1.0                      # how often glyphs change character
    ORBIT_SPEED = 1.0                  # speed of the circling thoughts
    NIGHTS = 3                         # moon cycles shown in scene 2
    YEARS = (2013, 2026)               # calendar years flipped in scene 3
    REPLAYS = 5                        # times the words close in on her (scene 4)
    MAN_WALK_SPEED = 16.0              # px/s

    # ---- look ---------------------------------------------------------------
    TRAIL = 0.045                      # motion-trail time constant (0 = none)
    VIGNETTE = 0.55                    # 0 = none, 1 = strong
    GLYPH_SIZES = (7, 9, 12, 16)       # px, smallest → largest
    DATE_SIZE = 15
    BRIGHT_LEVELS = 16
    BACKGROUND = (6, 6, 7)             # near-black charcoal

    PALETTE = [                        # muted, warm, no neon
        (232, 224, 204),   # 0 ivory        – the woman
        (214, 158, 84),    # 1 amber        – thoughts / memory
        (198, 140, 140),   # 2 dusty rose   – feeling
        (186, 98, 58),     # 3 burnt orange – words that hurt
        (124, 148, 118),   # 4 muted green  – time
        (138, 136, 130),   # 5 ash          – him
        (236, 218, 172),   # 6 moonlight
    ]

    # ---- optional extras ----------------------------------------------------
    SHOW_HUD = False
    SHOW_LYRICS = False                # needs: pip install arabic-reshaper python-bidi
    AUDIO_FILE = None                  # e.g. "song.mp3" to play along (optional)
    AUDIO_LYRIC_START = 0.0            # second in the audio where lyric 1 begins
    SEED = 7


IVORY, AMBER, ROSE, ORANGE, GREEN, ASH, MOON = range(7)

# Design space is always 1000 x 1000; it is scaled to the window.
GROUND = 792.0
W_HEAD = np.array([470.0, 462.0])          # woman's head (home pose)
W_PIVOT = np.array([480.0, GROUND])        # her feet — pivot for the S4 move
W_LEFT_SHIFT = np.array([-150.0, 0.0])
W_LEFT_SCALE = 0.94
MOON_C = np.array([735.0, 235.0])
MOON_R = 86.0
CAL = (250.0, 245.0, 750.0, 765.0)          # calendar box x0, y0, x1, y1


# =============================================================================
# 2. SMALL HELPERS
# =============================================================================
def ss(x):
    """Vectorised smoothstep on [0,1]."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def ssf(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def mask_points(img, n, rng, edge_frac=0.35, edge_px=2, blur=14):
    """Sample n points from a white-on-black PIL mask.
    Returns (points, is_edge, depth) — depth≈1 deep inside, ≈0 at the rim."""
    a = np.asarray(img) > 127
    er = np.asarray(img.filter(ImageFilter.MinFilter(edge_px * 2 + 1))) > 127
    edge = a & ~er
    ys, xs = np.nonzero(a)
    eys, exs = np.nonzero(edge)
    ne = int(n * edge_frac) if len(exs) else 0
    ie = rng.integers(0, len(exs), ne) if ne else np.zeros(0, int)
    ii = rng.integers(0, len(xs), n - ne)
    pts = np.concatenate([np.stack([exs[ie], eys[ie]], 1),
                          np.stack([xs[ii], ys[ii]], 1)]).astype(np.float64)
    pts += rng.uniform(-0.5, 0.5, pts.shape)
    is_edge = np.concatenate([np.ones(ne, bool), np.zeros(n - ne, bool)])
    bl = np.asarray(img.filter(ImageFilter.GaussianBlur(blur)), np.float64) / 255.0
    h, w = bl.shape
    depth = bl[np.clip(pts[:, 1].astype(int), 0, h - 1), np.clip(pts[:, 0].astype(int), 0, w - 1)]
    perm = rng.permutation(n)
    return pts[perm], is_edge[perm], depth[perm]


def pil_font(size):
    """A TTF that is guaranteed to exist: pygame's bundled default font."""
    path = os.path.join(os.path.dirname(pygame.__file__), pygame.font.get_default_font())
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def flow_field(p, t):
    """Smooth pseudo-curl flow (unit-ish velocities)."""
    x, y = p[:, 0], p[:, 1]
    vx = np.sin(y * 0.0075 + t * 0.55) + 0.7 * np.sin((x + y) * 0.0042 - t * 0.35)
    vy = 0.9 * np.cos(x * 0.0068 - t * 0.45) + 0.6 * np.cos((x - y) * 0.0051 + t * 0.6)
    return np.stack([vx, vy], 1)


def pile_y(x, u):
    """Height of the slowly accumulating mound of fallen memories."""
    h = 6.0 + 24.0 * np.exp(-((x - 470.0) / 210.0) ** 2)
    return GROUND + 18.0 - h * u


# =============================================================================
# 3. SHAPES — silhouettes and objects, drawn as masks, then sampled
# =============================================================================
def woman_mask():
    """A woman sitting on the ground hugging her knees, head lowered."""
    img = Image.new("L", (1000, 1000), 0)
    d = ImageDraw.Draw(img)
    G = int(GROUND)
    # torso / back
    d.polygon([(436, 500), (404, 510), (380, 532), (362, 570), (350, 615), (343, 668),
               (343, 718), (350, 756), (366, 780), (396, G), (476, G), (480, 700),
               (478, 610), (472, 560), (462, 528), (452, 508)], fill=255)
    # thigh, shin, foot
    d.line([(404, 760), (556, 592)], fill=255, width=80)
    d.ellipse((520, 552, 596, 628), fill=255)                       # knee
    d.line([(566, 596), (600, 766)], fill=255, width=50)
    d.polygon([(572, 750), (618, 764), (646, 780), (646, G), (566, G)], fill=255)
    # neck + head (lowered, face toward the knees)
    d.line([(452, 498), (444, 532)], fill=255, width=32)
    d.ellipse((428, 412, 512, 512), fill=255)
    d.polygon([(506, 468), (520, 487), (508, 492)], fill=255)        # nose
    d.polygon([(498, 498), (508, 506), (494, 512)], fill=255)        # chin
    # long hair falling down the back
    d.polygon([(446, 410), (418, 420), (398, 446), (388, 486), (382, 530), (372, 578),
               (360, 625), (350, 664), (346, 690), (360, 676), (372, 646), (386, 604),
               (400, 562), (414, 524), (428, 494), (448, 470)], fill=255)
    for pts in ([(392, 560), (378, 620), (370, 676), (374, 712)],
                (([(404, 540), (396, 600), (392, 650)])),
                ([(500, 432), (514, 462), (520, 496), (522, 528)])):
        d.line(pts, fill=255, width=4, joint="curve")
    # the arm wrapped round her knees — a thin dark gap makes it readable
    arm = [(424, 540), (494, 626), (590, 652)]
    d.line(arm, fill=0, width=40, joint="curve")
    d.ellipse((580, 630, 622, 672), fill=0)
    # repair what the gap cut away from the body edges
    d.line([(566, 640), (580, 700)], fill=255, width=48)
    d.line(arm, fill=255, width=26, joint="curve")
    d.ellipse((586, 636, 614, 664), fill=255)                       # hand
    return img


def year_mask(year):
    img = Image.new("L", (1000, 1000), 0)
    d = ImageDraw.Draw(img)
    x0, y0, x1, _ = CAL
    d.text(((x0 + x1) / 2, y0 + 62), str(year), font=pil_font(118), fill=255, anchor="mm")
    return img


def calendar_lines():
    """Calendar frame + grid as line segments with the character used on each."""
    x0, y0, x1, y1 = CAL
    hy = y0 + 122                        # header separator
    segs = []
    for o, w in ((0, 1.0), (7, 0.7)):    # double outer border
        segs += [(x0 - o, y0 - o, x1 + o, y0 - o, "-", w), (x0 - o, y1 + o, x1 + o, y1 + o, "-", w),
                 (x0 - o, y0 - o, x0 - o, y1 + o, "|", w), (x1 + o, y0 - o, x1 + o, y1 + o, "|", w)]
    segs.append((x0, hy, x1, hy, "=", 0.9))
    gy0 = hy + 36
    segs.append((x0, gy0, x1, gy0, "-", 0.6))
    rows, cols = 6, 7
    ch = (y1 - gy0) / rows
    cw = (x1 - x0) / cols
    for r in range(1, rows):
        segs.append((x0, gy0 + r * ch, x1, gy0 + r * ch, "-", 0.5))
    for c in range(1, cols):
        segs.append((x0 + c * cw, gy0, x0 + c * cw, y1, "|", 0.5))
    # ring binders on top
    for k in range(5):
        bx = x0 + (k + 1) * (x1 - x0) / 6
        segs.append((bx, y0 - 26, bx, y0 + 10, "|", 0.9))
    grid = dict(gy0=gy0, ch=ch, cw=cw, hy=hy)
    return segs, grid


def sample_segments(segs, n, rng):
    lengths = np.array([math.hypot(s[2] - s[0], s[3] - s[1]) * (0.6 + 0.4 * s[5]) for s in segs])
    counts = np.maximum(1, np.round(lengths / lengths.sum() * n).astype(int))
    while counts.sum() > n:
        counts[np.argmax(counts)] -= 1
    while counts.sum() < n:
        counts[np.argmax(lengths)] += 1
    pts, chars, wts = [], [], []
    for s, c in zip(segs, counts):
        u = rng.random(c)
        x = s[0] + (s[2] - s[0]) * u
        y = s[1] + (s[3] - s[1]) * u
        pts.append(np.stack([x, y], 1) + rng.normal(0, 0.6, (c, 2)))
        chars += [s[4]] * c
        wts.append(np.full(c, s[5]))
    return np.concatenate(pts), chars, np.concatenate(wts)


def fragment_templates(rng):
    """Small masks: quotation marks, speech waveforms, broken lines of 'text'."""
    temps = []
    W, H = 180, 90

    def new():
        im = Image.new("L", (W, H), 0)
        return im, ImageDraw.Draw(im)

    # 0 opening quote “
    im, d = new()
    for cx in (68, 108):
        d.ellipse((cx - 13, 44, cx + 13, 70), fill=255)
        d.polygon([(cx - 12, 52), (cx - 4, 26), (cx + 8, 18), (cx + 2, 30), (cx + 10, 50)], fill=255)
    temps.append(im)
    # 1 closing quote ”
    im, d = new()
    for cx in (72, 112):
        d.ellipse((cx - 13, 20, cx + 13, 46), fill=255)
        d.polygon([(cx + 12, 38), (cx + 4, 62), (cx - 8, 72), (cx - 2, 60), (cx - 10, 40)], fill=255)
    temps.append(im)
    # 2 speech waveform
    im, d = new()
    for i in range(24):
        x = 12 + i * 6.6
        env = math.sin(math.pi * (i + 0.5) / 24) ** 0.8
        h = 4 + 36 * env * (0.35 + 0.65 * rng.random())
        d.rectangle((x, 45 - h, x + 2.5, 45 + h), fill=255)
    temps.append(im)
    # 3 two broken lines of 'text'
    im, d = new()
    for yy in (30, 56):
        x = 8
        while x < W - 20:
            wl = int(rng.integers(12, 40))
            d.rounded_rectangle((x, yy - 5, min(W - 8, x + wl), yy + 5), radius=4, fill=255)
            x += wl + int(rng.integers(6, 12))
    temps.append(im)
    # 4 ellipsis + dash  … —
    im, d = new()
    for cx in (40, 64, 88):
        d.ellipse((cx - 7, 50, cx + 7, 64), fill=255)
    d.rectangle((104, 54, 168, 60), fill=255)
    temps.append(im)
    return temps


# =============================================================================
# 4. THE MAN — a parametric skeleton so he can walk with stable particles
# =============================================================================
class ManRig:
    # (joint a, joint b, width at a, width at b, weight, depth/brightness)
    LIMBS = [("hip", "neck", 30, 37, 6.0, 1.0),
             ("neck", "headb", 9, 9, 0.4, 1.0),
             ("shoulder", "elbow_f", 12, 10, 1.2, 0.55),
             ("elbow_f", "wrist_f", 10, 7, 1.0, 0.55),
             ("hip", "knee_f", 18, 15, 1.8, 0.55),
             ("knee_f", "ankle_f", 14, 10, 1.6, 0.55),
             ("ankle_f", "toe_f", 7, 6, 0.4, 0.55),
             ("hip", "knee_n", 19, 15, 1.8, 1.0),
             ("knee_n", "ankle_n", 14, 10, 1.6, 1.0),
             ("ankle_n", "toe_n", 7, 6, 0.4, 1.0),
             ("shoulder", "elbow_n", 12, 10, 1.2, 1.0),
             ("elbow_n", "wrist_n", 10, 7, 1.0, 1.0)]
    HEAD_R = 25.0

    def __init__(self, n, rng):
        weights = np.array([l[4] for l in self.LIMBS] + [2.6])
        counts = np.floor(weights / weights.sum() * n).astype(int)
        counts[0] += n - counts.sum()
        self.limb = np.concatenate([np.full(c, i) for i, c in enumerate(counts)])
        self.u = rng.random(n)
        edge = rng.random(n) < 0.42
        v = rng.uniform(-1, 1, n)
        v[edge] = np.sign(v[edge] + 1e-9) * rng.uniform(0.82, 1.0, edge.sum())
        self.v = v
        self.ang = rng.uniform(0, 2 * math.pi, n)
        self.rad = np.where(edge, rng.uniform(0.85, 1.0, n), np.sqrt(rng.random(n)))
        self.depth = np.ones(n)
        for i, l in enumerate(self.LIMBS):
            self.depth[self.limb == i] = l[5]
        self.edge = edge

    @staticmethod
    def joints(phi):
        j = {"hip": np.array([0.0, 0.0])}
        lean = 0.07
        j["neck"] = np.array([math.sin(lean) * 150, -150.0])
        j["headb"] = j["neck"] + (4.0, -14.0)
        j["head"] = j["neck"] + (7.0, -36.0)
        j["shoulder"] = j["neck"] + (0.0, 14.0)
        for side, off in (("n", 0.0), ("f", math.pi)):
            p = phi + off
            a_th = 0.40 * math.sin(p)
            knee = j["hip"] + 102 * np.array([math.sin(a_th), math.cos(a_th)])
            bend = 0.6 * max(0.0, math.sin(p - 1.3))
            a_sh = a_th - bend
            ankle = knee + 100 * np.array([math.sin(a_sh), math.cos(a_sh)])
            j["knee_" + side], j["ankle_" + side] = knee, ankle
            j["toe_" + side] = ankle + (25.0, 3.0)
            a_ua = -0.32 * math.sin(p)
            elbow = j["shoulder"] + 70 * np.array([math.sin(a_ua), math.cos(a_ua)])
            a_fa = a_ua + 0.28
            j["elbow_" + side] = elbow
            j["wrist_" + side] = elbow + 64 * np.array([math.sin(a_fa), math.cos(a_fa)])
        low = max(j["ankle_n"][1], j["ankle_f"][1]) + 6
        for k in j:
            j[k] = j[k] - (0, low)            # feet on the ground (y=0)
        return j

    def points(self, phi, base, scale):
        j = self.joints(phi)
        out = np.zeros((len(self.u), 2))
        for i, (a, b, wa, wb, _, _) in enumerate(self.LIMBS):
            m = self.limb == i
            A, B = j[a], j[b]
            dvec = B - A
            L = np.hypot(*dvec) + 1e-9
            nrm = np.array([-dvec[1], dvec[0]]) / L
            u = self.u[m]
            w = (wa + (wb - wa) * u) * 0.5
            out[m] = A + np.outer(u, dvec) + np.outer(self.v[m] * w, nrm)
        m = self.limb == len(self.LIMBS)
        r = self.rad[m] * self.HEAD_R
        out[m] = j["head"] + np.stack([np.cos(self.ang[m]) * r, np.sin(self.ang[m]) * r * 1.12], 1)
        return base + out * scale


# =============================================================================
# 5. PARTICLE GROUP
# =============================================================================
class Group:
    """A population of glyph particles. Modes: 0 spring→target, 1 flow, 2 fall."""

    def __init__(self, n, rng, charset):
        self.n = n
        self.rng = rng
        self.pos = np.zeros((n, 2))
        self.tgt = np.zeros((n, 2))
        self.rate = np.full(n, 3.0)
        self.bright = np.zeros(n)
        self.btgt = np.zeros(n)
        self.brate = np.full(n, 3.0)
        self.mode = np.zeros(n, np.int8)
        self.pal = np.zeros(n, np.int32)
        self.size = np.zeros(n, np.int32)
        self.charset = np.asarray(charset)
        self.char = rng.choice(self.charset, n)
        self.flicker = 0.5
        self.flow_speed = 1.0
        self.flow_bias = np.zeros((n, 2))
        self.fall_v = rng.uniform(70, 150, n)
        self.fall_on = np.zeros(n, bool)
        self.fall_tgt = np.zeros((n, 2))
        self.r1, self.r2, self.r3, self.r4 = (rng.random(n) for _ in range(4))
        self.lock_chars = None            # optional mask of particles whose chars don't flicker

    def set_charset(self, cs):
        cs = np.asarray(cs)
        if cs.shape != self.charset.shape or np.any(cs != self.charset):
            self.charset = cs

    def update(self, t, dt):
        k = 1.0 - np.exp(-self.rate * CFG.MOTION * dt)
        m0 = self.mode == 0
        self.pos[m0] += (self.tgt[m0] - self.pos[m0]) * k[m0, None]
        m1 = self.mode == 1
        if m1.any():
            v = flow_field(self.pos[m1], t) + self.flow_bias[m1]
            self.pos[m1] += v * CFG.FLOW_SPEED * self.flow_speed * dt
        m2 = self.mode == 2
        new = m2 & ~self.fall_on
        if new.any():
            x = np.clip(self.pos[new, 0] + self.rng.normal(0, 25, new.sum()), 70, 930)
            self.fall_tgt[new, 0] = x
            self.fall_tgt[new, 1] = pile_y(x, self.rng.random(new.sum()) ** 0.7)
        self.fall_on = m2
        if m2.any():
            p = self.pos[m2]
            ft = self.fall_tgt[m2]
            p[:, 1] = np.minimum(p[:, 1] + self.fall_v[m2] * dt, np.maximum(ft[:, 1], p[:, 1]))
            p[:, 0] += (ft[:, 0] - p[:, 0]) * (1 - math.exp(-1.2 * dt))
            self.pos[m2] = p
        self.bright += (self.btgt - self.bright) * (1.0 - np.exp(-self.brate * dt))
        # glyph flicker
        lam = self.n * self.flicker * CFG.FLICKER * dt
        c = int(lam) + (1 if self.rng.random() < lam - int(lam) else 0)
        if c:
            idx = self.rng.integers(0, self.n, c)
            if self.lock_chars is not None:
                idx = idx[~self.lock_chars[idx]]
            self.char[idx] = self.rng.choice(self.charset, len(idx))


# =============================================================================
# 6. THE FILM — timeline + scene choreography
# =============================================================================
class Film:
    def __init__(self, screen):
        self.screen = screen
        W, H = screen.get_size()
        self.S = min(W, H) / 1000.0
        self.off = np.array([(W - 1000 * self.S) / 2, (H - 1000 * self.S) / 2])
        self.canvas = pygame.Surface((W, H)).convert()
        self.canvas.fill(CFG.BACKGROUND)
        self._fonts()
        self._vignette(W, H)
        self._lyrics_init()
        self.show_hud = CFG.SHOW_HUD
        self.show_lyrics = CFG.SHOW_LYRICS
        self.reset()

    # ---------------------------------------------------------------- setup
    def _fonts(self):
        names = "dejavusansmono,menlo,consolas,couriernew,liberationmono,monospace"
        self.fonts = []
        for s in CFG.GLYPH_SIZES:
            px = max(6, int(round(s * self.S)))
            f = pygame.font.SysFont(names, px)
            self.fonts.append(f)
        base = ("0123456789" ".,:;'\"`!?-_=+*/\\|()[]{}<>~^%#&"
                "×÷±∞∑π√∆≈≠°·…“”‘’—")
        f = self.fonts[1]
        ok = []
        for ch in base:
            try:
                m = f.metrics(ch)
                if m and m[0] is not None and m[0][4] > 0:
                    ok.append(ch)
            except Exception:
                pass
        self.CH = "".join(ok)
        self.ci = {c: i for i, c in enumerate(self.CH)}
        self.NS = len(self.fonts)
        self.NP = len(CFG.PALETTE)
        self.NB = CFG.BRIGHT_LEVELS
        self.cache = [None] * (len(self.CH) * self.NS * self.NP * self.NB)
        self.goff = np.array([[f.size("0")[0] / 2, f.size("0")[1] / 2] for f in self.fonts])
        self.date_font = pygame.font.SysFont(names, max(8, int(CFG.DATE_SIZE * self.S)))
        self.date_cache = {}
        self.hud_font = pygame.font.SysFont(names, 13)

    def cs(self, chars):
        idx = [self.ci[c] for c in chars if c in self.ci]
        return np.array(idx or [self.ci.get("0", 0)])

    def _vignette(self, W, H):
        self.vign = None
        if CFG.VIGNETTE <= 0:
            return
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)) / 1.414
        v = 1.0 - CFG.VIGNETTE * np.clip(r, 0, 1) ** 2.2
        a = (np.clip(v, 0, 1) * 255).astype(np.uint8)
        arr = np.repeat(a.T[:, :, None], 3, 2)
        self.vign = pygame.surfarray.make_surface(arr).convert()

    def _lyrics_init(self):
        self.lyric_surfs = []
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
        except Exception:
            return
        path = pygame.font.match_font("arial,tahoma,notonaskharabic,notosansarabic,"
                                      "dejavusans,geezapro,segoeui,amiri")
        if not path:
            return
        f = pygame.font.Font(path, max(14, int(22 * self.S)))
        for text, _ in CFG.SEGMENTS:
            shaped = get_display(arabic_reshaper.reshape(text))
            self.lyric_surfs.append(f.render(shaped, True, (190, 180, 160)))

    def reset(self):
        rng = np.random.default_rng(CFG.SEED)
        self.rng = rng
        D = CFG.DENSITY
        nW, nT, nX, nM = (max(50, int(v * D)) for v in
                          (CFG.N_WOMAN, CFG.N_THOUGHT, CFG.N_WORLD, CFG.N_MAN))
        digits = self.cs("0123456789")
        self.cs_woman = self.cs("0123456789" * 4 + ".:+")
        self.cs_thought = self.cs("0123456789" * 2 + "°∞~·o+%")
        self.cs_frag = self.cs("“”‘’\"',.;:…—-_~" + "0123456789")
        self.cs_hurt = self.cs("/\\|×÷#%!?—~")
        self.cs_moon = self.cs("0123456789()°.")
        self.cs_dust = self.cs("0123456789.·:")

        # ---------- timeline
        b = [CFG.INTRO]
        for _, d in CFG.SEGMENTS:
            b.append(b[-1] + d)
        self.B = b
        self.end = b[-1] + CFG.OUTRO
        self.total = self.end + CFG.END_HOLD

        # ---------- woman
        pts, edge, depth = mask_points(woman_mask(), nW, rng, edge_frac=0.38)
        self.w_home = pts
        self.w_left = (pts - W_PIVOT) * W_LEFT_SCALE + W_PIVOT + W_LEFT_SHIFT
        self.head_left = (W_HEAD - W_PIVOT) * W_LEFT_SCALE + W_PIVOT + W_LEFT_SHIFT
        rim = np.clip((pts[:, 0] - 380) / 220, 0, 1) * 0.15           # light from the right
        self.w_b = np.where(edge, 0.78, 0.26 + 0.30 * (1 - depth) + rim) * rng.uniform(0.8, 1.05, nW)
        self.w_upper = np.clip((640 - pts[:, 1]) / 220, 0, 1)
        Wg = Group(nW, rng, self.cs_woman)
        Wg.pos = rng.uniform(0, 1000, (nW, 2))
        Wg.tgt = Wg.pos.copy()
        Wg.size = np.where(edge, rng.choice([0, 0, 1], nW), rng.choice([0, 1, 1, 2, 2, 3], nW))
        Wg.pal = np.where(rng.random(nW) < 0.22, ROSE, IVORY)
        Wg.flicker = 0.35
        self.w_scatter = Wg.pos.copy()
        self.W = Wg

        # ---------- thoughts
        Tg = Group(nT, rng, self.cs_thought)
        fam = rng.integers(0, 8, nT)                      # 0-6 rings, 7 = rose curve
        self.t_fam = fam
        self.t_R = 68 + fam * 23 + rng.normal(0, 5, nT)
        self.t_tilt = fam * 0.62 + rng.normal(0, 0.05, nT)
        self.t_flat = (0.30 + 0.55 * ((fam * 0.37) % 1.0)) + rng.normal(0, 0.02, nT)
        dirs = np.where(fam % 3 == 2, -1.0, 1.0)
        self.t_w = (0.75 + 0.22 * fam) * dirs * (1 + rng.normal(0, 0.04, nT))
        self.t_th = rng.uniform(0, 2 * math.pi, nT)
        Tg.pal = rng.choice([AMBER, AMBER, AMBER, ROSE, ROSE, IVORY], nT)
        Tg.size = rng.choice([0, 0, 1, 1, 2], nT)
        Tg.flicker = 1.6
        sx = rng.uniform(50, 950, nT)
        sy = 40 + 520 * rng.random(nT) ** 1.4
        far = np.hypot(sx - MOON_C[0], sy - MOON_C[1]) < MOON_R + 40
        sy[far] = 40 + 80 * rng.random(far.sum())
        sx[far] = rng.uniform(50, 560, far.sum())
        self.stars = np.stack([sx, sy], 1)
        night = rng.integers(0, max(1, CFG.NIGHTS), nT).astype(float)
        self.t_night = np.where(rng.random(nT) < 0.62, night, -1)
        Tg.pos = self.orbit(Tg, W_HEAD, 1.0, 1.0)
        Tg.tgt = Tg.pos.copy()
        self.T = Tg

        # fragments (scene 4)
        temps = fragment_templates(rng)
        F = 20
        self.F = F
        fid = np.arange(nT) % F
        rng.shuffle(fid)
        self.f_id = fid
        self.f_type = rng.choice([0, 1, 2, 3, 3, 4, 2, 0, 1], F)
        self.f_local = np.zeros((nT, 2))
        self.f_edge = np.zeros(nT, bool)
        for f in range(F):
            m = fid == f
            p, e, _ = mask_points(temps[self.f_type[f]], m.sum(), rng, edge_frac=0.3, edge_px=1, blur=3)
            self.f_local[m] = p - (90, 45)
            self.f_edge[m] = e
        self.f_base = rng.uniform(0, 2 * math.pi, F)
        self.f_rand = rng.random(F)
        self.f_scale = rng.uniform(0.85, 1.4, F)
        self.f_split = rng.uniform(-30, 30, F)

        # ---------- world (dust, moon, ground, calendar, echoes)
        Xg = Group(nX, rng, self.cs_dust)
        Xg.pos = rng.uniform(0, 1000, (nX, 2))
        self.x_dust = Xg.pos.copy()
        Xg.size = rng.choice([0, 0, 1], nX)
        self.x_size = Xg.size.copy()
        Xg.pal = np.full(nX, ASH)
        Xg.flicker = 0.4
        nMoon = int(nX * 0.55)
        self.n_moon = nMoon
        ang = rng.uniform(0, 2 * math.pi, nX)
        rr = np.sqrt(rng.random(nX))
        rr[: nMoon // 3] = rng.uniform(0.93, 1.0, nMoon // 3)          # rim
        self.moon_pts = MOON_C + np.stack([np.cos(ang), np.sin(ang)], 1) * (rr * MOON_R)[:, None]
        gx = rng.uniform(60, 940, nX)
        self.ground_pts = np.stack([gx, GROUND + 20 + rng.normal(0, 1.5, nX)], 1)
        # calendar
        segs, grid = calendar_lines()
        self.grid = grid
        nYear = int(nX * 0.30)
        nLine = nX - nYear
        lp, lch, lw = sample_segments(segs, nLine, rng)
        perm = rng.permutation(nLine)
        lp, lw = lp[perm], lw[perm]
        lch = [lch[i] for i in perm]
        self.n_line = nLine
        self.cal_pts = np.zeros((nX, 2))
        self.cal_pts[:nLine] = lp
        self.cal_w = np.ones(nX)
        self.cal_w[:nLine] = lw
        self.cal_char = np.array([self.ci.get(c, self.ci.get("-", 0)) for c in lch])
        y0, y1 = CFG.YEARS
        self.years = list(range(y0, y1 + 1))
        self.year_pts = [mask_points(year_mask(y), nYear, rng, edge_frac=0.6, edge_px=1, blur=4)[0]
                         for y in self.years]
        self.cal_pal = np.where(rng.random(nX) < 0.6, GREEN, IVORY)
        self.cal_pal[nLine:] = IVORY
        self.cal_size = rng.choice([0, 0, 1], nX)
        self.year_digits = self.cs("0123456789")
        # echoes around the woman (scene 4)
        edge_idx = np.nonzero(edge)[0]
        self.e_src = edge_idx[rng.integers(0, len(edge_idx), nX)]
        self.e_k = 1.06 + 0.46 * rng.random(nX) ** 1.2
        self.e_pal = rng.choice([ROSE, ROSE, AMBER, IVORY], nX)
        self.X = Xg

        # ---------- man
        Mg = Group(nM, rng, self.cs("0123456789.·:"))
        self.rig = ManRig(nM, rng)
        Mg.size = np.where(self.rig.edge, 0, rng.choice([0, 1, 1, 2], nM))
        Mg.pal = np.where(rng.random(nM) < 0.3, GREEN, ASH)
        Mg.flicker = 0.25
        self.man_scale = 0.58
        self.man_x0 = 735.0
        Mg.pos = self.rig.points(0, np.array([self.man_x0, GROUND]), self.man_scale) + rng.normal(0, 60, (nM, 2))
        Mg.tgt = Mg.pos.copy()
        self.man_scatter = Mg.pos.copy()
        self.M = Mg

        self.groups = [self.X, self.W, self.T, self.M]
        self.dates = []
        self.scene = "intro"
        self.t = 0.0

    # ---------------------------------------------------------------- pieces
    def orbit(self, g, head, rscale, flat_mul):
        """Overlapping elliptical loops (and one rose curve) around her head."""
        C = head + np.array([4.0, -34.0])
        th = self.t_th
        R = self.t_R * rscale
        rose = self.t_fam == 7
        R = np.where(rose, (40 + 150 * np.abs(np.cos(2.5 * th))) * rscale, R)
        e = np.where(rose, 0.62, self.t_flat * flat_mul)
        x = R * np.cos(th)
        y = R * np.sin(th) * e
        c, s = np.cos(self.t_tilt), np.sin(self.t_tilt)
        return C + np.stack([x * c - y * s, x * s + y * c], 1)

    def flip_state(self, t):
        T2, T3 = self.B[2], self.B[3]
        a = T2 + 0.16 * (T3 - T2)
        b = T3 - 0.30 * (T3 - T2)
        n = len(self.years) - 1
        u = np.clip((t - a) / max(0.01, b - a), 0, 1)
        idx = n * u ** 1.7
        page = int(min(n, math.floor(idx + 0.5)))
        frac = idx - math.floor(idx)
        sx = abs(math.cos(math.pi * frac)) if 0 < u < 1 else 1.0
        return page, sx, frac, u

    def cal_transform(self, p, sx):
        x0, y0, x1, y1 = CAL
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        q = p.copy()
        q[:, 0] = cx + (p[:, 0] - cx) * max(sx, 0.02)
        k = 1 - sx
        q[:, 1] = cy + (p[:, 1] - cy) * (1 + 0.07 * k) + (p[:, 0] - cx) * 0.10 * k
        return q

    # ---------------------------------------------------------------- update
    def update(self, t, dt):
        self.t = t
        T0, T1, T2, T3, T4 = self.B
        S1, S2, S3, S4 = (T1 - T0, T2 - T1, T3 - T2, T4 - T3)
        self.scene = ("intro" if t < T0 else "1  ala bali" if t < T1 else "2  nights" if t < T2
                      else "3  years" if t < T3 else "4  every word" if t < T4 else "ending")
        self._woman(t, T0, T1, T2, T3, T4, S1, S2, S3, S4)
        self._thoughts(t, dt, T0, T1, T2, T3, T4, S1, S2, S3, S4)
        self._world(t, T0, T1, T2, T3, T4, S1, S2, S3, S4)
        self._man(t, T3, T4, S4)
        for g in self.groups:
            g.update(t, dt)

    # ---- the woman
    def _woman(self, t, T0, T1, T2, T3, T4, S1, S2, S3, S4):
        g = self.W
        n = g.n
        r1, r2, r3, r4 = g.r1, g.r2, g.r3, g.r4
        # where she is (centre → left for scene 4)
        sh = ss((t - (T3 - 0.14 * S3) - 0.35 * r2) / 1.1)[:, None]
        base = self.w_home * (1 - sh) + self.w_left * sh
        base = base.copy()
        base[:, 1] += math.sin(t * 1.25) * 1.3 * self.w_upper          # breathing
        p4 = ssf((t - T3) / S4)
        if p4 > 0:                                                        # under pressure
            base += self.rng.normal(0, 0.25 + 0.9 * p4 * p4, (n, 2)) * self.w_upper[:, None]
        # assembly from scattered dust
        a = ss((t - (T0 - 0.9) - r1 * 0.55 * S1 - 0.9 * r4 * r4) / 0.8)
        drift = self.w_scatter + np.stack([np.sin(t * 0.3 + r1 * 9), np.cos(t * 0.25 + r2 * 9)], 1) * 18
        tgt = drift + (base - drift) * a[:, None]
        bt = 0.09 * (r3 < 0.5) * (1 - a) + self.w_b * a
        rate = 1.6 + 2.2 * r2
        mode = np.zeros(n, np.int8)
        brate = np.full(n, 2.5)

        # scene 2: flow → fall into the pile of nights
        t_flow = T1 - 0.06 * S2 + (0.28 * S2) * (0.75 * r1 + 0.25 * np.clip(self.w_home[:, 1] - 410, 0, 380) / 380)
        t_fall = T1 + S2 * (0.30 + 0.60 * r2)
        t_rise = T2 - 0.05 * S2 + r3 * 0.30 * S3
        in_flow = (t >= t_flow) & (t < t_fall)
        in_fall = (t >= t_fall) & (t < t_rise)
        risen = t >= t_rise
        mode[in_flow] = 1
        mode[in_fall] = 2
        bt = np.where(in_flow, self.w_b * 0.5, bt)
        bt = np.where(in_fall, 0.30 * (0.6 + 0.4 * r4), bt)
        # scene 3: she reappears, faint, behind the calendar
        s4b = ss((t - (T3 - 0.12 * S3) - 0.5 * r1) / 0.9)
        bt = np.where(risen, self.w_b * (0.24 + 0.76 * s4b), bt)
        rate = np.where(risen, 1.8 + 1.4 * r2, rate)
        brate = np.where(in_fall, 1.2, brate)
        # ending: alone, then dissolving upward into the dark
        t_dis = T4 + 0.48 * CFG.OUTRO + r1 * 0.32 * CFG.OUTRO
        dis = t >= t_dis
        if dis.any():
            mode[dis] = 1
            bt = np.where(dis, 0.0, bt)
            brate = np.where(dis, 0.9 + 0.6 * r2, brate)
            g.flow_bias[:, 0] = -0.2
            g.flow_bias[:, 1] = -0.9
            g.flow_speed = 0.45
        else:
            g.flow_bias[:] = (0.55, 0.25)
            g.flow_speed = 1.0
        g.tgt = tgt
        g.btgt = bt
        g.rate = rate
        g.brate = brate
        g.mode = mode

    # ---- her thoughts
    def _thoughts(self, t, dt, T0, T1, T2, T3, T4, S1, S2, S3, S4):
        g = self.T
        n = g.n
        r1, r2, r3 = g.r1, g.r2, g.r3
        p1 = ssf((t - T0) / S1)
        if t < T1:
            mul = 1.0 + 1.0 * p1
        elif t < T2:
            mul = 2.0 - 1.6 * ssf((t - T1) / (0.4 * S2))
        else:
            mul = 0.6
        self.t_th += self.t_w * mul * CFG.ORBIT_SPEED * dt
        head = W_HEAD if t < T3 else self.head_left
        rscale = 1.0 - 0.12 * p1
        # scene 2: loops loosen into a sky full of stars
        bS = ss((t - (T1 - 0.1) - r2 * 0.35 * S2) / (0.3 * S2))
        orb = self.orbit(g, head, rscale * (1 + 2.0 * ssf((t - T1) / (0.5 * S2))), 1.0)
        tgt = orb + (self.stars - orb) * bS[:, None]
        # brightness: pulses travelling round the loops
        pulse = 0.55 + 0.45 * np.sin(self.t_th * 2.0 + t * 3.0 + r1 * 2)
        act = ss((t - (T0 + 0.45) - r1 * 0.72 * S1) / 0.5)
        b_orb = act * (0.32 + 0.5 * r3) * pulse
        tw = (r1 < 0.5) * (0.07 + 0.26 * r3) * (0.55 + 0.45 * np.sin(t * (2 + 3 * r1) + r2 * 20))
        bt = b_orb * (1 - bS) + tw * bS
        pal = np.where(bS > 0.5, MOON, g.pal)
        mode = np.zeros(n, np.int8)
        rate = np.full(n, 9.0)
        rate = np.where(bS > 0, 2.0 + 2.0 * r1, rate)
        # each night, some stars fall and pile up on the ground
        t_fallen = np.where(self.t_night >= 0,
                            T1 + S2 * (0.36 + 0.58 * (self.t_night + r3) / max(1, CFG.NIGHTS)), 1e9)
        t_back = T2 - 0.05 * S2 + r1 * 0.40 * S3
        falling = (t >= t_fallen) & (t < t_back)
        mode[falling] = 2
        bt = np.where(falling, 0.16 + 0.16 * r2, bt)
        # scene 3: faint loops return round her head, behind the calendar
        back = t >= t_back
        if back.any():
            orb3 = self.orbit(g, W_HEAD, 0.95, 1.0)
            tgt = np.where(back[:, None], orb3, tgt)
            bt = np.where(back, 0.22 * act * pulse + 0.04, bt)
            rate = np.where(back, 2.5, rate)
            pal = np.where(back, np.where(r2 < 0.7, AMBER, ROSE), pal)

        # scene 4: fragments of words close in on her, again and again
        t_frag = T3 - 0.25 + r1 * 0.9
        fr = t >= t_frag
        if fr.any():
            ftgt, fbright, fpal, distorted = self._fragments(t, T3, S4)
            tgt = np.where(fr[:, None], ftgt, tgt)
            bt = np.where(fr, fbright, bt)
            pal = np.where(fr, fpal, pal)
            rate = np.where(fr, 9.0 + 3.0 * r2, rate)
            g.set_charset(self.cs_frag)
            g.size = np.where(fr, np.where(self.f_edge, 0, 1), g.size)
            ch = self.rng.random(n) < 4.0 * dt
            m = ch & distorted
            if m.any():
                g.char[m] = self.rng.choice(self.cs_hurt, m.sum())
        else:
            g.set_charset(self.cs_thought)
        # ending: scatter outward and fade
        t_sc = T4 + r1 * 0.30 * CFG.OUTRO
        sc = t >= t_sc
        if sc.any():
            mode[sc] = 1
            bt = np.where(sc, 0.0, bt)
            d = g.pos - (self.head_left + (0, 60))
            d /= (np.hypot(d[:, 0], d[:, 1])[:, None] + 1e-6)
            g.flow_bias = d * 1.6
            g.flow_speed = 1.4
        else:
            g.flow_bias[:] = (0.3, 0.6)
            g.flow_speed = 1.0
        g.tgt, g.btgt, g.rate, g.mode = tgt, bt, rate, mode
        g.brate = np.where(sc, 1.6, 3.0)
        g.pal = np.asarray(pal, np.int32)
        g.flicker = 2.2 if t > T3 else 1.6

    def _fragments(self, t, T3, S4):
        g = self.T
        F = self.F
        fid = self.f_id
        p4 = ssf((t - T3) / S4)
        C = self.head_left + np.array([12.0, 70.0])
        cyc = S4 / max(1, CFG.REPLAYS)
        tt = max(0.0, t - T3)
        c = math.floor(tt / cyc)
        u = (tt / cyc) % 1.0
        uf = (u + self.f_rand * 0.35) % 1.0
        approach = np.sin(math.pi * uf) ** 2
        # rearrangement: each replay glides every fragment to a new place
        step = c + ssf(u / 0.32)
        ang = self.f_base + step * 1.15 + 0.12 * np.sin(t * 0.7 + np.arange(F))
        dfar = 190 + 55 * self.f_rand - 25 * p4
        dnear = 100 + 30 * self.f_rand - 25 * p4
        D = dfar - (dfar - dnear) * approach
        cx = C[0] + np.cos(ang) * D
        cy = C[1] + np.sin(ang) * D * 0.92
        cx = np.minimum(cx, 610)                                      # never reach his side
        scale = self.f_scale * (0.85 + 0.35 * approach)
        rot = 0.3 * np.sin(ang * 1.7 + c)
        cr, sr = np.cos(rot)[fid], np.sin(rot)[fid]
        L = self.f_local * scale[fid, None]
        distorted_f = self.f_rand < (0.15 + 0.55 * p4)
        distorted = distorted_f[fid]
        # broken words: split, shear, shiver
        split = (self.f_local[:, 0] > self.f_split[fid]) & distorted
        L = L.copy()
        L[split] += np.array([9.0, -6.0]) * (0.5 + p4)
        L[distorted] += self.rng.normal(0, 0.6 + 1.6 * p4, (distorted.sum(), 2))
        x = L[:, 0] * cr - L[:, 1] * sr + cx[fid]
        y = L[:, 0] * sr + L[:, 1] * cr + cy[fid]
        tgt = np.stack([x, y], 1)
        b = (0.55 + 0.4 * approach[fid]) * np.where(self.f_edge, 1.0, 0.7) * (0.75 + 0.25 * g.r3)
        pal = np.where(distorted, np.where(g.r2 < 0.55, ORANGE, ROSE), np.where(g.r2 < 0.6, IVORY, AMBER))
        return tgt, b, pal, distorted

    # ---- world: dust, moon, ground, calendar, echoes
    def _world(self, t, T0, T1, T2, T3, T4, S1, S2, S3, S4):
        g = self.X
        n = g.n
        r1, r2, r3, r4 = g.r1, g.r2, g.r3, g.r4
        idx = np.arange(n)
        is_moon = idx < self.n_moon
        dust = self.x_dust + np.stack([np.sin(t * 0.2 + r1 * 7), np.cos(t * 0.17 + r2 * 7)], 1) * 14
        tgt = dust.copy()
        bt = np.where(r3 < 0.22, 0.05 + 0.04 * r1, 0.0) * (1 - 0.6 * ssf((t - T0) / S1))
        pal = np.full(n, ASH)
        rate = np.full(n, 1.5)
        size = self.x_size
        mode = np.zeros(n, np.int8)
        brate = np.full(n, 2.5)
        g.set_charset(self.cs_dust)
        g.lock_chars = None

        # scene 2 — crescent moon and a thin ground line
        if t >= T1 - 0.2:
            a = ss((t - (T1 - 0.1) - r1 * 0.35 * S2) / (0.25 * S2))
            tm = np.where(is_moon[:, None], self.moon_pts, self.ground_pts)
            tgt = dust + (tm - dust) * a[:, None]
            nightp = (t - T1) / S2 * CFG.NIGHTS
            d = MOON_R * (0.28 + 0.55 * (0.5 - 0.5 * math.cos(2 * math.pi * nightp)))
            drift = np.array([math.sin(nightp * 2 * math.pi) * 6, 0.0])
            shadow = MOON_C + drift + d * np.array([0.8, -0.6])
            dist = np.hypot(g.pos[:, 0] - shadow[0], g.pos[:, 1] - shadow[1])
            lit = ss((dist - MOON_R * 0.97) / (MOON_R * 0.10))
            mb = (0.05 + 0.85 * lit) * (0.8 + 0.2 * r2)
            bt = np.where(is_moon, mb, 0.10 + 0.08 * r2) * a + bt * (1 - a)
            pal = np.where(is_moon, MOON, ASH)
            rate = np.where(is_moon, 2.5 + 2 * r2, 1.8)
            g.set_charset(self.cs_moon)

        # scene 3 — the calendar of years
        if t >= T2 - 0.3:
            page, sx, frac, u = self.flip_state(t)
            cal = self.cal_pts.copy()
            cal[self.n_line:] = self.year_pts[page]
            cal = self.cal_transform(cal, sx)
            a = ss((t - (T2 - 0.3) - r1 * 0.5) / 0.6)
            tgt = np.where(a[:, None] > 0, cal, tgt)
            calb = (0.30 + 0.45 * self.cal_w) * (0.45 + 0.55 * sx)
            calb[self.n_line:] = (0.26 + 0.22 * r2[self.n_line:]) * (0.45 + 0.55 * sx)
            bt = np.where(a > 0, calb * a, bt)
            pal = np.where(a > 0, self.cal_pal, pal)
            size = np.where(a > 0, self.cal_size, size)
            rate = np.where(a > 0, np.where(t < T2 + 0.6, 5.0 + 4.0 * r2, 22.0 + 6.0 * r2), rate)
            # line chars follow orientation, year chars are digits
            lock = np.zeros(n, bool)
            lock[: self.n_line] = True
            g.char[: self.n_line] = self.cal_char
            g.lock_chars = lock
            g.set_charset(self.year_digits)
            self._dates(t, page, sx, frac, u)
        else:
            self.dates = []

        # end of scene 3 — calendar dissolves into drifting particles
        t_dis = T3 - 0.30 * S3 + r1 * 0.30 * S3
        dis = t >= t_dis
        if dis.any():
            mode[dis] = 1
            bt = np.where(dis, 0.25 * (0.6 + 0.4 * r2), bt)
            pal = np.where(dis, np.where(r3 < 0.5, AMBER, pal), pal)
            g.flow_bias[:] = (-0.9, 0.3)
            g.lock_chars = None
            g.set_charset(self.cs_thought)
            if t >= T3:
                self.dates = []

        # scene 4 — echoes of her silhouette thicken around her
        t_echo = T3 + 0.04 * S4 + r2 * 0.5 * S4
        ec = t >= t_echo
        if ec.any():
            mode[ec] = 0
            Cw = self.head_left + np.array([30.0, 140.0])
            src = self.w_left[self.e_src]
            k = self.e_k * (1 + 0.035 * np.sin(t * 1.6 + r1 * 6))
            ang = 0.06 * np.sin(t * 0.45 + self.e_k * 4)
            v = src - Cw
            ca, sa = np.cos(ang), np.sin(ang)
            v = np.stack([v[:, 0] * ca - v[:, 1] * sa, v[:, 0] * sa + v[:, 1] * ca], 1)
            et = Cw + v * k[:, None]
            et[:, 0] = np.minimum(et[:, 0], 600)
            tgt = np.where(ec[:, None], et, tgt)
            eb = 0.55 * np.clip((1.56 - self.e_k) / 0.5, 0.05, 1.0) * (0.6 + 0.4 * r3)
            bt = np.where(ec, eb, bt)
            pal = np.where(ec, self.e_pal, pal)
            rate = np.where(ec, 2.6 + 1.6 * r3, rate)
            g.set_charset(self.cs_thought)
        # ending
        t_sc = T4 + 0.1 + r1 * 0.32 * CFG.OUTRO
        sc = t >= t_sc
        if sc.any():
            mode[sc] = 1
            bt = np.where(sc, 0.0, bt)
            brate = np.where(sc, 1.5, brate)
            d = g.pos - (self.head_left + (0, 100))
            d /= (np.hypot(d[:, 0], d[:, 1])[:, None] + 1e-6)
            g.flow_bias = d * 1.3
        g.tgt, g.btgt, g.rate, g.mode, g.brate = tgt, bt, rate, mode, brate
        g.pal = np.asarray(pal, np.int32)
        g.size = np.asarray(size, np.int32)

    def _dates(self, t, page, sx, frac, u):
        """Day numbers inside the calendar cells (also characters)."""
        T2, T3 = self.B[2], self.B[3]
        fade = ssf((t - (T2 + 0.2)) / 0.5) * (1 - ssf((t - (T3 - 0.3 * (T3 - T2))) / 0.6))
        if fade <= 0.01:
            self.dates = []
            return
        year = self.years[page]
        start = (year * 5 + 3) % 7
        ndays = 28 + (year * 7) % 4
        gr = self.grid
        x0 = CAL[0]
        cursor = int(((frac + 0.5) % 1.0) * ndays) if 0 < u < 1 else ndays
        pts, items = [], []
        for dday in range(ndays):
            cell = start + dday
            r, c = divmod(cell, 7)
            if r > 5:
                r -= 1
            pts.append((x0 + c * gr["cw"] + 14, gr["gy0"] + r * gr["ch"] + 16))
            items.append(dday)
        q = self.cal_transform(np.array(pts, float), sx)
        out = []
        for (x, y), dday in zip(q, items):
            if dday == cursor:
                b, pal = 1.0, AMBER
            elif dday < cursor:
                b, pal = 0.22, GREEN
            else:
                b, pal = 0.50, IVORY
            out.append((str(dday + 1), x, y, b * fade * (0.4 + 0.6 * sx), pal))
        self.dates = out

    # ---- the man, far away, walking out of her story
    def _man(self, t, T3, T4, S4):
        g = self.M
        n = g.n
        t_in = T3 + 0.06 * S4
        walk_t = max(0.0, t - t_in)
        phi = walk_t * 2 * math.pi * 0.85
        x = self.man_x0 + walk_t * CFG.MAN_WALK_SPEED
        pts = self.rig.points(phi, np.array([x, GROUND]), self.man_scale)
        fade = ssf((t - (T3 + 0.45 * S4)) / (0.55 * S4))             # losing detail
        blur = 0.3 + 7.0 * fade * fade
        pts = pts + self.rng.normal(0, 1, (n, 2)) * blur * (0.4 + g.r2[:, None])
        a = ss((t - t_in - g.r1 * 0.9) / 0.7)
        tgt = self.man_scatter + (pts - self.man_scatter) * a[:, None]
        keep = g.r3 > fade * 0.85                                     # fewer particles
        b = a * self.rig.depth * (0.55 - 0.25 * fade) * (0.7 + 0.3 * g.r4) * keep
        b = b * (1 - ssf((t - (T4 - 0.2)) / 1.4))
        g.tgt, g.btgt = tgt, b
        g.rate = np.full(n, 6.0)
        g.brate = np.full(n, 3.0)

    # ---------------------------------------------------------------- render
    def camera(self, t):
        T0, T1, T2, T3, T4 = self.B
        z = (1.0 + 0.03 * ssf((t - T0) / (T1 - T0)) - 0.035 * ssf((t - T1) / (0.6 * (T2 - T1)))
             + 0.005 * ssf((t - T2) / 1.0) + 0.045 * ssf((t - T3) / (T4 - T3))
             - 0.02 * ssf((t - T4) / CFG.OUTRO))
        pan = np.array([28.0 * ssf((t - T3) / (T4 - T3)), -8.0 * ssf((t - T0) / (T1 - T0))])
        return z, pan

    def glyph(self, k):
        s = self.cache[k]
        if s is None:
            lv = k % self.NB
            rest = k // self.NB
            pal = rest % self.NP
            rest //= self.NP
            size = rest % self.NS
            ch = rest // self.NS
            v = (lv / (self.NB - 1)) ** 1.25
            col = tuple(int(c * v) for c in CFG.PALETTE[pal])
            s = self.fonts[size].render(self.CH[ch], True, col, (0, 0, 0)).convert()
            self.cache[k] = s
        return s

    def draw_group(self, g, z, pan):
        vis = g.bright > 0.035
        if not vis.any():
            return
        lv = np.minimum((g.bright[vis] * (self.NB - 1) + 0.5).astype(np.int64), self.NB - 1)
        ok = lv > 0
        if not ok.any():
            return
        lv = lv[ok]
        ch = g.char[vis][ok]
        sz = g.size[vis][ok]
        pal = g.pal[vis][ok]
        p = g.pos[vis][ok]
        keys = ((ch * self.NS + sz) * self.NP + pal) * self.NB + lv
        scr = ((p - 500.0) * z + 500.0 + pan) * self.S + self.off - self.goff[sz]
        cache = self.cache
        for k in np.unique(keys).tolist():
            if cache[k] is None:
                self.glyph(k)
        flag = pygame.BLEND_RGB_ADD
        xs = scr[:, 0].astype(np.int32).tolist()
        ys = scr[:, 1].astype(np.int32).tolist()
        self.canvas.blits([(cache[k], (x, y), None, flag)
                           for k, x, y in zip(keys.tolist(), xs, ys)], doreturn=False)

    def draw_dates(self, z, pan):
        flag = pygame.BLEND_RGB_ADD
        for text, x, y, b, pal in self.dates:
            lv = int(min(self.NB - 1, b * (self.NB - 1) + 0.5))
            if lv <= 0:
                continue
            key = (text, pal, lv)
            s = self.date_cache.get(key)
            if s is None:
                v = (lv / (self.NB - 1)) ** 1.25
                col = tuple(int(c * v) for c in CFG.PALETTE[pal])
                s = self.date_font.render(text, True, col, (0, 0, 0)).convert()
                self.date_cache[key] = s
            sx = ((x - 500) * z + 500 + pan[0]) * self.S + self.off[0]
            sy = ((y - 500) * z + 500 + pan[1]) * self.S + self.off[1]
            self.canvas.blit(s, (int(sx), int(sy)), special_flags=flag)

    def render(self, dt, fps=0.0):
        t = self.t
        if CFG.TRAIL > 0:
            f = int(255 * math.exp(-dt / CFG.TRAIL))
            self.canvas.fill((f, f, f), special_flags=pygame.BLEND_RGB_MULT)
            bg = CFG.BACKGROUND
            self.canvas.fill(bg, special_flags=pygame.BLEND_RGB_MAX)
        else:
            self.canvas.fill(CFG.BACKGROUND)
        z, pan = self.camera(t)
        for g in self.groups:
            self.draw_group(g, z, pan)
        self.draw_dates(z, pan)
        self.screen.blit(self.canvas, (0, 0))
        if self.vign is not None:
            self.screen.blit(self.vign, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
        if self.show_lyrics and self.lyric_surfs:
            self._draw_lyric(t)
        if self.show_hud:
            self._draw_hud(t, fps)

    def _draw_lyric(self, t):
        W, H = self.screen.get_size()
        for i, s in enumerate(self.lyric_surfs):
            a, b = self.B[i], self.B[i + 1]
            if a - 0.3 <= t <= b + 0.3:
                al = ssf((t - (a - 0.3)) / 0.5) * (1 - ssf((t - (b - 0.2)) / 0.5))
                s.set_alpha(int(200 * al))
                self.screen.blit(s, ((W - s.get_width()) // 2, int(H * 0.93)))

    def _draw_hud(self, t, fps):
        lines = [f"t = {t:6.2f}s   scene: {self.scene}   fps: {fps:4.1f}",
                 "segments: " + "  ".join(f"{self.B[i]:.1f}-{self.B[i+1]:.1f}" for i in range(4)),
                 "SPACE pause · R restart · H hud · L lyrics · ESC quit"]
        for i, l in enumerate(lines):
            self.screen.blit(self.hud_font.render(l, True, (120, 116, 108)), (10, 8 + i * 16))


# =============================================================================
# 7. AUDIO (optional) + MAIN LOOP
# =============================================================================
class Audio:
    def __init__(self):
        self.ok = False
        self.started = False
        if not CFG.AUDIO_FILE:
            return
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(CFG.AUDIO_FILE)
            self.ok = True
        except Exception as e:
            print("Audio disabled:", e)

    def restart(self):
        self.started = False
        if self.ok:
            pygame.mixer.music.stop()

    def tick(self, t):
        if not self.ok or self.started:
            return
        start_at = CFG.INTRO - CFG.AUDIO_LYRIC_START      # film time when audio sec 0 plays
        if t >= start_at:
            try:
                pygame.mixer.music.play(start=max(0.0, t - start_at))
            except Exception:
                pygame.mixer.music.play()
            self.started = True

    def pause(self, p):
        if self.ok and self.started:
            pygame.mixer.music.pause() if p else pygame.mixer.music.unpause()


def main():
    pygame.init()
    screen = pygame.display.set_mode((CFG.WIDTH, CFG.HEIGHT))
    pygame.display.set_caption(CFG.WINDOW_TITLE)
    screen.fill(CFG.BACKGROUND)
    pygame.display.flip()
    film = Film(screen)
    audio = Audio()
    clock = pygame.time.Clock()
    t0 = time.perf_counter()
    paused, pause_at, paused_total = False, 0.0, 0.0
    t_prev = 0.0
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                    if paused:
                        pause_at = time.perf_counter()
                    else:
                        paused_total += time.perf_counter() - pause_at
                    audio.pause(paused)
                elif e.key == pygame.K_r:
                    film.reset()
                    audio.restart()
                    t0, paused, paused_total, t_prev = time.perf_counter(), False, 0.0, 0.0
                elif e.key == pygame.K_h:
                    film.show_hud = not film.show_hud
                elif e.key == pygame.K_l:
                    film.show_lyrics = not film.show_lyrics
        clock.tick(CFG.FPS)
        if not paused:
            t = (time.perf_counter() - t0 - paused_total) * CFG.SPEED
            if t > film.total:
                if CFG.LOOP:
                    film.reset()
                    audio.restart()
                    t0, paused_total, t_prev, t = time.perf_counter(), 0.0, 0.0, 0.0
                else:
                    t = film.total
            dt = min(0.05, max(0.0, t - t_prev))
            t_prev = t
            audio.tick(t)
            film.update(t, dt)
            film.render(dt, clock.get_fps())
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
