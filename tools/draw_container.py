"""Draw the side of a light-blue 40' high-cube container (12.192 x 2.896 m) as a PNG.

python tools/draw_container.py assets/images/container_40hc.png  (needs the macOS fonts)
"""

import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1]
L, H = 2048, 486
rng = np.random.default_rng(4012)

BASE = np.array((62, 160, 205), np.float32) / 255.0      # RGB, container blue
PAINT = np.array((246, 248, 250), np.float32) / 255.0    # white livery
RUST = np.array((118, 70, 38), np.float32) / 255.0

POST, TOP_RAIL, BOTTOM_RAIL = 30, 18, 26
CAST_W, CAST_H = 30, 20
PANEL_TOP, PANEL_BOTTOM = TOP_RAIL, H - BOTTOM_RAIL


def corrugation(width, period=45.5, oversample=4):
    """Brightness across trapezoidal corrugations lit from the upper left."""
    xs = (np.arange(width * oversample) + 0.5) / oversample
    phase = (xs % period) / period
    shade = np.select(
        [phase < 0.38, phase < 0.50, phase < 0.88],
        [1.00, 0.80, 0.93],
        1.09,
    )
    return shade.reshape(width, oversample).mean(axis=1).astype(np.float32)


# Base colour with corrugations, a soft top-to-bottom light falloff and low-frequency mottling.
shade = np.ones((H, L), np.float32)
shade[PANEL_TOP:PANEL_BOTTOM, POST:L - POST] = corrugation(L - 2 * POST)[None, :]
shade *= np.linspace(1.04, 0.93, H, dtype=np.float32)[:, None]
mottle = cv2.resize(rng.normal(0, 1, (10, 40)).astype(np.float32), (L, H), interpolation=cv2.INTER_CUBIC)
shade *= 1.0 + 0.025 * mottle

# Rails overhang the panel: highlight on top, shadow just below.
shade[:TOP_RAIL] *= 0.97
shade[1:3] *= 1.12
occlusion = np.linspace(0.74, 1.0, 7, dtype=np.float32)
shade[TOP_RAIL:TOP_RAIL + 7, POST:L - POST] *= occlusion[:, None]
shade[PANEL_BOTTOM:] *= 0.86
shade[PANEL_BOTTOM:PANEL_BOTTOM + 2] *= 1.14
shade[PANEL_BOTTOM - 6:PANEL_BOTTOM, POST:L - POST] *= occlusion[::-1][1:][:, None]
# Corner posts.
for x0, inner in ((0, POST - 3), (L - POST, L - POST)):
    shade[:, x0:x0 + POST] *= 0.88
    shade[:, inner:inner + 3] *= 0.70
shade[:, 1:3] *= 1.10
shade[:, L - POST + 4:L - POST + 6] *= 1.08

# Two subtle dents.
yy, xx = np.mgrid[0:H, 0:L].astype(np.float32)
for cx, cy, r, depth in ((560, 300, 38, 0.06), (1610, 190, 30, 0.05)):
    blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))
    shade *= 1.0 - depth * blob * np.tanh((xx - cx) / r)

image = BASE[None, None, :] * shade[:, :, None]

# White livery: seven-pointed star and wordmark, shaded by the same corrugations.
SS = 2
paint = Image.new("L", (L * SS, H * SS), 0)
draw = ImageDraw.Draw(paint)
font = ImageFont.truetype("/System/Library/Fonts/Avenir Next.ttc", 10, index=0)
cap_target = 0.27 * H * SS
font = ImageFont.truetype("/System/Library/Fonts/Avenir Next.ttc",
                          round(10 * cap_target / (font.getbbox("M")[3] - font.getbbox("M")[1])), index=0)
text = "MAERSK"
tracking = 0.07 * cap_target
advances = [font.getlength(ch) for ch in text]
text_width = sum(advances) + tracking * (len(text) - 1)
star_r = 0.70 * cap_target
gap = 0.42 * cap_target
group = 2 * star_r + gap + text_width
left = L * SS / 2 - group / 2
center_y = (PANEL_TOP + PANEL_BOTTOM) / 2 * SS - 0.01 * H * SS
angles = -np.pi / 2 + np.arange(14) * np.pi / 7
radii = np.where(np.arange(14) % 2 == 0, star_r, star_r * 0.43)
star_cx = left + star_r
draw.polygon([(star_cx + r * np.cos(a), center_y + r * np.sin(a)) for r, a in zip(radii, angles)], fill=255)
top = font.getbbox("M")[1]
x = left + 2 * star_r + gap
for ch, advance in zip(text, advances):
    draw.text((x, center_y - cap_target / 2 - top), ch, font=font, fill=255)
    x += advance + tracking


def check_digit(code):
    """ISO 6346 check digit."""
    values = {}
    value = 10
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if value % 11 == 0:
            value += 1
        values[letter] = value
        value += 1
    total = sum((values[c] if c.isalpha() else int(c)) * 2 ** i for i, c in enumerate(code))
    return total % 11 % 10


owner = "MSKU471225"
small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", int(0.075 * H * SS))
line_h = 0.085 * H * SS
right = (L - POST - 64) * SS
y0 = (TOP_RAIL + 22) * SS
ident = f"{owner[:4]} {owner[4:]}"
digit = str(check_digit(owner))
digit_w = small.getlength(digit)
ident_w = small.getlength(ident + "  ")
x0 = right - ident_w - digit_w
draw.text((x0, y0), ident, font=small, fill=255)
box = (right - digit_w - 0.12 * line_h, y0 + 0.05 * line_h, right + 0.12 * line_h, y0 + 0.95 * line_h)
draw.text((right - digit_w, y0), digit, font=small, fill=255)
draw.rectangle(box, outline=255, width=max(2, round(0.06 * line_h)))
draw.text((x0, y0 + line_h * 1.15), "45G1", font=small, fill=255)
draw.text((x0, y0 + line_h * 2.3), "9'6\"", font=small, fill=255)

paint = np.asarray(paint.resize((L, H), Image.LANCZOS), np.float32)[:, :, None] / 255.0
painted = PAINT[None, None, :] * shade[:, :, None] * 0.97
image = image * (1.0 - paint) + painted * paint

# High-cube warning stripes on the top rail at both corners.
stripes = (((xx + yy) // 11) % 2 == 0)
yellow = np.array((245, 196, 0), np.float32) / 255.0
black = np.array((28, 28, 28), np.float32) / 255.0
for x0 in (POST + 8, L - POST - 8 - 118):
    area = np.s_[2:TOP_RAIL - 2, x0:x0 + 118]
    hatch = np.where(stripes[area][:, :, None], yellow, black) * shade[area][:, :, None]
    image[area] = hatch

# Rust drips under the top rail and along the bottom rail, grime near the floor.
for _ in range(34):
    x = POST + 20 + rng.integers(0, (L - 2 * POST - 40) // 45) * 45.5 + 30
    length = rng.uniform(18, 95)
    width = rng.uniform(1.5, 3.5)
    strength = rng.uniform(0.12, 0.32)
    top_y = TOP_RAIL + rng.uniform(0, 4)
    falloff = np.clip(1.0 - (yy - top_y) / length, 0.0, 1.0) * (yy >= top_y)
    streak = np.exp(-((xx - x) ** 2) / (2 * width * width)) * falloff * strength
    image = image * (1.0 - streak[:, :, None]) + RUST * streak[:, :, None]
blotches = cv2.resize(rng.random((3, 160)).astype(np.float32), (L, BOTTOM_RAIL + 14),
                      interpolation=cv2.INTER_CUBIC)
rust = np.zeros((H, L), np.float32)
rust[H - BOTTOM_RAIL - 14:] = np.clip(blotches - 0.66, 0.0, 1.0) * 0.8
for cast_y in (0, H - CAST_H):
    for cast_x in (0, L - CAST_W):
        rust[cast_y:cast_y + CAST_H, cast_x:cast_x + CAST_W] += 0.25
rust = cv2.GaussianBlur(np.clip(rust, 0.0, 0.32), (0, 0), 1.6)
image = image * (1.0 - rust[:, :, None]) + RUST * rust[:, :, None]
image *= np.clip(1.0 - 0.10 * np.clip((yy - H * 0.82) / (H * 0.18), 0.0, 1.0), 0.0, 1.0)[:, :, None]

# Grey corner castings with their oval apertures.
casting = np.array((88, 98, 108), np.float32) / 255.0
for cast_y in (0, H - CAST_H):
    for cast_x in (0, L - CAST_W):
        area = np.s_[cast_y:cast_y + CAST_H, cast_x:cast_x + CAST_W]
        image[area] = image[area] * 0.35 + casting * 0.65
        hole = np.zeros((CAST_H, CAST_W), np.uint8)
        cv2.ellipse(hole, (CAST_W // 2, CAST_H // 2), (7, 4), 0, 0, 360, 255, -1, cv2.LINE_AA)
        hole = hole.astype(np.float32)[:, :, None] / 255.0
        image[area] = image[area] * (1.0 - hole * 0.85)

# Thin dark outline so the box reads against any background.
image[:2] *= 0.6
image[-2:] *= 0.55
image[:, :2] *= 0.6
image[:, -2:] *= 0.6

bgr = cv2.cvtColor(np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
cv2.imwrite(OUT, bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
print(OUT, bgr.shape, "check digit", digit)
