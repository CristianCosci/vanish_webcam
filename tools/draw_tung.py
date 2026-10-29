"""Draw Tung Tung Tung Sahur (body + swinging bat), the TUNG! burst and Bombardiro Crocodilo.

python tools/draw_tung.py assets/images  (needs the macOS fonts)
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(sys.argv[1])
SS = 2  # Draw at twice the size, then downsample for clean edges.
rng = np.random.default_rng(77)

WOOD = np.array((196, 142, 88), np.float32) / 255.0
LIMB = (150, 98, 56, 255)
LIMB_DARK = (112, 70, 38, 255)
OUTLINE_BROWN = (58, 34, 18)


def P(*values):
    """Scale 1x coordinates to the supersampled canvas."""
    return tuple(v * SS for v in values)


def finish(image, outline, width):
    """Dark cartoon outline around the silhouette, then downsample."""
    rgba = np.asarray(image, np.float32) / 255.0
    alpha = rgba[:, :, 3]
    grown = cv2.dilate((alpha > 0.5).astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * width * SS + 1, 2 * width * SS + 1))).astype(np.float32)
    grown = cv2.GaussianBlur(grown, (0, 0), 0.8 * SS)
    under = np.zeros_like(rgba)
    under[:, :, :3] = np.array(outline, np.float32) / 255.0
    under[:, :, 3] = grown
    out_alpha = alpha + grown * (1 - alpha)
    out = (rgba[:, :, :3] * alpha[:, :, None] + under[:, :, :3] * (grown * (1 - alpha))[:, :, None])
    out = out / np.maximum(out_alpha[:, :, None], 1e-6)
    merged = np.dstack((out, out_alpha))
    image = Image.fromarray((np.clip(merged, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA")
    return image.resize((image.width // SS, image.height // SS), Image.LANCZOS)


def wood_layer(shape, x0, x1, light_bias=0.0, grain_scale=13.0, seed=0):
    """Cylinder-shaded wood with wavy vertical grain, as an RGB float array."""
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    center, radius = (x0 + x1) / 2, (x1 - x0) / 2
    u = np.clip((xx - center) / radius, -1, 1)
    cylinder = np.sqrt(1 - u * u)
    light = 0.50 + 0.55 * cylinder ** 0.8 - 0.14 * u + light_bias
    local = np.random.default_rng(seed)
    wobble = cv2.resize(local.normal(0, 1, (12, 6)).astype(np.float32), (width, height),
                        interpolation=cv2.INTER_CUBIC)
    phase = (xx + 7 * SS * np.sin(yy / (45 * SS) + xx / (60 * SS)) + 4 * SS * wobble) * 2 * np.pi / (grain_scale * SS)
    grain = 0.5 + 0.5 * np.cos(phase)
    light *= 1 - 0.22 * np.clip((grain - 0.82) / 0.18, 0, 1)
    light *= 1 + 0.05 * cv2.resize(local.normal(0, 1, (40, 8)).astype(np.float32), (width, height),
                                   interpolation=cv2.INTER_CUBIC)
    return WOOD[None, None, :] * light[:, :, None]


def tung_body():
    """Log body with face, hanging arm and legs. Floor at (210, 975), shoulder at (332, 450)."""
    w, h = 420, 1000
    image = Image.new("RGBA", P(w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    # Legs and feet behind the log.
    for top, bottom, foot in (((165, 735), (152, 945), (150, 958)), ((262, 735), (272, 945), (292, 958))):
        draw.line(P(*top, *bottom), fill=LIMB, width=30 * SS)
        draw.ellipse(P(foot[0] - 44, foot[1] - 19, foot[0] + 44, foot[1] + 19), fill=LIMB_DARK)
    # The log: shaded cylinder with a cut top.
    body = np.zeros((h * SS, w * SS, 4), np.float32)
    yy, xx = np.mgrid[0:h * SS, 0:w * SS].astype(np.float32)
    x0, x1, top, bottom = 70 * SS, 350 * SS, 60 * SS, 760 * SS
    radius = (x1 - x0) / 2
    cx = (x0 + x1) / 2
    inside = ((xx >= x0) & (xx <= x1) & (yy >= top) & (yy <= bottom)) | (
        ((xx - cx) / radius) ** 2 + ((yy - bottom) / (26 * SS)) ** 2 <= 1)
    body[:, :, :3] = wood_layer((h * SS, w * SS), x0, x1, seed=3)
    body[:, :, 3] = inside
    # Knots.
    for kx, ky, kr in ((128, 540, 15), (300, 655, 11), (236, 470, 8)):
        d = np.sqrt(((xx - kx * SS) / (kr * SS)) ** 2 + ((yy - ky * SS) / (kr * 1.6 * SS)) ** 2)
        ring = (0.5 + 0.5 * np.cos(d * 2 * np.pi * 1.6)) * np.clip(1.6 - d, 0, 1)
        body[:, :, :3] *= (1 - 0.35 * ring)[:, :, None]
    # Cut end on top with growth rings.
    d = np.sqrt(((xx - cx) / (radius - 2 * SS)) ** 2 + ((yy - top) / (24 * SS)) ** 2)
    face = d <= 1
    end = np.array((232, 190, 134), np.float32) / 255.0
    rings = 1 - 0.18 * (0.5 + 0.5 * np.cos(d * 2 * np.pi * 6.5)) - 0.1 * d
    body[face, :3] = end * rings[face][:, None]
    body[face, 3] = 1
    body[(d > 1) & (d < 1.06) & (yy < top + 4 * SS), :3] *= 0.6
    log_image = Image.fromarray((np.clip(body, 0, 1) * 255).astype(np.uint8), "RGBA")
    image.alpha_composite(log_image)
    draw = ImageDraw.Draw(image)
    # Hanging arm on the far side.
    draw.line(P(80, 450, 40, 660), fill=LIMB, width=26 * SS)
    draw.ellipse(P(16, 640, 64, 688), fill=LIMB_DARK)
    draw.line(P(68, 460, 44, 620), fill=(176, 120, 72, 255), width=5 * SS)
    # Face, looking at the victim on the right.
    for ex in (178, 284):
        draw.ellipse(P(ex - 42, 180, ex + 42, 284), fill=(250, 248, 240, 255), outline=OUTLINE_BROWN, width=5 * SS)
        draw.ellipse(P(ex - 6, 216, ex + 34, 256), fill=(18, 14, 12, 255))
        draw.ellipse(P(ex + 4, 222, ex + 16, 234), fill=(255, 255, 255, 255))
    draw.line(P(130, 158, 214, 182), fill=(70, 40, 22, 255), width=15 * SS)
    draw.line(P(250, 182, 330, 156), fill=(70, 40, 22, 255), width=15 * SS)
    draw.ellipse(P(222, 268, 258, 322), fill=(168, 112, 64, 255))
    draw.arc(P(222, 268, 258, 322), 300, 120, fill=(120, 76, 40, 255), width=4 * SS)
    # Wide grin with teeth.
    draw.chord(P(150, 300, 320, 400), 0, 180, fill=(92, 28, 24, 255), outline=OUTLINE_BROWN, width=5 * SS)
    teeth = Image.new("L", image.size, 0)
    ImageDraw.Draw(teeth).chord(P(150, 300, 320, 400), 0, 180, fill=255)
    band = Image.new("L", image.size, 0)
    ImageDraw.Draw(band).rectangle(P(150, 350, 320, 366), fill=255)
    teeth = Image.fromarray((np.asarray(teeth) & np.asarray(band)).astype(np.uint8))
    image.paste((248, 246, 236, 255), (0, 0), teeth)
    for tx in range(178, 300, 22):
        draw.line(P(tx, 350, tx, 366), fill=(150, 140, 120, 255), width=2 * SS)
    return finish(image, OUTLINE_BROWN, 4)


def tung_bat():
    """Arm and baseball bat pointing right; the shoulder pivot is at (30, 60)."""
    w, h = 760, 120
    image = Image.new("RGBA", P(w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.line(P(30, 60, 150, 60), fill=LIMB, width=26 * SS)
    draw.ellipse(P(14, 44, 46, 76), fill=LIMB)
    xs = np.arange(w * SS, dtype=np.float32) / SS
    radius = np.interp(xs, (112, 113, 126, 127, 330, 580, 735, 755), (0, 17, 17, 10, 10, 34, 34, 0))
    end = xs > 735
    radius[end] = 34 * np.sqrt(np.clip(1 - ((xs[end] - 735) / 20) ** 2, 0, 1))
    yy, xx = np.mgrid[0:h * SS, 0:w * SS].astype(np.float32)
    v = (yy / SS - 60) / np.maximum(radius[None, :], 1e-3)
    inside = (np.abs(v) <= 1) & (radius[None, :] > 0)
    light = 0.62 + 0.45 * np.sqrt(np.clip(1 - v * v, 0, 1)) - 0.16 * v
    grain = 0.5 + 0.5 * np.cos((yy / SS * 2.2 + 6 * np.sin(xx / SS / 70)) * 2 * np.pi / 9)
    light *= 1 - 0.08 * np.clip((grain - 0.8) / 0.2, 0, 1)
    color = np.array((236, 204, 150), np.float32) / 255.0
    bat = np.zeros((h * SS, w * SS, 4), np.float32)
    bat[:, :, :3] = color * light[:, :, None]
    bat[:, :, 3] = inside
    image.alpha_composite(Image.fromarray((np.clip(bat, 0, 1) * 255).astype(np.uint8), "RGBA"))
    draw = ImageDraw.Draw(image)
    # Fist around the handle.
    draw.ellipse(P(124, 32, 182, 88), fill=LIMB_DARK)
    for fy in (46, 60, 74):
        draw.arc(P(150, fy - 9, 176, fy + 9), 270, 90, fill=(80, 48, 24, 255), width=3 * SS)
    return finish(image, OUTLINE_BROWN, 4)


def tung_burst():
    """Comic burst with TUNG!"""
    w, h = 380, 270
    image = Image.new("RGBA", P(w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    points = []
    for index in range(28):
        angle = index * 2 * math.pi / 28
        r = (175 if index % 2 == 0 else 118) * rng.uniform(0.9, 1.05)
        points.append(P(190 + r * math.cos(angle) * 0.98, 135 + r * math.sin(angle) * 0.7))
    draw.polygon(points, fill=(255, 224, 40, 255), outline=(220, 60, 30, 255), width=7 * SS)
    inner = [(190 * SS + (x - 190 * SS) * 0.72, 135 * SS + (y - 135 * SS) * 0.72) for x, y in points]
    draw.polygon(inner, fill=(255, 244, 150, 255))
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Impact.ttf", 96 * SS)
    text = Image.new("RGBA", image.size, (0, 0, 0, 0))
    ImageDraw.Draw(text).text(P(190, 138), "TUNG!", font=font, anchor="mm", fill=(255, 255, 255, 255),
                              stroke_width=8 * SS, stroke_fill=(150, 20, 20, 255))
    image.alpha_composite(text.rotate(-8, resample=Image.BICUBIC, center=P(190, 135)))
    return finish(image, (90, 20, 10), 2)


def scales(alpha, cell=(20, 15)):
    """Crocodile skin: irregular rounded scales, each a little lighter or darker."""
    height, width = alpha.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32) / SS
    warp = cv2.resize(rng.normal(0, 1, (24, 60)).astype(np.float32), (width, height),
                      interpolation=cv2.INTER_CUBIC)
    row = np.floor((yy + 2.5 * warp) / cell[1]).astype(int)
    shifted = xx + row * 7.3 + 3.0 * warp
    col = np.floor(shifted / cell[0]).astype(int)
    fx = (shifted % cell[0]) / cell[0]
    fy = ((yy + 2.5 * warp) % cell[1]) / cell[1]
    bump = (np.sin(np.pi * fx) * np.sin(np.pi * fy)) ** 0.5
    tint = rng.uniform(0.88, 1.1, (row.max() + 2, col.max() + 2))[row - row.min(), col - col.min()]
    return (0.80 + 0.26 * bump - 0.06 * (fy - 0.5)) * tint


def bombardiro():
    """Crocodile bomber facing left; the cable hangs from (590, 302)."""
    w, h = 1000, 440
    image = Image.new("RGBA", P(w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    green, dark, belly = (84, 118, 58, 255), (46, 70, 34, 255), (186, 178, 118, 255)
    olive, olive_dark = (104, 110, 62, 255), (70, 76, 40, 255)
    # Far wing and engine, in the shade behind the fuselage.
    draw.polygon([P(500, 178), P(612, 172), P(668, 100), P(612, 98)], fill=olive_dark)
    draw.ellipse(P(522, 124, 612, 158), fill=(66, 70, 48, 255))
    draw.ellipse(P(508, 70, 532, 212), fill=(190, 190, 185, 70))
    # Tail fin and stabiliser.
    draw.polygon([P(796, 196), P(866, 66), P(934, 60), P(940, 204)], fill=green)
    draw.polygon([P(850, 222), P(962, 210), P(978, 230), P(862, 248)], fill=dark)
    # Fuselage: the crocodile's body.
    draw.ellipse(P(200, 148, 860, 302), fill=green)
    draw.polygon([P(760, 182), P(940, 198), P(940, 240), P(760, 266)], fill=green)
    # Head: long snout with an open mouth full of teeth.
    draw.polygon([P(320, 166), P(220, 158), P(120, 170), P(40, 188), P(18, 204), P(30, 222),
                  P(150, 222), P(320, 236)], fill=green)
    draw.polygon([P(320, 248), P(150, 246), P(52, 250), P(34, 262), P(60, 276), P(200, 288),
                  P(320, 292)], fill=green)
    draw.polygon([P(320, 236), P(150, 222), P(30, 222), P(52, 250), P(150, 246), P(320, 248)],
                 fill=(118, 30, 30, 255))
    for tx in range(44, 300, 18):
        y_top = 222 + (tx - 30) * 14 / 290
        draw.polygon([P(tx, y_top - 1), P(tx + 13, y_top - 1), P(tx + 6.5, y_top + 11)], fill=(246, 244, 232, 255))
        y_bottom = 250 - (tx - 30) * 2 / 290
        draw.polygon([P(tx + 7, y_bottom + 1), P(tx + 20, y_bottom + 1), P(tx + 13.5, y_bottom - 10)],
                     fill=(246, 244, 232, 255))
    rgba = np.asarray(image, np.float32) / 255.0
    alpha = rgba[:, :, 3]
    # Skin texture on the green parts, a pale belly underneath.
    greenish = (rgba[:, :, 1] > rgba[:, :, 0] + 0.08) & (alpha > 0.5)
    rgba[:, :, :3] *= np.where(greenish, scales(alpha), 1.0)[:, :, None]
    yy, xx = np.mgrid[0:h * SS, 0:w * SS].astype(np.float32) / SS
    lower = greenish & (yy > 252 + (xx - 520) ** 2 / 9000) & (xx > 300)
    rgba[lower, :3] = rgba[lower, :3] * 0.35 + np.array(belly[:3], np.float32) / 255.0 * 0.75
    shade = np.clip(1.12 - 0.35 * (yy - 150) / 150, 0.75, 1.15)
    rgba[:, :, :3] *= np.where(alpha > 0, shade, 1.0)[:, :, None]
    image = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA")
    draw = ImageDraw.Draw(image)
    # Back scutes along the spine and up the tail fin.
    for x in range(340, 820, 34):
        base = 150 + ((x - 530) / 330) ** 2 * 40
        draw.polygon([P(x, base + 6), P(x + 26, base + 6), P(x + 13, base - 18)], fill=dark)
    for index in range(4):
        fx, fy = 812 + index * 17, 168 - index * 30
        draw.polygon([P(fx, fy), P(fx + 18, fy + 6), P(fx - 2, fy + 22)], fill=dark)
    # Eye on top of the head.
    draw.ellipse(P(200, 124, 274, 172), fill=green, outline=dark, width=3 * SS)
    draw.ellipse(P(220, 130, 256, 164), fill=(244, 204, 40, 255), outline=(40, 30, 10, 255), width=2 * SS)
    draw.ellipse(P(234, 132, 242, 162), fill=(12, 12, 12, 255))
    draw.ellipse(P(226, 136, 233, 143), fill=(255, 255, 255, 255))
    draw.arc(P(198, 112, 280, 160), 200, 340, fill=dark, width=6 * SS)
    draw.ellipse(P(30, 190, 48, 200), fill=(30, 40, 20, 255))
    # Panel lines and rivets of the bomber.
    for px in (430, 600, 750):
        draw.line(P(px, 160 + abs(px - 520) * 0.05, px, 296 - abs(px - 520) * 0.12), fill=(40, 58, 30, 255), width=2 * SS)
        for ry in range(178, 286, 14):
            draw.ellipse(P(px + 5, ry, px + 9, ry + 4), fill=(140, 160, 110, 255))
    # Bomb bay with the cable ring.
    draw.rectangle(P(536, 288, 648, 300), fill=(52, 60, 40, 255), outline=(30, 34, 22, 255), width=2 * SS)
    draw.ellipse(P(582, 296, 598, 310), outline=(60, 60, 64, 255), width=4 * SS)
    # Near wing with its engine and a spinning propeller.
    draw.polygon([P(428, 256), P(612, 250), P(486, 402), P(400, 404)], fill=olive)
    draw.line(P(428, 256, 400, 404), fill=(150, 156, 100, 255), width=5 * SS)
    for lx in (470, 520):
        draw.line(P(lx, 256, lx - 40, 402), fill=olive_dark, width=2 * SS)
    draw.ellipse(P(390, 296, 508, 344), fill=(88, 92, 70, 255))
    draw.ellipse(P(384, 298, 404, 342), fill=(52, 54, 44, 255))
    draw.polygon([P(386, 308), P(360, 320), P(386, 332)], fill=(160, 160, 156, 255))
    prop = Image.new("RGBA", image.size, (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(prop)
    pdraw.ellipse(P(362, 238, 386, 402), fill=(200, 200, 196, 90))
    pdraw.ellipse(P(366, 250, 380, 300), fill=(60, 60, 60, 120))
    pdraw.ellipse(P(368, 340, 378, 392), fill=(60, 60, 60, 120))
    image.alpha_composite(prop)
    return finish(image, (26, 40, 18), 3)


OUT.mkdir(parents=True, exist_ok=True)
for name, sprite in (("tung_body", tung_body()), ("tung_bat", tung_bat()),
                     ("tung_burst", tung_burst()), ("bombardiro", bombardiro())):
    sprite.save(OUT / f"{name}.png", optimize=True)
    print(name, sprite.size, (OUT / f"{name}.png").stat().st_size // 1024, "KB")
