"""Generate heightfield + sun/sky and patch scene.xml.

Examples:
    # gentle olive grove at noon
    python gen_terrain.py --slope 15 --roughness 0.08 --softness 0.2 --grip 0.7 --sun-elev 75

    # golden hour, sun low in the west
    python gen_terrain.py --sun-elev 10 --sun-azim 180

    # nighttime
    python gen_terrain.py --sun-elev -10

    # muddy rainy field, overcast (sun high but dim)
    python gen_terrain.py --softness 0.8 --grip 0.5 --sun-elev 60 --brightness 0.4

Terrain (0..1 unless noted):
    --slope         percent rise/run (0..60+)
    --roughness     bump amplitude in METERS (0=flat, 0.05=pebbly, 0.20=rocky)
    --bump-scale    bump wavelength in grid cells
    --softness      0=concrete, 1=mud
    --grip          0=ice, 1=sticky soil

Sun/sky:
    --sun-azim      degrees: 0=+X (east), 90=+Y (north), 180=-X (west), 270=-Y (south)
    --sun-elev      degrees: 90=zenith, 0=horizon, <0=below horizon (night)
    --brightness    0..1 multiplier on top of time-of-day base (default 1)
"""
import argparse
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom as sz_zoom
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--slope", type=float, default=0.0)
p.add_argument("--roughness", type=float, default=0.10)
p.add_argument("--bump-scale", type=float, default=20.0)
p.add_argument("--softness", type=float, default=0.3)
p.add_argument("--grip", type=float, default=0.7)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--sun-azim", type=float, default=135.0, help="degrees, 0=+X")
p.add_argument("--sun-elev", type=float, default=60.0, help="degrees, 90=zenith, <0=night")
p.add_argument("--brightness", type=float, default=1.0)
args = p.parse_args()

softness = max(0.0, min(1.0, args.softness))
grip = max(0.0, min(1.0, args.grip))
brightness = max(0.0, min(1.5, args.brightness))

SCENE = Path("data/scene.xml")
PNG = Path("data/terrain.png")
RES = 256
HALF_EXTENT = 5.0

# ---------- terrain heightfield ----------
rng = np.random.default_rng(args.seed)
base = rng.random((RES, RES))
sigma = max(1.0, args.bump_scale)
bumps = (
    0.7 * gaussian_filter(base, sigma=sigma)
  + 0.2 * gaussian_filter(base, sigma=sigma / 4)
  + 0.1 * gaussian_filter(base, sigma=1)
)
bumps = (bumps - bumps.min()) / (bumps.max() - bumps.min())
bumps *= args.roughness

tilt_total = (args.slope / 100.0) * (2 * HALF_EXTENT)
ramp = np.linspace(0, tilt_total, RES)[None, :]
heights = bumps + ramp

zmax = float(heights.max() - heights.min())
if zmax < 1e-6:
    zmax = 1e-3
norm = (heights - heights.min()) / zmax
Image.fromarray((norm * 255).astype(np.uint8), mode="L").save(PNG)

# ---------- top-surface-only OBJ mesh (visual geom, no sides/bottom) ----------
OBJ = Path("data/terrain_surface.obj")
MESH_RES = 128
norm_mesh = sz_zoom(norm, MESH_RES / RES, order=1)
Δ = 2 * HALF_EXTENT / (MESH_RES - 1)
with OBJ.open("w") as f:
    for i in range(MESH_RES):
        for j in range(MESH_RES):
            x = -HALF_EXTENT + j * Δ
            y =  HALF_EXTENT - i * Δ
            z = float(norm_mesh[i, j]) * zmax
            # vertex normal from heightfield gradient (smooth shading)
            dzdj = float(norm_mesh[i, min(j+1,MESH_RES-1)] - norm_mesh[i, max(j-1,0)]) * zmax / (2*Δ if 0<j<MESH_RES-1 else Δ)
            dzdi = float(norm_mesh[max(i-1,0), j] - norm_mesh[min(i+1,MESH_RES-1), j]) * zmax / (2*Δ if 0<i<MESH_RES-1 else Δ)
            nl = (dzdj**2 + dzdi**2 + 1) ** 0.5
            f.write(f"v  {x:.5f} {y:.5f} {z:.5f}\n")
            f.write(f"vn {-dzdj/nl:.5f} {-dzdi/nl:.5f} {1/nl:.5f}\n")
    for i in range(MESH_RES - 1):
        for j in range(MESH_RES - 1):
            v00 = i * MESH_RES + j + 1
            v10 = (i+1) * MESH_RES + j + 1
            v01 = i * MESH_RES + (j+1) + 1
            v11 = (i+1) * MESH_RES + (j+1) + 1
            # top side (CCW from above → normals up)
            f.write(f"f {v00}//{v00} {v10}//{v10} {v11}//{v11}\n")
            f.write(f"f {v00}//{v00} {v11}//{v11} {v01}//{v01}\n")
            # bottom side (reversed winding so back-face culling doesn't hide it)
            f.write(f"f {v00}//{v00} {v11}//{v11} {v10}//{v10}\n")
            f.write(f"f {v00}//{v00} {v01}//{v01} {v11}//{v11}\n")

# ---------- spawn z so robot lands on top of terrain at (0,0) ----------
# world-space terrain height at hfield pixel (i,j) is norm[i,j] * zmax
# sample a small window around centre to be robust to local bumps
c = RES // 2
win = max(1, int(0.3 / (2 * HALF_EXTENT) * RES))   # ~0.3 m radius
patch = norm[c - win:c + win + 1, c - win:c + win + 1]
terrain_at_spawn = float(patch.max()) * zmax
# torso half-thickness 0.05 + limbs extend ~0.35m; 0.50m clears even steep slopes
spawn_z = terrain_at_spawn + 0.50

# ---------- contact physics ----------
# d0   = stiffness at zero penetration (gentle initial contact)
# dmax = asymptotic stiffness — kept near 1.0 so the surface stays impenetrable
#        no matter how soft the cushion is; "softness" only stretches the
#        cushion layer (width), it does not let the wall yield indefinitely
# width = depth of the soft cushion in metres (5mm hard → 30mm mud)
d0    = 0.85
dmax  = 0.999
width = 0.005 + 0.025 * softness
solimp = f"{d0:.3f} {dmax:.3f} {width:.4f}"

# response time: how quickly contact pushes back. Soft soil = a bit slower.
timeconst = 0.005 + 0.020 * softness
solref = f"{timeconst:.4f} 1"

slide = 0.3 + 1.7 * grip
torsional = 0.005 + 0.095 * grip
rolling = 0.0001 + 0.0199 * grip
friction = f"{slide:.3f} {torsional:.4f} {rolling:.5f}"

# ---------- sun direction ----------
# Sun position direction (unit vector from origin towards sun)
az_rad = np.radians(args.sun_azim)
el_rad = np.radians(args.sun_elev)
sun_x = np.cos(el_rad) * np.cos(az_rad)
sun_y = np.cos(el_rad) * np.sin(az_rad)
sun_z = np.sin(el_rad)

# Light direction = from sun towards scene = -sun_pos
light_dir = (-sun_x, -sun_y, -sun_z)
# Place light far away in sun direction (only matters for shadow origin)
light_pos = (sun_x * 10, sun_y * 10, max(sun_z * 10, 0.5))

# ---------- time-of-day palette ----------
# elev_norm: 1 at zenith, 0 at horizon, negative at night
elev_norm = np.sin(el_rad)

def lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))

NIGHT_TOP = (0.02, 0.03, 0.08)
NIGHT_HORIZON = (0.05, 0.06, 0.12)
DAWN_TOP = (0.25, 0.35, 0.55)
DAWN_HORIZON = (0.95, 0.55, 0.30)   # warm orange near horizon
DAY_TOP = (0.35, 0.55, 0.85)
DAY_HORIZON = (0.80, 0.90, 0.98)

NIGHT_LIGHT = (0.05, 0.07, 0.15)    # cool moonlight
DAWN_LIGHT = (1.00, 0.65, 0.40)     # warm orange
DAY_LIGHT = (0.95, 0.92, 0.85)      # warm white

NIGHT_AMBIENT = (0.05, 0.06, 0.10)
DAWN_AMBIENT = (0.30, 0.25, 0.25)
DAY_AMBIENT = (0.40, 0.42, 0.45)

if args.sun_elev < 0:
    # below horizon → night, fade by how far below
    night_depth = min(1.0, -args.sun_elev / 18.0)   # full night by -18°
    sky_top = lerp(DAWN_TOP, NIGHT_TOP, night_depth)
    sky_horizon = lerp(DAWN_HORIZON, NIGHT_HORIZON, night_depth)
    light_rgb = lerp(DAWN_LIGHT, NIGHT_LIGHT, night_depth)
    ambient = lerp(DAWN_AMBIENT, NIGHT_AMBIENT, night_depth)
    light_scale = lerp((0.4, 0.4, 0.4), (0.15, 0.15, 0.2), night_depth)[0]
elif args.sun_elev < 20:
    # golden hour: lerp from horizon palette to dawn-mid
    t = args.sun_elev / 20.0
    sky_top = lerp(DAWN_TOP, DAY_TOP, t)
    sky_horizon = lerp(DAWN_HORIZON, DAY_HORIZON, t)
    light_rgb = lerp(DAWN_LIGHT, DAY_LIGHT, t)
    ambient = lerp(DAWN_AMBIENT, DAY_AMBIENT, t)
    light_scale = 0.7 + 0.3 * t
else:
    # daytime
    t = min(1.0, (args.sun_elev - 20) / 60.0)
    sky_top = DAY_TOP
    sky_horizon = lerp(DAY_HORIZON, (0.65, 0.80, 0.95), t)
    light_rgb = DAY_LIGHT
    ambient = DAY_AMBIENT
    light_scale = 1.0

light_rgb = tuple(c * light_scale * brightness for c in light_rgb)
ambient = tuple(c * brightness for c in ambient)

def fmt(v):
    return " ".join(f"{x:.3f}" for x in v)

# ---------- patch scene.xml ----------
text = SCENE.read_text()

# hfield size
new_size = f'size="{HALF_EXTENT} {HALF_EXTENT} {zmax:.4f} 0.05"'
text, n = re.subn(r'(<hfield\b[^>]*?)\s+size="[^"]*"',
                  lambda m: m.group(1) + " " + new_size, text, count=1)
if n != 1:
    raise RuntimeError("could not patch hfield size")

# floor physics
def patch_floor(s, attr, value):
    return re.subn(
        r'(<geom\b[^>]*\bname="floor"[^>]*?\b' + attr + r'=")[^"]*(")',
        lambda m: m.group(1) + value + m.group(2), s, count=1,
    )

for attr, val in [("friction", friction), ("solimp", solimp), ("solref", solref)]:
    text, k = patch_floor(text, attr, val)
    if k != 1:
        raise RuntimeError(f"could not patch floor {attr}")

# skybox colors (rgb1=top, rgb2=horizon)
text, k = re.subn(
    r'(<texture\b[^>]*\bname="skybox"[^>]*\brgb1=")[^"]*(")',
    lambda m: m.group(1) + fmt(sky_top) + m.group(2), text, count=1)
if k != 1:
    raise RuntimeError("could not patch skybox rgb1")
text, k = re.subn(
    r'(<texture\b[^>]*\bname="skybox"[^>]*\brgb2=")[^"]*(")',
    lambda m: m.group(1) + fmt(sky_horizon) + m.group(2), text, count=1)
if k != 1:
    raise RuntimeError("could not patch skybox rgb2")

# sun light
def patch_sun(s, attr, value):
    return re.subn(
        r'(<light\b[^>]*\bname="sun"[^>]*?\b' + attr + r'=")[^"]*(")',
        lambda m: m.group(1) + value + m.group(2), s, count=1,
    )

text, k = patch_sun(text, "pos", fmt(light_pos))
if k != 1:
    raise RuntimeError("could not patch sun pos")
text, k = patch_sun(text, "dir", fmt(light_dir))
if k != 1:
    raise RuntimeError("could not patch sun dir")
text, k = patch_sun(text, "diffuse", fmt(light_rgb))
if k != 1:
    raise RuntimeError("could not patch sun diffuse")

# headlight ambient (controls overall scene fill)
text, k = re.subn(
    r'(<headlight\b[^>]*\bambient=")[^"]*(")',
    lambda m: m.group(1) + fmt(ambient) + m.group(2), text, count=1)
if k != 1:
    raise RuntimeError("could not patch headlight ambient")

# torso spawn pos — keep robot above terrain at spawn point
text, k = re.subn(
    r'(<body\b[^>]*\bname="torso"[^>]*\bpos=")[^"]*(")',
    lambda m: m.group(1) + f"0 0 {spawn_z:.4f}" + m.group(2), text, count=1)
if k != 1:
    raise RuntimeError("could not patch torso pos")

# keyframe "home" — first 3 numbers of qpos are the torso x y z
text, k = re.subn(
    r'(<key\b[^>]*\bname="home"[^>]*\bqpos=")[^ ]+ [^ ]+ [^ ]+( )',
    lambda m: m.group(1) + f"0 0 {spawn_z:.4f}" + m.group(2), text, count=1)
if k != 1:
    raise RuntimeError("could not patch keyframe home qpos")

SCENE.write_text(text)

angle_deg = np.degrees(np.arctan(args.slope / 100.0))
print(f"Saved {PNG}")
print(f"  slope:      {args.slope}% ({angle_deg:.1f}°)")
print(f"  roughness:  ±{args.roughness} m  bump-scale {sigma:.0f}")
print(f"  softness:   {softness:.2f}  -> solimp={solimp}  solref={solref}")
print(f"  grip:       {grip:.2f}  -> friction={friction}")
print(f"  sun:        azim={args.sun_azim}° elev={args.sun_elev}°  brightness×{brightness}")
print(f"              dir={fmt(light_dir)}  diffuse={fmt(light_rgb)}")
print(f"  sky:        top={fmt(sky_top)}  horizon={fmt(sky_horizon)}")
print(f"  spawn z:    {spawn_z:.3f} m  (terrain at (0,0): {terrain_at_spawn:.3f} m)")
print(f"Patched {SCENE}")
