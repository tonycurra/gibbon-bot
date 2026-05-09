"""Olive-tree variety presets.

A variety bundles the visual + skeletal defaults for an age class. CLI
flags (--size, --branches, --leaves, --detachment) jitter on top of these.
"""

VARIETIES = {
    # young sapling: thin, mostly straight, sparse canopy
    "young": dict(
        height=1.4,
        base_r=0.07,
        top_r=0.04,
        gnarl=0.08,
        canopy=0.55,
        fol_r=0.30,
        n_branches=3,
    ),
    # mature working tree: typical olive grove specimen
    "mature": dict(
        height=2.4,
        base_r=0.18,
        top_r=0.10,
        gnarl=0.22,
        canopy=0.95,
        fol_r=0.55,
        n_branches=5,
    ),
    # ancient: gnarled, fat trunk, wide canopy
    "ancient": dict(
        height=2.0,
        base_r=0.30,
        top_r=0.18,
        gnarl=0.55,
        canopy=1.10,
        fol_r=0.75,
        n_branches=6,
    ),
    # olive: real Mediterranean olive — squat thick burly multi-trunk
    "olive": dict(
        height=4.5,
        base_r=0.50,
        top_r=0.20,
        gnarl=0.80,
        canopy=1.40,
        fol_r=0.85,
        n_branches=6,
    ),
}
