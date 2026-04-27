# Floor Alignment Issue Analysis

## The Problem

The floor alignment issue was caused by a mismatch between:
1. **Visual floor position** (in XML files) - default at z=0 for plane geoms
2. **Physics floor position** (in Python code) - consistently at FLOOR_Z = 0.04

## Key Findings

### All historical versions had the same issue:
- Visual floor at z=0 (default plane position)
- Physics floor at z=0.04 (FLOOR_Z constant in code)
- This created a 4cm gap between visual and physics floors

### The fix (in scene_fixed_current.xml):
Added explicit position: pos="0 0 0.04" to match FLOOR_Z

## File Comparison

| File | Floor Position | Status |
|------|----------------|--------|
| scene_5287eb3.xml | z=0 (default) | ❌ Misaligned |
| scene_fixed_current.xml | z=0.04 (explicit) | ✅ Fixed |
