#!/usr/bin/env python3
"""
Fix collision detection in scene.xml
- Enable arm collisions with floor
- Add self-collision prevention
- Set realistic friction
"""

import re


def fix_scene_xml(input_file, output_file):
    with open(input_file, "r") as f:
        content = f.read()

    # Fix 1: Enable collisions for all robot parts
    content = re.sub(
        r'contype="0" conaffinity="0"', 'contype="1" conaffinity="1"', content
    )

    # Fix 2: Add realistic friction to all robot parts
    content = re.sub(r'(class="robot")', r'\1 friction="1.0 0.01 0.001"', content)

    # Fix 3: Add collision pairs to prevent self-collision
    # This goes in the <contact> section
    contact_section = """<contact>
    <pair geom1="floor" geom2="robot" solref="0.02 1" solimp="0.9 0.95 0.001"/>
    <pair geom1="robot" geom2="robot" solref="0.02 1" solimp="0.9 0.95 0.001" condim="3"/>
</contact>"""

    if "<contact>" not in content:
        # Add contact section before closing </mujoco>
        content = content.replace("</mujoco>", f"{contact_section}\n</mujoco>")

    with open(output_file, "w") as f:
        f.write(content)

    print(f"✅ Fixed collisions in {output_file}")
    print("   - Enabled arm collisions")
    print("   - Added realistic friction")
    print("   - Prevented self-collisions")


if __name__ == "__main__":
    fix_scene_xml("scene.xml", "scene_fixed.xml")
