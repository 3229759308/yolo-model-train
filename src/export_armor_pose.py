"""Compatibility command; prefer python src/armor.py export."""
from armor_pose.inference.export import main

if __name__ == "__main__":
    raise SystemExit(main())
