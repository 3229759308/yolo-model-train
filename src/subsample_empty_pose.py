"""Compatibility command; prefer python src/armor.py subsample."""
from armor_pose.data.subsample import main

if __name__ == "__main__":
    raise SystemExit(main())
