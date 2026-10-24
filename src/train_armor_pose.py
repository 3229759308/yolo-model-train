"""Compatibility command; prefer python src/armor.py train."""
from armor_pose.training.engine import main

if __name__ == "__main__":
    raise SystemExit(main())
