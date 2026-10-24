"""Compatibility command; prefer python src/armor.py predict."""
from armor_pose.inference.predict import main

if __name__ == "__main__":
    raise SystemExit(main())
