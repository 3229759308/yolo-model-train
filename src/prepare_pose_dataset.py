"""Compatibility command; prefer python src/armor.py prepare."""
from armor_pose.data.prepare import main

if __name__ == "__main__":
    raise SystemExit(main())
