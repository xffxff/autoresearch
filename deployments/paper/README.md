# Paper Deployments

This directory stores approved paper-trading strategy snapshots.

- `current.json` points to the currently active paper version.
- `versions/<version_id>/features.py` is the approved feature snapshot.
- `versions/<version_id>/strategy.py` is the approved strategy snapshot.
- `versions/<version_id>/manifest.json` records approval metadata and file hashes.

These files are part of the audit trail and should remain tracked by git.
