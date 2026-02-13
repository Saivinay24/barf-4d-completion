# Core Pipeline — Vinay (R0)

This folder contains the integration code that connects all components.

## What Goes Here

- `architecture.md` — System design and technical approach
- `pipeline.py` — Main integration pipeline
- `temporal_smooth.py` — Temporal consistency implementation
- `gap_detection.py` — Gap detection algorithm
- Integration tests and utilities

## Current Status

Coming soon. See my notes for integration tasks.

## Architecture Overview

```
video → 4D reconstruction → gap detection → novel view generation → 
temporal consistency → merge → output
```

Each component is built by a different team member, integrated here.
