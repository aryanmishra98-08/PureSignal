"""
pipeline/ — orchestration split out of main.py.

  session.py  user selection, config validation, startup and shutdown
  stages.py   SegmentPipeline: encoder submission, gating, forwarding
  loops.py    the extractor-mode and gatekeeper-mode consumer loops

Nothing here may be imported before config.rebind() has run — these modules
pull in audio.* and speaker.*, which read config constants at import time.
"""
