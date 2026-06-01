# =============================================================================
# tests/conftest.py — pytest session bootstrap
#
# Inserts src/ onto sys.path so that all test modules can import pipeline
# packages (config, audio, speaker, utils) without a package install.
# =============================================================================
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
