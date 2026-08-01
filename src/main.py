# =============================================================================
# main.py — Pipeline entry point
#
# Usage:
#   python src/main.py [--source mic|path/to/file.wav]
#                      [--config config/base.yaml]
#                      [--set extractor.enabled=false]   # revert to gatekeeper
#                      [--set vad.hangover_ms=300]
#                      [--no-ultravox]
#
# EXTRACTOR MODE (extractor.enabled = true, default):
#   mic → SlidingWindowBuffer → window_queue
#       → parallel Conv-TasNet separation (EXTRACTOR_MAX_WORKERS)
#       → ResequencingBuffer (restore order) → target source selection
#       → new-sample tail → VAD → encoder → tracker → policy → Ultravox
#
# GATEKEEPER MODE (extractor.enabled = false):
#   mic → frame_queue → VAD → encoder → tracker → policy → Ultravox
#
# IMPORTANT — import ordering:
#   Only argparse and config may be imported at module scope.  audio.*,
#   speaker.*, llm.* and pipeline.* all read config constants when they load, so
#   they must not be imported until --config / --set have been applied.
#   config.rebind() raises if that ordering is violated.
# =============================================================================
import argparse
import sys

import config


def _parse_args():
    """Parse and return CLI arguments for the pipeline."""
    p = argparse.ArgumentParser(
        description="PureSignal speaker-ID pipeline v2")
    p.add_argument("--source", default="mic", metavar="mic|PATH")
    p.add_argument("--config", default=None, metavar="PATH")
    p.add_argument("--set", dest="set_args", action="append",
                   default=[], metavar="key=value")
    p.add_argument("--no-ultravox", action="store_true")
    return p.parse_args()


def main():
    """
    Entry point — wire together config, the session, and the pipeline loop.

    Flow:
      1. Parse CLI args and rebind config BEFORE importing the pipeline.
      2. Seed RNG, validate config, select enrolled users.
      3. Run startup() to load models and start audio.
      4. Dispatch to the extractor or gatekeeper loop.
      5. On exit (KeyboardInterrupt or EOF), call shutdown() exactly once.
    """
    args = _parse_args()
    if args.config or args.set_args:
        config.load_and_rebind(path=args.config, set_args=args.set_args)

    # Safe to import the pipeline now that the config is final.
    from pipeline import session
    from pipeline.loops import process_loop_extractor, process_loop_gatekeeper

    session.seed(config.RANDOM_SEED)
    session.validate_config(args.no_ultravox)
    usernames = session.select_users()

    source_arg, no_ultravox = args.source, args.no_ultravox
    target_emb = session.get_target_embedding(
        usernames) if config.EXTRACTOR_ENABLED else None

    source, frame_queue, file_thread, ultravox_thread = session.startup(
        usernames, source_arg, no_ultravox)

    # No custom SIGINT handler: the default raises KeyboardInterrupt, which the
    # try/finally below already handles. A handler calling sys.exit(0) raises
    # SystemExit, which `except KeyboardInterrupt` does not catch, so the finally
    # block ran shutdown a second time on every Ctrl+C.
    try:
        if config.EXTRACTOR_ENABLED:
            process_loop_extractor(target_emb, no_ultravox)
        else:
            process_loop_gatekeeper(no_ultravox)
    except KeyboardInterrupt:
        pass
    finally:
        session.shutdown(source, source_arg, no_ultravox, ultravox_thread)
        if file_thread:
            file_thread.join(timeout=2)

    sys.exit(0)


if __name__ == "__main__":
    main()
