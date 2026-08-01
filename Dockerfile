# =============================================================================
# Dockerfile — offline file-mode evaluation only.
#
# This image deliberately does NOT support microphone capture or Ultravox
# playback: a container has no audio device, and there is no MPS on Linux, so
# the encoder and separator would both fall back to CPU. PortAudio is therefore
# not installed, and config/container.yaml pins both devices to cpu and runs
# gatekeeper mode. audio/capture.py imports sounddevice lazily, so file mode
# works fine without it.
#
# Supported uses:
#   docker build -t puresignal .
#
#   # run the pipeline over a mounted WAV
#   docker run --rm -v /data/audio:/data puresignal \
#       python src/main.py --config config/container.yaml \
#              --source /data/recording.wav --no-ultravox
#
#   # run the test suite
#   docker run --rm puresignal pytest tests/ -v
# =============================================================================
FROM python:3.11-slim

# libsndfile / ffmpeg for file I/O. No libportaudio2 — see the header.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/
COPY eval/ ./eval/
COPY tests/ ./tests/

# Create runtime directories
RUN mkdir -p logs eval/results profiles

# CMD rather than ENTRYPOINT: the previous ENTRYPOINT ended in "--source", which
# forced the first user argument to be a source path and made it impossible to
# run the tests or the eval scripts in this image.
CMD ["python", "src/main.py", "--config", "config/container.yaml", \
     "--source", "/data/recording.wav", "--no-ultravox"]
