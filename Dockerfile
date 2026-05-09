FROM python:3.11-slim

# System dependencies for audio (PortAudio) and scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
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

# No mic access in container — file source only.
# Evaluation: point --source at a mounted audio file.
# Example:
#   docker run -v /data/audio:/data puresignal \
#       /data/recording.wav --no-ultravox
ENTRYPOINT ["python", "src/main.py", "--source"]
