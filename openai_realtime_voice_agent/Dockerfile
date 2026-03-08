ARG PYTHON_BASE=python:3.12-slim-bookworm

# Build stage - contains dependency installation
FROM ${PYTHON_BASE} AS builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN \
    python -m pip install --no-cache-dir poetry \
    && poetry config virtualenvs.create false \
    && poetry config virtualenvs.in-project false

# Copy Poetry configuration files and app directory for package installation
COPY pyproject.toml poetry.lock* /tmp/
RUN mkdir -p /tmp/openai_realtime_voice_agent
COPY app/ /tmp/openai_realtime_voice_agent/app/

# Install dependencies using Poetry
WORKDIR /tmp

ENV POETRY_VIRTUALENVS_CREATE=false
RUN poetry install --only=main --no-root --no-interaction --no-ansi

# Runtime stage - minimal image without build tools
FROM ${PYTHON_BASE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        curl \
        ffmpeg \
        libasound2 \
        libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local

# Copy application code
COPY app/ /app/

# Copy root filesystem
COPY root/ /
RUN chmod a+x /run.sh

# Set entrypoint
CMD ["/run.sh"]
