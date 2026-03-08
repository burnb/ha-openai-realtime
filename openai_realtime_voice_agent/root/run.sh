#!/bin/bash
set -e

# Validate required configuration
if [ -z "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY is required but not set" >&2
    exit 1
fi

# Start the application
export PYTHONUNBUFFERED=1
exec python3 -m app.main

