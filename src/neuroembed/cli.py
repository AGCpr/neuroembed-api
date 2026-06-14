"""Command-line entry point for running the NeuroEmbed API server."""
from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroEmbed API server")
    parser.add_argument("--host", default=os.environ.get("NEUROEMBED_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("NEUROEMBED_PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for dev")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    uvicorn.run(
        "neuroembed.main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
