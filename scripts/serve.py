#!/usr/bin/env python3
"""Run the API server."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Run log risk scoring API server")
    parser.add_argument("-m", "--model", default="models/v2/model.onnx", help="Path to ONNX model")
    parser.add_argument("-t", "--tokenizer", default="models/v2/tokenizer", help="Path to tokenizer")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--threads", type=int, default=4, help="Number of inference threads")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    import uvicorn
    from src.api.server import create_app

    # Create app with custom settings
    app = create_app(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        num_threads=args.threads,
    )

    print(f"Starting server on {args.host}:{args.port}")
    print(f"Model: {args.model}")
    print(f"Threads: {args.threads}")
    print(f"Docs: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
