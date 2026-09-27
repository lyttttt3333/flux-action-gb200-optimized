#!/usr/bin/env python3
"""Run the serial GPT-6 Astra planner -> FLUX Action WebSocket proxy."""

from __future__ import annotations

import argparse
import asyncio

from flux_action.serving.gpt6_vla import AstraPlanner, FixedPlanner, serve_proxy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", default="ws://127.0.0.1:8000")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max"), default="low")
    parser.add_argument("--image-detail", choices=("low", "high", "auto"), default="high")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--log-path")
    parser.add_argument(
        "--fixed-subtask",
        help="offline plumbing test only: bypass the API and always use this subtask",
    )
    args = parser.parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("jpeg-quality must be in [1, 100]")
    planner = (
        FixedPlanner(args.fixed_subtask)
        if args.fixed_subtask is not None
        else AstraPlanner(
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            reasoning_effort=args.reasoning_effort,
            image_detail=args.image_detail,
            jpeg_quality=args.jpeg_quality,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
    )
    asyncio.run(
        serve_proxy(
            planner=planner,
            upstream=args.upstream,
            host=args.host,
            port=args.port,
            log_path=args.log_path,
        )
    )


if __name__ == "__main__":
    main()
