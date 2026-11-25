"""Run the live sglang_generate transport tests."""

import asyncio
import os

# common selects its fixtures at import time.
os.environ["INFERENCE_ENGINE"] = "sglang"
os.environ["TRANSPORT_MODE"] = "sglang_generate"

from common import run_generate_gauntlet
from proxyserver.rollout_provider import SGLangRolloutProvider


async def main() -> None:
    await run_generate_gauntlet(SGLangRolloutProvider, "sglang_generate")


if __name__ == "__main__":
    asyncio.run(main())
