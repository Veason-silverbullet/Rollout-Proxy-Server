"""Run the live vllm_generate transport tests."""

import asyncio
import os

# common selects its fixtures at import time.
os.environ["INFERENCE_ENGINE"] = "vllm"
os.environ["TRANSPORT_MODE"] = "vllm_generate"

from common import run_generate_gauntlet
from proxyserver.rollout_provider import VLLMRolloutProvider


async def main() -> None:
    await run_generate_gauntlet(VLLMRolloutProvider, "vllm_generate")


if __name__ == "__main__":
    asyncio.run(main())
