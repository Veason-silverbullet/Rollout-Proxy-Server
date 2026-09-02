"""Offline test of the routed-experts capture path (R3).

What this pins, engine wire to record blob:

* ``_pack_routed_experts`` — SGLang's base64 int32 meta (the format slime's
  own ``decode_int32_meta_array`` reads) repacked to the uint8 record blob,
  and every refusal: a missing payload (engines launched without
  ``enable_return_routed_experts``), a misaligned element count, and ids
  that do not fit uint8.  A refused payload must never be recorded — a
  silently short tensor crashes the trainer's replay pass hours later.
* ``_sglang_generate`` — ``return_routed_experts`` rides the ``/generate``
  body only when asked, and the packed blob rides the returned engine meta.
* ``SlimeRolloutProvider`` — the capture layers (config baseline, runtime
  toggle, effective) behind ``GET/PUT /routed_experts``.

Run:
    python test/test-routed-experts.py
"""

from __future__ import annotations
import asyncio
import base64
import sys
from pathlib import Path
from typing import Any
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from proxyserver.engines import EngineError  # noqa: E402
from proxyserver.rollout_provider import (  # noqa: E402
    SlimeRolloutProvider,
    _pack_routed_experts,
    _sglang_generate,
)
from offline_common import check  # noqa: E402


def sglang_meta(values: list[int]) -> dict[str, Any]:
    """meta_info as the engine sends it: base64 of little-endian int32."""
    return {"routed_experts": base64.b64encode(np.asarray(values, dtype="<i4").tobytes()).decode("ascii")}


def test_pack_round_trip() -> None:
    print("int32 engine payload -> uint8 record blob, losslessly")
    # 3 prompt + 2 completion tokens -> 4 rows; 6 selections per token.
    values = [(i * 37) % 256 for i in range(4 * 6)]
    blob = _pack_routed_experts(sglang_meta(values), 3, 2, "s")

    check("rows = prompt + completion - 1", blob["rows"] == 4)
    check("cols is inferred from the element count", blob["cols"] == 6)
    check("dtype says uint8", blob["dtype"] == "uint8")
    unpacked = np.frombuffer(base64.b64decode(blob["data"]), dtype=np.uint8)
    check("every expert id survives the repack", unpacked.tolist() == values)
    check("the blob is 4x smaller than the wire payload",
          len(base64.b64decode(blob["data"])) * 4
          == len(base64.b64decode(sglang_meta(values)["routed_experts"])))


def test_pack_refusals() -> None:
    print("\nA payload that cannot be replayed is refused, not recorded")
    for label, meta, prompt_len, completion_len, needle in (
        ("a missing payload names the engine flag",
         {}, 3, 2, "enable_return_routed_experts"),
        ("an element count that does not divide the rows is refused",
         sglang_meta([1] * 23), 3, 2, "does not line up"),
        ("an id over 255 cannot be packed as uint8",
         sglang_meta([0] * 23 + [256]), 3, 2, "uint8"),
        ("a negative id is refused too",
         sglang_meta([-1] + [0] * 23), 3, 2, "uint8"),
    ):
        try:
            _pack_routed_experts(meta, prompt_len, completion_len, "s")
        except EngineError as e:
            check(label, needle in str(e))
        else:
            raise AssertionError(f"FAIL: {label}: no EngineError raised")


class _FakeResponse:
    def __init__(self, output: dict[str, Any]):
        self._output = output

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._output


class _FakeHttp:
    """Records the POST body and returns a scripted /generate response."""

    def __init__(self, output: dict[str, Any]):
        self._output = output
        self.bodies: list[dict[str, Any]] = []

    async def post(self, url: str, json: dict[str, Any], headers=None) -> _FakeResponse:
        self.bodies.append(json)
        return _FakeResponse(self._output)


def test_generate_carries_the_flag_and_the_blob() -> None:
    print("\n/generate asks for routed experts only when capture is on")
    values = list(range(4 * 6))
    output = {"output_ids": [900, 901], "meta_info": {**sglang_meta(values), "finish_reason": {"type": "stop"}}}

    http = _FakeHttp(output)
    _, _, _, meta = asyncio.run(_sglang_generate(
        http, [1, 2, 3], {"max_tokens": 8}, "s", return_routed_experts=True))
    check("the request body carries return_routed_experts",
          http.bodies[0].get("return_routed_experts") is True)
    check("the packed blob rides the engine meta",
          meta["routed_experts"]["rows"] == 4 and meta["routed_experts"]["dtype"] == "uint8")

    http = _FakeHttp({"output_ids": [900, 901], "meta_info": {"finish_reason": {"type": "stop"}}})
    _, _, _, meta = asyncio.run(_sglang_generate(http, [1, 2, 3], {"max_tokens": 8}, "s"))
    check("capture off sends no request key", "return_routed_experts" not in http.bodies[0])
    check("and attaches no meta", "routed_experts" not in meta)

    http = _FakeHttp({"output_ids": [900, 901], "meta_info": {"finish_reason": {"type": "stop"}}})
    try:
        asyncio.run(_sglang_generate(http, [1, 2, 3], {"max_tokens": 8}, "s", return_routed_experts=True))
    except EngineError as e:
        check("capture on against engines without the server flag fails the first turn",
              "enable_return_routed_experts" in str(e))
    else:
        raise AssertionError("FAIL: missing routed_experts did not raise")


def test_provider_capture_layers() -> None:
    print("\nThe capture layers behind GET/PUT /routed_experts")
    provider = SlimeRolloutProvider("http://127.0.0.1:1", return_routed_experts=True)
    check("the config baseline is effective by itself",
          provider.get_routed_experts_config() == {"config": True, "runtime": None, "effective": True})
    check("the runtime layer wins while set",
          provider.set_runtime_routed_experts(False)["effective"] is False)
    check("clearing the layer falls back to the config",
          provider.set_runtime_routed_experts(None)["effective"] is True)
    try:
        provider.set_runtime_routed_experts("yes")  # type: ignore[arg-type]
    except ValueError:
        check("a non-bool toggle is refused", True)
    else:
        raise AssertionError("FAIL: non-bool toggle accepted")


def test_pack_delta_and_full_range_fallback() -> None:
    print("\nA delta request packs only the new rows; a full-range answer is sliced")
    # Stream: 5 prompt + 2 completion tokens -> 6 rows in total; the recorder
    # already holds 4 rows x 6 cols, so this turn's delta is rows 4..6.
    full = [(i * 37) % 256 for i in range(6 * 6)]
    delta = full[4 * 6:]

    blob = _pack_routed_experts(sglang_meta(delta), 5, 2, "s", prior=(4, 6))
    check("rows = prompt + completion - 1 - start", blob["rows"] == 2 and blob["start"] == 4)
    check("the delta's ids are packed as-is",
          np.frombuffer(base64.b64decode(blob["data"]), dtype=np.uint8).tolist() == delta)

    blob = _pack_routed_experts(sglang_meta(full), 5, 2, "s", prior=(4, 6))
    check("an engine that ignored start_len is sliced down to the delta",
          blob["rows"] == 2 and blob["start"] == 4
          and np.frombuffer(base64.b64decode(blob["data"]), dtype=np.uint8).tolist() == delta)

    blob = _pack_routed_experts(sglang_meta(full), 5, 2, "s")
    check("no prior: the whole stream, start 0", blob["rows"] == 6 and blob["start"] == 0)

    try:
        _pack_routed_experts(sglang_meta([1] * 10), 5, 2, "s", prior=(4, 6))
    except EngineError as e:
        check("a delta with the wrong column count is refused", "expected 6 cols" in str(e))
    else:
        raise AssertionError("FAIL: mismatched delta did not raise")

    output = {"output_ids": [900, 901], "meta_info": {**sglang_meta(delta), "finish_reason": {"type": "stop"}}}
    http = _FakeHttp(output)
    _, _, _, meta = asyncio.run(_sglang_generate(
        http, [1, 2, 3, 4, 5], {"max_tokens": 8}, "s",
        return_routed_experts=True, routed_experts_prior=(4, 6)))
    check("the request carries routed_experts_start_len = stored rows",
          http.bodies[0].get("routed_experts_start_len") == 4)
    check("and the meta carries the delta blob", meta["routed_experts"]["rows"] == 2)
    http = _FakeHttp({"output_ids": [900, 901], "meta_info": {**sglang_meta(full), "finish_reason": {"type": "stop"}}})
    asyncio.run(_sglang_generate(http, [1, 2, 3, 4, 5], {"max_tokens": 8}, "s", return_routed_experts=True))
    check("no prior sends no start_len", "routed_experts_start_len" not in http.bodies[0])


def main() -> None:
    print("=" * 70)
    print("Routed-experts capture (R3): pack, wire, layers")
    print("=" * 70)
    test_pack_round_trip()
    test_pack_refusals()
    test_pack_delta_and_full_range_fallback()
    test_generate_carries_the_flag_and_the_blob()
    test_provider_capture_layers()
    print("\n" + "=" * 70)
    print("PASS: the engine's int32 routed-experts meta repacks losslessly to\n"
          "      the uint8 record blob; misaligned, out-of-range, or missing\n"
          "      payloads are refused before anything is recorded; /generate\n"
          "      carries the request flag only while capture is effective; and\n"
          "      the provider's config/runtime layers behave like the\n"
          "      sampling-override layers they mirror.")
    print("=" * 70)


if __name__ == "__main__":
    main()
