"""Text-only stop boundaries; sampled token IDs and logprobs stay intact."""


def stop_text(text: str, stops: tuple[str, ...]) -> str:
    """Exclude the first completed stop and any remainder of its final token."""
    matches = ((pos + len(stop), index, pos) for index, stop in enumerate(stops)
               if stop and (pos := text.find(stop)) >= 0)
    match = min(matches, default=None)
    return text[:match[2]] if match is not None else text


def special_token_ids(tokenizer) -> set[int]:
    # Fast tokenizers can omit special added tokens from all_special_ids.
    return set(getattr(tokenizer, "all_special_ids", ())) | {
        token for token, added in getattr(tokenizer, "added_tokens_decoder", {}).items() if added.special
    }


def stop_boundary(token_ids, tokenizer, text, limit, skip_special_tokens, specials):
    """Locate a character cut without breaking the tokenizer's decoder state."""
    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is not None:
        from tokenizers.decoders import DecodeStream

        decoder = DecodeStream(skip_special_tokens=skip_special_tokens)
    emitted, before = "", 0
    for pos, token in enumerate(token_ids):
        if backend is not None:
            delta = decoder.step(backend, token) or ""
        else:
            # Custom/slow tokenizers expose only decode. Defer incomplete UTF-8
            # until the prefix stabilizes, as incremental engine decoding does.
            prefix = tokenizer.decode(token_ids[:pos + 1], skip_special_tokens=skip_special_tokens)
            delta = prefix[len(emitted):] if prefix.startswith(emitted) and not prefix.endswith("�") else ""
        if pos == len(token_ids) - 1:
            # SGLang can match an unfinished byte sequence's replacement
            # character in its raw decode even when DecodeStream buffers it.
            delta = text[len(emitted):]
        if len(emitted) + len(delta) > limit:
            offset = limit - len(emitted)
            return before, pos + 1, delta[:offset], delta[offset:]
        emitted += delta
        if delta or (before == pos and skip_special_tokens and token in specials):
            before = pos + 1
    raise ValueError("Tokenizer did not expose the matched stop boundary")


def decode_stopped(token_ids: list[int], tokenizer, stops: tuple[str, ...], *,
                   skip_special_tokens: bool, stop_skip_special_tokens: bool) -> tuple[str, str]:
    """Project one engine stop boundary into parser text and raw reasoning text."""
    engine_text = tokenizer.decode(token_ids, skip_special_tokens=stop_skip_special_tokens)
    visible = stop_text(engine_text, stops)
    if len(visible) == len(engine_text):
        raw = tokenizer.decode(token_ids, skip_special_tokens=False)
        content = tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        return content, raw

    specials = special_token_ids(tokenizer)
    before, end, fragment, suffix = stop_boundary(
        token_ids, tokenizer, engine_text, len(visible), stop_skip_special_tokens, specials,
    )

    def project(skip):
        if skip == stop_skip_special_tokens:
            return visible
        prefix = tokenizer.decode(token_ids[:before], skip_special_tokens=skip)
        if not fragment:
            return prefix
        if not stop_skip_special_tokens and token_ids[end - 1] in specials:
            # A cut inside a special marker leaves literal text, not a token
            # that can be dropped from the parser view.
            return prefix + fragment
        decoded = tokenizer.decode(token_ids[:end], skip_special_tokens=skip)
        if decoded.endswith(suffix):
            return decoded[:-len(suffix)]
        # An interrupted byte sequence may render differently in the two
        # views. Keep only the engine-visible portion of this partial group.
        return prefix + fragment

    return project(skip_special_tokens), project(False)
