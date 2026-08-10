def head_tail_truncate(tokenizer, text: str, head_n: int = 64, tail_n: int = 190) -> str:
    ids = tokenizer(text, truncation=False)['input_ids']
    if len(ids) <= head_n + tail_n:
        return text
    kept = ids[:head_n] + ids[-tail_n:]
    return tokenizer.decode(kept, skip_special_tokens=True)
