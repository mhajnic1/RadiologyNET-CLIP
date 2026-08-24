import pandas as pd

# CLIP's context is 77, minus the SOT and EOT tokens leaves 75 usable. phase 3 settled on
# a 25/75 head-to-tail split (64+190 of 254), so the same ratio scaled here is 19+56.
# deliberately the same strategy: otherwise the comparison would confound the tokenizer's
# context length with how cleverly the text was cut
HEAD_N = 19
TAIL_N = 56
COLUMN = 'DIAGNOSIS_TRUNCATED_77'


def clip_head_tail_truncate(tokenizer, text: str, head_n=HEAD_N, tail_n=TAIL_N) -> str:
    """head+tail truncation against open_clip's SimpleTokenizer.

    Can't reuse src/data/text_truncation.py here, that one expects a HuggingFace
    tokenizer (tokenizer(text, truncation=False)['input_ids']). SimpleTokenizer has a
    plain encode/decode pair instead, and adds no special tokens.
    """
    ids = tokenizer.encode(text)
    if len(ids) <= head_n + tail_n:
        return text
    return tokenizer.decode(ids[:head_n] + ids[-tail_n:])


def add_clip_column(path='data/diagnoses_final.csv', head_n=HEAD_N, tail_n=TAIL_N,
                    column=COLUMN):
    import open_clip
    from src.models.original_clip import MODEL_NAME

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    df = pd.read_csv(path, encoding='utf-8-sig')

    lengths = df.DIAGNOSIS_EN.apply(lambda t: len(tokenizer.encode(t)))
    over = int((lengths > head_n + tail_n).sum())
    print(f'diagnoses: {len(df)}, token length median {lengths.median():.0f}, '
          f'max {lengths.max()}')
    print(f'over the {head_n + tail_n} usable token budget: {over} ({over/len(df):.1%})')

    df[column] = df.DIAGNOSIS_EN.apply(
        lambda t: clip_head_tail_truncate(tokenizer, t, head_n, tail_n)
    )
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'wrote column {column} to {path}')
    return df


if __name__ == '__main__':
    add_clip_column()
