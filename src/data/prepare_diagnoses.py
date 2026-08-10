import pandas as pd

from src.data.text_truncation import head_tail_truncate

ERROR_PATTERNS = ['#N/A', '#VALUE', '#ERROR', '#REF']


def export_for_translation(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates('ExamID')[['ExamID', 'DIAGNOSIS']].rename(columns={'DIAGNOSIS': 'DIAGNOSIS_HR'})


def find_translation_errors(translated: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(False, index=translated.index)
    for pat in ERROR_PATTERNS:
        mask |= translated.DIAGNOSIS_EN.str.contains(pat, case=False, na=False)
    return translated[mask]


def finalize_diagnoses(translated: pd.DataFrame, tokenizer, head_n: int = 64, tail_n: int = 190) -> pd.DataFrame:
    out = translated.copy()
    out['DIAGNOSIS_TRUNCATED'] = out.DIAGNOSIS_EN.apply(
        lambda t: head_tail_truncate(tokenizer, t, head_n=head_n, tail_n=tail_n)
    )
    return out
