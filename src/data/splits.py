import pandas as pd
from sklearn.model_selection import train_test_split


def make_exam_level_splits(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
    exam_ids = df.drop_duplicates('ExamID')[['ExamID', 'Modality']].reset_index(drop=True)

    train_ids, temp_ids = train_test_split(
        exam_ids, test_size=test_size, stratify=exam_ids.Modality, random_state=random_state
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5, stratify=temp_ids.Modality, random_state=random_state
    )

    return pd.concat([
        train_ids.assign(split='train'),
        val_ids.assign(split='val'),
        test_ids.assign(split='test'),
    ])[['ExamID', 'split']]
