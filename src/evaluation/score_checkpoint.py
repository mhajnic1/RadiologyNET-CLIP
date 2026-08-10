import json
import os
import sys

from src.evaluation.embed import load_trained_model
from src.evaluation.validate import score_split


def main(checkpoint_path, out_path=None, split='val'):
    model, preprocess_val, tokenizer = load_trained_model(checkpoint_path)
    score, t2i, i2t = score_split(model, preprocess_val, tokenizer, split, num_workers=4)

    result = {
        'checkpoint': checkpoint_path,
        'split': split,
        'recall_score': score,
        'text_to_image_recall': t2i,
        'image_to_text_recall': i2t,
    }
    print(json.dumps(result, indent=2))

    if out_path:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == '__main__':
    main(sys.argv[1],
         sys.argv[2] if len(sys.argv) > 2 else None,
         sys.argv[3] if len(sys.argv) > 3 else 'val')
