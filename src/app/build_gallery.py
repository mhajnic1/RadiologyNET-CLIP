import os

import torch

from src.evaluation.embed import load_trained_model, compute_split_embeddings

CHECKPOINT = 'checkpoints/tuning/v2_head_lr.pt'
OUT = 'data/embeddings/gallery_all.pt'


def main(checkpoint=CHECKPOINT, out=OUT, batch_size=96, num_workers=4):
    # embeds every row in the dataset, not just one split, so the demo app can search
    # the whole thing. the split label rides along in the metadata so the app can still
    # narrow to test-only, which is the honest setting since the model never saw those
    model, preprocess_val, tokenizer = load_trained_model(checkpoint)

    image_embeds, text_embeds, metadata = compute_split_embeddings(
        split=None, model=model, preprocess_val=preprocess_val, tokenizer=tokenizer,
        batch_size=batch_size, num_workers=num_workers,
    )

    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({'image_embeds': image_embeds, 'text_embeds': text_embeds,
                'metadata': metadata, 'checkpoint': checkpoint}, out)

    print(f'\nwrote {out}')
    print(f'  images: {image_embeds.shape}, text: {text_embeds.shape}')
    print(f'  rows per split:\n{metadata["split"].value_counts().to_string()}')
    print(f'  unique diagnoses: {metadata.DIAGNOSIS_TRUNCATED.nunique()}')
    print(f'  file size: {os.path.getsize(out)/1e6:.1f} MB')


if __name__ == '__main__':
    main()
