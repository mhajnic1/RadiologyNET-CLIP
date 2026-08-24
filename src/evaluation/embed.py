import os

import torch
from torch.utils.data import DataLoader

from src.data.dataset import RadiologyNETDataset
from src.models.biomedclip import load_biomedclip


def load_trained_model(checkpoint_path='checkpoints/best.pt', base_model='biomedclip'):
    if base_model == 'clip':
        from src.models.original_clip import load_original_clip
        model, _, preprocess_val, tokenizer = load_original_clip()
    else:
        model, _, preprocess_val, tokenizer = load_biomedclip()
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict']

    # newer checkpoints only store the trainable tensors, the rest are already the
    # pretrained weights load_biomedclip just gave us
    if ckpt.get('partial_state'):
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected, f'checkpoint has keys the model does not: {unexpected[:5]}'
        assert len(state) > 0, 'checkpoint had no weights in it'
    else:
        model.load_state_dict(state)

    model = model.cuda().eval()
    return model, preprocess_val, tokenizer


def load_zeroshot_model():
    # pretrained weights only, no checkpoint - for the phase 8 baseline
    model, _, preprocess_val, tokenizer = load_biomedclip()
    model = model.cuda().eval()
    return model, preprocess_val, tokenizer


def load_zeroshot_original_clip():
    # the general-purpose CLIP openai released, no medical pretraining and no
    # fine-tuning. the other end of the comparison from BiomedCLIP
    from src.models.original_clip import load_original_clip
    model, _, preprocess_val, tokenizer = load_original_clip()
    model = model.cuda().eval()
    return model, preprocess_val, tokenizer


@torch.no_grad()
def compute_split_embeddings(split, model, preprocess_val, tokenizer,
                              data_root='data', images_root='data/images', batch_size=96,
                              num_workers=0, text_column='DIAGNOSIS_TRUNCATED'):
    ds = RadiologyNETDataset(split=split, data_root=data_root, images_root=images_root,
                              tokenizer=tokenizer, image_transform=preprocess_val,
                              random_slice=False, text_column=text_column)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    image_embeds = []
    text_embeds = []
    for images, texts in loader:
        images, texts = images.cuda(), texts.cuda()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            img_f = model.encode_image(images, normalize=True)
            txt_f = model.encode_text(texts, normalize=True)
        image_embeds.append(img_f.float().cpu())
        text_embeds.append(txt_f.float().cpu())

    image_embeds = torch.cat(image_embeds)
    text_embeds = torch.cat(text_embeds)
    metadata = ds.rows.reset_index().rename(columns={'index': 'id'})

    return image_embeds, text_embeds, metadata


def get_split_embeddings(split, model, preprocess_val, tokenizer, cache_name,
                          data_root='data', images_root='data/images', batch_size=96,
                          cache_dir='data/embeddings',
                          text_column='DIAGNOSIS_TRUNCATED'):
    # cached under data/, not results/, since these are derived from real diagnosis
    # text and images, same sensitivity as the rest of data/
    cache_path = os.path.join(cache_dir, f'{cache_name}_{split}.pt')
    if os.path.exists(cache_path):
        cached = torch.load(cache_path, weights_only=False)
        return cached['image_embeds'], cached['text_embeds'], cached['metadata']

    image_embeds, text_embeds, metadata = compute_split_embeddings(
        split, model, preprocess_val, tokenizer, data_root, images_root, batch_size,
        text_column=text_column,
    )
    os.makedirs(cache_dir, exist_ok=True)
    torch.save({'image_embeds': image_embeds, 'text_embeds': text_embeds, 'metadata': metadata}, cache_path)
    return image_embeds, text_embeds, metadata
