import open_clip

MODEL_NAME = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'


def load_biomedclip():
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(MODEL_NAME)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess_train, preprocess_val, tokenizer


def set_freeze_mode(model, mode: str, n_unfrozen_blocks: int = 2):
    for p in model.parameters():
        p.requires_grad = True

    if mode == 'full':
        return model

    for p in model.visual.trunk.parameters():
        p.requires_grad = False
    for p in model.text.transformer.parameters():
        p.requires_grad = False

    if mode == 'frozen_backbone':
        return model

    if mode == 'partial':
        for block in model.visual.trunk.blocks[-n_unfrozen_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        for layer in model.text.transformer.encoder.layer[-n_unfrozen_blocks:]:
            for p in layer.parameters():
                p.requires_grad = True
        return model

    raise ValueError(f"unknown freeze mode: {mode}")
