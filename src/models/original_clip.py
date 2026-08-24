import open_clip

# quickgelu, not plain ViT-B-16. openai trained the original CLIP with QuickGELU and
# open_clip warns about the mismatch if you ask for the plain variant with these weights,
# which would quietly give a slightly wrong model and an unfair baseline
MODEL_NAME = 'ViT-B-16-quickgelu'
PRETRAINED = 'openai'

# same image tower size as BiomedCLIP (ViT-B/16), so the comparison isolates the
# pretraining domain rather than confounding it with architecture
CONTEXT_LENGTH = 77


def load_original_clip(pretrained=PRETRAINED):
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, preprocess_train, preprocess_val, tokenizer


def set_freeze_mode(model, mode: str, n_unfrozen_blocks: int = 2):
    """Same three modes as the BiomedCLIP version, mapped onto CLIP's layout.

    BiomedCLIP is a CustomTextCLIP (timm trunk + HF BERT), so it has model.visual.trunk
    and model.text.transformer. The original is a plain CLIP: model.visual holds the
    vision tower and model.transformer is the text tower at the top level.
    """
    for p in model.parameters():
        p.requires_grad = True

    if mode == 'full':
        return model

    # freeze both trunks plus the text embeddings
    for p in model.visual.parameters():
        p.requires_grad = False
    for p in model.transformer.parameters():
        p.requires_grad = False
    model.token_embedding.weight.requires_grad = False
    model.positional_embedding.requires_grad = False

    # projections, final norms and the temperature are the "heads" and stay trainable,
    # matching what frozen_backbone means for BiomedCLIP
    model.visual.proj.requires_grad = True
    for p in model.visual.ln_post.parameters():
        p.requires_grad = True
    model.text_projection.requires_grad = True
    for p in model.ln_final.parameters():
        p.requires_grad = True
    model.logit_scale.requires_grad = True

    if mode == 'frozen_backbone':
        return model

    if mode == 'partial':
        for block in model.visual.transformer.resblocks[-n_unfrozen_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        for block in model.transformer.resblocks[-n_unfrozen_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        return model

    raise ValueError(f"unknown freeze mode: {mode}")


# prefixes that count as "backbone" for the layer-wise learning rate split in training
BACKBONE_PREFIXES = ('visual.transformer', 'visual.conv1', 'visual.class_embedding',
                     'visual.positional_embedding', 'visual.ln_pre',
                     'transformer.', 'token_embedding', 'positional_embedding')
