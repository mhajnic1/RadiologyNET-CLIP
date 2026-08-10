from src.evaluation.embed import compute_split_embeddings
from src.evaluation.retrieval import text_to_image_recall, image_to_text_recall


def recall_score(image_embeds, text_embeds, metadata):
    # one number to rank variants on, averaging both directions so neither gets
    # optimised at the other's expense
    t2i, _ = text_to_image_recall(image_embeds, text_embeds, metadata)
    i2t, _ = image_to_text_recall(image_embeds, text_embeds, metadata)
    return (t2i[10] + i2t[10]) / 2, t2i, i2t


def score_split(model, preprocess_val, tokenizer, split='val',
                data_root='data', images_root='data/images', batch_size=96,
                num_workers=0):
    # ranking variants on the real task metric instead of val_loss, and on val only -
    # test stays untouched so the phase 7/8 numbers keep meaning something
    was_training = model.training
    model.eval()

    image_embeds, text_embeds, metadata = compute_split_embeddings(
        split, model, preprocess_val, tokenizer, data_root, images_root, batch_size,
        num_workers=num_workers,
    )

    if was_training:
        model.train()

    return recall_score(image_embeds, text_embeds, metadata)
