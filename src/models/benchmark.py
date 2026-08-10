import time

import torch
from open_clip.loss import ClipLoss
from torch.utils.data import DataLoader

from src.data.dataset import RadiologyNETDataset
from src.models.biomedclip import load_biomedclip, set_freeze_mode

clip_loss = ClipLoss()


def measure_memory(mode, batch_size, data_root='data', images_root='data/images', amp_dtype=None):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model, preprocess_train, _, tokenizer = load_biomedclip()
    set_freeze_mode(model, mode)
    model = model.cuda().train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    ds = RadiologyNETDataset(split='train', data_root=data_root, images_root=images_root,
                              tokenizer=tokenizer, image_transform=preprocess_train)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    images, texts = next(iter(loader))
    images, texts = images.cuda(), texts.cuda()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    try:
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
            image_features = model.encode_image(images, normalize=True)
            text_features = model.encode_text(texts, normalize=True)
            loss = clip_loss(image_features, text_features, model.logit_scale.exp())
        loss.backward()
        optimizer.step()
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        status = 'ok'
    except torch.cuda.OutOfMemoryError:
        peak_gb = None
        status = 'OOM'

    del model, optimizer, images, texts
    torch.cuda.empty_cache()
    return trainable, total, peak_gb, status


def time_iterations(mode, batch_size, n_iters=5, data_root='data', images_root='data/images', amp_dtype=None):
    model, preprocess_train, _, tokenizer = load_biomedclip()
    set_freeze_mode(model, mode)
    model = model.cuda().train()

    ds = RadiologyNETDataset(split='train', data_root=data_root, images_root=images_root,
                              tokenizer=tokenizer, image_transform=preprocess_train)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)

    def step(images, texts):
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_dtype is not None):
            image_features = model.encode_image(images, normalize=True)
            text_features = model.encode_text(texts, normalize=True)
            loss = clip_loss(image_features, text_features, model.logit_scale.exp())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    it = iter(loader)
    images, texts = next(it)
    images, texts = images.cuda(), texts.cuda()
    step(images, texts)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iters):
        images, texts = next(it)
        images, texts = images.cuda(), texts.cuda()
        torch.cuda.synchronize()
        t0 = time.time()
        step(images, texts)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    del model, optimizer
    torch.cuda.empty_cache()
    avg = sum(times) / len(times)
    return avg, batch_size / avg, avg / batch_size * 1000


if __name__ == '__main__':
    gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}, total VRAM: {gpu_total_gb:.2f} GB\n")

    for mode, bs in [('full', 32), ('frozen_backbone', 32), ('partial', 32),
                      ('partial', 64), ('partial', 96), ('partial', 128),
                      ('frozen_backbone', 128), ('frozen_backbone', 256), ('frozen_backbone', 384)]:
        trainable, total, peak_gb, status = measure_memory(mode, bs)
        peak_str = f"{peak_gb:.2f} GB" if peak_gb else "N/A"
        print(f"mode={mode:16s} batch={bs:3d}  trainable={trainable:>12,} ({trainable/total:.1%})  peak_vram={peak_str:>9s}  {status}")

    print()
    for mode, bs in [('partial', 64), ('partial', 96)]:
        avg, sps, ms_per_sample = time_iterations(mode, bs)
        print(f"mode={mode:16s} batch={bs:3d}  per_sample={ms_per_sample:.2f}ms  samples/sec={sps:.1f}")

    print("\nbf16 amp sweep:")
    for mode, bs in [('partial', 64), ('partial', 96), ('partial', 128), ('partial', 160), ('full', 32)]:
        trainable, total, peak_gb, status = measure_memory(mode, bs, amp_dtype=torch.bfloat16)
        peak_str = f"{peak_gb:.2f} GB" if peak_gb else "N/A"
        print(f"[bf16] mode={mode:16s} batch={bs:3d}  peak_vram={peak_str:>9s}  {status}")

    print()
    for bs in [64, 96, 128, 160]:
        avg, sps, ms_per_sample = time_iterations('partial', bs, amp_dtype=torch.bfloat16)
        print(f"[bf16] partial batch={bs:3d}  per_sample={ms_per_sample:.2f}ms  samples/sec={sps:.1f}")
