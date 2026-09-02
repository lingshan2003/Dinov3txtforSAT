# AutoDL Server Assets

Last updated: 2026-09-02

This document records the assets reported as available on the AutoDL server. It is
an inventory, not a guarantee that every archive has been extracted correctly. Run
the verification commands below after changing files on the server.

## Server layout

- Persistent workspace: `/root/autodl-tmp`
- Project checkout: `/root/autodl-tmp/Dinov3txtforSAT`
- Model assets: `/root/autodl-tmp/Dinov3txtforSAT/assets/checkpoints`
- Raw datasets: `/root/autodl-tmp/Dinov3txtforSAT/assets/data/raw`
- Generated manifests: `/root/autodl-tmp/Dinov3txtforSAT/assets/data/manifests`
- DINOv3 source checkout: `/root/autodl-tmp/Dinov3txtforSAT/external/dinov3`
- Experiment outputs: `/root/autodl-tmp/Dinov3txtforSAT/outputs`

Large assets under `assets/checkpoints`, `assets/data`, `external`, and `outputs`
are intentionally excluded from Git.

## Model assets reported available

The following files have been downloaded and uploaded to the server:

| Asset | Expected project path | Purpose |
| --- | --- | --- |
| DINOv3 ViT-L/16 Web backbone | `assets/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | Web-pretrained visual baseline |
| DINOv3 ViT-L/16 SAT backbone | `assets/checkpoints/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` | Satellite-pretrained visual backbone |
| dino.txt alignment components | `assets/checkpoints/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth` | Vision alignment head and text encoder initialization |
| BPE vocabulary | `assets/checkpoints/bpe_simple_vocab_16e6.txt.gz` | Text tokenizer vocabulary; keep compressed |

## Dataset assets reported available

### ChatEarthNet training data

- Uploaded archive: `s2_rgb_images.zip`
- Expected extraction root: `assets/data/raw/chatearthnet`
- Role: remote-sensing image-text alignment training data
- Required in addition to images: the corresponding ChatEarthNet caption/annotation
  metadata. Its exact filename and extracted path still need to be verified.

### EuroSAT evaluation data

- Uploaded archive: `EuroSAT.zip`
- Expected extraction root: `assets/data/raw/eurosat`
- Role: zero-shot remote-sensing scene classification evaluation
- The extracted class-directory layout and image count still need to be verified.

### RSICD evaluation data

- Uploaded image archive: `RSICD_images.zip`
- Uploaded annotation archive: `annotations_rsicd.rar`
- Annotation archive contains at least `dataset_rsicd.json` and `readme.txt`
- Expected extraction root: `assets/data/raw/rsicd`
- Role: image-to-text and text-to-image retrieval evaluation
- `p7zip 16.02` cannot extract the annotation archive's compression method; use
  `unrar` or `unar` instead.

## Verification commands on AutoDL

Run from the project root:

```bash
cd /root/autodl-tmp/Dinov3txtforSAT

find assets/checkpoints -maxdepth 1 -type f -exec ls -lh {} \;
find assets/data/raw -maxdepth 4 -type d | sort

find assets/data/raw/chatearthnet -type f | wc -l
find assets/data/raw/eurosat -type f | wc -l
find assets/data/raw/rsicd -type f | wc -l
find assets/data/raw/rsicd -name 'dataset_rsicd.json' -print

du -sh assets/checkpoints assets/data/raw/*
```

Before formal experiments, record exact file counts, extracted paths, archive
checksums, and dataset split definitions in this document.

