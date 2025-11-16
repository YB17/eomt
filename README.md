# Your ViT is Secretly an Image Segmentation Model  
**CVPR 2025 · Highlight Paper**

**[Tommie Kerssies](https://tommiekerssies.com)¹, [Niccolò Cavagnero](https://scholar.google.com/citations?user=Pr4XHRAAAAAJ)²\*, [Alexander Hermans](https://scholar.google.de/citations?user=V0iMeYsAAAAJ)³, [Narges Norouzi](https://scholar.google.com/citations?user=q7sm490AAAAJ)¹, [Giuseppe Averta](https://www.giuseppeaverta.me/)², [Bastian Leibe](https://scholar.google.com/citations?user=ZcULDB0AAAAJ)³, [Gijs Dubbelman](https://scholar.google.nl/citations?user=wy57br8AAAAJ)¹, [Daan de Geus](https://ddegeus.github.io)¹,³**

¹ Eindhoven University of Technology  
² Polytechnic of Turin  
³ RWTH Aachen University  
\* Work done while visiting RWTH Aachen University

📄 **Paper:** [arXiv:2503.19108](https://arxiv.org/abs/2503.19108)

## Overview

We present the **Encoder-only Mask Transformer (EoMT)**, a minimalist image segmentation model that repurposes a plain Vision Transformer (ViT) to jointly encode image patches and segmentation queries as tokens. No adapters. No decoders. Just the ViT.

Leveraging large-scale pre-trained ViTs, EoMT achieves accuracy similar to state-of-the-art methods that rely on complex, task-specific components. At the same time, it is significantly faster thanks to its simplicity, for example up to 4× faster with ViT-L.  

Turns out, *your ViT is secretly an image segmentation model*. EoMT shows that architectural complexity isn’t necessary. For segmentation, a plain Transformer is all you need.

## Installation

If you don't have Conda installed, install Miniconda and restart your shell:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Then create the environment, activate it, and install the dependencies:

```bash
conda create -n eomt python==3.13.2
conda activate eomt
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements_extra.txt
```

[Weights & Biases](https://wandb.ai/) (wandb) is used for experiment logging and visualization. To enable wandb, log in to your account:

```bash
wandb login
```

## Data preparation

Download the datasets below depending on which datasets you plan to use.  
You do **not** need to unzip any of the downloaded files.  
Simply place them in a directory of your choice and provide that path via the `--data.path` argument.  
The code will read the `.zip` files directly.

**COCO**
```bash
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
```

**ADE20K**
```bash
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
wget http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar
tar -xf annotations_instance.tar
zip -r -0 annotations_instance.zip annotations_instance/
rm -rf annotations_instance.tar
rm -rf annotations_instance
```

**Cityscapes**
```bash
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=<your_username>&password=<your_password>&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
```

🔧 Replace `<your_username>` and `<your_password>` with your actual [Cityscapes](https://www.cityscapes-dataset.com/) login credentials.  

## Usage

### Training

To train EoMT from scratch, run:

```bash
python3 main.py fit \
  -c configs/coco/panoptic/eomt_large_640.yaml \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset
```

This command trains the `EoMT-L` model with a 640×640 input size on COCO panoptic segmentation using 4 GPUs. Each GPU processes a batch of 4 images, for a total batch size of 16.  

✅ Make sure the total batch size is `devices × batch_size = 16`  
🔧 Replace `/path/to/dataset` with the directory containing the dataset zip files.

> This configuration takes ~6 hours on 4×NVIDIA H100 GPUs, each using ~26GB VRAM.

To fine-tune a pre-trained EoMT model, add:

```bash
  --model.ckpt_path /path/to/pytorch_model.bin \
  --model.load_ckpt_class_head False
```

🔧 Replace `/path/to/pytorch_model.bin` with the path to the checkpoint to fine-tune.  
> `--model.load_ckpt_class_head False` skips loading the classification head when fine-tuning on a dataset with different classes. 

### Evaluating

To evaluate a pre-trained EoMT model, run:

```bash
python3 main.py validate \
  -c configs/coco/panoptic/eomt_large_640.yaml \
  --model.network.masked_attn_enabled False \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset \
  --model.ckpt_path /path/to/pytorch_model.bin
```

This command evaluates the same `EoMT-L` model using 4 GPUs with a batch size of 4 per GPU.

🔧 Replace `/path/to/dataset` with the directory containing the dataset zip files.  
🔧 Replace `/path/to/pytorch_model.bin` with the path to the checkpoint to evaluate.

A [notebook](inference.ipynb) is available for quick inference and visualization with auto-downloaded pre-trained models.

## SigLIP2-EoMT-OVPanoptic

SigLIP2-EoMT keeps the encoder-only Mask Transformer philosophy while swapping the vanilla ViT for SigLIP2’s ViT tower (MAP pooling bypassed). Image patch tokens and segmentation queries share the very same encoder blocks; masked attention is trained with polynomial annealing (`P_mask: 1 → 0` with factor `0.9`) and disabled during inference. The final representation fans out into two light-weight heads:

* a mask projector that upsamples query embeddings back to the patch grid, and
* an open-vocabulary head that performs temperature-controlled mask pooling followed by similarity against SigLIP2 text embeddings (with optional multilingual prompts and per-class bias calibration).

### Three-stage recipe

1. **Stage A – Frozen backbone warm-up.** Freeze the SigLIP2 vision tower. Train mask/query heads plus the open-vocab classifier with mask annealing enabled. Queries can be initialised from SigLIP2 text embeddings (`QUERY_INIT=text+learnable`) to stabilise optimisation.
2. **Stage B – LoRA fine-tuning.** Enable LoRA on the last 12 transformer blocks with rank 16 for Q/K/V and rank 32 for FFN (`fc1`, `fc2`). Only LoRA parameters remain trainable; the teacher tower provides light feature and ITC distillation.
3. **Optional Stage C – Wider adapters.** Increase FFN rank (e.g. 48) and optionally fuse the last three block outputs before mask projection for extra headroom.

### Prompt templates & synonyms

We follow CLIP-style prompt engineering. English templates:

```
a photo of a {}.
a {} in the scene.
```

Stuff categories use contextual prompts such as `a patch of {}` or `the {} background`. Synonyms help recover alternative phrasings, e.g. `"cell phone" → ["mobile phone", "smartphone"]`, `"couch" → ["sofa"]`. When `OPEN_VOCAB.MULTILINGUAL=true`, the head concatenates Mandarin templates like `"一张关于{}的照片。"` to the prompt set. Open-vocabulary splits (`OPEN_VOCAB_SPLIT=ovp_val`) keep 20 thing classes and 17 stuff classes unseen during training.

### Resource footprint

* **Stage A (frozen backbone):** ~14–16 GB per 512×512 batch of four on modern GPUs.
* **Stage B (LoRA rank 8):** +1.5 GB compared to Stage A.
* **Stage B (LoRA rank 16 + FFN):** +3–3.5 GB with a minor throughput drop (~5%).

### Frequently asked questions

* **Why “NAFlex”?** SigLIP2 ships MAP pooling by default. We bypass that head and interpolate positional encodings so arbitrary resolutions (e.g. 640–1024) remain valid.
* **Which layers receive LoRA?** The default config adapts Q/K/V and FFN (`fc1`/`fc2`) in the last 12 blocks with rank `(16, 32)`. `LAST_N_LAYERS`, `RANK_ATTN`, and `RANK_FFN` expose further control.
* **Stuff taxonomy looks noisy.** Adjust templates, synonyms, or the seen/unseen split in `eomt/data/coco_ov_vocab.py`.
* **Logit calibration.** Use `OPEN_VOCAB.TEMP`, `OPEN_VOCAB.CALIBRATION_BIAS`, and `OPEN_VOCAB.ENERGY_REJECT_THR` to calibrate similarity scores at validation time.

### Training & evaluation commands

```bash
# Stage A – frozen tower warm-up
python tools/train_net.py \
  --config-file configs/coco_panoptic_siglip2_eomt_ov.yaml \
  MODEL.BACKBONE.LORA.ENABLED false \
  OUTPUT_DIR runs/coco_eomt_siglip2_ov/stageA

# Stage B – LoRA finetuning
python tools/train_net.py \
  --config-file configs/coco_panoptic_siglip2_eomt_ov.yaml \
  MODEL.BACKBONE.LORA.ENABLED true \
  MODEL.BACKBONE.LORA.LAST_N_LAYERS 12 \
  MODEL.BACKBONE.LORA.RANK_ATTN 16 \
  MODEL.BACKBONE.LORA.RANK_FFN 32 \
  OUTPUT_DIR runs/coco_eomt_siglip2_ov/stageB

# Stage C – LoRA FFN rank sweep + multi-layer aggregation
python tools/train_net.py \
  --config-file configs/coco_panoptic_siglip2_eomt_ov.yaml \
  MODEL.BACKBONE.LORA.ENABLED true \
  MODEL.BACKBONE.LORA.LAST_N_LAYERS 16 \
  MODEL.BACKBONE.LORA.RANK_ATTN 16 \
  MODEL.BACKBONE.LORA.RANK_FFN 48 \
  OUTPUT_DIR runs/coco_eomt_siglip2_ov/stageC

# Validation
python tools/test_net.py \
  --config-file configs/coco_panoptic_siglip2_eomt_ov.yaml \
  OUTPUT_DIR runs/coco_eomt_siglip2_ov/eval

# Smoke test for the SigLIP2 + Open-Vocab stack
pytest -q tests/test_siglip2_ov.py
```

### Weights & Biases logging

Enable `LOGGING.WANDB.ENABLED` and provide your project metadata to capture Stage A/B training statistics (loss curves, PQ, LR schedulers, dataset sizes, parameter counts, gradients, etc.) in Weights & Biases. The integration automatically uploads:

* the resolved YAML configuration alongside derived training metadata (steps per epoch, warmup schedule, etc.);
* dataset statistics (train/val counts, resolution, batch size, worker count);
* model statistics (trainable vs. frozen parameter totals, stuff/thing class counts);
* gradient and parameter norms through `wandb.watch` (disable via `LOGGING.WANDB.WATCH.ENABLED false`).

Example Stage A run that logs to W&B:

```bash
export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
python tools/train_net.py \
  --config-file configs/coco_panoptic_siglip2_eomt_ov.yaml \
  MODEL.BACKBONE.LORA.ENABLED false \
  MODEL.BACKBONE.FREEZE true \
  OUTPUT_DIR runs/coco_eomt_siglip2_ov/stageA \
  LOGGING.WANDB.ENABLED true \
  LOGGING.WANDB.PROJECT eomt-panoptic \
  LOGGING.WANDB.NAME stageA-freeze \
  LOGGING.WANDB.TAGS [stageA,coco]
```

Optional knobs include `LOGGING.WANDB.GROUP`, `LOGGING.WANDB.NOTES`, `LOGGING.WANDB.MODE`, and `LOGGING.WANDB.RESUME` for distributed or offline workflows.

**Results placeholder.** Full COCO panoptic metrics (`PQ_all`, `PQ_th`, `PQ_st`, `PQ_unseen`) are logged during training; populate the table below after running Stage B/Stage C on your hardware.

## Model Zoo

> All FPS values were measured on an NVIDIA H100 GPU.

### Panoptic Segmentation

#### COCO

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">PQ</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 640x640 -->
<tr><td align="left"><a href="configs/coco_panoptic_eomt_large_640.yaml">EoMT-L</a></td>
<td align="center">640x640</td>
<td align="center">128</td>
<td align="center">56.0</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_large_640/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-L 1280x1280 -->
<tr><td align="left"><a href="configs/coco_panoptic_eomt_large_1280.yaml">EoMT-L</a></td>
<td align="center">1280x1280</td>
<td align="center">30</td>
<td align="center">58.3</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_large_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-g 640x640 -->
<tr><td align="left"><a href="configs/coco_panoptic_eomt_giant_640.yaml">EoMT-g</a></td>
<td align="center">640x640</td>
<td align="center">55</td>
<td align="center">57.0</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_giant_640/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-g 1280x1280 -->
<tr><td align="left"><a href="configs/coco_panoptic_eomt_giant_1280.yaml">EoMT-g</a></td>
<td align="center">1280x1280</td>
<td align="center">12</td>
<td align="center">59.2</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_giant_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

#### ADE20K

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">PQ</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 640x640 -->
<tr><td align="left"><a href="configs/ade20k_panoptic_eomt_large_640.yaml">EoMT-L</a></td>
<td align="center">640x640</td>
<td align="center">128</td>
<td align="center">50.6<sup>C</sup></td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_panoptic_eomt_large_640/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-L 1280x1280 -->
<tr><td align="left"><a href="configs/ade20k_panoptic_eomt_large_1280.yaml">EoMT-L</a></td>
<td align="center">1280x1280</td>
<td align="center">30</td>
<td align="center">51.7<sup>C</sup></td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_panoptic_eomt_large_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-g 640x640 -->
<tr><td align="left"><a href="configs/ade20k_panoptic_eomt_giant_640.yaml">EoMT-g</a></td>
<td align="center">640x640</td>
<td align="center">55</td>
<td align="center">51.3<sup>C</sup></td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_panoptic_eomt_giant_640/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-g 1280x1280 -->
<tr><td align="left"><a href="configs/ade20k_panoptic_eomt_giant_1280.yaml">EoMT-g</a></td>
<td align="center">1280x1280</td>
<td align="center">12</td>
<td align="center">52.8<sup>C</sup></td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_panoptic_eomt_giant_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

*<sub><sup>C</sup> models pre-trained on COCO panoptic segmentation. See above for how to load a checkpoint.</sub>*

### Semantic Segmentation

#### Cityscapes

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 1024x1024 -->
<tr><td align="left"><a href="configs/cityscapes_semantic_eomt_large_1024.yaml">EoMT-L</a></td>
<td align="center">1024x1024</td>
<td align="center">25</td>
<td align="center">84.2</td>
<td align="center"><a href="https://huggingface.co/tue-mps/cityscapes_semantic_eomt_large_1024/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

#### ADE20K

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 512x512 -->
<tr><td align="left"><a href="configs/ade20k_semantic_eomt_large_512.yaml">EoMT-L</a></td>
<td align="center">512x512</td>
<td align="center">92</td>
<td align="center">58.4</td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_semantic_eomt_large_512/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

### Instance Segmentation

#### COCO

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">mAP</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 640x640 -->
<tr><td align="left"><a href="configs/coco_instance_eomt_large_640.yaml">EoMT-L</a></td>
<td align="center">640x640</td>
<td align="center">128</td>
<td align="center">45.2*</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_instance_eomt_large_640/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-L 1280x1280 -->
<tr><td align="left"><a href="configs/coco_instance_eomt_large_1280.yaml">EoMT-L</a></td>
<td align="center">1280x1280</td>
<td align="center">30</td>
<td align="center">48.8*</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_instance_eomt_large_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

*<sub>\* mAP reported using pycocotools; TorchMetrics (used by default) yields ~0.7 lower.</sub>*

## Citation
If you find this work useful in your research, please cite it using the BibTeX entry below:

```BibTeX
@inproceedings{kerssies2025eomt,
  author    = {Kerssies, Tommie and Cavagnero, Niccolò and Hermans, Alexander and Norouzi, Narges and Averta, Giuseppe and Leibe, Bastian and Dubbelman, Gijs and de Geus, Daan},
  title     = {Your ViT is Secretly an Image Segmentation Model},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
}
```

## Acknowledgements

This project builds upon code from the following libraries and repositories:

- [Hugging Face Transformers](https://github.com/huggingface/transformers) (Apache-2.0 License)  
- [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) (Apache-2.0 License)  
- [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) (Apache-2.0 License)  
- [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) (Apache-2.0 License)  
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) (Apache-2.0 License)
- [Detectron2](https://github.com/facebookresearch/detectron2) (Apache-2.0 License)

### 可复制的一键运行示例命令（贴到 README 的最后）

```bash
# 0) 安装
pip install -r requirements.txt
pip install -r requirements_extra.txt

# 1) 冻塔热身（Stage A）
python tools/train_net.py \
  --config-file configs/coco_panoptic_siglip2_eomt_ov.yaml \
  MODEL.BACKBONE.LORA.ENABLED false \
  OUTPUT_DIR <OUTPUT_DIR>/stageA

# 2) LoRA 微调（Stage B）
python tools/train_net.py \
  --config-file configs/coco_panoptic_siglip2_eomt_ov.yaml \
  MODEL.BACKBONE.LORA.ENABLED true \
  MODEL.BACKBONE.LORA.RANK 8 \
  MODEL.BACKBONE.LORA.LAYERS_LAST_N 8 \
  LOSS.DISTILL.FEAT_ALIGN 0.001 \
  LOSS.DISTILL.ITC_WEIGHT 0.05 \
  OUTPUT_DIR <OUTPUT_DIR>/stageB

# 3) 验证
python tools/test_net.py \
  --config-file configs/coco_panoptic_siglip2_eomt_ov.yaml \
  OUTPUT_DIR <OUTPUT_DIR>/eval
```
