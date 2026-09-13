<h1 align="center">Language-Guided Representation Learning<br>for Robust Cross-Sensor Tactile Perception</h1>

<p align="center">
  <img src="https://img.shields.io/badge/IROS_2026-Accepted-164E63" alt="Accepted at IROS 2026">
  <a href="https://github.com/Mashood3624/Language_Tactile/blob/main/environment.yml"><img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&amp;logoColor=white" alt="Python 3.10"></a>
  <a href="https://github.com/Mashood3624/Language_Tactile/blob/main/environment.yml"><img src="https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C?logo=pytorch&amp;logoColor=white" alt="PyTorch 2.0.1"></a>
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/mashood3624/">Mashood M. Mohsan</a> · Muhayy Ud Din · Binzhao Xu · Ahmad Abubakar · Irfan Hussain<br>
  Khalifa University Center for Autonomous Robotic Systems (KUCARS)<br>
  Khalifa University, UAE
</p>

<p align="center">
  <a href="https://www.ku.ac.ae/"><img src="assets/logos/Khalifa_logo.png" alt="Khalifa University" height="48"></a>
  &nbsp;&nbsp;
  <a href="https://touchripe.com/"><img src="assets/logos/TouchRIPE.png" alt="TouchRIPE" height="48"></a>
  &nbsp;&nbsp;
  <a href="https://www.ku.ac.ae/kucars"><img src="assets/logos/KUCARS.jpg" alt="KUCARS" height="48"></a>
  &nbsp;&nbsp;
  <a href="https://www.linkedin.com/company/ihlab/"><img src="assets/logos/ihlab_logo.jpeg" alt="AERIS Lab" height="48"></a>
</p>

<p align="center">
  <a href="https://mashood3624.github.io/Language_Tactile/">Project website</a> ·
  <a href="https://youtu.be/QaMzg2h5LKA">Video</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#dataset">Dataset</a> ·
  <a href="#citation">Citation</a>
</p>

Language descriptions guide a tactile image encoder to learn material representations across sensors. Training has two stages: distill a frozen BART language teacher into a ViT tactile student, then freeze the tactile encoder and train a material classifier. Inference uses tactile images alone.

![Language–tactile distillation and downstream classification](assets/main.png)

## Quick start

### 1. Install the environment

```bash
git clone https://github.com/Mashood3624/Language_Tactile.git
cd Language_Tactile
conda env create -f environment.yml
conda activate Mashood_LT
```

The environment pins Python 3.10, PyTorch 2.0.1 and CUDA 11.7. GPU training requires a compatible NVIDIA driver; CUDA is selected automatically when available.

### 2. Add the dataset

[**Download dataset from here**](https://huggingface.co/datasets/YOUR_HF_USERNAME/Language_Tactile)

Save `dataset.zip` inside `Language_Tactile/` and extract it from the project directory:

```bash
unzip -o dataset.zip
```

The folder layout should be:

```text
Language_Tactile/
├── dataset/
│   ├── images/
│   │   ├── hct/
│   │   └── ssvtp/
│   ├── splits/
│   └── additional/
├── configs/
├── environment.yml
├── train_distillation.py
└── train_fewshot.py
```

The image paths and CSVs are already configured for this layout. No path edits are needed.

### 3. Train in two stages

Run distillation first, then classification:

```bash
python train_distillation.py
python train_fewshot.py
```

Distillation saves the encoder to `weights/distillation/best_weights/`. The second script loads it automatically, freezes the encoder and saves its results to `weights/fewshot/`. Pretrained models download on the first run.

Change training settings in [distillation.json](https://github.com/Mashood3624/Language_Tactile/blob/main/configs/distillation.json) and [fewshot.json](https://github.com/Mashood3624/Language_Tactile/blob/main/configs/fewshot.json).

## Dataset

The four experiment CSVs contain **39,717 tactile–vision pairs**, with language descriptions and material labels:

| Source | Distillation | Few-shot | Total |
| --- | ---: | ---: | ---: |
| TVL/HCT | 27,223 | 8,683 | 35,906 |
| SSVTP | 2,475 | 1,336 | 3,811 |
| **Combined** | **29,698** | **10,019** | **39,717** |

Counts include training and evaluation. Distillation uses **21,035 / 8,663** train/evaluation pairs; classification uses **7,147 / 2,872**.

The `additional/` folder contains **2,462 unused labeled pairs** and **16 pairs requiring label confirmation**, excluded from the experiment splits. The complete download contains **42,195 pairs / 84,390 images**, approximately **14.3 GB**.

## Citation

```bibtex
@inproceedings{mohsan2026language,
  title={Language-Guided Representation Learning for Robust Cross-Sensor Tactile Perception},
  author={Mohsan, Mashood M. and Din, Muhayy Ud and Xu, Binzhao and Abubakar, Ahmad and Hussain, Irfan},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2026}
}
```

## Acknowledgements

Data sources: [TVL/HCT](https://tactile-vlm.github.io/) and [SSVTP](https://sites.google.com/berkeley.edu/ssvtp). Code builds on [MDistiller](https://github.com/megvii-research/mdistiller), [Transformers](https://github.com/huggingface/transformers) and the [distillation example](https://github.com/philschmid/knowledge-distillation-transformers-pytorch-sagemaker).
