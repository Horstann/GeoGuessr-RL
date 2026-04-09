# WanderBench: Learning to Wander

### Improving the Global Image Geolocation Ability of LMMs via Actionable Reasoning

<p align="center">
  <a href="https://arxiv.org/abs/2603.10463"><img src="https://img.shields.io/badge/arXiv-2603.10463-b31b1b.svg" alt="arXiv"></a>
  <a href="#"><img src="https://img.shields.io/badge/CVPR%20Findings-2026-4b44ce.svg" alt="CVPR Findings 2026"></a>
  <a href="https://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg" alt="License: CC BY 4.0"></a>
  <a href="https://huggingface.co/datasets/Yushuo-Zheng/WanderBench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow.svg" alt="Dataset"></a>
</p>

> **WanderBench** is a geolocation benchmark containing over 32K panoramas across six continents, organized as navigable graphs that enable interactive exploration. We propose **GeoAoT** (Geolocation with Action of Thought), a framework that couples reasoning with embodied actions such as approaching landmarks or adjusting viewpoints to reduce uncertainty. Experiments across 19 large multimodal models demonstrate that GeoAoT achieves improved localization accuracy and stronger performance in dynamic environments.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [GeoAoT Exploration](#geoaot-exploration)
  - [Baseline (Direct Prediction)](#baseline-direct-prediction)
  - [Batch Geocoding](#batch-geocoding)
- [Configuration](#configuration)
  - [Model Configuration](#model-configuration)
  - [Batch Processing](#batch-processing)
  - [API Keys](#api-keys)
- [Supported Models](#supported-models)
- [Citation](#citation)

## Overview

Traditional image geolocation methods rely on static, single-view prediction. **GeoAoT** instead performs multi-step interactive exploration over a graph of connected panoramas, allowing the model to navigate (move and rotate) to gather more visual evidence before making a location prediction.

```
                    +-----------------+
                    |  Load Panorama  |
                    |   Graph (JSON)  |
                    +--------+--------+
                             |
                    +--------v--------+
                    |  Render Current |
                    |  View + Arrows  |
                    +--------+--------+
                             |
                    +--------v--------+
                    | LMM Observes &  |
              +---->| Decides Action  |
              |     +--------+--------+
              |              |
              |     +--------v--------+
              |     | rotate / move / |
              |     |     guess?      |
              |     +--------+--------+
              |        |           |
              |   rotate/move     guess
              |        |           |
              +--------+  +--------v--------+
                          | Final Location  |
                          |   Prediction    |
                          +-----------------+
```

## Installation

```bash
git clone https://github.com/YushuoZheng/WanderBench.git
cd WanderBench
pip install -r requirements.txt
```

**Dependencies:**

```
hydra-core
omegaconf
openai
httpx
Pillow
numpy
opencv-python
geopy
requests
streetview
```

## Data Preparation

The WanderBench dataset is available on Hugging Face:

**[Yushuo-Zheng/WanderBench](https://huggingface.co/datasets/Yushuo-Zheng/WanderBench)**

```bash
# Download via huggingface-cli
pip install huggingface_hub
huggingface-cli download Yushuo-Zheng/WanderBench --repo-type dataset --local-dir ./data
```

The dataset provides **graph JSON files** that define navigable graphs with node coordinates and adjacency matrices. Each node references a Google Street View panorama by its pano ID. Due to Google's Terms of Service, panorama images are **not** included in the dataset -- they must be fetched locally using the provided pano IDs via the [Google Street View API](https://developers.google.com/maps/documentation/streetview).

**Step 1: Download graph data**

```bash
# Download via huggingface-cli
pip install huggingface_hub
huggingface-cli download Yushuo-Zheng/WanderBench --repo-type dataset --local-dir ./data
```

**Step 2: Fetch panorama images**

Use the `streetview` Python package (or the Google Street View Static API) to download panoramas by pano ID. The code will automatically fetch and cache panorama images at runtime via `utils.py`.

**Step 3: Update config paths**

```yaml
# conf/batch_process/default.yaml
input_graphs_folder: './data'
pano_folder: './pano_images'
```

<details>
<summary>Graph JSON format</summary>

```json
{
  "center_pano_id": "pano_id_123",
  "nodes": [
    {
      "pano_id": "pano_id_123",
      "matrix_id": 0,
      "coordinate": {
        "lat": 40.7128,
        "lon": -74.0060,
        "heading": 1.5708
      }
    }
  ],
  "adjacency_matrix": [
    [-1, 1.57, 0.0],
    [4.71, -1, 1.57],
    [3.14, 4.71, -1]
  ]
}
```

- `nodes`: List of panorama nodes with coordinates (lat/lon in degrees, heading in radians). Each `pano_id` corresponds to a Google Street View panorama.
- `adjacency_matrix`: Directional angles (radians) between connected nodes; `-1` indicates no connection.

</details>

## Usage

### GeoAoT Exploration

Run multi-step interactive geolocation:

```bash
python main.py
```

Override configuration via command line:

```bash
python main.py \
  ai_config=InternVL3-78B-Instruct \
  batch_process=default \
  output_folder=/path/to/output \
  max_steps=4 \
  debug=true
```

- `max_steps`: Maximum navigation steps before forcing a guess (default: 4).
- `debug`: When `true`, saves panorama renders and action logs for each step.

### Baseline (Direct Prediction)

Run single-step direct prediction without exploration:

```bash
python batch_process_baseline.py
```

This uses the same configuration system and output format as GeoAoT, enabling direct comparison.

### Batch Geocoding

Post-process results to convert location descriptions into coordinates:

```bash
python batch_geocoding.py \
  --input_folder /path/to/results \
  --google_api_key YOUR_KEY \
  --method google
```

Use `--method nominatim` for a free alternative via OpenStreetMap.

## Configuration

All configuration is managed via [Hydra](https://hydra.cc/) with YAML files under `conf/`.

### Model Configuration

Edit or create files in `conf/ai_config/`:

```yaml
# conf/ai_config/gemini_flash.yaml
model: "gemini-2.5-flash"
temperature: 0.0
max_tokens: 2000
base_url: "https://your-api-endpoint/v1/"
model_type: "closed_source"  # or "open_source"
```

### Batch Processing

Edit `conf/batch_process/default.yaml`:

```yaml
input_graphs_folder: '/path/to/your/graph/jsons'
pano_folder: '/path/to/your/panoramas'
max_workers: 4
```

### API Keys

Edit `conf/ai_keys/default.yaml`:

```yaml
api_key: "YOUR_API_KEY_HERE"           # For closed-source models
google_api_key: "YOUR_GOOGLE_API_KEY"  # For geocoding
server_keys:                            # For open-source models
  ak: "YOUR_ACCESS_KEY"
  sk: "YOUR_SECRET_KEY"
```

## Supported Models

WanderBench includes configurations for **19+ models** across closed-source and open-source families:

| Family | Models |
|--------|--------|
| **Gemini** | Gemini-2.5-Flash |
| **GPT** | o3 |
| **InternVL2** | 2B, 4B, 8B, 40B |
| **InternVL3** | 8B, 38B, 78B |
| **Qwen2-VL** | 7B, 72B |
| **Qwen2.5-VL** | 7B, 72B |
| **LLaVA-OneVision** | 7B |

Pre-packaged configurations are available under `conf/ai_config/PackToGo/`.

## Citation

If you find WanderBench or GeoAoT useful in your research, please cite our paper:

```bibtex
@misc{zheng2026learningwanderimprovingglobal,
      title={Learning to Wander: Improving the Global Image Geolocation Ability of LMMs via Actionable Reasoning},
      author={Yushuo Zheng and Huiyu Duan and Zicheng Zhang and Xiaohong Liu and Xiongkuo Min},
      year={2026},
      eprint={2603.10463},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.10463},
}
```
