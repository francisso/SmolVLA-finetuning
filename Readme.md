# SmolVLA-finetuning

## Overview

This repository contains code for fine-tuning SmolVLA (Small Vision-Language-Action) models. SmolVLA is designed for robotics tasks involving vision, language, and action understanding. The project includes scripts for data conversion, training, inference, and visualization.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd SmolVLA-finetuning
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install additional system dependencies:
   - ffmpeg (for video processing)
   - Note: Use transformers==5.3.0 (or compatible version). Newer versions may have compatibility issues.

## Usage

### Data Preparation

- **convert_to_lerobot.py**: Converts datasets to the LeRobot format for training.
  ```bash
  python convert_to_lerobot.py
  ```

- **split.sh**: Splits the dataset into training and validation sets.
  ```bash
  ./split.sh
  ```

### Training

- **train.sh**: Runs the fine-tuning process.
  ```bash
  ./train.sh
  ```

### Inference

- **run_model.py**: Runs inference with the trained model.
  ```bash
  python run_model.py
  ```

### Visualization

- **vis.sh**: Generates visualizations of the model's outputs or training data.
  ```bash
  ./vis.sh
  ```

## Requirements

- Python 3.10+ (works on 3.12.13)
- CUDA-compatible GPU 
- Key Python packages: transformers, lerobot, etc. (see requirements.txt)

