# GRPO VLM Research Project

## Overview

This project is a research environment for fine-tuning Vision Language Models (VLMs) using Group Relative Policy Optimization (GRPO). It provides a structured setup for data processing, model training, evaluation, and inference.

## Project Structure

```
.
├── configs/            # Experiment configuration files (e.g., hyperparameters)
├── data/               # Datasets (raw and processed)
├── models/             # Saved model checkpoints (local)
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── src/                # Reusable source code
│   ├── data/           # Data loading and preprocessing
│   ├── evaluation/     # Model evaluation scripts
│   ├── inference/      # Inference scripts
│   └── training/       # Core training logic
├── tests/              # Unit and integration tests
├── .github/workflows/  # CI/CD workflows
├── Dockerfile          # Docker container definition
├── README.md           # This file
└── requirements.txt    # Python dependencies
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd grpo-vlm-research
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    conda create --name venv python=3.10 -y
    conda activate venv
    ```

3.  **Install dependencies:**
    ```