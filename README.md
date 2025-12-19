# Fine-Tuning OpenCLIP on Fashion Product Images

This repository contains code to fine-tune an OpenCLIP (`ViT-B-32`) model on the Fashion Product Images dataset. It demonstrates how to adapt a powerful pre-trained vision-language model to a specific domain using efficient optimization techniques.

## Files

- **`OpenCLIP_finetune_improved.py`**: The main training script. It implements best practices for stable fine-tuning on small datasets.
- **`OpenCLIP_finetune_food101.ipynb`**: The original exploratory notebook.
- **`requirements.txt`**: Python dependencies.

## Key Features & Optimization

Fine-tuning foundation models like CLIP on small datasets (e.g., ~2k images) is prone to **Model Collapse** and **Catastrophic Forgetting**. The `improved` script implements several advanced techniques to solve this:

1.  **Linear Probing (Partial Freezing)**:
    *   **Frozen Vision Encoder**: The ViT backbone is frozen to preserve its powerful pre-trained features.
    *   **Frozen Text Transformer**: The deep text layers are frozen to prevent overfitting to the small text corpus.
    *   **Trainable Layers**: Only the `text_projection`, `ln_final` (normalization), and `token_embedding` are trained.

2.  **Stable Optimization**:
    *   **Smart Weight Decay**: Applied *only* to weight matrices. Biases, LayerNorms, and Logit Scale are excluded from decay to prevent shrinking critical parameters.
    *   **Gradient Clipping**: gradients are clipped to `1.0` to prevent "exploding gradient" spikes that destroy weights.
    *   **Logit Scale Initialization**: Manually initialized to `ln(10)` (~2.3) instead of `100` to avoid gradient saturation.
    *   **Lookahead Scheduler**: Uses a linear warmup (10% of steps) to gently introduce updates.

## Performance
With these optimizations, the model avoids collapse (where Recall drops to 0.0) and maintains high retrieval accuracy even on a small subset of data.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the improved training script:

```bash
python OpenCLIP_finetune_improved.py
```

The script will:
1.  Load the `ashraq/fashion-product-images-small` dataset.
2.  Initialize the OpenCLIP ViT-B-32 model.
3.  Apply the freezing and optimization strategies.
4.  Train for 5 epochs.
5.  Save the best model to `./model/openclip_food101_improved.pt`.