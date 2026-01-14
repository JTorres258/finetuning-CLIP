# OpenCLIP Fine-tuning and Interaction

This repository contains Jupyter notebooks for fine-tuning and interacting with OpenCLIP models. It demonstrates how to adapt pre-trained vision-language models to specific datasets and how to perform various tasks like zero-shot classification and similarity calculation.

## Files

- **`OpenCLIP_finetune_flickr30k.ipynb`**: An end-to-end notebook for fine-tuning an OpenCLIP (`ViT-B-32`) model on the Hugging Face `nlphuji/flickr30k` dataset. It covers data loading, model initialization, partial freezing strategies (Linear Probing), and training.
- **`interacting-clip.ipynb`**: A self-contained notebook showing how to download and run OpenCLIP models, calculate similarity between images and text, and perform zero-shot image classification.
- **`requirements.txt`**: Python dependencies.

## Key Features & Optimization

Fine-tuning foundation models like CLIP on smaller specific datasets can benefit from optimization techniques demonstrated in the fine-tuning notebook:

1.  **Linear Probing (Partial Freezing)**:
    *   **Frozen Vision Encoder**: The ViT backbone is frozen to preserve pre-trained features.
    *   **Frozen Text Transformer**: Deep text layers can be frozen to prevent overfitting.
    *   **Trainable Layers**: Specific layers like `text_projection` are targeted for training.

2.  **Stable Optimization**:
    *   **Smart Weight Decay**: Improving generalization.
    *   **Gradient Clipping**: Preventing exploding gradients.
    *   **Logit Scale Initialization**: optimizing the scaling factor for better convergence.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Fine-tuning
Open `OpenCLIP_finetune_flickr30k.ipynb` in multiple environments (Jupyter, Colab, etc.) to follow the step-by-step guide for fine-tuning the model. The notebook is configured to use the `nlphuji/flickr30k` dataset but can be adapted for other Hugging Face image-text datasets.

### Interacting
Open `interacting-clip.ipynb` to explore the capabilities of pre-trained OpenCLIP models. You can use it to:
- List available pretrained models.
- Perform multiclass classification on custom images.
- Calculate cosine similarity between arbitrary text and images.