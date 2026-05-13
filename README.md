# BERTSum: Extractive Summarization on CNN

This repository contains an implementation of the **BERTSum** model for **Extractive Summarization**, based on the paper "Fine-tune BERT for Extractive Summarization" by Yang Liu (2019). The goal is to identify and extract the most important sentences from a document to form a concise summary.

## Project Overview
BERTSum modifies the vanilla BERT architecture by inserting multiple `[CLS]` tokens at the start of each sentence and utilizing **Interval Segment Embeddings** to distinguish multiple sentences within a document. This implementation uses a simple linear classifier on top of BERT outputs to predict the importance of each sentence.

## Environment Setup
The project requires a specific environment to ensure compatibility with older PyTorch and ROUGE evaluation scripts. It is recommended to run this on Google Colab with a GPU (e.g., Tesla T4).

### Dependencies
- Python 3.10+
- PyTorch 1.11.0+cu113
- torchvision 0.12.0+cu113
- pytorch_transformers 1.2.0
- pyrouge (requires ROUGE-1.5.5 and Perl environment)
- libxml-parser-perl

## Critical Code Modification
To support newer versions of PyTorch, the following modification in `src/models/data_loader.py` is required to handle boolean masking correctly:

```python
# Before
mask = 1 - (src == 0)
mask_cls = 1 - (clss == 0)

# After (Fix for newer PyTorch versions)
mask = ~(src == 0)
mask_cls = ~(clss == 0)
```

## Experimental Results

The model was evaluated on the **CNN/DailyMail** validation and test sets. The following table summarizes the performance improvement across different training checkpoints compared to the original paper's baseline.

### Performance Metrics

| Checkpoint (Steps) | Validation Xent ↓ | ROUGE-1 (F1) ↑ | ROUGE-2 (F1) ↑ | ROUGE-L (F1) ↑ |
| :--- | :---: | :---: | :---: | :---: |
| Step 1000 | 5.8020 | 41.93 | 19.06 | 38.30 |
| Step 1500 | 5.6017 | 42.32 | 19.34 | 38.70 |
| **Step 2000** | **5.5791** | **42.46** | **19.45** | **38.79** |
| **Paper (50,000 Steps)** | -- | **43.23** | **20.22** | **39.60** |

### Key Observations

* **Training Efficiency**: After only **2,000 steps** of fine-tuning, the model achieved a ROUGE-1 score of **42.46**. This is remarkably close to the original paper's reported **43.23** which required **50,000 steps**.
* **Steady Convergence**: The **Validation Cross-Entropy (Xent)** decreased consistently from **5.98** (at step 500) to **5.57** (at step 2000). This indicates a stable learning process and effective feature extraction from the BERT-based encoder.
* **Impact of Checkpoints**: The most significant jump in ROUGE scores occurred between **Step 1000 and 1500**. While the improvement rate slowed slightly toward Step 2000, the overall trend suggests that further training could bridge the remaining gap to the SOTA (State-of-the-art) performance.

## Acknowledgments

* **Original Paper**: [Fine-tune BERT for Extractive Summarization](https://arxiv.org/abs/1903.10318) by Yang Liu (2019).
* **Base Code**: This implementation is based on the open-source repository [nlpyang/BertSum](https://github.com/nlpyang/BertSum).
* **Course Support**: Developed as part of the coursework for the Natural Language Processing program.

## Author

**Pei-Hsiu Lu** * **Affiliation**: Undergraduate student at National Taiwan Normal University (NTNU).
* **Interests**: Software Development, LLM Integration, and Web Detection.
* **Contact**: Paul234jj@gmail.com
