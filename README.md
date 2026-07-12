# Deep Learning & GenAI Project

Welcome to the Deep Learning & GenAI Project repository. This project tackles natural language processing and multiple-choice question-answering tasks using various machine learning and deep learning approaches, progressing from simple baselines to advanced fine-tuned Large Language Models (LLMs).

## Project Overview

This repository contains the complete pipeline for our project, including Exploratory Data Analysis (EDA), model training, evaluation, and reporting. The modeling journey covers:
1. **Baseline Model:** TF-IDF with traditional classifiers
2. **Deep Learning Model from Scratch:** Bi-LSTM implemented in PyTorch
3. **Advanced GenAI Model:** BERT-base-uncased fine-tuned with LoRA (Low-Rank Adaptation)

## Repository Structure

```
.
├── milestones/                 # Milestone notebooks tracking project progress
│   ├── GenAI_Milestone_2.ipynb # Milestone 2 implementation and analysis
│   └── GenAI_Milestone_3.ipynb # Milestone 3 implementation and analysis
├── models/                     # Saved model weights and artifacts
│   ├── adapter_model.safetensors       # LoRA weights for the BERT model
│   ├── bilstm_scratch_model.pt         # PyTorch weights for the Bi-LSTM model
│   └── tfidf_baseline_model.joblib     # Serialized TF-IDF baseline model
├── notebooks/                  # Core project notebooks
│   ├── 01_EDA_Preprocessing.ipynb      # Data exploration and preprocessing steps
│   ├── 02_TF-IDF_Baseline.ipynb        # TF-IDF baseline model training and evaluation
│   ├── 03_Bi-LSTM_Scratch_model.ipynb  # Bi-LSTM model training and evaluation
│   ├── 04_BERT-base-uncased + LoRA_finetuned.ipynb # BERT fine-tuning using LoRA
│   └── final_notebook.ipynb            # Final combined notebook bringing it all together
├── reports/                    # Project reports and documentation
│   ├── 01_EDA_Report.pdf               # Report on Exploratory Data Analysis
│   ├── 02_TFIDF_MODEL.pdf              # Details on TF-IDF baseline
│   ├── 03_BILSTM_MODEL.pdf             # Details on Bi-LSTM scratch implementation
│   ├── 04_BERT_LORA_MODEL_REPORT.pdf   # Details on BERT + LoRA fine-tuning
│   ├── Milestone 2 Report.pdf          # Summary report for Milestone 2
│   └── final-report.pdf                # Comprehensive final project report
├── src/                        # Source code scripts for training and inference
│   ├── inference.py                    # Script for running model inferences
│   ├── train.py                        # Script for training models
│   └── utils.py                        # Utility functions for data handling and metrics
├── requirements.txt            # Python dependencies needed for the project
└── README.md                   # Project documentation
```

## Getting Started

### Prerequisites

Ensure you have Python installed (Python 3.8+ recommended). You can install the required dependencies using the provided `requirements.txt` file (if populated):

```bash
pip install -r requirements.txt
```

### Navigating the Codebase

- **Exploratory Data Analysis:** Start with `notebooks/01_EDA_Preprocessing.ipynb` to understand the dataset, class distributions, and preprocessing steps.
- **Model Training & Evaluation:** Check the sequential notebooks in the `notebooks/` directory to see how each model was developed and trained:
  - `02_TF-IDF_Baseline.ipynb`: Traditional Machine Learning approach.
  - `03_Bi-LSTM_Scratch_model.ipynb`: Deep Learning approach built from scratch.
  - `04_BERT-base-uncased + LoRA_finetuned.ipynb`: Modern GenAI approach using parameter-efficient fine-tuning (LoRA) for LLMs.
- **Reports:** For detailed analysis, methodologies, metrics, and insights, refer to the PDF reports in the `reports/` directory.
- **Source Code:** Reusable components, training loops, and inference logic can be found in the `src/` directory.

## Models Implemented

1. **TF-IDF Baseline:** Uses Term Frequency-Inverse Document Frequency features alongside standard classification models to establish a reliable performance baseline.
2. **Bi-LSTM from Scratch:** A Bidirectional Long Short-Term Memory network developed entirely from scratch in PyTorch to capture sequential context and long-range dependencies in textual data.
3. **BERT + LoRA:** A pre-trained Large Language Model (`bert-base-uncased`) fine-tuned specifically for the target task using Low-Rank Adaptation (LoRA), ensuring high parameter efficiency and robust state-of-the-art performance.

## License

This project is licensed under the terms of the applicable project license.
