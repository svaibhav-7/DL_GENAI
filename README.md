---
title: Smart MCQ Models
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# Check out the live deployed application [Smart-MCQ-Solver-Models] that uses the trained models to answer a Question/Prompt over given choice of options:
https://dlgenai-2upn9rjd5gpm2azoh2namr.streamlit.app/

# Deep Learning & GenAI Project

Welcome to the Deep Learning & GenAI Project repository. This project tackles natural language processing and question-answering tasks using various machine learning and deep learning approaches, progressing from simple baselines to advanced fine-tuned Large Language Models (LLMs).

## Project Overview

This repository contains the complete pipeline for our project, including Exploratory Data Analysis (EDA), model training, evaluation, and reporting. The modeling journey covers:
1. **Baseline Model:** TF-IDF
2. **Deep Learning Model from Scratch:** Bi-LSTM
3. **Advanced GenAI Model:** BERT-base-uncased fine-tuned with LoRA (Low-Rank Adaptation)
4. **RAG Model:** Retrieval-Augmented Generation with Reranking

## Repository Structure

```
.
├── milestones/             # Milestone notebooks tracking project progress
│   ├── GenAI_Milestone_2.ipynb
│   └── GenAI_Milestone_3.ipynb
├── models/                 # Saved model weights and artifacts
│   ├── adapter_model.safetensors       # LoRA weights for the BERT model
│   ├── bilstm_scratch_model.pt         # PyTorch weights for the Bi-LSTM model
│   └── tfidf_baseline_model.joblib     # Serialized TF-IDF baseline model
├── notebooks/              # Core project notebooks
│   ├── 01_EDA_Preprocessing.ipynb      # Data exploration and preprocessing steps
│   ├── 02_TF-IDF_Baseline.ipynb        # TF-IDF baseline model training and evaluation
│   ├── 03_Bi-LSTM_Scratch_model.ipynb  # Bi-LSTM model training and evaluation
│   ├── 04_BERT-base-uncased + LoRA_finetuned.ipynb # BERT fine-tuning using LoRA
│   ├── 05_RAG_RERANKED.ipynb           # RAG with Reranking implementation
│   └── final_notebook.ipynb            # Final combined notebook
├── reports/                # Project reports and documentation
│   ├── 01_EDA_Report.pdf
│   ├── 02_TFIDF_MODEL.pdf
│   ├── 03_BILSTM_MODEL.pdf
│   ├── 04_BERT_LORA_MODEL_REPORT.pdf
│   ├── Milestone 2 Report.pdf
│   └── final-report.pdf
├── src/                    # Source code scripts for training and inference
│   ├── inference.py
│   ├── train.py
│   └── utils.py
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Getting Started

### Prerequisites

Ensure you have Python installed. You can install the required dependencies using the provided `requirements.txt` file (if populated):

```bash
pip install -r requirements.txt
```

### Navigating the Codebase

- **Exploratory Data Analysis:** Start with `notebooks/01_EDA_Preprocessing.ipynb` to understand the dataset, distributions, and preprocessing steps.
- **Model Training & Evaluation:** Check the sequential notebooks in the `notebooks/` directory to see how each model was developed:
  - `02_TF-IDF_Baseline.ipynb`: Traditional Machine Learning approach.
  - `03_Bi-LSTM_Scratch_model.ipynb`: Deep Learning approach built from scratch.
  - `04_BERT-base-uncased + LoRA_finetuned.ipynb`: Modern LLM approach using parameter-efficient fine-tuning (LoRA).
  - `05_RAG_RERANKED.ipynb`: Retrieval-Augmented Generation approach with document reranking.
- **Reports:** For detailed analysis, methodologies, and performance metrics, refer to the PDF reports in the `reports/` directory. Each model and milestone has a corresponding detailed report.
- **Source Code:** Reusable functions and scripts for training and inference are located in the `src/` directory.

## Models Implemented

The trained models are deployed on Hugging Face Hub: [smart-mcq-models](https://huggingface.co/SVS8907/smart-mcq-models).

1. **TF-IDF Baseline:** Uses Term Frequency-Inverse Document Frequency features with traditional classifiers to establish a performance baseline.
2. **Bi-LSTM:** A Bidirectional Long Short-Term Memory network built in PyTorch to capture sequential context in text.
3. **BERT + LoRA:** A pre-trained `bert-base-uncased` model fine-tuned using Low-Rank Adaptation, allowing for efficient training with high performance on the task.
4. **RAG with Reranking:** A Retrieval-Augmented Generation pipeline using document retrieval and a reranking stage to select the most relevant context for question answering.


# Deep Learning & GenAI Project

Welcome to the Deep Learning & GenAI Project repository. This project tackles natural language processing and question-answering tasks using various machine learning and deep learning approaches, progressing from simple baselines to advanced fine-tuned Large Language Models (LLMs).

## Project Overview

This repository contains the complete pipeline for our project, including Exploratory Data Analysis (EDA), model training, evaluation, and reporting. The modeling journey covers:
1. **Baseline Model:** TF-IDF
2. **Deep Learning Model from Scratch:** Bi-LSTM
3. **Advanced GenAI Model:** BERT-base-uncased fine-tuned with LoRA (Low-Rank Adaptation)
4. **RAG Model:** Retrieval-Augmented Generation with Reranking

## Repository Structure

```
.
├── milestones/             # Milestone notebooks tracking project progress
│   ├── GenAI_Milestone_2.ipynb
│   └── GenAI_Milestone_3.ipynb
├── models/                 # Saved model weights and artifacts
│   ├── adapter_model.safetensors       # LoRA weights for the BERT model
│   ├── bilstm_scratch_model.pt         # PyTorch weights for the Bi-LSTM model
│   └── tfidf_baseline_model.joblib     # Serialized TF-IDF baseline model
├── notebooks/              # Core project notebooks
│   ├── 01_EDA_Preprocessing.ipynb      # Data exploration and preprocessing steps
│   ├── 02_TF-IDF_Baseline.ipynb        # TF-IDF baseline model training and evaluation
│   ├── 03_Bi-LSTM_Scratch_model.ipynb  # Bi-LSTM model training and evaluation
│   ├── 04_BERT-base-uncased + LoRA_finetuned.ipynb # BERT fine-tuning using LoRA
│   ├── 05_RAG_RERANKED.ipynb           # RAG with Reranking implementation
│   └── final_notebook.ipynb            # Final combined notebook
├── reports/                # Project reports and documentation
│   ├── 01_EDA_Report.pdf
│   ├── 02_TFIDF_MODEL.pdf
│   ├── 03_BILSTM_MODEL.pdf
│   ├── 04_BERT_LORA_MODEL_REPORT.pdf
│   ├── Milestone 2 Report.pdf
│   └── final-report.pdf
├── src/                    # Source code scripts for training and inference
│   ├── inference.py
│   ├── train.py
│   └── utils.py
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Getting Started

### Prerequisites

Ensure you have Python installed. You can install the required dependencies using the provided `requirements.txt` file (if populated):

```bash
pip install -r requirements.txt
```

### Navigating the Codebase

- **Exploratory Data Analysis:** Start with `notebooks/01_EDA_Preprocessing.ipynb` to understand the dataset, distributions, and preprocessing steps.
- **Model Training & Evaluation:** Check the sequential notebooks in the `notebooks/` directory to see how each model was developed:
  - `02_TF-IDF_Baseline.ipynb`: Traditional Machine Learning approach.
  - `03_Bi-LSTM_Scratch_model.ipynb`: Deep Learning approach built from scratch.
  - `04_BERT-base-uncased + LoRA_finetuned.ipynb`: Modern LLM approach using parameter-efficient fine-tuning (LoRA).
  - `05_RAG_RERANKED.ipynb`: Retrieval-Augmented Generation approach with document reranking.
- **Reports:** For detailed analysis, methodologies, and performance metrics, refer to the PDF reports in the `reports/` directory. Each model and milestone has a corresponding detailed report.
- **Source Code:** Reusable functions and scripts for training and inference are located in the `src/` directory.

## Models Implemented

The trained models are deployed on Hugging Face Hub: [smart-mcq-models](https://huggingface.co/SVS8907/smart-mcq-models).

1. **TF-IDF Baseline:** Uses Term Frequency-Inverse Document Frequency features with traditional classifiers to establish a performance baseline.
2. **Bi-LSTM:** A Bidirectional Long Short-Term Memory network built in PyTorch to capture sequential context in text.
3. **BERT + LoRA:** A pre-trained `bert-base-uncased` model fine-tuned using Low-Rank Adaptation, allowing for efficient training with high performance on the task.
4. **RAG with Reranking:** A Retrieval-Augmented Generation pipeline using document retrieval and a reranking stage to select the most relevant context for question answering.

