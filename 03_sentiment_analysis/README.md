# LLM-based Sentiment Classification

A comprehensive demonstration of using Large Language Models for zero-shot sentiment classification on movie reviews. This implementation uses **Ollama** for free, local inference.

## 📋 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [The Core Algorithm](#the-core-algorithm)
- [Understanding the Output](#understanding-the-output)
- [Metrics Deep Dive](#metrics-deep-dive)
- [Key Concepts](#key-concepts)
- [Installation & Usage](#installation--usage)
- [Improving Accuracy](#improving-accuracy)
- [Limitations](#limitations)

---

## 🎯 Overview

This script performs **zero-shot sentiment classification** - it uses a language model to classify movie reviews as positive or negative *without* training the model on labeled examples. The model relies purely on its pre-existing knowledge from pre-training.

### The Big Picture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌─────────────┐
│  Movie Review   │ ──▶ │  Prompt Template │ ──▶ │   LLM (Llama)   │ ──▶ │  "0" or "1" │
│  "Great film!"  │     │  Instructions +  │     │  Understands &  │     │  Prediction │
│                 │     │  [DOCUMENT]      │     │  Responds       │     │             │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └─────────────┘
```

### What Does It Do?

1. **Takes a movie review** (text input)
2. **Constructs a prompt** with classification instructions
3. **Sends it to a local LLM** via Ollama
4. **Receives a prediction** (positive=1 or negative=0)
5. **Evaluates performance** against ground truth labels from SST-2 dataset

---

## 🔧 How It Works

### Step 1: Server Check & Model Loading

```
Checking Ollama server status...
✓ Ollama server is running

Available models: ['llama3.2:1b']
✓ Model 'llama3.2:1b' is available
```

**What's happening:**
- The script checks if Ollama (a local LLM server) is running on `localhost:11434`
- Lists available models and verifies `llama3.2:1b` (a 1 billion parameter model) is ready
- **Why Ollama?** It lets you run LLMs locally, completely free, no API costs

### Step 2: Prompt Engineering

This is the **key to zero-shot classification**. We craft a specific prompt:

```python
prompt = """You are a sentiment classifier. Classify the following movie review as positive or negative.

Movie review: [DOCUMENT]

If the review is positive, output only: 1
If the review is negative, output only: 0

Only output the number, nothing else."""
```

**What makes this prompt work:**

| Element | Purpose |
|---------|---------|
| `"You are a sentiment classifier"` | Sets the model's "role" or persona |
| `"positive or negative"` | Defines the binary classification task |
| `[DOCUMENT]` | Placeholder replaced with actual review text |
| `"output only: 1" / "output only: 0"` | Enforces simple, parseable output |
| `"Only output the number"` | Prevents verbose explanations |

### Step 3: Demo Classification

```
Review: "unpretentious, charming, quirky, original"
  Expected: Positive
  Model Response: '1'
  Parsed Prediction: Positive ✓

Review: "terrible, boring, waste of time"
  Expected: Negative
  Model Response: '1'  ← INCORRECT
  Parsed Prediction: Positive ✗
```

**What's happening here:**

1. Each review is inserted into the prompt template (replacing `[DOCUMENT]`)
2. The complete prompt is sent to Llama via Ollama's REST API
3. The model generates a response (`'1'` or `'0'`)
4. We parse the response into an integer prediction

### Step 4: Full Evaluation on SST-2 Dataset

```
Loading SST-2 dataset from HuggingFace...
Evaluating on 50 samples...
```

**SST-2 Dataset (Stanford Sentiment Treebank v2):**
- **Source:** Movie reviews from Rotten Tomatoes
- **Task:** Binary sentiment classification
- **Labels:** 0 = Negative, 1 = Positive
- **Size:** ~67K training, ~872 validation samples

---

## 🧠 The Core Algorithm

```python
# 1. Create the classification prompt template
prompt = """You are a sentiment classifier...
Movie review: [DOCUMENT]
If positive, output: 1
If negative, output: 0"""

# 2. For each review in the dataset:
for review in reviews:
    # Insert review into prompt
    full_prompt = prompt.replace("[DOCUMENT]", review)
    
    # Send to Ollama API (local server)
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:1b",
            "prompt": full_prompt,
            "temperature": 0  # Deterministic output
        }
    )
    
    # Parse response ("0" or "1") → integer
    prediction = parse_prediction(response.text)
    predictions.append(prediction)

# 3. Evaluate using sklearn metrics
accuracy = accuracy_score(true_labels, predictions)
report = classification_report(true_labels, predictions)
```

### API Call Details

```python
payload = {
    "model": "llama3.2:1b",     # Which model to use
    "prompt": full_prompt,       # The complete prompt with review
    "stream": False,             # Get complete response at once
    "options": {
        "temperature": 0,        # No randomness (deterministic)
        "num_predict": 10        # Limit output tokens
    }
}
```

---

## 📊 Understanding the Output

### Sample Output

```
============================================================
CLASSIFICATION PERFORMANCE REPORT
============================================================

Overall Accuracy: 0.6200 (62.00%)

Detailed Metrics:
------------------------------------------------------------
                 precision    recall  f1-score   support

Negative Review       1.00      0.14      0.24        22
Positive Review       0.60      1.00      0.75        28

       accuracy                           0.62        50
      macro avg       0.80      0.57      0.49        50
   weighted avg       0.77      0.62      0.52        50
```

### The Confusion Matrix

```
                      Predicted
                   Neg (0)   Pos (1)
                 ┌─────────┬─────────┐
Actual Neg (0):  │    3    │   19    │  ← Only 3 correct negatives!
                 ├─────────┼─────────┤
Actual Pos (1):  │    0    │   28    │  ← All 28 positives correct
                 └─────────┴─────────┘
```

**Key Observation:** The model predicts positive (1) for almost everything, showing a severe positive bias.

---

## 📈 Metrics Deep Dive

### Metric Definitions

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision** | TP / (TP + FP) | Of all predicted positives, how many were correct? |
| **Recall** | TP / (TP + FN) | Of all actual positives, how many did we find? |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Support** | Count | Number of actual samples in each class |

### Applied to Our Results

| Metric | Negative Review | Positive Review | Interpretation |
|--------|-----------------|-----------------|----------------|
| **Precision** | 1.00 | 0.60 | When it predicts negative, it's always right. But 40% of positive predictions are wrong. |
| **Recall** | 0.14 | 1.00 | Only catches 14% of negatives! Catches 100% of positives. |
| **F1-Score** | 0.24 | 0.75 | Very poor for negatives (imbalanced precision/recall) |
| **Support** | 22 | 28 | More positive samples in test set |

### Why the Model Struggles

The `llama3.2:1b` model has a **severe positive bias**:
- **Recall for Negative = 0.14**: Out of 22 actual negative reviews, only 3 were correctly identified
- **Recall for Positive = 1.00**: All 28 positive reviews were correctly identified
- **The model almost always outputs "1"**

Reasons for this:
1. **Model Size**: Only 1 billion parameters (GPT-4 has ~1.7 trillion)
2. **Training Bias**: May have seen more positive sentiment during pre-training
3. **Prompt Sensitivity**: Small models need more precise prompting

---

## 💡 Key Concepts

| Concept | Definition | In This Script |
|---------|------------|----------------|
| **Zero-shot Learning** | Classify without task-specific training | Uses prompt instructions only, no fine-tuning |
| **Prompt Engineering** | Crafting prompts to guide model behavior | The classification template with clear instructions |
| **Temperature = 0** | Deterministic output (no randomness) | Same input → same output every time |
| **Local Inference** | Running models on your own machine | Via Ollama, no cloud API needed |
| **Binary Classification** | Two-class prediction problem | Positive (1) vs Negative (0) |

### Zero-shot vs Few-shot vs Fine-tuning

| Approach | Description | Accuracy | Cost |
|----------|-------------|----------|------|
| **Zero-shot** | Prompt only, no examples | Lower | Free |
| **Few-shot** | Include 3-5 examples in prompt | Medium | Free |
| **Fine-tuning** | Train on labeled dataset | Highest | GPU time |

---

## 🛠️ Installation & Usage

### Prerequisites

```bash
# Install Ollama
brew install ollama

# Start Ollama server
brew services start ollama

# Pull a model
ollama pull llama3.2:1b

# Install Python dependencies
pip install datasets scikit-learn matplotlib seaborn tqdm numpy requests
```

### Running the Script

```bash
python gpt_sentiment_classifier.py
```

### Programmatic Usage

```python
from gpt_sentiment_classifier import (
    ollama_generation,
    create_classification_prompt,
    parse_prediction
)

# Classify a single review
prompt = create_classification_prompt()
review = "This movie was absolutely fantastic!"
response = ollama_generation(prompt, review, model="llama3.2:1b")
prediction = parse_prediction(response)

print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")
```

---

## 🚀 Improving Accuracy

### 1. Use a Larger Model

```bash
ollama pull llama3.2      # 3B parameters (~2GB)
ollama pull phi3          # 3.8B parameters (~2.3GB)
ollama pull mistral       # 7B parameters (~4GB)
```

Then update the script:
```python
MODEL = "llama3.2"  # Instead of "llama3.2:1b"
```

### 2. Use Few-shot Learning (Add Examples)

```python
prompt = """Classify movie review sentiment as 0 (negative) or 1 (positive).

Examples:
"Great movie, loved every minute!" → 1
"Terrible film, waste of time" → 0
"Boring and poorly acted" → 0
"A masterpiece of cinema!" → 1

Movie review: [DOCUMENT]
Sentiment (0 or 1):"""
```

### 3. Use OpenAI API (Paid, Higher Accuracy)

If you add credits to your OpenAI account, GPT-3.5/4 typically achieves ~90%+ accuracy on sentiment classification.

### Model Size vs Accuracy Comparison

| Model | Parameters | Expected Accuracy | Speed |
|-------|------------|-------------------|-------|
| llama3.2:1b | 1B | ~60-65% | Very Fast |
| llama3.2 | 3B | ~70-75% | Fast |
| phi3 | 3.8B | ~75-80% | Medium |
| mistral | 7B | ~80-85% | Slower |
| GPT-3.5-turbo | ~175B | ~88-92% | API latency |
| GPT-4 | ~1.7T | ~94-97% | API latency |

---

## ⚠️ Limitations

### 1. Model Size

Small local models (1B-7B params) are less accurate than large cloud models (GPT-4). Trade-off: Free & private vs. accurate & paid.

### 2. Positive Bias

The llama3.2:1b model shows strong positive bias. This is a known issue with smaller models on sentiment tasks.

### 3. Evaluation Concerns

> **Warning**: LLMs may have been trained on public datasets like SST-2. High performance doesn't necessarily prove generalization ability.

### 4. Local Hardware Requirements

Larger models require more RAM:
- 1B model: ~2GB RAM
- 7B model: ~8GB RAM
- 13B model: ~16GB RAM

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `gpt_sentiment_classifier.py` | Main script with all functions |
| `README_GPT_SENTIMENT.md` | This documentation |
| `ollama_sentiment_confusion_matrix.png` | Confusion matrix visualization |

---

## 🔗 References

- [Ollama Documentation](https://ollama.ai/)
- [SST-2 Dataset Paper](https://nlp.stanford.edu/sentiment/)
- [HuggingFace GLUE Benchmark](https://huggingface.co/datasets/glue)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

---

## 📝 Summary

This script demonstrates:
1. **Zero-shot classification** using prompt engineering
2. **Local LLM inference** via Ollama (free, private)
3. **Performance evaluation** using sklearn metrics
4. **Practical ML insights** about model limitations

The 62% accuracy with a 1B model is expected - larger models or fine-tuning would significantly improve results.
