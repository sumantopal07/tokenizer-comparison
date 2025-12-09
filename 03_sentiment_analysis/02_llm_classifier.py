"""
LLM-based Sentiment Classification (Using Ollama - Free & Local)

This module demonstrates how to use local LLM models via Ollama for binary sentiment 
classification on movie reviews. It showcases a zero-shot classification approach
where the model classifies text without being explicitly trained on the specific dataset.

Key Concepts:
    - Zero-shot Classification: The model performs classification using its pre-trained
      knowledge without task-specific fine-tuning.
    - Prompt Engineering: We craft a specific prompt template that instructs the model
      on how to perform the classification task.
    - Local Inference: Uses Ollama for free, local model inference (no API costs!)

Usage:
    python gpt_sentiment_classifier.py

Requirements:
    - Ollama installed: brew install ollama
    - Model pulled: ollama pull llama3.2:1b (or similar)

Author: Tokenizer Comparison Project
Date: December 2024
"""

import requests
import json
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import time
import subprocess


def check_ollama_running() -> bool:
    """
    Check if Ollama server is running.
    
    Returns:
        bool: True if Ollama is running, False otherwise.
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def start_ollama_server() -> bool:
    """
    Attempt to start the Ollama server if not running.
    
    Returns:
        bool: True if server started successfully, False otherwise.
    """
    print("Attempting to start Ollama server...")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(3)  # Wait for server to start
        return check_ollama_running()
    except Exception as e:
        print(f"Failed to start Ollama: {e}")
        return False


def list_available_models() -> list:
    """
    List all available Ollama models.
    
    Returns:
        list: List of available model names.
    """
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [m["name"] for m in models]
        return []
    except:
        return []


def pull_model(model_name: str = "llama3.2:1b") -> bool:
    """
    Pull a model from Ollama registry.
    
    Args:
        model_name (str): Name of the model to pull.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    print(f"Pulling model: {model_name} (this may take a few minutes)...")
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Model pull timed out. Please run 'ollama pull llama3.2:1b' manually.")
        return False
    except Exception as e:
        print(f"Error pulling model: {e}")
        return False


def ollama_generation(
    prompt: str,
    document: str,
    model: str = "llama3.2:1b"
) -> str:
    """
    Generate a response from a local Ollama model based on a prompt template.
    
    This function sends a request to the locally running Ollama server using the
    specified model. The prompt template should contain a [DOCUMENT] placeholder
    that gets replaced with the actual document text.
    
    Args:
        prompt (str): The prompt template containing instructions for the model.
                      Must include [DOCUMENT] placeholder for document insertion.
        document (str): The input text to be classified/processed.
        model (str, optional): The Ollama model to use. Defaults to "llama3.2:1b".
                              Other options: "llama3.2", "mistral", "phi3", etc.
    
    Returns:
        str: The model's generated response text.
    
    Technical Details:
        - Uses Ollama's REST API at localhost:11434
        - No API costs - runs entirely locally!
        - Performance depends on your machine's specs
    
    Example:
        >>> prompt = "Classify this as positive or negative: [DOCUMENT]"
        >>> response = ollama_generation(prompt, "Great movie!")
        >>> print(response)
        "1"
    """
    full_prompt = prompt.replace("[DOCUMENT]", document)
    
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0,  # Deterministic output
            "num_predict": 10  # Limit output tokens for classification
        }
    }
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            print(f"API Error: {response.status_code}")
            return ""
    except requests.exceptions.Timeout:
        print("Request timed out")
        return ""
    except Exception as e:
        print(f"Error: {e}")
        return ""


def create_classification_prompt() -> str:
    """
    Create the prompt template for sentiment classification.
    
    This function returns a carefully crafted prompt that instructs the model
    to perform binary sentiment classification on movie reviews. The prompt uses
    the [DOCUMENT] placeholder which gets replaced with actual review text.
    
    Returns:
        str: The prompt template with [DOCUMENT] placeholder.
    
    Prompt Engineering Considerations:
        1. Clear Task Definition: Explicitly states the classification task
        2. Binary Output: Restricts output to just "1" or "0"
        3. Label Mapping:
           - 1 = Positive review
           - 0 = Negative review
        4. Constraint: "Only output the number" prevents verbose responses
    """
    return """You are a sentiment classifier. Classify the following movie review as positive or negative.

Movie review: [DOCUMENT]

If the review is positive, output only: 1
If the review is negative, output only: 0

Only output the number, nothing else."""


def parse_prediction(response: str) -> int:
    """
    Parse the model's response into a binary prediction.
    
    The model should return either "0" or "1", but sometimes may include
    extra text or formatting. This function extracts the numeric prediction.
    
    Args:
        response (str): The raw response from the model.
    
    Returns:
        int: The parsed prediction (0 for negative, 1 for positive).
             Returns -1 if parsing fails (invalid response).
    """
    response = response.strip()
    
    # Direct match
    if response == "0":
        return 0
    elif response == "1":
        return 1
    
    # Check if response starts with 0 or 1
    if response.startswith("0"):
        return 0
    elif response.startswith("1"):
        return 1
    
    # Search for 0 or 1 in the response
    if "1" in response and "0" not in response:
        return 1
    elif "0" in response and "1" not in response:
        return 0
    
    # Look for keywords
    response_lower = response.lower()
    if "positive" in response_lower:
        return 1
    elif "negative" in response_lower:
        return 0
    
    # If both or neither are present, return -1 (parsing failed)
    return -1


def evaluate_performance(y_true: list, y_pred: list) -> dict:
    """
    Evaluate and display the classification performance metrics.
    
    Args:
        y_true (list): Ground truth labels (actual sentiment: 0 or 1).
        y_pred (list): Predicted labels from the model (0 or 1).
    
    Returns:
        dict: Dictionary containing accuracy, report, and confusion_matrix.
    
    Metrics Explained:
        - Precision: Of all predicted positives, how many were actually positive?
        - Recall: Of all actual positives, how many were correctly identified?
        - F1-Score: Harmonic mean of precision and recall
        - Accuracy: Overall correct predictions / total predictions
    """
    # Generate classification report
    target_names = ["Negative Review", "Positive Review"]
    report = classification_report(y_true, y_pred, target_names=target_names)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print results
    print("\n" + "=" * 60)
    print("CLASSIFICATION PERFORMANCE REPORT")
    print("=" * 60)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nDetailed Metrics:")
    print("-" * 60)
    print(report)
    
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm: np.ndarray, save_path: str = None) -> None:
    """
    Visualize the confusion matrix as a heatmap.
    
    Args:
        cm (np.ndarray): 2x2 confusion matrix from sklearn.
        save_path (str, optional): Path to save the figure. If None, displays plot.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive'],
        annot_kws={'size': 14}
    )
    plt.title('Confusion Matrix - Ollama Sentiment Classification', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\nConfusion matrix saved to: {save_path}")
    else:
        plt.show()


def run_demo(model: str = "llama3.2:1b") -> None:
    """
    Run a quick demonstration of the sentiment classifier.
    
    Args:
        model (str): The Ollama model to use.
    """
    print("\n" + "=" * 60)
    print("DEMO: Ollama Sentiment Classification")
    print("=" * 60)
    
    prompt = create_classification_prompt()
    
    demo_reviews = [
        ("unpretentious, charming, quirky, original", "Positive"),
        ("terrible, boring, waste of time", "Negative"),
        ("has great acting but poor script", "Mixed/Ambiguous"),
        ("a masterpiece of modern cinema", "Positive"),
        ("not worth the ticket price", "Negative")
    ]
    
    print(f"\nUsing model: {model}")
    print("\nClassifying sample reviews...\n")
    
    for review, expected in demo_reviews:
        response = ollama_generation(prompt, review, model)
        prediction = parse_prediction(response)
        label = "Positive" if prediction == 1 else "Negative" if prediction == 0 else "Unknown"
        
        print(f"Review: \"{review}\"")
        print(f"  Expected: {expected}")
        print(f"  Model Response: '{response}'")
        print(f"  Parsed Prediction: {label}")
        print()


def run_full_evaluation(
    model: str = "llama3.2:1b",
    num_samples: int = 50,
    dataset_split: str = "validation"
) -> dict:
    """
    Run a full evaluation on the SST-2 sentiment dataset.
    
    Args:
        model (str): The Ollama model to use.
        num_samples (int, optional): Number of samples to evaluate. Defaults to 50.
        dataset_split (str, optional): Which split to use. Defaults to "validation".
    
    Returns:
        dict: Evaluation results including accuracy, metrics, and predictions.
    
    Note:
        - SST-2 is Stanford Sentiment Treebank (binary classification)
        - Labels: 0 = Negative, 1 = Positive
        - Using fewer samples since local inference is slower than API
    """
    print("\n" + "=" * 60)
    print(f"FULL EVALUATION: SST-2 Dataset ({num_samples} samples)")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading SST-2 dataset from HuggingFace...")
    dataset = load_dataset("glue", "sst2", trust_remote_code=True)
    
    # Get validation data (test set doesn't have public labels)
    data = dataset[dataset_split]
    
    # Sample subset
    if num_samples < len(data):
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(len(data), num_samples, replace=False)
        # Convert numpy int64 to Python int for HuggingFace dataset compatibility
        sentences = [data["sentence"][int(i)] for i in indices]
        labels = [data["label"][int(i)] for i in indices]
    else:
        sentences = data["sentence"]
        labels = data["label"]
    
    print(f"Evaluating on {len(sentences)} samples...")
    print(f"Using model: {model}")
    
    # Create prompt
    prompt = create_classification_prompt()
    
    # Run predictions
    predictions = []
    failed_parses = 0
    
    print("\nRunning classification (this may take a while for local inference)...")
    for i, sentence in enumerate(tqdm(sentences, desc="Classifying")):
        response = ollama_generation(prompt, sentence, model)
        pred = parse_prediction(response)
        
        if pred == -1:
            failed_parses += 1
            pred = 0  # Default to negative if parsing fails
            
        predictions.append(pred)
    
    if failed_parses > 0:
        print(f"\nWarning: {failed_parses} responses couldn't be parsed")
    
    # Evaluate
    results = evaluate_performance(labels, predictions)
    
    # Add additional info
    results['predictions'] = predictions
    results['labels'] = labels
    results['sentences'] = sentences
    results['failed_parses'] = failed_parses
    
    return results


def main():
    """
    Main function to run the Ollama sentiment classification demo and evaluation.
    
    Workflow:
        1. Check if Ollama is running
        2. Ensure a model is available (pull if needed)
        3. Run quick demo on sample reviews
        4. Run full evaluation on SST-2 dataset
        5. Display and visualize results
    """
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    # Model to use (smaller = faster, larger = more accurate)
    # Options: "llama3.2:1b", "llama3.2", "phi3", "mistral", etc.
    MODEL = "llama3.2:1b"
    
    # Number of samples to evaluate (reduce for faster execution)
    NUM_SAMPLES = 50
    
    # =========================================================================
    # SETUP
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("OLLAMA SENTIMENT CLASSIFICATION SYSTEM")
    print("=" * 60)
    print("\nThis script demonstrates zero-shot sentiment classification")
    print("using local LLM models via Ollama (FREE - no API costs!).")
    
    # Check if Ollama is running
    print("\nChecking Ollama server status...")
    if not check_ollama_running():
        print("Ollama server not running. Attempting to start...")
        if not start_ollama_server():
            print("\n❌ Failed to start Ollama server.")
            print("Please run 'ollama serve' in a separate terminal.")
            return
    
    print("✓ Ollama server is running")
    
    # Check available models
    available = list_available_models()
    print(f"\nAvailable models: {available if available else '(none)'}")
    
    # Pull model if needed
    if MODEL not in available:
        print(f"\nModel '{MODEL}' not found locally.")
        if not pull_model(MODEL):
            print(f"\n❌ Failed to pull model. Please run: ollama pull {MODEL}")
            return
        print(f"✓ Model '{MODEL}' pulled successfully")
    else:
        print(f"✓ Model '{MODEL}' is available")
    
    # =========================================================================
    # EXECUTION
    # =========================================================================
    
    # Run demo
    run_demo(MODEL)
    
    # Run full evaluation
    print("\n" + "-" * 60)
    print("Starting full evaluation...")
    print(f"This will classify {NUM_SAMPLES} movie reviews.")
    print("-" * 60)
    
    results = run_full_evaluation(MODEL, num_samples=NUM_SAMPLES)
    
    # Visualize results
    print("\nGenerating confusion matrix visualization...")
    plot_confusion_matrix(
        results['confusion_matrix'],
        save_path="ollama_sentiment_confusion_matrix.png"
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nModel used: {MODEL}")
    print(f"Total samples evaluated: {len(results['predictions'])}")
    print(f"Final Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Failed parses: {results['failed_parses']}")
    
    print("\n" + "-" * 60)
    print("💡 ADVANTAGES OF LOCAL INFERENCE")
    print("-" * 60)
    print("""
✓ Completely FREE - no API costs
✓ Privacy - data never leaves your machine  
✓ No rate limits
✓ Works offline

Note: Local models may be less accurate than GPT-4, but are
great for learning and experimentation!
    """)


if __name__ == "__main__":
    main()
