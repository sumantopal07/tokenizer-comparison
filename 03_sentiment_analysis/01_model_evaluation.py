"""
Sentiment Analysis Model Evaluation Demo

This script demonstrates:
1. Running inference on a test dataset using a sentiment analysis pipeline
2. Extracting predictions from model outputs
3. Evaluating model performance using various metrics
4. Understanding precision, recall, accuracy, and F1 score
5. Visualizing results with a confusion matrix
"""

import numpy as np
from tqdm import tqdm
from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# STEP 1: Load Dataset
# ============================================================================
print("=" * 70)
print("STEP 1: Loading Dataset")
print("=" * 70)

# Load a sentiment analysis dataset (we'll use IMDB as an example)
# The original code uses `data["test"]`, so we'll load the test split
print("\nLoading IMDB dataset (this may take a moment)...")
data = load_dataset("imdb", split="test")  # Load full test set
data = data.shuffle(seed=42).select(range(1000))  # Shuffle and select 1000 balanced samples

print(f"✓ Loaded {len(data)} test examples")
print(f"✓ Dataset columns: {data.column_names}")
print(f"\nExample entry:")
print(f"  Text: {data[0]['text'][:100]}...")
print(f"  Label: {data[0]['label']} (0=Negative, 1=Positive)")

# ============================================================================
# STEP 2: Load Sentiment Analysis Model
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Loading Pre-trained Sentiment Analysis Model")
print("=" * 70)

# Load a pre-trained sentiment analysis pipeline from HuggingFace
# This uses a model fine-tuned specifically for sentiment analysis
print("\nInitializing pipeline (downloading model if needed)...")
pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    return_all_scores=True,  # Return scores for all possible labels
    truncation=True  # Truncate texts longer than 512 tokens
)

print("✓ Model loaded successfully")
print(f"✓ Model: {pipe.model.config.name_or_path}")

# ============================================================================
# STEP 3: Run Inference on Test Data
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Running Inference on Test Data")
print("=" * 70)

print("\nProcessing test examples...")
print("For each text, the model outputs scores for NEGATIVE and POSITIVE classes.")

# Store predictions
y_pred = []
y_true = data["label"]  # Ground truth labels

# Process each example in the test set
# Using KeyDataset is more memory efficient than loading all texts at once
from transformers.pipelines.pt_utils import KeyDataset

for i, output in enumerate(tqdm(
    pipe(KeyDataset(data, "text")), 
    total=len(data),
    desc="Running inference"
)):
    # output is a list of dictionaries with 'label' and 'score' keys
    # Example: [{'label': 'NEGATIVE', 'score': 0.8}, {'label': 'POSITIVE', 'score': 0.2}]
    
    negative_score = output[0]["score"]  # Score for negative class
    positive_score = output[1]["score"]  # Score for positive class
    
    # Determine prediction by choosing the class with higher score
    # argmax([negative, positive]) returns 0 for negative, 1 for positive
    assignment = np.argmax([negative_score, positive_score])
    y_pred.append(assignment)
    
    # Show first few predictions for demonstration
    if i < 3:
        print(f"\nExample {i+1}:")
        print(f"  Negative score: {negative_score:.4f}")
        print(f"  Positive score: {positive_score:.4f}")
        print(f"  Predicted: {'POSITIVE' if assignment == 1 else 'NEGATIVE'}")
        print(f"  Actual: {'POSITIVE' if y_true[i] == 1 else 'NEGATIVE'}")
        print(f"  Correct: {'✓' if assignment == y_true[i] else '✗'}")

print(f"\n✓ Generated {len(y_pred)} predictions")

# ============================================================================
# STEP 4: Evaluate Model Performance
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Evaluating Model Performance")
print("=" * 70)

def evaluate_performance(y_true, y_pred):
    """
    Create and print the classification report
    
    The classification report includes:
    - Precision: Of all items predicted as positive, how many were actually positive?
    - Recall: Of all actual positive items, how many did we correctly identify?
    - F1-score: Harmonic mean of precision and recall (balanced metric)
    - Support: Number of actual occurrences of each class in the dataset
    """
    performance = classification_report(
        y_true, 
        y_pred,
        target_names=["Negative Review", "Positive Review"],
        digits=2
    )
    print(performance)

print("\nClassification Report:")
print("-" * 70)
evaluate_performance(y_true, y_pred)

# ============================================================================
# STEP 5: Understanding the Confusion Matrix
# ============================================================================
print("\n" + "=" * 70)
print("STEP 5: Understanding the Confusion Matrix")
print("=" * 70)

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print("-" * 70)
print(f"\n{cm}\n")

# Extract values from confusion matrix
tn, fp, fn, tp = cm.ravel()

print("Breaking down the confusion matrix:")
print(f"  True Negatives (TN):  {tn} - Correctly predicted negative reviews")
print(f"  False Positives (FP): {fp} - Incorrectly predicted as positive")
print(f"  False Negatives (FN): {fn} - Incorrectly predicted as negative")
print(f"  True Positives (TP):  {tp} - Correctly predicted positive reviews")

# ============================================================================
# STEP 6: Calculating Metrics Manually
# ============================================================================
print("\n" + "=" * 70)
print("STEP 6: Understanding Metrics with Manual Calculations")
print("=" * 70)

# For negative class (class 0)
precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else 0

# For positive class (class 1)
precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0

# Overall accuracy
accuracy = (tp + tn) / (tp + tn + fp + fn)

print("\nNegative Class Metrics:")
print(f"  Precision = TN / (TN + FN) = {tn} / {tn + fn} = {precision_neg:.2f}")
print(f"    → Of all predicted negatives, {precision_neg*100:.1f}% were actually negative")
print(f"  Recall = TN / (TN + FP) = {tn} / {tn + fp} = {recall_neg:.2f}")
print(f"    → Of all actual negatives, we found {recall_neg*100:.1f}%")
print(f"  F1-Score = 2 * (P * R) / (P + R) = {f1_neg:.2f}")

print("\nPositive Class Metrics:")
print(f"  Precision = TP / (TP + FP) = {tp} / {tp + fp} = {precision_pos:.2f}")
print(f"    → Of all predicted positives, {precision_pos*100:.1f}% were actually positive")
print(f"  Recall = TP / (TP + FN) = {tp} / {tp + fn} = {recall_pos:.2f}")
print(f"    → Of all actual positives, we found {recall_pos*100:.1f}%")
print(f"  F1-Score = 2 * (P * R) / (P + R) = {f1_pos:.2f}")

print("\nOverall Metrics:")
print(f"  Accuracy = (TP + TN) / Total = ({tp} + {tn}) / {tp + tn + fp + fn} = {accuracy:.2f}")
print(f"    → Model is correct {accuracy*100:.1f}% of the time")

# ============================================================================
# STEP 7: Visualize Confusion Matrix
# ============================================================================
print("\n" + "=" * 70)
print("STEP 7: Visualizing Confusion Matrix")
print("=" * 70)

plt.figure(figsize=(10, 8))

# Create heatmap
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=['Negative', 'Positive'],
    yticklabels=['Negative', 'Positive'],
    cbar_kws={'label': 'Count'}
)

plt.title('Confusion Matrix for Sentiment Analysis', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

# Add annotations explaining each quadrant
plt.text(0.5, 0.25, f'TN = {tn}\n(Correct)', ha='center', va='center', fontsize=10, color='white' if cm[0,0] > cm.max()/2 else 'black')
plt.text(1.5, 0.25, f'FP = {fp}\n(Type I Error)', ha='center', va='center', fontsize=10, color='white' if cm[0,1] > cm.max()/2 else 'black')
plt.text(0.5, 1.25, f'FN = {fn}\n(Type II Error)', ha='center', va='center', fontsize=10, color='white' if cm[1,0] > cm.max()/2 else 'black')
plt.text(1.5, 1.25, f'TP = {tp}\n(Correct)', ha='center', va='center', fontsize=10, color='white' if cm[1,1] > cm.max()/2 else 'black')

plt.tight_layout()
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(script_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
print("\n✓ Confusion matrix saved to 'confusion_matrix.png'")

# ============================================================================
# STEP 8: Key Takeaways
# ============================================================================
print("\n" + "=" * 70)
print("STEP 8: Key Takeaways")
print("=" * 70)

print("""
📊 METRICS EXPLAINED:

1. PRECISION - "How accurate are our positive predictions?"
   • Formula: TP / (TP + FP)
   • High precision = Few false alarms
   • Important when false positives are costly

2. RECALL - "How many actual positives did we find?"
   • Formula: TP / (TP + FN)
   • High recall = We find most positive cases
   • Important when missing positives is costly

3. F1-SCORE - "Balanced measure of precision and recall"
   • Formula: 2 × (Precision × Recall) / (Precision + Recall)
   • Harmonic mean gives equal weight to both metrics
   • Good for imbalanced datasets

4. ACCURACY - "Overall correctness"
   • Formula: (TP + TN) / Total
   • Can be misleading with imbalanced data
   • Simple but not always the best metric

🎯 TRADE-OFFS:
   • High precision, low recall: Conservative model (few predictions, but accurate)
   • Low precision, high recall: Aggressive model (many predictions, some wrong)
   • F1-score helps balance these trade-offs

💡 CONFUSION MATRIX:
   • Shows where the model succeeds and fails
   • Diagonal elements (TN, TP) are correct predictions
   • Off-diagonal elements (FP, FN) are errors
   • Helps identify if model is biased toward one class
""")

print("=" * 70)
print("Script completed successfully!")
print("=" * 70)
