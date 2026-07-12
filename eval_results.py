import pandas as pd
import os
import sys

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
# Paths to the benchmark result files (both can coexist)
PRO_CSV   = os.path.join("results", "mmlu_pro_benchmark_results.csv")
NORMAL_CSV = os.path.join("results", "mmlu_normal_benchmark_results.csv")

def load_and_analyze(csv_path, dataset_name):
    """Load a CSV, compute accuracy and return a summary DataFrame."""
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return None, None

    df = pd.read_csv(csv_path, on_bad_lines='skip')
    
    # Mark correct predictions
    df['correct'] = df['correct_answer_letter'] == df['predicted_answer']

    total = len(df)
    correct = df['correct'].sum()
    accuracy = (correct / total) * 100
    avg_time = df['time_seconds'].mean()

    # Group by category
    category_stats = df.groupby('category').agg(
        total_questions=('correct', 'count'),
        correct_answers=('correct', 'sum'),
        accuracy=('correct', lambda x: (x.sum() / len(x)) * 100),
        avg_time=('time_seconds', 'mean')
    ).round(2)

    return total, accuracy, avg_time, category_stats

def print_results(total, accuracy, avg_time, category_stats, dataset_name):
    """Print a formatted results table."""
    print("=" * 50)
    print(f"📊 FINAL MAGI RESULTS – {dataset_name.upper()}")
    print("=" * 50)
    print(f"Total questions: {total}")
    print(f"General Accuracy: {accuracy:.2f}%")
    print(f"Average time per question: {avg_time:.2f}s")
    print("=" * 50)
    print("\n📈 Breakdown by category:")
    print(category_stats)

# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------
if __name__ == "__main__":
    # Try to load both result files, if they exist
    pro_exists = os.path.exists(PRO_CSV)
    normal_exists = os.path.exists(NORMAL_CSV)

    if not pro_exists and not normal_exists:
        print("❌ No benchmark result files found in 'results/' folder.")
        print("   Expected files: mmlu_pro_benchmark_results.csv or mmlu_normal_benchmark_results.csv")
        sys.exit(1)

    # If both exist, print both. Otherwise, just print the one that exists.
    if pro_exists:
        total, acc, avg_t, stats = load_and_analyze(PRO_CSV, "MMLU Pro")
        if stats is not None:
            print_results(total, acc, avg_t, stats, "MMLU Pro")
            print("\n")

    if normal_exists:
        total, acc, avg_t, stats = load_and_analyze(NORMAL_CSV, "MMLU Normal")
        if stats is not None:
            print_results(total, acc, avg_t, stats, "MMLU Normal")