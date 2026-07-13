import pandas as pd
import os
import sys

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
PRO_CSV   = os.path.join("results", "modal_magi_mmlu_pro_results_700.csv")
NORMAL_CSV = os.path.join("results", "modal_magi_mmlu_results.csv")

def load_and_analyze(csv_path, dataset_name):
    """Load a CSV, compute accuracy and return a summary DataFrame."""
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return None, None

    df = pd.read_csv(csv_path, on_bad_lines='skip')
    
    # Use the existing 'is_correct' column (already boolean)
    # If it's not boolean, convert: df['is_correct'] = df['is_correct'].astype(bool)
    total = len(df)
    correct = df['is_correct'].sum()
    accuracy = (correct / total) * 100

    # Time column is missing – we'll set avg_time to 0.0 and warn
    if 'time_seconds' in df.columns:
        avg_time = df['time_seconds'].mean()
    else:
        avg_time = 0.0
        print(f"⚠️  Warning: No 'time_seconds' column in {dataset_name}. Skipping average time.")

    # Group by category (using 'category' column)
    category_stats = df.groupby('category').agg(
        total_questions=('is_correct', 'count'),
        correct_answers=('is_correct', 'sum'),
        accuracy=('is_correct', lambda x: (x.sum() / len(x)) * 100),
        avg_time=('time_seconds', 'mean') if 'time_seconds' in df.columns else ('is_correct', lambda x: 0.0)
    ).round(2)

    # If time column missing, drop the avg_time column from category_stats
    if 'time_seconds' not in df.columns:
        category_stats = category_stats.drop(columns=['avg_time'])

    return total, accuracy, avg_time, category_stats

def print_results(total, accuracy, avg_time, category_stats, dataset_name):
    """Print a formatted results table."""
    print("=" * 50)
    print(f"📊 FINAL MAGI RESULTS – {dataset_name.upper()}")
    print("=" * 50)
    print(f"Total questions: {total}")
    print(f"General Accuracy: {accuracy:.2f}%")
    if avg_time > 0:
        print(f"Average time per question: {avg_time:.2f}s")
    else:
        print("Average time: not available")
    print("=" * 50)
    print("\n📈 Breakdown by category:")
    print(category_stats)

# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------
if __name__ == "__main__":
    pro_exists = os.path.exists(PRO_CSV)
    normal_exists = os.path.exists(NORMAL_CSV)

    if not pro_exists and not normal_exists:
        print("❌ No benchmark result files found in 'results/' folder.")
        sys.exit(1)

    if pro_exists:
        total, acc, avg_t, stats = load_and_analyze(PRO_CSV, "MMLU Pro")
        if stats is not None:
            print_results(total, acc, avg_t, stats, "MMLU Pro")
            print("\n")

    if normal_exists:
        total, acc, avg_t, stats = load_and_analyze(NORMAL_CSV, "MMLU Normal")
        if stats is not None:
            print_results(total, acc, avg_t, stats, "MMLU Normal")