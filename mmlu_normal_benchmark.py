import time
import csv
import os
import sys
import re

sys.path.append(os.path.dirname(__file__))
from magi import graph, ORCHESTRATOR_ID, SPECIALIST_IDS, get_llm_answer_for_agent

# ==========================================
# CONFIGURATION MMLU (NORMAL)
# ==========================================
CATEGORIES = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    # "college_computer_science", "college_mathematics", "college_medicine",
    # "college_physics", "computer_security", "conceptual_physics",
    # "econometrics", "electrical_engineering", "elementary_mathematics",
    # "formal_logic", "global_facts", "high_school_biology",
    # "high_school_chemistry", "high_school_computer_science",
    # "high_school_european_history", "high_school_geography",
    # "high_school_government_and_politics", "high_school_macroeconomics",
    # "high_school_mathematics", "high_school_microeconomics",
    # "high_school_physics", "high_school_psychology", "high_school_statistics",
    # "high_school_us_history", "high_school_world_history", "human_aging",
    # "human_sexuality", "international_law", "jurisprudence",
    # "logical_fallacies", "machine_learning", "management", "marketing",
    # "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    # "nutrition", "philosophy", "prehistory", "professional_accounting",
    # "professional_law", "professional_medicine", "professional_psychology",
    # "public_relations", "security_studies", "sociology", "us_foreign_policy",
    # "virology", "world_religions"
]              # Empty list = fetch all categories          
SAMPLES_PER_CATEGORY = 5 
RESULTS_DIR = "results"
GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(RESULTS_DIR, "mmlu_normal_benchmark_results.csv")
# ==========================================

def build_llm_funcs():
    llm_funcs = {}
    for aid in SPECIALIST_IDS:
        def make_func(aid):
            return lambda p, aid=aid: get_llm_answer_for_agent(aid, p)[0]
        llm_funcs[aid] = make_func(aid)
    llm_funcs[ORCHESTRATOR_ID] = lambda p: get_llm_answer_for_agent(ORCHESTRATOR_ID, p)[0]
    return llm_funcs

def load_mmlu_normal(categories, n_samples=5):
    try:
        from datasets import load_dataset
        # KEY CHANGE: Uses "cais/mmlu" instead of "TIGER-Lab/MMLU-Pro"
        dataset = load_dataset("cais/mmlu", "all", split="test")
        
        if not categories:
            categories = list(set(dataset['subject']))
        
        final_data = []
        for cat in categories:
            cat_data = dataset.filter(lambda x: x['subject'] == cat)
            if len(cat_data) > 0:
                n = min(n_samples, len(cat_data))
                final_data.extend(cat_data.select(range(n)))
            else:
                print(f"⚠️ Category '{cat}' not found in MMLU")
        return final_data
    except ImportError:
        print("❌ Please install the dataset library: pip install datasets")
        return []

def format_prompt(question, choices):
    prompt = f"Question: {question}\nChoices:\n"
    for i, opt in enumerate(choices):
        letter = chr(65 + i)
        prompt += f"{letter}) {opt}\n"
    prompt += "\nAnswer (respond only with a single capital letter A-D. STRICTLY FOLLOW THIS INSTRUCTION: Do NOT add any other text, explanations, or punctuation):"
    return prompt

def extract_letter(response):
    # 1. Strict regex: Look for "Answer: X", "Correct choice X", etc.
    match = re.search(r'[Cc]orrect\s*(?:choice|option)?\s*(?:is|:)?\s*([A-D])', response)
    if match:
        return match.group(1)

    # 2. Isolated letter regex: Look for any A-D bounded by word boundaries
    match = re.search(r'\b([A-D])\b', response)
    if match:
        return match.group(1)

    # 3. Relentless fallback: Search line by line for answer/choice contexts
    lines = response.split('\n')
    for line in lines:
        if 'answer' in line.lower() or 'choice' in line.lower():
            match = re.search(r'([A-D])', line)
            if match:
                return match.group(1)
    
    # 4. Last resort: return the first A-D found in the whole string
    found_letters = [char for char in response if char in 'ABCD']
    if found_letters:
        return found_letters[0]

    return ''

def run_benchmark():
    llm_funcs = build_llm_funcs()
    data = load_mmlu_normal(CATEGORIES, n_samples=SAMPLES_PER_CATEGORY)
    
    if not data:
        return

    total_questions = len(data)
    print(f"📊 MMLU (Normal) Benchmark started. Total questions: {total_questions}\n")
    
    file_exists = os.path.isfile(OUTPUT_CSV)
    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["category", "question", "correct_answer_letter", "predicted_answer", "time_seconds", "full_response"])
    
    processed = 0
    total_time_spent = 0.0
    
    for item in data:
        processed += 1
        start_time = time.perf_counter()
        
        # KEY CHANGE: Normal MMLU uses "choices" (not "options")
        prompt = format_prompt(item['question'], item['choices'])
        
        # === CALL TO MAGI ===
        response, _ = graph.interact_workflow(
            query=prompt,
            orchestrator_id=ORCHESTRATOR_ID,
            specialist_ids=SPECIALIST_IDS,
            llm_funcs=llm_funcs,
            max_rounds=3,
            verbose=False,
            log_func=None,
            live_graph=True,
            script_dir=GRAPHS_DIR  # Saves graphs inside results/graphs/
        )
        # ==========================
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        total_time_spent += duration
        
        predicted = extract_letter(response)
        
        # KEY CHANGE: Normal MMLU uses "answer" (0-3 index)
        correct_idx = item['answer']
        correct_letter = chr(65 + correct_idx)
        
        with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([item['subject'], item['question'], correct_letter, predicted, round(duration, 2), response])
        
        avg_time = total_time_spent / processed if processed > 0 else 0
        remaining = total_questions - processed
        eta_sec = avg_time * remaining
        eta_m = int(eta_sec // 60)
        eta_s = int(eta_sec % 60)
        
        print(f"[{processed}/{total_questions}] | Time: {duration:.2f}s | Average: {avg_time:.1f}s | ETA: {eta_m}m {eta_s}s")

    print(f"\n✅ Benchmark finished. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_benchmark()