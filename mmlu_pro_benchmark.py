import time
import csv
import os
import sys
import re

sys.path.append(os.path.dirname(__file__))
from magi import graph, ORCHESTRATOR_ID, SPECIALIST_IDS, get_llm_answer_for_agent

# ==========================================
# CONFIGURATION MMLU PRO
# ==========================================
CATEGORIES = [
    # "history", "physics", "philosophy", "engineering", "computer science", 
    # "chemistry", "health", "business", "math", "biology", "law", "other", 
    # "economics", "psychology"
]              # Empty list = fetch all categories
SAMPLES_PER_CATEGORY = 5
RESULTS_DIR = "results"
GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(RESULTS_DIR, "mmlu_pro_benchmark_results.csv")
# ==========================================

def build_llm_funcs():
    llm_funcs = {}
    for aid in SPECIALIST_IDS:
        def make_func(aid):
            return lambda p, aid=aid: get_llm_answer_for_agent(aid, p)[0]
        llm_funcs[aid] = make_func(aid)
    llm_funcs[ORCHESTRATOR_ID] = lambda p: get_llm_answer_for_agent(ORCHESTRATOR_ID, p)[0]
    return llm_funcs

def load_mmlu_pro(categories, n_samples=5):
    try:
        from datasets import load_dataset
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        
        if not categories:
            categories = list(set(dataset['category']))
        
        filtered_data = dataset.filter(lambda x: x['category'] in categories)
        final_data = []
        for cat in categories:
            cat_data = filtered_data.filter(lambda x: x['category'] == cat)
            final_data.extend(cat_data.select(range(min(n_samples, len(cat_data)))))
        return final_data
    except ImportError:
        print("❌ Please install the dataset library: pip install datasets")
        return []

def format_prompt(question, options):
    prompt = f"Question: {question}\nChoices:\n"
    for i, opt in enumerate(options):
        letter = chr(65 + i)
        prompt += f"{letter}) {opt}\n"
    # Force the model to respond with just the letter
    prompt += "\nAnswer (respond only with a single capital letter A-J. STRICTLY FOLLOW THIS INSTRUCTION: Do NOT add any other text, explanations, or punctuation):"
    return prompt

def extract_letter(response):
    # 1. Strict regex: Look for "Answer: X", "Correct choice X", etc.
    match = re.search(r'[Cc]orrect\s*(?:choice|option)?\s*(?:is|:)?\s*([A-J])', response)
    if match:
        return match.group(1)

    # 2. Isolated letter regex: Look for any A-J bounded by word boundaries
    match = re.search(r'\b([A-J])\b', response)
    if match:
        return match.group(1)

    # 3. Relentless fallback: Search line by line for answer/choice contexts
    lines = response.split('\n')
    for line in lines:
        if 'answer' in line.lower() or 'choice' in line.lower():
            match = re.search(r'([A-J])', line)
            if match:
                return match.group(1)
    
    # 4. Last resort: return the first A-J found in the whole string
    found_letters = [char for char in response if char in 'ABCDEFGHIJ']
    if found_letters:
        return found_letters[0]

    return ''

def run_benchmark():
    llm_funcs = build_llm_funcs()
    data = load_mmlu_pro(CATEGORIES, n_samples=SAMPLES_PER_CATEGORY)
    
    if not data:
        return

    total_questions = len(data)
    print(f"📊 MMLU Pro Benchmark started. Total questions: {total_questions}\n")
    
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
        
        prompt = format_prompt(item['question'], item['options'])
        
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
        
        # MMLU Pro returns answer as a number string or a letter string
        raw_answer = item['answer']
        if isinstance(raw_answer, str) and raw_answer.isdigit():
            correct_idx = int(raw_answer)
            correct_letter = chr(65 + correct_idx)
        else:
            correct_letter = str(raw_answer).strip().upper()
        
        with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([item['category'], item['question'], correct_letter, predicted, round(duration, 2), response])
        
        avg_time = total_time_spent / processed if processed > 0 else 0
        remaining = total_questions - processed
        eta_sec = avg_time * remaining
        eta_m = int(eta_sec // 60)
        eta_s = int(eta_sec % 60)
        
        print(f"[{processed}/{total_questions}] | Time: {duration:.2f}s | Average: {avg_time:.1f}s | ETA: {eta_m}m {eta_s}s")

    print(f"\n✅ Benchmark finished. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_benchmark()