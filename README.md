# MAGI: Multi-Agentic Generative Inference

MAGI is a modular, ontology-driven multi-agent framework for complex reasoning and non-linear thought processes. The system uses a topological knowledge graph where specialized agents delegate tasks based on confidence thresholds, synthesizing answers through a dynamic ontological base. Its core objective is to explore reasoning architectures that emulate distributed cognition.

## Key Features

- **Topological Knowledge Graph:** A persistent memory system (ChromaDB + NetworkX) that stores conceptual connections and dynamically merges similar ideas to avoid explosion.
- **Specialist Agents:** Modular experts in domains such as Social, Academic, Linguistic, and Domain-specific fields (Law, Biology, Philosophy, etc.).
- **Autonomous Delegation:** Agents evaluate their own confidence and delegate to peers using a high-confidence threshold (>0.95).
- **Live Visualization:** Generate interactive graphs to observe the growth of the ontological network in real-time.
- **Benchmarking:** Included scripts to evaluate performance on both **MMLU Normal** and **MMLU Pro** benchmarks.

## System Architecture

- **Orchestrator:** The central node that receives queries, analyzes context, and decides which specialists to consult or if it can infer directly.
- **Specialists:** Six base agents (Social, Academic, Emotional, Aesthetic, Linguistic) + domain-specific agents (Biology, Law, Philosophy, Physics, etc.).
- **KBS (Knowledge Base System):** ChromaDB stores vector embeddings of concepts and their interactions, allowing the system to retrieve relevant past insights for new queries.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/LAIn-co/MAGI.git
   cd MAGI
   ```

2. Create and activate a Python virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. (_Optional but required to run not listed models_) Download a GGUF model (e.g., Llama-3.1-8B or Llama-3.2-1B) and place it inside the pre_trained_models/ folder.


## Usage

### Interactive Chat

To start the interactive agent system:
```bash
python magi.py
```

_(Use /help inside the chat for a list of commands like /live, /orchestrate, and /visualize)_


### Running Benchmarks

To run the MMLU Normal or MMLU Pro benchmarks: 
```bash
python mmlu_normal_benchmark.py
python mmlu_pro_benchmark.py
```

_(Results are automatically saved in the results/ folder, and graphs in results/graphs/)_


### Evaluating results
To view the performance summary from the CSV files: 
```bash
python eval_results.py
```

_(Remember to edit the correct path)_


## Results & Outputs

- CSV: Results for MMLU Pro and Normal are stored in results/
- Graphs: Interactive HTML and JSON graph data are stored in results/graphs/
- Logs: Chat histories, thought traces, and errors are saved in .logs/ (git ignored)

## Version:

Multi-Agent Generative Inference System 
| Type: Pseudo-OneShot
| alpha-v1.1.1
| Alias: _Orchestrator_

## License
Licensed under the AGPLv3 license.
