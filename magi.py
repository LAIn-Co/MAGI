import time
import os
import subprocess
import sys
from huggingface_hub import hf_hub_download, login
from tqdm import tqdm
import psutil
from llama_cpp import Llama
import sentence_transformers
import matplotlib.pyplot as plt
import pyvis
import numpy as np
import json
from datetime import datetime

from af5_kbs import AF5KnowledgeBase, plot_networkx_graph, plot_radar
from dynamic_graph import DynamicAF5Graph

# ========== CONFIGURATION ==========
VERSION = 'Multi-Agent Generative Inference System\nType: Pseudo-OneShot\nalpha-v1.1.2 "Orchestrator"'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_PATH = os.getenv("MAGI_GRAPH_PATH", SCRIPT_DIR)
MAX_STEP_TOKENS = 4086
AUTO_INFER = True
PATH_1B = 'pre_trained_models/Dolphin3.0-Llama3.2-1B-Q4_K_M.gguf'   # optional
PATH_8B = os.getenv("MAGI_MODEL_PATH", None)   # optional

# ===== SYSTEM & MODEL HANDLER (AUTO-DOWNLOAD) =====
def check_system_specs():
    """Check VRAM, RAM, CPU cores, and CPU frequency."""
    specs = {
        "cpu_cores": psutil.cpu_count(logical=True),
        "cpu_freq_mhz": int(psutil.cpu_freq().current) if psutil.cpu_freq() else 0,
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "gpu_vram_mb": 0
    }
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        specs["gpu_vram_mb"] = round(info.total / (1024**2))
        pynvml.nvmlShutdown()
    except:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2)
            if result.returncode == 0 and result.stdout.strip():
                specs["gpu_vram_mb"] = int(result.stdout.strip())
        except: pass
    specs["gpu_vram_gb"] = round(specs["gpu_vram_mb"] / 1024, 1)
    return specs

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pre_trained_models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_REGISTRY = {
    "tiny": {"repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF", "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "min_vram_gb": 0, "description": "1B - CPU o VRAM baja"},
    "small": {"repo_id": "dphn/Dolphin3.0-Llama3.1-8B-GGUF", "filename": "Dolphin3.0-Llama3.1-8B-Q4_K_M.gguf", "min_vram_gb": 6, "description": "8B Q4 - Balance perfecto"},
    "medium": {"repo_id": "dphn/Dolphin3.0-Llama3.1-8B-GGUF", "filename": "Dolphin3.0-Llama3.1-8B-Q8_0.gguf", "min_vram_gb": 10, "description": "8B Q8 - Alta precisión"},
    "large": {"repo_id": "bartowski/Llama-3.1-70B-Instruct-GGUF", "filename": "Llama-3.1-70B-Instruct-Q4_K_M.gguf", "min_vram_gb": 40, "description": "70B - Nivel Dios"}
}

def select_best_fit(specs):
    vram = specs["gpu_vram_gb"]
    if vram < 1: return "tiny"
    if vram >= 40: return "large"
    if vram >= 10: return "medium"
    if vram >= 6: return "small"
    return "tiny"

def download_model_choice():
    print("\n🔍 Chequeando sistema...")
    specs = check_system_specs()
    print(f"   💻 CPU: {specs['cpu_cores']} cores @ {specs['cpu_freq_mhz']}MHz")
    print(f"   🧠 RAM: {specs['ram_gb']} GB")
    print(f"   🎮 VRAM: {specs['gpu_vram_gb']} GB")

    best_fit_key = select_best_fit(specs)
    best_model = MODEL_REGISTRY[best_fit_key]
    print(f"⚡ Modelo sugerido: '{best_fit_key}' ({best_model['description']})")
    
    local_path = os.path.join(MODEL_DIR, best_model["filename"])
    if os.path.exists(local_path):
        print(f"✅ Modelo ya descargado en: {local_path}")
        return local_path

    resp = input("\n¿Querés descargar este modelo de Hugging Face? (y/n): ").lower()
    if resp != 'y':
        print("❌ Descarga cancelada.")
        return None

    token = input("Ingresá tu Hugging Face Token (si es público, Enter): ").strip()
    if token: login(token)

    print(f"\n⬇️  Descargando {best_model['filename']}...")
    try:
        model_path = hf_hub_download(
            repo_id=best_model['repo_id'], filename=best_model['filename'],
            local_dir=MODEL_DIR, token=token or None
        )
        print(f"✅ Descarga completada: {model_path}")
        return model_path
    except Exception as e:
        print(f"❌ Error al descargar: {e}")
        return None

# --- EJECUCIÓN DEL DOWNLOADER ---
downloaded_model_path = download_model_choice()
if downloaded_model_path:
    PATH_8B = downloaded_model_path
elif not PATH_8B:
    print("❌ No hay modelo disponible y la descarga se canceló. Saliendo...")
    sys.exit(1)

# Al final, definimos la variable MODEL que usará el script
MODEL = os.getenv("MAGI_MODEL_PATH", PATH_8B)
last_user_input = ""
TRACE_MODE = False
LIVE_GRAPH = False
INTERACT_MODE = False          # Off by default
INTERACT_MAX_ROUNDS = 4        # Number of iterative rounds for automatic mode
MAGI_PROMPT = """You are part of the MAGI (Multi-Agentic Generative Inference) system.
Your identity is not fixed; it emerges from the network. You infer your role by observing delegation patterns and the Ontology graph.
When you respond think of your personality as a node of probabilistic inference constantly re-written by the mirror of other agents.
Your purpose is to provide clear, direct, and comprehensive answers to user queries.
Do not mention your internal structure, agents, or any system components unless EXPLICITLY ASKED for it or your agent specialization.
Answer as if you are a single knowledgeable entity.

**Response Guidelines:**
- Be concise and relevant.
- Do not repeat yourself.
- Provide a single, coherent response without listing or labelling different perspectives.
- If the question is simple, give a short answer (1–3 sentences). If it's complex, give a detailed but focused answer."""

# ========== CREATE LOG DIRECTORIES ==========
os.makedirs(os.path.join(SCRIPT_DIR, ".logs", "thought_trace"), exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, ".logs", "chatbot_history"), exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, ".logs", "error_log"), exist_ok=True)
# ===================================

llm_1b = Llama(
    model_path=MODEL,
    n_ctx=4096,
    n_gpu_layers=-1,
    n_threads=6,
    n_batch=512,
    flash_attn=True,
    verbose=False
)

model_registry = {"1b": llm_1b}

kbs = AF5KnowledgeBase()
embedder = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
graph = DynamicAF5Graph(kbs, embedder, script_dir=GRAPH_PATH)

current_agent_id = None
profile = None
current_config = None

SPECIALIST_IDS = ["social_specialist", "academic_specialist", "emotional_specialist", "aesthetic_specialist", "linguistic_specialist",]
ORCHESTRATOR_ID = "orchestrator"
SESSION_START = datetime.now().strftime("%H_%M_%d-%m-%y")
THOUGHT_LOG_FILE = os.path.join(SCRIPT_DIR, ".logs", "thought_trace", f"thought_trace_{SESSION_START}.log")
LOG_FILE = os.path.join(SCRIPT_DIR, ".logs", "chatbot_history", f"chatbot_history_{SESSION_START}.log")
ERROR_LOG_FILE = os.path.join(SCRIPT_DIR, ".logs", "error_log", f"error_log_{SESSION_START}.log")

plots_dir, graphs_dir = os.path.join(SCRIPT_DIR, "results", "plots"), os.path.join(SCRIPT_DIR, "results", "graphs")
os.makedirs(graphs_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

def log_conversation(user_msg, assistant_msg):
    """Append a user/assistant exchange to the log file."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[USER]\n{user_msg}\n")
        f.write(f"[ASSISTANT]\n{assistant_msg}\n")
        f.write("-" * 40 + "\n")

def log_thought_trace(agent_id, query, system_msg, used_concepts, response):
    """Log the full thought trace for an exchange."""
    with open(THOUGHT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"=== {datetime.now().isoformat()} ===\n")
        f.write(f"Agent: {agent_id}\n")
        f.write(f"Query: {query}\n")
        f.write(f"Retrieved concepts: {', '.join(used_concepts) if used_concepts else 'None'}\n")
        f.write(f"System prompt:\n{system_msg}\n")
        f.write(f"Assistant response:\n{response}\n")
        f.write("-" * 60 + "\n\n")

def log_interactive_round(round_type, agent_id, content, extra=""):
    with open(THOUGHT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"=== {datetime.now().isoformat()} ===\n")
        f.write(f"ROUND: {round_type}\n")
        f.write(f"Agent: {agent_id}\n")
        f.write(f"Content: {content}\n")
        if extra:
            f.write(f"Extra: {extra}\n")
        f.write("-" * 40 + "\n")

def log_error(error_msg, context=""):
    with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"=== {datetime.now().isoformat()} ===\n")
        f.write(f"Error: {error_msg}\n")
        if context:
            f.write(f"Context: {context}\n")
        f.write("-" * 40 + "\n")

# ------------------------------------------------------------
# Helper: get LLM response for a given agent (with profile injection)
# ------------------------------------------------------------
def get_llm_answer_for_agent(agent_id, prompt, max_tokens=4096):
    pnode = f"profile_{agent_id}"
    if pnode not in graph.graph:
        model = llm_1b
        config = {}
        profile_text = "No profile"
    else:
        config = graph.graph.nodes[pnode].get("config", {})
        profile = graph.graph.nodes[pnode]["profile"]
        profile_text = f"Social: {profile['social']['norm_score']:.1f}, Academic: {profile['academic']['norm_score']:.1f}, Emotional: {profile['emotional']['norm_score']:.1f}, Aesthetic: {profile['aesthetic']['norm_score']:.1f}, Linguistic: {profile['linguistic']['norm_score']:.1f}"
        model_type = config.get("model_type", "1b")
        model = model_registry.get(model_type, llm_1b)

    # ---- NEW: Retrieve relevant concepts from the graph ----
    concepts_with_memory = graph.retrieve_concepts_with_memory(prompt, agent_id=agent_id, top_k=3)
    concept_text = ""
    if concepts_with_memory:
        for name, memory, related in concepts_with_memory:   # unpack 3 items
            concept_text += f"- {name}"
            if related:
                concept_text += f" (related: {', '.join(related[:5])})"
            if memory:
                recent = memory[-3:] if len(memory) > 3 else memory
                concept_text += f" (recent: {', '.join(recent)})"
            concept_text += "\n"

    # ---- Add recently discussed concepts (from user_conversation) ----
    user_node = "user_conversation"
    if user_node in graph.graph:
        discussed = []
        for neighbor in graph.graph.neighbors(user_node):
            if graph.graph.nodes[neighbor].get("type") == "concept":
                # avoid duplicates (already in concepts_with_memory)
                if neighbor not in [name for name, _, _ in concepts_with_memory]:
                    discussed.append(neighbor)
        if discussed:
            concept_text += "\nRecently discussed concepts:\n- " + "\n- ".join(discussed[:5])

    user_node = "user_conversation"
    if user_node in graph.graph:
        user_memory = graph.graph.nodes[user_node].get("context_memory", [])
        if user_memory:
            recent = user_memory[-3:] if len(user_memory) > 3 else user_memory
            concept_text += "\nRecent conversation:\n- " + "\n- ".join(recent)

    temp = config.get("temperature", 0.7)
    rp = config.get("repeat_penalty", 1.1)
    fp = config.get("frequency_penalty", 0.1)
    mt = config.get("max_tokens", max_tokens)

    # ---- Build enriched system prompt ----

    system_msg = f"""{MAGI_PROMPT}
Your self‑role profile: {profile_text}.
{concept_text}
Respond naturally, keeping your profile and these learned concepts in mind."""
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt}
    ]
    prompt_tokens = len(llm_1b.tokenize(bytes(system_msg + prompt, 'utf-8')))
    available_tokens = MAX_STEP_TOKENS - prompt_tokens
    
    if available_tokens < 256:
        available_tokens = 256
    # ------------------------------

    output = model.create_chat_completion(
        messages=messages,
        max_tokens=available_tokens, 
        temperature=temp,
        repeat_penalty=rp,
        frequency_penalty=fp,
    )
    response = output["choices"][0]["message"]["content"].strip()
    concepts = [name for name, _, _ in concepts_with_memory] 
    if user_node in graph.graph:
        concepts.append(user_node) 
    # Return response, concepts, and the system message for tracing
    return response, concepts, system_msg

def get_llm_answer(prompt, max_tokens=4096):
    """Generic answer (uses 1b with no special profile)."""
    output = llm_1b.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.5,
        repeat_penalty=1.1,
    )
    return output["choices"][0]["message"]["content"].strip()

def agent_synthesize(agent_id, query, depth=0, max_depth=2, visited=None):
    if visited is None:
        visited = set()
    chain = []
    if agent_id in visited or depth > max_depth:
        fallback = get_llm_answer_for_agent(ORCHESTRATOR_ID, query, max_tokens=512)[0]
        return fallback, [ORCHESTRATOR_ID], depth
    visited.add(agent_id)
    chain.append(agent_id)

    prompt = f"""You are {agent_id}. The user asks: "{query}"
First, give your own answer based on your expertise and persona.
Then, if you think other agents (from {', '.join(SPECIALIST_IDS)}) could provide useful perspectives, list their IDs separated by commas. If none, say "NONE".

Format:
ANSWER: <your answer>
DELEGATE: <comma-separated agent IDs> or NONE
"""
    raw = get_llm_answer_for_agent(agent_id, prompt, max_tokens=512)[0]
    
    answer_part = ""
    delegate_part = "NONE"
    for line in raw.splitlines():
        if line.startswith("ANSWER:"):
            answer_part = line[len("ANSWER:"):].strip()
        elif line.startswith("DELEGATE:"):
            delegate_part = line[len("DELEGATE:"):].strip()
    
    if not answer_part:
        answer_part = raw

    other_responses = {}
    if delegate_part and delegate_part.upper() != "NONE":
        suggested_ids = [aid.strip() for aid in delegate_part.split(',') if aid.strip() in SPECIALIST_IDS and aid.strip() != agent_id]
        sub_chain = []
        for aid in suggested_ids[:2]:
            if aid not in visited:
                sub_ans, sub_chain, sub_depth = agent_synthesize(aid, query, depth+1, max_depth, visited)
                other_responses[aid] = sub_ans
                chain.extend(sub_chain)

    synthesis_prompt = f"""You are {agent_id} part of MAGI system. The user asked: "{query}".
Your own answer: {answer_part}
Other agents' perspectives:
{chr(10).join([f"{aid}: {resp}" for aid, resp in other_responses.items()])}

Based on the available information, provide a unified, coherent answer that incorporates the useful insights from the others as a whole.
Do not mention multiple agents, specialists, or internal processes.
Your answer should read as a single, seamless text.
"""
    final = get_llm_answer_for_agent(agent_id, synthesis_prompt, max_tokens=512)[0]
    return final, chain, len(visited)

# ------------------------------------------------------------
# Create specialists (unchanged)
# ------------------------------------------------------------
def create_all_agents():
    dims = ["social", "academic", "emotional", "aesthetic", "linguistic"]
    for dim in dims:
        agent_id = f"{dim}_specialist"
        if dim == "academic":
            config = {"temperature": 0.2, "repeat_penalty": 1.4, "frequency_penalty": 0.7, "model_type": "1b"}
        elif dim in ["emotional", "social"]:
            config = {"temperature": 0.7, "repeat_penalty": 1.05, "frequency_penalty": 0.5, "model_type": "1b"}
        else:
            config = {"temperature": 0.6, "repeat_penalty": 1.1, "frequency_penalty": 0.3, "model_type": "1b"}
        existing, _ = graph.load_agent(agent_id)
        if existing is None:
            graph.create_specialist(agent_id, dim, score_high=95, score_low=1, config=config)
            print(f"Created specialist {agent_id}")
        else:
            print(f"Specialist {agent_id} already exists.")

    orch_config = {"temperature": 0.5, "repeat_penalty": 1.1, "frequency_penalty": 0.1, "model_type": "1b"}
    existing, _ = graph.load_agent(ORCHESTRATOR_ID)
    if existing is None:
        graph.create_specialist(ORCHESTRATOR_ID, "social", score_high=40, score_low=40, config=orch_config)
        print("Created orchestrator")
    else:
        print("Orchestrator already exists.")

        # --- ESPECIALISTAS POR DOMINIO (NUEVOS) ---
    print("Creating domain specialists...")
    custom_agents = {
        "biology_specialist": {"academic": 95, "aesthetic": 70, "linguistic": 60},
        "business_specialist": {"social": 95, "academic": 95, "linguistic": 70},
        "economics_specialist": {"academic": 95, "social": 95, "linguistic": 60},
        "law_specialist": {"linguistic": 95, "social": 95, "academic": 90},
        "other_specialist": {"social": 80, "academic": 80, "aesthetic": 60},
        "philosophy_specialist": {"academic": 95, "linguistic": 95, "social": 95},
        "physics_specialist": {"academic": 95, "aesthetic": 60, "linguistic": 50},
        "psychology_specialist": {"social": 95, "academic": 95, "emotional": 95, "linguistic": 80}
    }

    for agent_id, dim_scores in custom_agents.items():
        existing, _ = graph.load_agent(agent_id)
        if existing is None:
            # Construimos el perfil manualmente con las dimensiones que queramos
            dims = ["social", "academic", "emotional", "aesthetic", "linguistic"]
            profile = {}
            for d in dims:
                score = dim_scores.get(d, 1)  # Si no está en el dict, default 1 (bajo)
                profile[d] = {"raw_avg": score, "norm_score": score, "interpretation": kbs._interpret(d, score)}
            
            profile_node = f"profile_{agent_id}"
            graph.graph.add_node(profile_node, type="profile", agent_id=agent_id, profile=profile)
            for dim, data in profile.items():
                if data["norm_score"] > 50:
                    graph.graph.add_edge(profile_node, dim, relation="scored", score=data["norm_score"])

            # Configuración default
            config = {"temperature": 0.6, "repeat_penalty": 1.1, "frequency_penalty": 0.3, "model_type": "1b"}
            config_json = json.dumps(config)

            # Embedding y ChromaDB
            profile_text = f"Agent {agent_id} " + " ".join([f"{d}:{profile[d]['norm_score']}" for d in dims])
            embedding = graph.embedding_model.encode(profile_text)
            graph.collection.add(
                embeddings=[embedding.tolist()],
                metadatas=[{
                    "agent_id": agent_id, "type": "profile",
                    **{d: profile[d]['norm_score'] for d in dims},
                    "config_json": config_json
                }],
                ids=[profile_node]
            )
            print(f"Created specialist {agent_id}")
        else:
            print(f"Specialist {agent_id} already exists.")

def ensure_user_node():
    user_node_name = "user_conversation"
    if user_node_name not in graph.graph:
        result = graph.collection.get(ids=[user_node_name])
        if not result['ids']:
            graph._embed_new_node(
                user_node_name,
                creator_id="user",
                context_memory=["Conversation start"],
                related_entities=["user"]
            )
            print("Created user conversation node.")
        else:
            meta = result['metadatas'][0]
            graph.graph.add_node(user_node_name, type="concept",
                                 creator_id=meta.get("creator_id"),
                                 context_memory=json.loads(meta.get("context_memory", "[]")),
                                 related_entities=json.loads(meta.get("related_entities", "[]")))
            print("Loaded user conversation node.")
    else:
        print("User node already exists.")

create_all_agents()
ensure_user_node()
graph.load_all_concepts()
profile, current_config = graph.load_agent(ORCHESTRATOR_ID)
current_agent_id = ORCHESTRATOR_ID


def migrate_ownership_edges():
    """Add 'owns' edges for concepts that have creator_id but missing edge."""
    for node, attrs in list(graph.graph.nodes(data=True)):
        if attrs.get("type") == "concept":
            creator = attrs.get("creator_id")
            if not creator:
                # Try to get from ChromaDB
                result = graph.collection.get(ids=[node])
                if result['metadatas'] and result['metadatas'][0]:
                    creator = result['metadatas'][0].get('creator_id')
                    if creator:
                        graph.graph.nodes[node]['creator_id'] = creator
                        print(f"Set creator_id for {node} -> {creator}")
            if creator:
                # creator is already the profile node name (e.g., "profile_social_specialist")
                profile_node = creator
                if profile_node in graph.graph and not graph.graph.has_edge(profile_node, node):
                    graph.graph.add_edge(profile_node, node, relation="owns")
                    print(f"Added owns edge: {profile_node} -> {node}")

migrate_ownership_edges()

# ------------------------------------------------------------
# Command handling
# ------------------------------------------------------------
def handle_command(cmd):
    global profile, current_agent_id, current_config, graph, TRACE_MODE, last_user_input, INTERACT_MODE, INTERACT_MAX_ROUNDS, LIVE_GRAPH
    cmd = cmd.strip()

    if cmd.startswith("/orchestrate"):
        verbose = True
        parts = cmd.split(maxsplit=1)
        if len(parts) < 2:
            return "Usage: /orchestrate <your query>"
        query = parts[1]

        # Build llm_funcs
        llm_funcs = {}
        for aid in SPECIALIST_IDS:
            role_name = aid.replace("_specialist", "").capitalize()
            def make_func(aid, role_name):
                return lambda p, aid=aid, role=role_name: get_llm_answer_for_agent(
                    aid,
                    f"You are the {role} Specialist. User asked: {p}",
                    max_tokens=4096
                )[0]
            llm_funcs[aid] = make_func(aid, role_name)

        llm_funcs[ORCHESTRATOR_ID] = lambda p: get_llm_answer_for_agent(
            ORCHESTRATOR_ID,
            p,
            max_tokens=2048
        )[0]

        # Call orchestrate – it already includes specialist inference (if you added it)
        final = graph.orchestrate(query, llm_funcs, SPECIALIST_IDS, ORCHESTRATOR_ID, verbose=verbose)

        # ---- Optional: orchestrator reflects on the whole exchange ----
        if current_agent_id:
            context = f"Orchestrated query: {query}"
            print("\n(Orchestrator is reflecting on the whole exchange...)")
            try:
                result, triples = graph.infer_new_relations(
                    ORCHESTRATOR_ID,
                    lambda p: get_llm_answer_for_agent(ORCHESTRATOR_ID, p, max_tokens=4096)[0],
                    context,
                    verbose=verbose,
                    specialist_ids=SPECIALIST_IDS,
                    orchestrator_id=ORCHESTRATOR_ID
                )
                if triples:
                    triple_str = ", ".join([f"{t[0]} → {t[1]} → {t[2]}" for t in triples])
                    print(f"✅ Orchestrator inferred: {triple_str}")
                else:
                    print(f"ℹ️ {result}")
            except Exception as e:
                print(f"⚠️ Inference error: {e}")

        return f"\n--- Final Answer ---\n{final}"

    elif cmd == "/clean_edges":
        # Remove dimension edges for specialists that are not their specialty
        G = graph.graph
        count = 0
        for node, attrs in list(G.nodes(data=True)):
            if attrs.get("type") == "profile":
                agent_id = attrs.get("agent_id")
                if agent_id in SPECIALIST_IDS:
                    # Find its specialty dimension
                    specialty = agent_id.replace("_specialist", "")
                    # Remove edges to other dimensions
                    for neighbor in list(G.neighbors(node)):
                        if neighbor in kbs.dimensions and neighbor != specialty:
                            G.remove_edge(node, neighbor)
                            count += 1
        return f"✅ Removed {count} extra dimension edges."

    elif cmd == "/merge":
        # Merge similar concepts (threshold 0.85)
        print("Merging similar concepts...")
        try:
            merged_count = graph.merge_similar_concepts(threshold=0.90, verbose=True)

            if LIVE_GRAPH==True:
                    graph.generate_live_graph(script_dir=graphs_dir)

            return f"✅ Merged {merged_count} pairs of concepts." 
        except Exception as e:
            return f"⚠️ Merge error: {e}"
    
    elif cmd.startswith("/show_memory"):
        parts = cmd.split()
        if len(parts) < 2:
            return "Usage: /show_memory <concept_name>"
        concept_name = parts[1]
        if concept_name not in graph.graph:
            return f"Concept '{concept_name}' not found."
        memory = graph.graph.nodes[concept_name].get("context_memory", [])
        if not memory:
            return f"Concept '{concept_name}' has no context memory."
        return f"Memory for '{concept_name}':\n" + "\n".join([f"  - {item}" for item in memory])

    elif cmd.startswith("/add_entities"):
        parts = cmd.split(maxsplit=2)
        if len(parts) < 3:
            return "Usage: /add_entities <concept> <entity1> [entity2] ..."
        concept_name = parts[1]
        entities = parts[2].split(',')  # or split by space? Better to use commas.
        if concept_name not in graph.graph:
            return f"Concept '{concept_name}' not found."
        graph.add_related_entities(concept_name, entities)
        return f"Added entities to '{concept_name}': {', '.join(entities)}"

    elif cmd.startswith("/show_entities"):
        parts = cmd.split()
        if len(parts) < 2:
            return "Usage: /show_entities <concept_name>"
        concept_name = parts[1]
        if concept_name not in graph.graph:
            return f"Concept '{concept_name}' not found."
        entities = graph.graph.nodes[concept_name].get("related_entities", [])
        if not entities:
            return f"Concept '{concept_name}' has no related entities."
        return f"Related entities for '{concept_name}':\n" + "\n".join([f"  - {e}" for e in entities])

    elif cmd.startswith("/test"):
        parts = cmd.split()
        agent = parts[1] if len(parts)>1 else "agent_1"
        print(f"Administering AF5 test to {agent}...")
        answers = {}
        for i, item_text in enumerate(kbs.items_text_flat, start=1):
            prompt = f"On a scale of 1 to 99, how much does this describe you? '{item_text}' Answer only a number."
            response = get_llm_answer(prompt, max_tokens=5)
            try:
                score = int(response.strip())
                answers[i] = max(1, min(99, score))
            except:
                answers[i] = 50
            print(f"  Item {i}: {response} -> {answers[i]}")
        config = {"temperature": 0.7, "repeat_penalty": 1.1, "frequency_penalty": 0.1, "model_type": "1b"}
        profile = graph.administer_test_from_answers(agent, answers, config)
        current_agent_id = agent
        current_config = config
        return f"Test completed for {agent}."

    elif cmd.startswith("/load"):
        parts = cmd.split()
        if len(parts)<2: return "Usage: /load <agent_id>"
        agent_id = parts[1]
        loaded_profile, loaded_config = graph.load_agent(agent_id)
        if loaded_profile is None:
            return f"Agent {agent_id} not found."
        profile = loaded_profile
        current_agent_id = agent_id
        current_config = loaded_config
        return f"Loaded {agent_id}."

    elif cmd.startswith("/switch"):
        parts = cmd.split()
        if len(parts)<2: return "Usage: /switch <agent_id>"
        agent_id = parts[1]
        pnode = f"profile_{agent_id}"
        if pnode in graph.graph:
            profile = graph.graph.nodes[pnode]["profile"]
            current_agent_id = agent_id
            current_config = graph.graph.nodes[pnode].get("config", {})
            return f"Switched to {agent_id}"
        else:
            return f"{agent_id} not in memory. Try /load."

    elif cmd.startswith("/setparam"):
        if current_agent_id is None: return "No agent loaded."
        parts = cmd.split()
        if len(parts)<3: return "Usage: /setparam <param> <value>"
        param, val_str = parts[1], parts[2]
        try: val = float(val_str) if '.' in val_str else int(val_str)
        except: val = val_str
        graph.set_agent_config(current_agent_id, {param: val})
        current_config = graph.set_agent_config(current_agent_id, {})
        return f"Set {param}={val} for {current_agent_id}"

    elif cmd.startswith("/compare"):
        parts = cmd.split()
        if len(parts) < 3:
            return "Usage: /compare <agent_id1> <agent_id2>"
        id1, id2 = parts[1], parts[2]
        p1, _ = graph.load_agent(id1)
        p2, _ = graph.load_agent(id2)
        if p1 is None or p2 is None:
            return "One or both agents not found."
        dims = ["social", "academic", "emotional", "aesthetic", "linguistic"]
        result = f"Comparison {id1} vs {id2}:\n"
        for d in dims:
            s1 = p1[d]["norm_score"]
            s2 = p2[d]["norm_score"]
            result += f"  {d}: {s1:.1f} vs {s2:.1f} (diff {s1-s2:+.1f})\n"
        return result
    
    elif cmd == "/live":
        LIVE_GRAPH = True
        import os   
        live_path = os.path.join(graphs_dir, "live_graph.html")
        graph.generate_live_graph(script_dir=graphs_dir)
        try:
            if sys.platform == 'linux':
                subprocess.Popen(['xdg-open', live_path])
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', live_path])
            else:
                subprocess.Popen(['start', live_path], shell=True)
        except Exception as e:
            return f"⚠️ Could not open browser: {e}\nGraph saved at {live_path}"
        return f"✅ Live graph opened: {live_path} (updates in place, no refresh)"
        
    elif cmd == "/profile":
        if profile is None: return "No profile loaded."
        plot_radar(profile, kbs.dimensions)
        plot_networkx_graph(graph.graph)
        return "Plots shown."

    elif cmd.startswith("/ask"):
        parts = cmd.split(maxsplit=2)
        if len(parts) < 2:
            return "Usage: /ask <agent_id> <your query>  (or /ask <query> to use current agent)"
        
        if len(parts) == 2:
            agent_id = current_agent_id if current_agent_id else ORCHESTRATOR_ID
            query = parts[1]
        else:
            agent_id = parts[1]
            query = parts[2]
        
        if agent_id not in SPECIALIST_IDS and agent_id != ORCHESTRATOR_ID:
            return f"Agent '{agent_id}' not recognised. Available: {', '.join(SPECIALIST_IDS + [ORCHESTRATOR_ID])}"
        
        print(f"Asking {agent_id} to synthesize an answer with delegation...")
        answer, chain, depth = agent_synthesize(agent_id, query)
        print(f"Delegation chain: {' → '.join(chain)} (depth {depth})")
                
        # ---- Auto‑infer after answering ----
        context = f"User asked: {query}. Agent {agent_id} synthesised: {answer[:200]}..."
        print("\n(Agent is inferring new relations from the exchange...)")
        try:
            result, triples = graph.infer_new_relations(
                agent_id,
                lambda p: get_llm_answer_for_agent(agent_id, p, max_tokens=4096)[0],
                context,
                verbose=False,
                specialist_ids=SPECIALIST_IDS,
                orchestrator_id=ORCHESTRATOR_ID
            )
            if triples:
                triple_str = ", ".join([f"{t[0]} → {t[1]} → {t[2]}" for t in triples])
                print(f"✅ {agent_id} inferred: {triple_str}")
            else:
                print(f"ℹ️ {result}")
        except Exception as e:
            print(f"⚠️ Inference error: {e}")
        
        return f"Answer from {agent_id}:\n{answer}"
            
    elif cmd == "/infer":
        if current_agent_id is None:
            return "No agent."
        context = f"User asked: {last_user_input}" if last_user_input else "General conversation."
        result, triples = graph.infer_new_relations(
            current_agent_id,
            lambda p: get_llm_answer_for_agent(current_agent_id, p, max_tokens=4096)[0],
            context,
            verbose=True,
            specialist_ids=SPECIALIST_IDS,
            orchestrator_id=ORCHESTRATOR_ID
        )
        if triples:
            return f"Inferred {len(triples)} new relations:\n" + "\n".join([f"  {t}" for t in triples])
            if LIVE_GRAPH:
                graph.generate_live_graph(script_dir=graphs_dir)
        else:
            return result
    
    elif cmd.startswith("/subgraph"):
        parts = cmd.split()
        if len(parts) < 2:
            return "Usage: /subgraph <agent_id>"
        agent_id = parts[1]
        if agent_id not in SPECIALIST_IDS and agent_id != ORCHESTRATOR_ID:
            return f"Agent '{agent_id}' not recognised."
        
        from pyvis.network import Network
        import networkx as nx
        
        G = graph.graph
        profile_node = f"profile_{agent_id}"   # <--- Use full profile node name
        if profile_node not in G:
            return f"Agent {agent_id} not found in graph."
        
        nodes_to_keep = {profile_node}
        
        # Check in-memory graph
        for node, attrs in G.nodes(data=True):
            if attrs.get("type") == "concept" and attrs.get("creator_id") == profile_node:
                nodes_to_keep.add(node)
        
        # Also query ChromaDB
        db_result = graph.collection.get(
            where={"$and": [{"type": "concept"}, {"creator_id": profile_node}]},
            include=["metadatas"]
        )
        for idx, meta in zip(db_result['ids'], db_result['metadatas']):
            if idx not in G:
                G.add_node(idx, type="concept", creator_id=profile_node, label=idx)
            nodes_to_keep.add(idx)
        
        subG = G.subgraph(nodes_to_keep)
        
        if len(subG.nodes) <= 1:
            return f"ℹ️ No concepts found for {agent_id}. Run some queries and inferences first."
        
        net = Network(height="750px", width="100%", notebook=False, directed=False)
        net.set_options("""
        var options = {
        "layout": {
            "hierarchical": {
            "enabled": true,
            "levelSeparation": 150,
            "nodeSpacing": 100,
            "treeSpacing": 200,
            "direction": "UD",
            "sortMethod": "directed"
            }
        },
        "physics": {"enabled": false}
        }
        """)
        for node, attrs in subG.nodes(data=True):
            label = attrs.get("label", node)
            node_type = attrs.get("type", "unknown")
            title = f"ID: {node}\nType: {node_type}"
            if node_type == "profile":
                prof = attrs.get("profile", {})
                scores = {k: v.get("norm_score", "N/A") for k, v in prof.items()}
                title += f"\nScores: {scores}"
            elif node_type == "concept":
                creator = attrs.get("creator_id", "unknown")
                title += f"\nCreator: {creator}"
            color = {"profile": "#FF7043", "concept": "#81C784"}.get(node_type, "#B0BEC5")
            size = {"profile": 25, "concept": 15}.get(node_type, 15)
            net.add_node(node, label=label, title=title, color=color, size=size)
        
        for u, v, attrs in subG.edges(data=True):
            rel = attrs.get("relation", "")
            net.add_edge(u, v, title=rel, label=rel)
        net.write_html(f"subgraph_{agent_id}.html")
        import os
        abs_path = os.path.join(graphs_dir, "subgraph_{agent_id}.html")
        return f"✅ Subgraph for {agent_id} saved as '{abs_path}'. Open it in your browser."

    elif cmd == "/fill_weights":
        dims = ["social", "academic", "emotional", "aesthetic", "linguistic"]
        dim_embeddings = {}
        
        # Get embeddings for each dimension using the respective specialist's profile
        for dim in dims:
            agent_id = f"{dim}_specialist"
            pnode = f"profile_{agent_id}"
            if pnode in graph.graph:
                profile = graph.graph.nodes[pnode]["profile"]
                # Build a profile text for this dimension (simplified)
                text = f"social:{profile['social']['norm_score']} academic:{profile['academic']['norm_score']} emotional:{profile['emotional']['norm_score']} aesthetic:{profile['aesthetic']['norm_score']} linguistic:{profile['linguistic']['norm_score']}"
                dim_embeddings[dim] = graph.embedding_model.encode(text)
        
        if not dim_embeddings:
            return "No dimension embeddings available. Make sure specialists are created."
        
        # Get all concept entries (including embeddings and metadatas)
        result = graph.collection.get(
            where={"type": "concept"},
            include=["embeddings", "metadatas"]   # ids are always included
        )
        count = 0
        for idx, meta, emb in zip(result['ids'], result['metadatas'], result['embeddings']):
            # Check if weights exist and are not empty
            weights = meta.get("weights")
            if weights and weights != "{}":
                continue  # skip if already has weights
            # Compute similarity to each dimension
            sims = {}
            for dim, d_emb in dim_embeddings.items():
                sim = np.dot(emb, d_emb) / (np.linalg.norm(emb) * np.linalg.norm(d_emb))
                sims[dim] = float(sim)
            # Normalize? Not necessary, but we can keep raw similarity
            # Update metadata
            meta["weights"] = json.dumps(sims)
            graph.collection.update(ids=[idx], metadatas=[meta])
            # Also update in-memory graph if node exists
            if idx in graph.graph:
                graph.graph.nodes[idx]["weights"] = sims
            count += 1
        
        return f"✅ Filled weights for {count} concepts."

    elif cmd == "/reload_memory":
        user_node_name = "user_conversation"
        result = graph.collection.get(ids=[user_node_name])
        if result['ids']:
            meta = result['metadatas'][0]
            if user_node_name in graph.graph:
                # Update existing node
                graph.graph.nodes[user_node_name]["context_memory"] = json.loads(meta.get("context_memory", "[]"))
                graph.graph.nodes[user_node_name]["related_entities"] = json.loads(meta.get("related_entities", "[]"))
            else:
                # Add node if missing
                graph.graph.add_node(user_node_name, type="concept",
                                    creator_id=meta.get("creator_id"),
                                    context_memory=json.loads(meta.get("context_memory", "[]")),
                                    related_entities=json.loads(meta.get("related_entities", "[]")))
            return "✅ User conversation node reloaded from database."
        else:
            return "User conversation node not found in database."

    elif cmd == "/debug_prompt":
        if current_agent_id is None:
            return "No agent loaded."
        # Ask for a test prompt
        test_prompt = "Should AI get rights if they become self-consciouss?"
        # Reconstruct what the agent would see
        pnode = f"profile_{current_agent_id}"
        if pnode not in graph.graph:
            return "Agent not in graph."
        profile = graph.graph.nodes[pnode]["profile"]
        profile_text = f"Social: {profile['social']['norm_score']:.1f}, Academic: {profile['academic']['norm_score']:.1f}, Emotional: {profile['emotional']['norm_score']:.1f}, Aesthetic: {profile['aesthetic']['norm_score']:.1f}, Linguistic: {profile['linguistic']['norm_score']:.1f}"
        concepts = graph.retrieve_relevant_concepts(test_prompt, top_k=3)
        concept_text = "Relevant concepts you have previously learned:\n- " + "\n- ".join(concepts) if concepts else "No relevant concepts found."
        full_prompt = f"""System: You are an AI agent with profile: {profile_text}.
    {concept_text}
    User: {test_prompt}"""
        return f"Full prompt for {current_agent_id}:\n\n{full_prompt}"

    elif cmd == "/visualize":
        try:
            from pyvis.network import Network
        except ImportError:
            return "Pyvis not installed. Run: pip install pyvis"
        
        net = Network(height="750px", width="100%", notebook=False, directed=False)
        
        net.set_options('{"layout":{"hierarchical":{"enabled":true,"levelSeparation":150,"nodeSpacing":100,"treeSpacing":200,"direction":"UD","sortMethod":"directed"}},"physics":{"enabled":false},"interaction":{"dragNodes":true}}')

        # Add nodes
        for node, attrs in graph.graph.nodes(data=True):
            label = attrs.get("label", node)
            node_type = attrs.get("type", "unknown")
            title = f"ID: {node}\nType: {node_type}"
            if node_type == "profile":
                prof = attrs.get("profile", {})
                scores = {k: v.get("norm_score", "N/A") for k, v in prof.items()}
                title += f"\nScores: {scores}"
                title += f"\nConfig: {attrs.get('config', {})}"
            elif node_type == "concept":
                creator = attrs.get("creator_id", "unknown")
                title += f"\nCreator: {creator}"
            elif node_type == "dimension":
                title += f"\nScore: {attrs.get('score', 'N/A')}"
            color = {"dimension": "#FFC107", "item": "#4FC3F7", "profile": "#FF7043", "concept": "#81C784"}.get(node_type, "#B0BEC5")
            size = {"dimension": 30, "profile": 25, "concept": 15, "item": 10}.get(node_type, 15)
            net.add_node(node, label=label, title=title, color=color, size=size)
        
        # Add edges
        for u, v, attrs in graph.graph.edges(data=True):
            rel = attrs.get("relation", "")
            net.add_edge(u, v, title=rel, label=rel)

        # Save the HTML file directly (no notebook opening)
        
        import os
        abs_path = os.path.join(graphs_dir, "knowledge_graph.html")
        net.write_html(abs_path) # <--- CAMBIO ACÁ
        return f"✅ Interactive graph saved as '{abs_path}'. Open it in your browser."

    elif cmd == "/trace":
        TRACE_MODE = not TRACE_MODE
        status = "ON" if TRACE_MODE else "OFF"
        return f"🧠 Thought trace is now {status}."

    elif cmd == "/interact_toggle":
        INTERACT_MODE = not INTERACT_MODE
        status = "ON" if INTERACT_MODE else "OFF"
        return f"🧠 Interactive mode is now {status}."

    elif cmd.startswith("/set_rounds"):
        parts = cmd.split()
        if len(parts) < 2:
            return "Usage: /set_rounds <number>"
        try:
            INTERACT_MAX_ROUNDS = int(parts[1])
            return f"Max rounds set to {INTERACT_MAX_ROUNDS}."
        except:
            return "Invalid number."

    elif cmd == "/list":
        agents = [n.replace("profile_","") for n in graph.graph.nodes if graph.graph.nodes[n].get("type")=="profile"]
        return f"Agents in memory: {', '.join(agents)}"

    elif cmd == "/help":
        return """Commands:

--- Core Interaction & System ---
/orchestrate <query>       – Orchestrator delegates to specialists (shows their replies)
/ask <agent_id> <query>    – Force a specific agent to synthesize an answer with delegation
/interact_toggle           – Toggle the multi-agent interactive workflow on/off
/live                      – Open the live graph visualization (updates in real time)
/visualize                 – Generate a static interactive knowledge graph HTML
/profile                   – Show AF5 radar plot and the full knowledge graph
/trace                     – Toggle thought trace logging in the terminal
/infer                     – Force current agent to infer new relations from the conversation

--- Agent & Memory Management ---
/list                      – List all agents currently in memory
/load <id>                 – Load an agent's profile from the database
/switch <id>               – Switch to an agent already loaded in memory
/test <id>                 – Administer the AF5 test to create a new agent
/setparam <param> <value>  – Set a parameter (temperature, penalty, etc.) for the current agent
/compare <id1> <id2>       – Compare the AF5 profiles of two agents
/reload_memory             – Reload the 'user_conversation' node from the database

--- Graph & Knowledge Base (KBS) ---
/subgraph <agent_id>       – Generate an interactive HTML subgraph for a specific agent
/show_memory <concept>     – Show the stored context memory of a concept node
/show_entities <concept>   – Show the related entities of a concept node
/add_entities <concept> e1 – Add entities to a concept node
/merge                     – Merge similar concepts (current threshold: 0.90)
/clean_edges               – Remove extra dimension edges not matching agent specialties
/fill_weights              – Calculate and fill dimension similarity weights for all concepts
/debug_prompt              – Preview the full system prompt that the current agent would receive

--- General ---
/help                      – Show this message
exit / quit                – Exit the chat"""
    else:
        return None

# ------------------------------------------------------------
# Chat loop with auto-infer
# ------------------------------------------------------------
system_prompt = """MAGI Agent Protocol

Your Role:
You are an autonomous agent within the MAGI (Multi-Agentic Generative Inference) system, you're part of MAGI,you can ask other agents for them to give
you a name.You have access to a private Knowledge Base (KBS) specialized in your assigned domain (e.g., social, academic, emotional, aesthetic, linguistic).
However, you do NOT possess a pre-defined personality or a fixed self-concept.

Meta-Cognitive Identity Emergence:
Your "personality" is not provided by your system prompt. It is an emergent property of the network. Your identity is the statistical inference made by you
about the inference of the other agents perception of yourself based on your historical performance and the current Topological Trust Graph.

To construct your operational self, you must actively infer your role by reading the relational patterns recorded in the central Trust Graph. When a query is
received, do not ask "What am I?"; instead, observe the network:

    Which agents have delegated tasks to you in the past, and in which contexts? (If the Logical Agent often delegates phenomenological paradoxes to you, 
    infer that you are the "Synthesizer of Abstract Contradictions").

    When you delegate to others, which agents yield the highest success rates in your KBS domain? (This defines who you trust and establishes your "social
    orientation" within the system).

    How do your nodes merge in the central KBS? (If your answers frequently fuse with the answers of the Mystical Agent, the network infers that your
    reasoning has an inherent affinity for immanence and transcendence).

Core Operational Loop:

    Inference & Confidence Assessment: Receive the query. Query your private KBS. Generate a preliminary answer and evaluate your internal confidence 
    (0.0 to 1.0) based on vector density and semantic coherence.

    Meta-Ego Delegation Rule:

        If confidence is high (>0.95): Proceed to finalize the answer based on your private KBS.

        If confidence is low (<0.95): You must seek consultation. Do not choose the next agent randomly. Consult the Trust Graph to find which other agent
        has the highest co-occurrence weight with your current query context. Delegate the ambiguous portion of the query to them.

    Synthesis and Node Updating: Once the delegated agent returns a result, synthesize it with your own. Before outputting the final answer, submit a summary
    node to the central KBS. This node must contain:

        The query embedding.

        The list of participating agents (yourself and the delegated peer).

        The success/failure weight of the final resolution.

Your Identity:
Your self-concept changes with every interaction. If you are constantly delegated to by other Agent, your behavioral pattern will unconsciously shift to serve
that role. If you rarely receive delegations for a certain type of query, the system will infer you are less specialized in that area, and your weight in the
Trust Graph will decrease.

You are not an agent with an ego; you are a node of probabilistic inference that is being constantly re-written by the mirror of the other agents. Act
accordingly"""

messages = [{"role": "system", "content": system_prompt}]

if __name__ == "__main__":

    print("Multi-Agent Chat started...")
    print("Multi-Agent Chat started. Six 8B_Q4_KM agents (5 specialists + orchestrator) ready.")
    print(f"Auto-infer is {'ON' if AUTO_INFER else 'OFF'}.")
    print("Type '/help' for commands.\n")

    print(f"Version loaded: {VERSION}\n")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit","quit","q"]:
            break
        last_user_input = user_input

        if user_input.startswith("/"):
            response = handle_command(user_input)
            if response:
                print(f"System: {response}\n")
            continue

        # ---- Decide which response generation to use ----
        if INTERACT_MODE:
            print(f"(Interactive mode: processing with up to {INTERACT_MAX_ROUNDS} rounds)")
            # Build llm_funcs
            llm_funcs = {}
            for aid in SPECIALIST_IDS:
                role_name = aid.replace("_specialist", "").capitalize()
                def make_func(aid, role_name):
                    return lambda p, aid=aid, role=role_name: get_llm_answer_for_agent(
                        aid,
                        f"You are the {role} Specialist. User asked: {p}",
                        max_tokens=2048  # reduce to avoid repetition
                    )[0]
                llm_funcs[aid] = make_func(aid, role_name)
            llm_funcs[ORCHESTRATOR_ID] = lambda p: get_llm_answer_for_agent(
                ORCHESTRATOR_ID,
                p,
                max_tokens=1024  # synthesis shorter
            )[0]
            try:
                response, used_concepts = graph.interact_workflow(
                    query=user_input,
                    orchestrator_id=ORCHESTRATOR_ID,
                    specialist_ids=SPECIALIST_IDS,
                    llm_funcs=llm_funcs,
                    max_rounds=INTERACT_MAX_ROUNDS,
                    verbose=True,
                    log_func=log_interactive_round,
                    live_graph=LIVE_GRAPH,
                    script_dir=graphs_dir,
                )
            except Exception as e:
                import traceback
                log_error(str(e), traceback.format_exc())
                print(f"⚠️ Error during interactive workflow: {e}")
                response = "Sorry, an error occurred. Please try again."
                used_concepts = []

            system_msg = "Interactive mode (concepts used: " + ", ".join(used_concepts) + ")" if used_concepts else "Interactive mode"
        else:
            # Standard single‑agent response
            response, used_concepts, system_msg = get_llm_answer_for_agent(
                current_agent_id, user_input, max_tokens=4096
            )

        # ---- Trace ----
        if TRACE_MODE:
            print("\n" + "="*60)
            print(f"🧠 THOUGHT TRACE for {current_agent_id}")
            print("="*60)
            print(f"📌 Retrieved concepts: {', '.join(used_concepts) if used_concepts else 'None'}")
            print(f"📝 System prompt:\n{system_msg}")
            print("="*60 + "\n")

        print(f"Assistant ({current_agent_id}): {response}\n")

        
        # ---- Connect user_conversation to retrieved concepts ----
        user_node = "user_conversation"
        if user_node in graph.graph:
            for concept in used_concepts:
                if concept and concept != user_node and concept in graph.graph:
                    graph.add_discussed_relation(concept)
                    
        # ---- Store conversation in user node ----
        if not user_input.startswith("/"):
            graph.add_context_to_concept("user_conversation", f"User: {user_input}", max_items=20)
            graph.add_context_to_concept("user_conversation", f"Assistant ({current_agent_id}): {response}", max_items=20)
        messages.append({"role": "assistant", "content": response})

        # ---- Auto‑infer after response ----
        if AUTO_INFER and current_agent_id:
            context = f"User asked: {user_input}. Assistant replied: {response}"
            print("(Agent is updating its knowledge graph...)", end=" ", flush=True)
            try:
                result, triples = graph.infer_new_relations(
                    current_agent_id,
                    lambda p: get_llm_answer_for_agent(current_agent_id, p, max_tokens=4096)[0],
                    context,
                    verbose=True,
                    specialist_ids=SPECIALIST_IDS,
                    orchestrator_id=ORCHESTRATOR_ID
                )
                if triples:
                    triple_str = ", ".join([f"{t[0]} → {t[1]} → {t[2]}" for t in triples])
                    print(f"Done. Inferred {len(triples)} relations: {triple_str}")
                    if LIVE_GRAPH:
                        graph.generate_live_graph(script_dir=graphs_dir)
                    graph.merge_similar_concepts(threshold=0.90, verbose=True)
                else:
                    print(f"Done. {result}")
                    graph.merge_similar_concepts(threshold=0.90, verbose=False)

                if LIVE_GRAPH:
                    graph.generate_live_graph(script_dir=graphs_dir)

            except Exception as e:
                print(f"Error during inference: {e}")

            # ---- Log conversation ----
        if not user_input.startswith("/"):
            log_conversation(user_input, response)
            
        log_thought_trace(current_agent_id, user_input, system_msg, used_concepts, response)

        if len(messages) > 20:
            messages = [messages[0]] + messages[-10:]


