import networkx as nx
import chromadb
from chromadb.utils import embedding_functions
import json
from pyvis.network import Network
from af5_kbs import plot_networkx_graph, plot_radar
import re
import ast
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

class DynamicAF5Graph:
    def __init__(self, kbs, embedding_model, script_dir):
        self.kbs = kbs
        self.graph = nx.Graph()
        db_path = os.path.join(script_dir, "af5_chroma_db") if script_dir else "./af5_chroma_db"

        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="af5_nodes",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction()
        )
        self.embedding_model = embedding_model
        self._init_static_structure()
        self._seen_triples = set()

    def _init_static_structure(self):
        for dim, data in self.kbs.dimensions.items():
            self.graph.add_node(dim, type="dimension", label=data["label"], score=50)
            for idx, item_num in enumerate(data["items"]):
                item_label = f"I{item_num}"
                self.graph.add_node(item_label, type="item", text=data["items_text"][idx], item_num=item_num)
                self.graph.add_edge(dim, item_label, relation="belongs_to")

    # ---------- Specialist creation ----------
    def create_specialist(self, agent_id, dimension, score_high=95, score_low=1, config=None):
        dims = ["social", "academic", "emotional", "aesthetic", "linguistic"]
        profile = {}
        for d in dims:
            score = score_high if d == dimension else score_low
            profile[d] = {"raw_avg": score, "norm_score": score, "interpretation": self.kbs._interpret(d, score)}
        profile_node = f"profile_{agent_id}"
        self.graph.add_node(profile_node, type="profile", agent_id=agent_id, profile=profile)
        for dim, data in profile.items():
            if data["norm_score"] == score_high:
                self.graph.add_edge(profile_node, dim, relation="scored", score=data["norm_score"])

        if config is None:
            config = {"max_tokens": 2048, "temperature": 0.7, "repeat_penalty": 1.1, "frequency_penalty": 0.1, "model_type": "1b"}

        config_json = json.dumps(config)
        profile_text = f"Agent {agent_id} social:{profile['social']['norm_score']} academic:{profile['academic']['norm_score']} emotional:{profile['emotional']['norm_score']} aesthetic:{profile['aesthetic']['norm_score']} linguistic:{profile['linguistic']['norm_score']}"
        embedding = self.embedding_model.encode(profile_text)
        self.collection.add(
            embeddings=[embedding.tolist()],
            metadatas=[{
                "agent_id": agent_id, "type": "profile",
                "social": profile['social']['norm_score'],
                "academic": profile['academic']['norm_score'],
                "emotional": profile['emotional']['norm_score'],
                "aesthetic": profile['aesthetic']['norm_score'],
                "linguistic": profile['linguistic']['norm_score'],
                "config_json": config_json
            }],
            ids=[f"profile_{agent_id}"]
        )
        return profile

    # ---------- Store from test answers ----------
    def administer_test_from_answers(self, agent_id, answers, config=None):
        profile = self.kbs.compute_profile(answers)
        profile_node = f"profile_{agent_id}"
        self.graph.add_node(profile_node, type="profile", agent_id=agent_id, profile=profile)
        for dim, data in profile.items():
            self.graph.add_edge(profile_node, dim, relation="scored", score=data["norm_score"])
        if config is None:
            config = {"max_tokens": 2048, "temperature": 0.7, "repeat_penalty": 1.1, "frequency_penalty": 0.1, "model_type": "1b"}
        config_json = json.dumps(config)
        profile_text = f"Agent {agent_id} social:{profile['social']['norm_score']} academic:{profile['academic']['norm_score']} emotional:{profile['emotional']['norm_score']} aesthetic:{profile['aesthetic']['norm_score']} linguistic:{profile['linguistic']['norm_score']}"
        embedding = self.embedding_model.encode(profile_text)
        self.collection.add(
            embeddings=[embedding.tolist()],
            metadatas=[{
                "agent_id": agent_id, "type": "profile",
                "social": profile['social']['norm_score'],
                "academic": profile['academic']['norm_score'],
                "emotional": profile['emotional']['norm_score'],
                "aesthetic": profile['aesthetic']['norm_score'],
                "linguistic": profile['linguistic']['norm_score'],
                "config_json": config_json
            }],
            ids=[f"profile_{agent_id}"]
        )
        return profile

    # ---------- Load / Update config ----------
    def load_agent(self, agent_id):
        result = self.collection.get(ids=[f"profile_{agent_id}"])
        if not result['ids']:
            return None, None
        meta = result['metadatas'][0]
        profile = {}
        for dim in ["social","academic","emotional","aesthetic","linguistic"]:
            score = meta.get(dim, 50.0)
            profile[dim] = {"norm_score": score, "raw_avg": None, "interpretation": self.kbs._interpret(dim, score)}
        config = json.loads(meta.get("config_json", "{}"))
        profile_node = f"profile_{agent_id}"
        self.graph.add_node(profile_node, type="profile", agent_id=agent_id, profile=profile, config=config)
        for dim, data in profile.items():
            self.graph.add_edge(profile_node, dim, relation="scored", score=data["norm_score"])
        return profile, config

    def set_agent_config(self, agent_id, config_updates):
        existing = self.collection.get(ids=[f"profile_{agent_id}"])
        if not existing['ids']: return None
        meta = existing['metadatas'][0]
        old_config = json.loads(meta.get("config_json", "{}"))
        old_config.update(config_updates)
        meta["config_json"] = json.dumps(old_config)
        self.collection.update(ids=[f"profile_{agent_id}"], metadatas=[meta])
        profile_node = f"profile_{agent_id}"
        if profile_node in self.graph:
            self.graph.nodes[profile_node]["config"] = old_config
        return old_config

    # ---------- Orchestration ----------
    def orchestrate(self, query, llm_funcs, specialist_ids, orchestrator_id, verbose=True):
        orch_node = f"profile_{orchestrator_id}"
        if orch_node not in self.graph:
            return "Orchestrator not found."

        agents_info = []
        for aid in specialist_ids:
            pnode = f"profile_{aid}"
            if pnode in self.graph:
                prof = self.graph.nodes[pnode]["profile"]
                agents_info.append(f"{aid}: social={prof['social']['norm_score']:.1f}, academic={prof['academic']['norm_score']:.1f}, emotional={prof['emotional']['norm_score']:.1f}, aesthetic={prof['aesthetic']['norm_score']:.1f}, linguistic={prof['linguistic']['norm_score']:.1f}")

        prompt = f"""
You are the orchestrator. Given the user query, decide which specialist agents should be consulted.
Query: {query}
Agents:
{chr(10).join(agents_info)}
Return a list of agent IDs, separated by commas, in order of relevance.
"""
        response = llm_funcs[orchestrator_id](prompt)
        selected_ids = [aid.strip() for aid in response.split(',') if aid.strip() in specialist_ids]
        if not selected_ids:
            selected_ids = specialist_ids[:4]

        raw_responses = {}
        if verbose:
            print("\n--- Specialist consultation ---")
        for aid in selected_ids:
            if aid not in llm_funcs:
                continue
            resp = llm_funcs[aid](query)
            raw_responses[aid] = resp
            if verbose:
                print(f"  {aid} says: {resp[:300]}...\n")

        # Let each specialist infer new relations
        if verbose:
            print("\n--- Specialists inferring new relations ---")
        for aid, resp in raw_responses.items():
            context = f"Specialist {aid} answered: {resp}"
            if aid in llm_funcs:
                try:
                    result_str, triples = self.infer_new_relations(
                        aid,
                        llm_funcs[aid],
                        context=context,
                        verbose=False,
                        specialist_ids=specialist_ids,
                        orchestrator_id=orchestrator_id
                    )
                    if triples:
                        triple_str = ", ".join([f"{t[0]} → {t[1]} → {t[2]}" for t in triples])
                        if verbose:
                            print(f"  {aid} inferred: {triple_str}")
                    elif verbose:
                        print(f"  {aid} inferred nothing new.")
                except Exception as e:
                    if verbose:
                        print(f"  Inference error for {aid}: {e}")

        # Retrieve concepts with memory for synthesis
        concepts_with_memory = self.retrieve_concepts_with_memory(query, top_k=3)
        concept_text = ""
        if concepts_with_memory:
            concept_names = [name for name, _, _ in concepts_with_memory]
            concept_text = "Relevant concepts from the knowledge graph:\n- " + "\n- ".join(concept_names)

        synthesis_prompt = f"""
You are the orchestrator. The user asked: {query}
{concept_text}
The specialists provided the following responses:
{chr(10).join([f"{aid}: {resp}" for aid, resp in raw_responses.items()])}

**Response Length Guideline:**
- If the query is simple, factual, or irrelevant, provide a **concise synthesis** (1–3 sentences).
- If the query is complex, open‑ended, or nuanced, provide a **comprehensive, detailed synthesis** that integrates the specialists' perspectives thoroughly.

Synthesize a coherent, concise final answer to the user.
"""
        final = llm_funcs[orchestrator_id](synthesis_prompt)
        return final, selected_ids

    # ---------- Auto-infer ----------
    def infer_new_relations(self, agent_id, llm_func, context="", verbose=True, specialist_ids=None, orchestrator_id=None):
        if specialist_ids is None:
            specialist_ids = []
        if orchestrator_id is None:
            orchestrator_id = ""

        profile_node = f"profile_{agent_id}"
        if profile_node not in self.graph:
            return "No profile found.", []
        profile = self.graph.nodes[profile_node]["profile"]
        prompt = f"""
Given an AI agent with the following self-role profile:
Social: {profile['social']['norm_score']:.1f}
Academic: {profile['academic']['norm_score']:.1f}
Emotional: {profile['emotional']['norm_score']:.1f}
Aesthetic: {profile['aesthetic']['norm_score']:.1f}
Linguistic: {profile['linguistic']['norm_score']:.1f}

Context: {context}

Generate exactly 3 new conceptual connections (triples) between any existing concepts, agents, dimensions, or new topics.
The subject and object can be any node: concept, agent, dimension, or new topic, ideas, etc.
Use this EXACT format, one per line:
('subject', 'relation', 'object')
or
('subject', 'relation', 'object') | [weight_social, weight_academic, weight_emotional, weight_aesthetic, weight_linguistic] | [entity1, entity2, ...]

Only output these lines, no other text.
"""

        raw_response = llm_func(prompt)
        if verbose:
            print("\n--- Raw inference response ---")
            print(raw_response)
            print("-------------------------------\n")

        triples, weights, entities = self._parse_triples_with_weights_and_entities(raw_response)
        new_triples = []
        for (n1, rel, n2), w, e in zip(triples, weights, entities):
            # Determine source and target
            if n1 in specialist_ids or n1 == orchestrator_id:
                source = f"profile_{n1}"
            else:
                source = n1
            if n2 in specialist_ids or n2 == orchestrator_id:
                target = f"profile_{n2}"
            else:
                target = n2

            # Check if this triple already exists (cache or edge)
            key = (source, rel, target)
            if key in self._seen_triples or self.graph.has_edge(source, target):
                continue
            self._seen_triples.add(key)

            # Ensure nodes exist for n1 and n2 (if they are not agents)
            for node_name in [n1, n2]:
                if node_name not in self.graph:
                    if node_name in specialist_ids or node_name == orchestrator_id:
                        pass
                    else:
                        self.graph.add_node(node_name, type="concept", creator_id=f"profile_{agent_id}")

            # Add edge
            self.graph.add_edge(source, target, relation=rel)

            # Store relation in concept nodes (if they are concepts)
            if source in self.graph and self.graph.nodes[source].get("type") == "concept":
                self.add_relation_to_concept(source, n1, rel, n2)
            if target in self.graph and self.graph.nodes[target].get("type") == "concept":
                self.add_relation_to_concept(target, n1, rel, n2)

            # Clean entities
            cleaned_entities = []
            if e:
                cleaned_entities = [
                    str(item) for item in e
                    if isinstance(item, str) and item not in ['...', '...', ''] and item.strip()
                ]

            # Handle concept n2 (if it's not an agent)
            if n2 not in specialist_ids and n2 != orchestrator_id:
                profile_node_agent = f"profile_{agent_id}"
                if profile_node_agent in self.graph and not self.graph.has_edge(profile_node_agent, n2):
                    self.graph.add_edge(profile_node_agent, n2, relation="owns")

                # Check if n2 already exists in ChromaDB
                existing = self.collection.get(ids=[n2])
                if existing['ids']:
                    # Concept exists – add context and entities
                    context_phrase = f"Relation: {n1} {rel} {n2} (from agent {agent_id})"
                    self.add_context_to_concept(n2, context_phrase)
                    if cleaned_entities:
                        self.add_related_entities(n2, cleaned_entities)
                else:
                    # New concept: create with initial context and entities
                    weights_dict = None
                    if w:
                        dims = ["social", "academic", "emotional", "aesthetic", "linguistic"]
                        weights_dict = {dims[i]: w[i] for i in range(len(w)) if i < len(dims)}
                    initial_context = [f"Created via: {n1} {rel} {n2} (from agent {agent_id})"]
                    self._embed_new_node(
                        n2,
                        creator_id=f"profile_{agent_id}",
                        weights=weights_dict,
                        context_memory=initial_context,
                        related_entities=cleaned_entities
                    )
            new_triples.append((n1, rel, n2))
        return f"Inferred {len(new_triples)} new relations.", new_triples

    # ---------- Parser ----------
    def _parse_triples_with_weights_and_entities(self, text):
        triples = []
        weights = []
        entities = []
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Try to parse with regex and ast
            try:
                # First, match the full format with optional weights and entities
                match = re.match(r"\(\s*'([^']+)'\s*,\s*'([^']+)'\s*,\s*(.+?)\s*\)\s*(?:\|\s*\[([^\]]+)\]\s*(?:\|\s*\[([^\]]+)\])?)?", line)
                if match:
                    n1 = match.group(1)
                    rel = match.group(2)
                    n2_raw = match.group(3)
                    w_str = match.group(4)
                    e_str = match.group(5)

                    # Sanitize n2: if it's a list, take the first element as string
                    n2 = n2_raw.strip()
                    if n2.startswith('['):
                        try:
                            parsed = ast.literal_eval(n2)
                            if isinstance(parsed, list) and parsed:
                                n2 = str(parsed[0])
                            else:
                                continue
                        except:
                            continue
                    else:
                        # strip quotes if present
                        n2 = n2.strip("'\"")

                    # Parse weights
                    w = None
                    if w_str:
                        w = self._normalize_weights(w_str)


                    # Parse entities
                    e_list = []
                    if e_str:
                        try:
                            e_list = ast.literal_eval("[" + e_str + "]")
                            e_list = [str(item) for item in e_list if isinstance(item, str) and item.strip() and item not in ['...', '']]
                        except:
                            e_list = []

                    # Only add if n2 is a valid string
                    if n2 and isinstance(n2, str) and n2.strip():
                        triples.append((n1, rel, n2))
                        weights.append(w)
                        entities.append(e_list)
                    continue
            except Exception:
                continue
        return triples, weights, entities

    # ---------- Embed new node ----------
    def _embed_new_node(self, node_name, creator_id=None, weights=None, context_memory=None, related_entities=None):
        existing = self.collection.get(ids=[node_name])
        if existing['ids']:
            return
        embedding = self.embedding_model.encode(node_name)
        metadata = {
            "type": "concept",
            "name": node_name,
            "creator_id": creator_id or "unknown",
            "weights": json.dumps(weights) if weights else "{}",
            "context_memory": json.dumps(context_memory[:10] if context_memory else []),
            "related_entities": json.dumps(related_entities[:20] if related_entities else []),
            "relations": json.dumps([]),
            "usage_count": 0
        }
        self.collection.add(
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            ids=[node_name]
        )
        if node_name not in self.graph:
            self.graph.add_node(node_name, type="concept", creator_id=creator_id, weights=weights,
                                context_memory=context_memory or [], related_entities=related_entities or [],
                                relations=[], usage_count=0)

    # ---------- Add context to existing concept ----------
    def add_context_to_concept(self, concept_name, context_phrase, max_items=10):
        if concept_name not in self.graph:
            return False
        memory = self.graph.nodes[concept_name].get("context_memory", [])
        if not isinstance(memory, list):
            memory = []
        if context_phrase not in memory:
            memory.append(context_phrase)
        if len(memory) > max_items:
            memory = memory[-max_items:]
        self.graph.nodes[concept_name]["context_memory"] = memory
        # Update ChromaDB
        meta = self.collection.get(ids=[concept_name], include=["metadatas"])['metadatas'][0]
        meta["context_memory"] = json.dumps(memory)
        self.collection.update(ids=[concept_name], metadatas=[meta])
        return True

    # ---------- Add related entities ----------
    def add_related_entities(self, concept_name, entities_list):
        if concept_name not in self.graph:
            return False
        current = self.graph.nodes[concept_name].get("related_entities", [])
        for e in entities_list:
            if e not in current:
                current.append(e)
        current = current[-20:]  # keep last 20
        self.graph.nodes[concept_name]["related_entities"] = current
        meta = self.collection.get(ids=[concept_name], include=["metadatas"])['metadatas'][0]
        meta["related_entities"] = json.dumps(current)
        self.collection.update(ids=[concept_name], metadatas=[meta])
        return True

    # ---------- Retrieval ----------
    def retrieve_relevant_concepts(self, query, agent_id=None, top_k=3):
        if agent_id:
            where_filter = {"$and": [{"type": "concept"}, {"creator_id": f"profile_{agent_id}"}]}
        else:
            where_filter = {"type": "concept"}
        query_embedding = self.embedding_model.encode(query)
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_filter,
            include=["metadatas"]
        )
        if results["metadatas"] and results["metadatas"][0]:
            concepts = []
            for meta in results["metadatas"][0]:
                name = meta.get("name", "Unknown")
                memory = json.loads(meta.get("context_memory", "[]"))
                concepts.append((name, memory))
            return concepts
        return []

    def retrieve_concepts_with_memory(self, query, agent_id=None, top_k=3):
        if agent_id:
            where_filter = {"$and": [{"type": "concept"}, {"creator_id": f"profile_{agent_id}"}]}
        else:
            where_filter = {"type": "concept"}
        query_embedding = self.embedding_model.encode(query)
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_filter,
            include=["metadatas"]
        )
        if results["metadatas"] and results["metadatas"][0]:
            concepts = []
            for meta in results["metadatas"][0]:
                name = meta.get("name", "Unknown")
                memory = json.loads(meta.get("context_memory", "[]"))
                related = json.loads(meta.get("related_entities", "[]"))
                # Increment usage_count
                if name in self.graph:
                    current = self.graph.nodes[name].get("usage_count", 0) + 1
                    self.graph.nodes[name]["usage_count"] = current
                    meta["usage_count"] = current
                    self.collection.update(ids=[name], metadatas=[meta])
                concepts.append((name, memory, related))
            return concepts
        return []

    # ---------- Similarity ----------
    def find_similar_agents(self, query_profile, top_k=3):
        query_text = f"social:{query_profile['social']['norm_score']} academic:{query_profile['academic']['norm_score']} emotional:{query_profile['emotional']['norm_score']} aesthetic:{query_profile['aesthetic']['norm_score']} linguistic:{query_profile['linguistic']['norm_score']}"
        results = self.collection.query(query_texts=[query_text], n_results=top_k)
        return results["metadatas"]

    def find_best_agent_for_query(self, query, exclude=[], top_k=1):
        query_embedding = self.embedding_model.encode(query)
        all_profiles = self.collection.get(where={"type": "profile"}, include=["metadatas"])
        if not all_profiles['ids']:
            return None
        best_id = None
        best_score = -1
        for idx, meta in zip(all_profiles['ids'], all_profiles['metadatas']):
            agent_id = meta.get("agent_id")
            if agent_id in exclude:
                continue
            profile_text = f"social:{meta.get('social',0)} academic:{meta.get('academic',0)} emotional:{meta.get('emotional',0)} aesthetic:{meta.get('aesthetic',0)} linguistic:{meta.get('linguistic',0)}"
            profile_emb = self.embedding_model.encode(profile_text)
            sim = self.embedding_model.similarity(query_embedding, profile_emb)
            if sim > best_score:
                best_score = sim
                best_id = agent_id
        return best_id

    def plot_graph(self):
        plot_networkx_graph(self.graph)
        profiles = [n for n, attr in self.graph.nodes(data=True) if attr.get("type") == "profile"]
        if profiles:
            plot_radar(self.graph.nodes[profiles[0]]["profile"], self.kbs.dimensions)

    # ---------- Merge ----------
    def merge_similar_concepts(self, threshold=0.80, verbose=True):
        result = self.collection.get(where={"type": "concept"}, include=["embeddings", "metadatas"])
        if not result['ids']:
            return 0
        ids = result['ids']
        embeddings = result['embeddings']
        # Only keep nodes that are currently in the graph
        valid_pairs = [(id, emb) for id, emb in zip(ids, embeddings) if id in self.graph]
        if not valid_pairs:
            return 0
        valid_ids = [p[0] for p in valid_pairs]
        valid_embs = [p[1] for p in valid_pairs]

        sim_matrix = cosine_similarity(np.array(valid_embs))
        merged = set()
        merged_count = 0

        for i in range(len(valid_ids)):
            node_i = valid_ids[i]
            if node_i in merged or node_i == "user_conversation":
                continue
            # Extra safety: check if node still exists
            if node_i not in self.graph:
                continue
            usage_i = self.graph.nodes[node_i].get("usage_count", 0) if node_i in self.graph else 0
            for j in range(i+1, len(valid_ids)):
                node_j = valid_ids[j]
                if node_j in merged or node_j == "user_conversation":
                    continue
                if node_j not in self.graph:
                    continue
                if sim_matrix[i][j] < (1 - threshold):
                    continue
                usage_j = self.graph.nodes[node_j].get("usage_count", 0) if node_j in self.graph else 0
                if usage_i >= usage_j:
                    survivor = node_i
                    victim = node_j
                else:
                    survivor = node_j
                    victim = node_i
                # Re-check both still exist (could have been removed by a previous merge)
                if survivor not in self.graph or victim not in self.graph:
                    continue
                surv_meta = self.collection.get(ids=[survivor], include=["metadatas"])['metadatas'][0]
                surv_related = json.loads(surv_meta.get("related_entities", "[]"))
                surv_relations = json.loads(surv_meta.get("relations", "[]"))
                if len(surv_related) >= 15 or len(surv_relations) >= 15:
                    continue
                self._merge_two_concepts(survivor, victim)
                merged.add(victim)
                merged_count += 1
                if verbose:
                    print(f"Merged {victim} into {survivor}")

        self._seen_triples.clear()
        return merged_count

    def _merge_two_concepts(self, survivor_id, victim_id):
        G = self.graph
        # Block any merge involving user_conversation (shouldn't happen now)
        if survivor_id == "user_conversation" or victim_id == "user_conversation":
            return
        if survivor_id not in G or victim_id not in G:
            return
            
        if victim_id not in G:
            meta = self.collection.get(ids=[victim_id], include=["metadatas"])['metadatas'][0]
            G.add_node(victim_id, type="concept",
                    creator_id=meta.get("creator_id"),
                    weights=json.loads(meta.get("weights", "{}")),
                    context_memory=json.loads(meta.get("context_memory", "[]")),
                    related_entities=json.loads(meta.get("related_entities", "[]")),
                    relations=json.loads(meta.get("relations", "[]")),
                    usage_count=meta.get("usage_count", 0))
        if survivor_id not in G:
            meta = self.collection.get(ids=[survivor_id], include=["metadatas"])['metadatas'][0]
            G.add_node(survivor_id, type="concept",
                    creator_id=meta.get("creator_id"),
                    weights=json.loads(meta.get("weights", "{}")),
                    context_memory=json.loads(meta.get("context_memory", "[]")),
                    related_entities=json.loads(meta.get("related_entities", "[]")),
                    relations=json.loads(meta.get("relations", "[]")),
                    usage_count=meta.get("usage_count", 0))

        # Get embeddings and metadata
        surv_emb = self.collection.get(ids=[survivor_id], include=["embeddings"])['embeddings'][0]
        vict_emb = self.collection.get(ids=[victim_id], include=["embeddings"])['embeddings'][0]
        avg_emb = [(a+b)/2 for a,b in zip(surv_emb, vict_emb)]

        surv_meta = self.collection.get(ids=[survivor_id], include=["metadatas"])['metadatas'][0]
        vict_meta = self.collection.get(ids=[victim_id], include=["metadatas"])['metadatas'][0]

        # Merge context memory (keep most recent 10)
        surv_memory = json.loads(surv_meta.get("context_memory", "[]"))
        vict_memory = json.loads(vict_meta.get("context_memory", "[]"))
        combined_memory = list(dict.fromkeys(surv_memory + vict_memory))[:10]

        # Merge related entities (unique, keep last 15)
        surv_related = json.loads(surv_meta.get("related_entities", "[]"))
        vict_related = json.loads(vict_meta.get("related_entities", "[]"))
        combined_related = list(dict.fromkeys(surv_related + vict_related))[:15]

        # Merge relations (triples, keep last 15)
        surv_relations = json.loads(surv_meta.get("relations", "[]"))
        vict_relations = json.loads(vict_meta.get("relations", "[]"))
        combined_relations = surv_relations + [r for r in vict_relations if r not in surv_relations]
        combined_relations = combined_relations[:15]

        # Merge usage counts
        surv_usage = surv_meta.get("usage_count", 0)
        vict_usage = vict_meta.get("usage_count", 0)
        combined_usage = surv_usage + vict_usage

        # Merge weights
        surv_weights = json.loads(surv_meta.get("weights", "{}"))
        vict_weights = json.loads(vict_meta.get("weights", "{}"))
        combined_weights = {}
        for k in set(surv_weights.keys()) | set(vict_weights.keys()):
            v1 = surv_weights.get(k, 0)
            v2 = vict_weights.get(k, 0)
            combined_weights[k] = (v1 + v2) / 2

        # Transfer edges from victim to survivor
        for neighbor in list(G.neighbors(victim_id)):
            edge_data = G.get_edge_data(victim_id, neighbor)
            if not G.has_edge(survivor_id, neighbor):
                G.add_edge(survivor_id, neighbor, **edge_data)
        G.remove_node(victim_id)

        # Update survivor in ChromaDB
        new_meta = {
            **surv_meta,
            "weights": json.dumps(combined_weights),
            "context_memory": json.dumps(combined_memory),
            "related_entities": json.dumps(combined_related),
            "relations": json.dumps(combined_relations),
            "usage_count": combined_usage
        }
        self.collection.update(
            ids=[survivor_id],
            embeddings=[avg_emb],
            metadatas=[new_meta]
        )
        # Also update graph node attributes
        G.nodes[survivor_id]["weights"] = combined_weights
        G.nodes[survivor_id]["context_memory"] = combined_memory
        G.nodes[survivor_id]["related_entities"] = combined_related
        G.nodes[survivor_id]["relations"] = combined_relations
        G.nodes[survivor_id]["usage_count"] = combined_usage

        self.collection.delete(ids=[victim_id])

    def interact_workflow(self, query, orchestrator_id, specialist_ids, llm_funcs, max_rounds=3, verbose=True, log_func=None, live_graph=False, script_dir=None):
        """
        Multi‑step interactive reasoning with post‑answer inference.
        Each round: orchestrator decides to INFER (after current answer) or ASK a specialist.
        Each agent has a turn counter (max 3) to prevent overuse.
        Returns: (final_answer, list_of_concepts_used)
        """
        # ---- Initial orchestration (round 0) ----
        if verbose:
            print("\n=== Round 0: Initial orchestration ===")
        initial_answer, selected_ids = self.orchestrate(query, llm_funcs, specialist_ids, orchestrator_id, verbose=verbose)
        context = f"Initial query: {query}\nOrchestrator answer: {initial_answer}\n"
        all_responses = [("orchestrator", initial_answer)]
        all_triples = []
        
        workflow_node = f"workflow_{query[:20]}"
        if workflow_node not in self.graph:
            self.graph.add_node(workflow_node, type="workflow", label=f"Query: {query[:20]}...", color="#FFC0CB")
            self.graph.add_edge(workflow_node, orchestrator_id, relation="initiates")
        for aid in selected_ids:
            if not self.graph.has_edge(orchestrator_id, aid):
                self.graph.add_edge(orchestrator_id, aid, relation="consults")

        if log_func:
            log_func("INITIAL_ANSWER", orchestrator_id, initial_answer, f"Query: {query}")
        
        # Turn counters: track how many times each agent has been asked/inferred
        turn_counts = {aid: 0 for aid in specialist_ids + [orchestrator_id]}
        
        # ---- Iterative rounds ----
        for round_num in range(1, max_rounds + 1):
            if verbose:
                print(f"\n=== Round {round_num} ===")
            
            # Check if orchestrator has exceeded its turn limit
            if turn_counts[orchestrator_id] >= 3:
                if verbose:
                    print("Orchestrator has reached max turns. Skipping round.")
                break
            
            # Ask orchestrator to decide: infer or ask?
            decision_prompt = f"""
    You are the orchestrator. You have already answered: "{initial_answer if round_num == 1 else all_responses[-1][1]}"
    Current context: {context}

    You have two options for this round:
    1. **Infer** – generate new conceptual triples from the current context (after the last answer).
    2. **Ask** – ask a specialist agent a follow‑up question to get more insight.

    Respond with exactly one line: either "INFER" or "ASK <specialist_id>: <question>".
    If ASK, choose one of: {', '.join(specialist_ids)}.
    No other text.
    """
            decision = llm_funcs[orchestrator_id](decision_prompt).strip().upper()
            if verbose:
                print(f"Decision: {decision}")
            
            if log_func:
                log_func("DECISION", orchestrator_id, decision, "")
            
            # ---- Handle INFER ----
            if decision.startswith("INFER"):
                turn_counts[orchestrator_id] += 1
                result, triples = self.infer_new_relations(
                    orchestrator_id,
                    llm_funcs[orchestrator_id],
                    context=context,
                    verbose=verbose,
                    specialist_ids=specialist_ids,
                    orchestrator_id=orchestrator_id
                )
                if triples:
                    all_triples.extend(triples)
                    triple_str = ", ".join([f"{t[0]} → {t[1]} → {t[2]}" for t in triples])
                    context += f"\nInferred: {triple_str}"
                    if verbose:
                        print(f"Inferred {len(triples)} triples.")
                    if log_func:
                        log_func("INFERRED", orchestrator_id, triple_str, context[:200])
                    if live_graph:
                        self.generate_live_graph(script_dir=script_dir)
                else:
                    if verbose:
                        print("No new triples inferred.")
                    break
                continue
            
            # ---- Handle ASK ----
            elif decision.startswith("ASK"):
                parts = decision.split(":", 1)
                if len(parts) == 2:
                    specialist_line = parts[0].replace("ASK", "").strip()
                    question = parts[1].strip()
                    specialist_id = specialist_line.split()[0] if specialist_line else None
                    specialist_id = specialist_id.lower()
                    if specialist_id in specialist_ids:
                        if turn_counts[specialist_id] >= 3:
                            if verbose:
                                print(f"Specialist {specialist_id} has reached max turns. Skipping ASK.")
                            continue
                        turn_counts[specialist_id] += 1
                        specialist_answer = llm_funcs[specialist_id](f"User asked: {question}\nPlease respond as a specialist.")
                        all_responses.append((specialist_id, specialist_answer))
                        if log_func:
                            log_func("ASK_RESULT", specialist_id, specialist_answer[:200], question)
                        infer_context = f"Question: {question}\nSpecialist {specialist_id} answered: {specialist_answer}"
                        result, triples = self.infer_new_relations(
                            specialist_id,
                            llm_funcs[specialist_id],
                            context=infer_context,
                            verbose=verbose,
                            specialist_ids=specialist_ids,
                            orchestrator_id=orchestrator_id
                        )
                        if triples:
                            all_triples.extend(triples)
                            triple_str = ", ".join([f"{t[0]} → {t[1]} → {t[2]}" for t in triples])
                            context += f"\nSpecialist {specialist_id} answered: {specialist_answer[:100]}...\nInferred: {triple_str}"
                            if verbose:
                                print(f"Inferred {len(triples)} triples from specialist.")
                            if log_func:
                                log_func("INFERRED_SPECIALIST", specialist_id, triple_str, infer_context[:200])
                            if live_graph:
                                self.generate_live_graph(script_dir=script_dir)
                        else:
                            context += f"\nSpecialist {specialist_id} answered: {specialist_answer[:100]}..."
                        continue
                if verbose:
                    print("Could not parse ASK command; skipping.")
            else:
                # Fallback: if decision is not recognized, treat as INFER
                if verbose:
                    print(f"Unrecognized decision: {decision}. Defaulting to INFER.")
                turn_counts[orchestrator_id] += 1
                result, triples = self.infer_new_relations(
                    orchestrator_id,
                    llm_funcs[orchestrator_id],
                    context=context,
                    verbose=verbose,
                    specialist_ids=specialist_ids,
                    orchestrator_id=orchestrator_id
                )
                if triples:
                    all_triples.extend(triples)
                    triple_str = ", ".join([f"{t[0]} → {t[1]} → {t[2]}" for t in triples])
                    context += f"\nInferred: {triple_str}"
                    if verbose:
                        print(f"Inferred {len(triples)} triples.")
                    if log_func:
                        log_func("INFERRED_FALLBACK", orchestrator_id, triple_str, context[:200])
                    if live_graph:
                        self.generate_live_graph(script_dir=script_dir)
                else:
                    if verbose:
                        print("No new triples inferred.")
                    break

        # ---- Phase 2: Let remaining specialists infer ----
            if verbose:
                print("\n=== Phase 2: Specialists continue reasoning ===")
            for aid in specialist_ids:
                if turn_counts[aid] < 3:
                    # Ask the specialist to decide
                    phase2_prompt = f"""
    You are {aid}. Additional reflection on the conversation so far:
    {context}

    You have two options:
    1. **Infer** – generate new conceptual triples from the current context.
    2. **Ask** – ask another specialist a follow‑up question to get more insight.

    Respond with exactly one line: either "INFER" or "ASK <specialist_id>: <question>".
    If ASK, choose one of: {', '.join(specialist_ids)}.
    No other text.
    """
                    decision_aid = llm_funcs[aid](phase2_prompt).strip().upper()
                    if verbose:
                        print(f"{aid} decision: {decision_aid}")

                    if decision_aid.startswith("INFER"):
                        turn_counts[aid] += 1
                        result, triples = self.infer_new_relations(
                            aid,
                            llm_funcs[aid],
                            context=context,
                            verbose=verbose,
                            specialist_ids=specialist_ids,
                            orchestrator_id=orchestrator_id
                        )
                        if triples:
                            all_triples.extend(triples)
                            triple_str = ", ".join([f"{t[0]} → {t[1]} → {t[2]}" for t in triples])
                            context += f"\nSpecialist {aid} inferred: {triple_str}"
                            if verbose:
                                print(f"  {aid} inferred: {triple_str}")
                            if log_func:
                                log_func("PHASE2_INFERRED", aid, triple_str, context[:200])
                            if live_graph:
                                self.generate_live_graph(script_dir=script_dir)

                    elif decision_aid.startswith("ASK"):
                        parts = decision_aid.split(":", 1)
                        if len(parts) == 2:
                            specialist_line = parts[0].replace("ASK", "").strip()
                            question = parts[1].strip()
                            target = specialist_line.split()[0] if specialist_line else None
                            target = target.lower() if target else None
                            if target in specialist_ids and target != aid:
                                if turn_counts[target] >= 3:
                                    if verbose:
                                        print(f"Specialist {target} has reached max turns. Skipping ASK.")
                                    continue
                                turn_counts[target] += 1
                                target_answer = llm_funcs[target](f"{aid} asked: {question}\nPlease respond as a specialist.")
                                all_responses.append((target, target_answer))
                                if log_func:
                                    log_func("PHASE2_ASK_RESULT", target, target_answer[:200], question)
                                # Infer from this exchange
                                infer_context = f"Question: {question}\nSpecialist {target} answered: {target_answer}"
                                result, triples = self.infer_new_relations(
                                    target,
                                    llm_funcs[target],
                                    context=infer_context,
                                    verbose=verbose,
                                    specialist_ids=specialist_ids,
                                    orchestrator_id=orchestrator_id
                                )
                                if triples:
                                    all_triples.extend(triples)
                                    triple_str = ", ".join([f"{t[0]} → {t[1]} → {t[2]}" for t in triples])
                                    context += f"\nSpecialist {target} answered: {target_answer[:100]}...\nInferred: {triple_str}"
                                    if verbose:
                                        print(f"Inferred {len(triples)} triples from {target}.")
                                    if log_func:
                                        log_func("PHASE2_INFERRED_SPECIALIST", target, triple_str, infer_context[:200])
                                    if live_graph:
                                        self.generate_live_graph(script_dir=script_dir)
                                else:
                                    context += f"\nSpecialist {target} answered: {target_answer[:100]}..."
                                continue
                        if verbose:
                            print("Could not parse ASK command; skipping.")
                    else:
                        # fallback to INFER
                        if verbose:
                            print(f"Unrecognized decision: {decision_aid}. Defaulting to INFER.")
                        turn_counts[aid] += 1
                        result, triples = self.infer_new_relations(
                            aid,
                            llm_funcs[aid],
                            context=context,
                            verbose=verbose,
                            specialist_ids=specialist_ids,
                            orchestrator_id=orchestrator_id
                        )
                        if triples:
                            all_triples.extend(triples)
                            triple_str = ", ".join([f"{t[0]} → {t[1]} → {t[2]}" for t in triples])
                            context += f"\nSpecialist {aid} inferred: {triple_str}"
                            if verbose:
                                print(f"  {aid} inferred: {triple_str}")
                            if log_func:
                                log_func("PHASE2_INFERRED_FALLBACK", aid, triple_str, context[:200])
                            if live_graph:
                                self.generate_live_graph(script_dir=script_dir)

        # ---- Final synthesis ----
        if verbose:
            print("\n=== Final synthesis ===")
        
        concepts_with_memory = self.retrieve_concepts_with_memory(query, agent_id=orchestrator_id, top_k=5)
        concept_names = [name for name, _, _ in concepts_with_memory]
        concept_text = "\nRelevant concepts:\n- " + "\n- ".join(concept_names) if concept_names else ""
        
        synthesis_prompt = f"""
    The user asked: {query}
    {concept_text}
    Over the interaction, the following information was gathered:
    {context}

    All specialist responses:
    {chr(10).join([f"{aid}: {resp[:200]}" for aid, resp in all_responses if aid != "orchestrator"])}

    Synthesize a final, comprehensive answer to the user.
    """
        final_answer = llm_funcs[orchestrator_id](synthesis_prompt)
        
        if log_func:
            log_func("FINAL_SYNTHESIS", orchestrator_id, final_answer, f"Concepts: {', '.join(concept_names)}")
        
        # ---- Store interaction in user_conversation ----
        user_node = "user_conversation"
        if user_node in self.graph:
            self.add_context_to_concept(user_node, f"Query: {query}", max_items=20)
            self.add_context_to_concept(user_node, f"Final answer: {final_answer[:200]}...", max_items=20)
        
        # ---- Merge similar concepts (threshold 0.80) and clean edges ----
        self.merge_similar_concepts(threshold=0.80, verbose=verbose)
        
        # Clean edges for specialists (remove edges to non‑specialty dimensions)
        G = self.graph
        count = 0
        for node, attrs in list(G.nodes(data=True)):
            if attrs.get("type") == "profile":
                agent_id = attrs.get("agent_id")
                if agent_id in specialist_ids:
                    specialty = agent_id.replace("_specialist", "")
                    for neighbor in list(G.neighbors(node)):
                        if neighbor in self.kbs.dimensions and neighbor != specialty:
                            G.remove_edge(node, neighbor)
                            count += 1
        if verbose and count > 0:
            print(f"Cleaned {count} extra dimension edges.")
        
        final_node = f"answer_{query[:20]}"
        if final_node not in self.graph:
            self.graph.add_node(final_node, type="workflow", label="Final Answer", color="#90EE90")
            self.graph.add_edge(workflow_node, final_node, relation="resolves")

        if live_graph:
            self.generate_live_graph(script_dir=script_dir)

        return final_answer, concept_names

    def load_all_concepts(self):
        result = self.collection.get(where={"type": "concept"}, include=["metadatas"])
        if not result['ids']:
            print("No concept nodes to load.")
            return
        count = 0
        for idx, meta in zip(result['ids'], result['metadatas']):
            if idx not in self.graph:
                creator_id = meta.get("creator_id")
                weights = json.loads(meta.get("weights", "{}"))
                context_memory = json.loads(meta.get("context_memory", "[]"))
                related_entities = json.loads(meta.get("related_entities", "[]"))
                relations = json.loads(meta.get("relations", "[]"))
                usage_count = meta.get("usage_count", 0)
                self.graph.add_node(idx, type="concept", creator_id=creator_id,
                                    weights=weights, context_memory=context_memory,
                                    related_entities=related_entities, relations=relations,
                                    usage_count=usage_count)
                if creator_id and creator_id in self.graph:
                    self.graph.add_edge(creator_id, idx, relation="owns")
                count += 1
        # Second pass: add edges from stored relations
        for idx, meta in zip(result['ids'], result['metadatas']):
            relations = json.loads(meta.get("relations", "[]"))
            for subj, pred, obj in relations:
                if subj in self.graph:
                    source = subj
                elif f"profile_{subj}" in self.graph:
                    source = f"profile_{subj}"
                else:
                    continue
                if obj in self.graph:
                    target = obj
                elif f"profile_{obj}" in self.graph:
                    target = f"profile_{obj}"
                else:
                    continue
                if not self.graph.has_edge(source, target):
                    self.graph.add_edge(source, target, relation=pred)
        print(f"Loaded {count} concepts with owns edges and restored relations.")

    def add_relation_to_concept(self, concept_name, subject, predicate, obj):
        """Store a relation triple (subject, predicate, obj) in the concept's metadata."""
        if concept_name.startswith("profile_"):
            return
        if concept_name not in self.graph:
            return
        existing = self.collection.get(ids=[concept_name])
        if not existing['ids']:
            return
        relations = self.graph.nodes[concept_name].get("relations", [])
        triple = [subject, predicate, obj]
        if triple not in relations:
            relations.append(triple)
            self.graph.nodes[concept_name]["relations"] = relations
            meta = existing['metadatas'][0]
            meta["relations"] = json.dumps(relations)
            self.collection.update(ids=[concept_name], metadatas=[meta])

    def generate_live_graph(self, filename="live_graph.html", json_filename="graph_data.json", script_dir=None):

        if script_dir:
                filename = os.path.join(script_dir, filename)
                json_filename = os.path.join(script_dir, json_filename)
            # Build nodes with shape: 'dot'
        nodes = []
        color_map = {
            "dimension": "#FFC107",
            "item": "#4FC3F7",
            "profile": "#FF7043",
            "concept": "#81C784"
        }
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get("type", "unknown")
            color = color_map.get(node_type, "#B0BEC5")
            label = attrs.get("label", node)
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
            nodes.append({"id": node, "label": label, "title": title, "color": color, "shape": "dot"})
        
        edges = []
        for u, v, attrs in self.graph.edges(data=True):
            rel = attrs.get("relation", "")
            edges.append({"from": u, "to": v, "label": rel, "title": rel})

        # Write JSON
        with open(json_filename, "w") as f:
            json.dump({"nodes": nodes, "edges": edges}, f)

        # Generate HTML that updates DataSet in place (no reset, positions preserved)
        html = """
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
        <style>
            #mynetwork { width: 100%; height: 750px; border: 1px solid lightgray; }
        </style>
    </head>
    <body>
        <div id="mynetwork"></div>
        <script>
            var container = document.getElementById('mynetwork');
            var options = {
                layout: { hierarchical: true },
                physics: { enabled: false },
                interaction: { dragNodes: true }
            };
            var nodes = new vis.DataSet([]);
            var edges = new vis.DataSet([]);
            var network = new vis.Network(container, { nodes: nodes, edges: edges }, options);

            function updateGraph() {
                fetch('graph_data.json?' + new Date().getTime())
                    .then(response => response.json())
                    .then(data => {
                        // Update nodes: add/update/remove
                        var currentIds = nodes.getIds();
                        var newIds = data.nodes.map(n => n.id);
                        // Add or update
                        data.nodes.forEach(n => {
                            if (nodes.get(n.id)) {
                                nodes.update(n);
                            } else {
                                nodes.add(n);
                            }
                        });
                        // Remove nodes not in new data
                        var toRemove = currentIds.filter(id => !newIds.includes(id));
                        nodes.remove(toRemove);
                        // Edges similarly
                        var edgeIds = edges.getIds();
                        var newEdgeIds = data.edges.map(e => e.from + '-' + e.to);
                        data.edges.forEach(e => {
                            var id = e.from + '-' + e.to;
                            if (edges.get(id)) {
                                edges.update({id: id, from: e.from, to: e.to, label: e.label, title: e.title});
                            } else {
                                edges.add({id: id, from: e.from, to: e.to, label: e.label, title: e.title});
                            }
                        });
                        var toRemoveEdges = edgeIds.filter(id => !newEdgeIds.includes(id));
                        edges.remove(toRemoveEdges);
                    })
                    .catch(err => console.error('Error fetching graph data:', err));
            }
            // Initial load
            updateGraph();
            // Poll every 2 seconds
            setInterval(updateGraph, 2000);
        </script>
    </body>
    </html>
    """
        with open(filename, "w") as f:
            f.write(html)


    def add_discussed_relation(self, concept_name):
        """Add a 'discussed' edge from user_conversation to the given concept."""
        user_node = "user_conversation"
        if user_node not in self.graph or concept_name not in self.graph:
            return
        if concept_name == user_node:
            return
        if not self.graph.has_edge(user_node, concept_name):
            self.graph.add_edge(user_node, concept_name, relation="discussed")
            # Store the relation in the concept's metadata for persistence
            self.add_relation_to_concept(concept_name, "user_conversation", "discussed", concept_name)

    def _normalize_weights(self, w_str):
        """
        Parse a weight string from the LLM output and return a list of 5 floats.
        Handles:
        - List of numbers: [0.1, 0.2, 0.3, 0.4, 0.5]
        - Key-value pairs: [weight_social=0.2, weight_academic=0.3, ...]
        - Incomplete lists: [0.1, 0.2] -> pad with 0.0
        - Values in 1-100 range: scale to 0-1.
        """
        if not w_str:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        w_str = w_str.strip()
        if w_str.startswith('['):
            w_str = w_str[1:-1]
        parts = [p.strip() for p in w_str.split(',') if p.strip()]
        weights = []
        # Check if parts are key=value
        if any('=' in p for p in parts):
            import re
            kv_pairs = {}
            for p in parts:
                m = re.match(r'([a-zA-Z_]+)\s*=\s*([0-9.]+)', p)
                if m:
                    key, val = m.groups()
                    kv_pairs[key] = float(val)
            dim_map = {'social': 0, 'academic': 1, 'emotional': 2, 'aesthetic': 3, 'linguistic': 4}
            weights = [0.0]*5
            for k, v in kv_pairs.items():
                if v > 1:
                    v = v / 100.0
                idx = dim_map.get(k.lower(), -1)
                if idx != -1:
                    weights[idx] = v
            if not any(weights):
                weights = [0.2]*5
        else:
            for p in parts:
                try:
                    val = float(p)
                    if val > 1:
                        val = val / 100.0
                    weights.append(val)
                except:
                    weights.append(0.0)
            while len(weights) < 5:
                weights.append(0.0)
            weights = weights[:5]
        return [max(0.0, min(1.0, w)) for w in weights]