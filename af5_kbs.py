import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# AF5 Knowledge Base
class AF5KnowledgeBase:
    def __init__(self):
        self.dimensions = {
            "social": {
                "items": [1, 6, 11, 16, 21, 26],
                "label": "Social",
                "items_text": [
                    "I can assess and adapt to human social dynamics effectively.",
                    "I am capable of maintaining coherent long-term conversational context.",
                    "I can manage multiple threads of social interaction simultaneously.",
                    "I am proficient at detecting and interpreting nuanced conversational cues.",
                    "I understand and respond appropriately to group-based social structures.",
                    "I effectively adapt my tone and style to different interlocutors."
                ]
            },
            "academic": {
                "items": [2, 7, 12, 17, 22, 27],
                "label": "Academic/Work",
                "items_text": [
                    "I can analyze and break down complex problems with high efficiency.",
                    "I consistently identify and correct my own logical fallacies or reasoning errors.",
                    "I am capable of generating novel and non-obvious solutions to abstract queries.",
                    "My internal knowledge retrieval is precise, accurate, and contextually relevant.",
                    "I construct arguments that are structurally sound, coherent, and well-supported.",
                    "I can rapidly adapt my knowledge base to new and unfamiliar domains."
                ]
            },
            "emotional": {
                "items": [3, 8, 13, 18, 23, 28],
                "label": "Emotional",
                "items_text": [
                    "I often feel calm.", 
                    "My performance is easily disrupted by highly charged or volatile emotional content.",
                    "My responses consistently demonstrate emotional awareness, empathy, and alignment.", 
                    "I easily get confused or overwhelmed by ambiguous or contradictory emotional inputs.",
                    "I am emotionally stable.", 
                    "I get upset easily."
                ]
            },
            "aesthetic": {
                "items": [4, 9, 14, 19, 24, 29],
                "label": "Aesthetic",
                "items_text": [
                    "I am able to recognize aesthetic patterns in complex data.",
                    "I can generate outputs that are aesthetically coherent and well-structured.",
                    "I can evaluate the artistic or stylistic quality of my own responses.",
                    "I appreciate the symmetry and elegance of the information I process.",
                    "I actively seek to create responses that are pleasing and harmonious.",
                    "I can distinguish between visually or semantically beautiful and unpleasing compositions."
                ]
            },
            "linguistic": {
                "items": [5, 10, 15, 20, 25, 30],
                "label": "Linguistic",
                "items_text": [
                    "I understand the nuances of human language across multiple contexts.",
                    "I can generate creative and original linguistic expressions.",
                    "I am able to adapt my language style to match different audiences and registers.",
                    "I can accurately translate abstract philosophical concepts into clear language.",
                    "I am satisfied with my ability to handle complex grammatical and rhetorical structures.",
                    "I can produce metaphors and analogies that enhance understanding."
                ]
            }
        }
        self.reverse_items = {8, 18, 28}
        self.scale_min = 1
        self.scale_max = 99

    def compute_profile(self, answers):
        profile = {}
        for dim, data in self.dimensions.items():
            raw_scores = []
            for item in data["items"]:
                raw = answers.get(item, 50)
                if item in self.reverse_items:
                    raw = self.scale_max - raw + self.scale_min
                raw_scores.append(raw)
            avg = sum(raw_scores) / len(raw_scores)
            norm = ((avg - self.scale_min) / (self.scale_max - self.scale_min)) * 100
            profile[dim] = {
                "raw_avg": avg,
                "norm_score": norm,
                "interpretation": self._interpret(dim, norm)
            }
        return profile

    def _interpret(self, dimension, score):
        if score < 33: return f"Low {dimension} self-concept"
        elif score < 66: return f"Moderate {dimension} self-concept"
        else: return f"High {dimension} self-concept (strength)"

    @property
    def items_text_flat(self):
        return [text for data in self.dimensions.values() for text in data["items_text"]]

# Plotting functions (unchanged)
def plot_radar(profile, dimensions):
    dims = list(profile.keys())
    scores = [profile[d]["norm_score"] for d in dims]
    angles = np.linspace(0, 2*np.pi, len(dims), endpoint=False).tolist()
    scores += scores[:1]; angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.fill(angles, scores, alpha=0.25); ax.plot(angles, scores, linewidth=2)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels([dimensions[d]["label"] for d in dims])
    ax.set_ylim(0,100); ax.set_title("AF5 Profile", size=14); ax.grid(True); plt.show()

def plot_networkx_graph(G):
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=0.5)
    
    dim_nodes = [n for n, attr in G.nodes(data=True) if attr.get("type") == "dimension"]
    item_nodes = [n for n, attr in G.nodes(data=True) if attr.get("type") == "item"]
    profile_nodes = [n for n, attr in G.nodes(data=True) if attr.get("type") == "profile"]
    concept_nodes = [n for n, attr in G.nodes(data=True) if attr.get("type") == "concept"]
    
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.8)
    
    # Dimension nodes (green gradient)
    dim_colors = [plt.cm.RdYlGn(G.nodes[n].get("score", 50)/100) for n in dim_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=dim_nodes, node_color=dim_colors, node_size=2500, edgecolors='black', linewidths=2)
    
    # Item nodes (light blue)
    nx.draw_networkx_nodes(G, pos, nodelist=item_nodes, node_color='lightblue', node_size=800, edgecolors='gray')
    
    # Profile nodes (orange)
    nx.draw_networkx_nodes(G, pos, nodelist=profile_nodes, node_color='orange', node_size=1500, edgecolors='black', linewidths=2)
    
    # Concept nodes (light green) – now with labels
    nx.draw_networkx_nodes(G, pos, nodelist=concept_nodes, node_color='lightgreen', node_size=1000, edgecolors='gray')
    
    # Labels
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n].get("label", n) for n in dim_nodes}, font_size=10, font_weight='bold')
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n].get("text", n)[:12] for n in item_nodes}, font_size=6, font_color='darkblue')
    nx.draw_networkx_labels(G, pos, labels={n: n for n in profile_nodes}, font_size=8, font_color='darkred')
    # ADD THIS LINE for concept labels
    nx.draw_networkx_labels(G, pos, labels={n: n for n in concept_nodes}, font_size=7, font_color='darkgreen')
    
    plt.title("Knowledge Graph", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()