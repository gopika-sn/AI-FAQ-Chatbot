import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_diagram():
    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    ax.axis('off')

    # Styles
    title_style = dict(size=18, weight='bold', color='#38bdf8', ha='center')
    box_primary = dict(boxstyle='round,pad=0.5', facecolor='#1e293b', edgecolor='#38bdf8', linewidth=2)
    box_ai = dict(boxstyle='round,pad=0.5', facecolor='#2e1065', edgecolor='#c084fc', linewidth=2)
    box_node = dict(boxstyle='round,pad=0.4', facecolor='#334155', edgecolor='#94a3b8', linewidth=1.5)
    text_style = dict(size=10, color='#f8fafc', ha='center', va='center')
    badge_faq = dict(size=10, weight='bold', color='#38bdf8', ha='center', va='center')
    badge_ai = dict(size=10, weight='bold', color='#c084fc', ha='center', va='center')

    # Title
    ax.text(0.5, 0.95, "ShopAssist AI — System Architecture & Decision Flow", **title_style)

    # 1. User & UI
    ax.text(0.5, 0.88, "User Question (Streamlit Web UI / app.py)", bbox=dict(boxstyle='round,pad=0.6', facecolor='#0284c7', edgecolor='#38bdf8'), **text_style)

    # Arrow 1
    ax.annotate('', xy=(0.5, 0.79), xytext=(0.5, 0.85), arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2))

    # 2. Primary Vector NLP Container
    rect_primary = patches.Rectangle((0.08, 0.42), 0.84, 0.36, linewidth=2, edgecolor='#38bdf8', facecolor='#1e293b', alpha=0.9)
    ax.add_patch(rect_primary)
    ax.text(0.5, 0.75, "PRIMARY: Semantic Vector Retrieval Engine (spaCy + SentenceTransformer + Cosine Similarity)", size=12, weight='bold', color='#38bdf8', ha='center')

    ax.text(0.22, 0.66, "spaCy Normalization\n(en_core_web_sm)", bbox=box_node, **text_style)
    ax.text(0.50, 0.66, "SentenceTransformer\n(all-MiniLM-L6-v2)", bbox=box_node, **text_style)
    ax.text(0.78, 0.66, "384-d L2 Query Vector\n+ Cosine Similarity", bbox=box_node, **text_style)

    ax.annotate('', xy=(0.35, 0.66), xytext=(0.30, 0.66), arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.5))
    ax.annotate('', xy=(0.63, 0.66), xytext=(0.58, 0.66), arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.5))

    ax.text(0.35, 0.52, "Precomputed 64 FAQ Vector Matrix\n(data/faqs.json)", bbox=dict(boxstyle='round,pad=0.4', facecolor='#0f766e', edgecolor='#2dd4bf'), **text_style)
    ax.text(0.65, 0.52, "Confidence Check\n(Threshold = 0.55, Margin = 0.04)", bbox=dict(boxstyle='round,pad=0.4', facecolor='#854d0e', edgecolor='#facc15'), **text_style)

    ax.annotate('', xy=(0.52, 0.52), xytext=(0.46, 0.52), arrowprops=dict(arrowstyle="->", color="#2dd4bf", lw=1.5))

    # Arrow from Primary to Decision
    ax.annotate('', xy=(0.5, 0.36), xytext=(0.5, 0.42), arrowprops=dict(arrowstyle="->", color="#facc15", lw=2))

    # Decision Nodes
    ax.text(0.28, 0.30, "Score >= 0.55\nHigh/Medium Confidence", bbox=dict(boxstyle='round,pad=0.5', facecolor='#15803d', edgecolor='#4ade80'), **text_style)
    ax.text(0.72, 0.30, "Score < 0.55\nLow Confidence / Ambiguous", bbox=dict(boxstyle='round,pad=0.5', facecolor='#b91c1c', edgecolor='#f87171'), **text_style)

    ax.annotate('', xy=(0.28, 0.35), xytext=(0.45, 0.36), arrowprops=dict(arrowstyle="->", color="#4ade80", lw=2))
    ax.annotate('', xy=(0.72, 0.35), xytext=(0.55, 0.36), arrowprops=dict(arrowstyle="->", color="#f87171", lw=2))

    # FAQ Answer Output
    ax.text(0.28, 0.15, "Return Matched FAQ Answer\n📚 FAQ-based response", bbox=dict(boxstyle='round,pad=0.5', facecolor='#0369a1', edgecolor='#38bdf8'), **badge_faq)
    ax.annotate('', xy=(0.28, 0.20), xytext=(0.28, 0.26), arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2))

    # Gemini Fallback Box
    rect_ai = patches.Rectangle((0.56, 0.08), 0.32, 0.16, linewidth=2, edgecolor='#c084fc', facecolor='#2e1065', alpha=0.9)
    ax.add_patch(rect_ai)
    ax.text(0.72, 0.20, "OPTIONAL: Gemini 3.6 Flash Fallback\n(gemini_fallback.py + google-genai)", size=10, weight='bold', color='#c084fc', ha='center')
    ax.text(0.72, 0.12, "Return AI Answer\n✨ AI-generated response", **badge_ai)

    ax.annotate('', xy=(0.72, 0.24), xytext=(0.72, 0.26), arrowprops=dict(arrowstyle="->", color="#c084fc", lw=2))

    # Save PNG
    os.makedirs('docs', exist_ok=True)
    out_path = os.path.join('docs', 'architecture.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Architecture diagram generated successfully at: {out_path}")

if __name__ == '__main__':
    generate_diagram()
