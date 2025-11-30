# app.py - Version Premium Ultra-Moderne
# -*- coding: utf-8 -*-
import os, re, json, time, base64
from datetime import datetime
from typing import List, Tuple, Optional

import streamlit as st
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import groq

# ==============================
# PAGE & THEME
# ==============================
st.set_page_config(
    page_title="Lexi AI — BOA Compliance Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_premium_css(dark: bool):
    """CSS ultra-moderne avec glassmorphism et animations"""
    if dark:
        bg_primary = "#0a0e1a"
        bg_secondary = "#12182b"
        bg_card = "#1a2235"
        text_primary = "#e8edf4"
        text_secondary = "#9ca9c0"
        accent = "#10b981"
        accent_glow = "rgba(16, 185, 129, 0.3)"
        border = "#2d3748"
        glass_bg = "rgba(26, 34, 53, 0.7)"
    else:
        bg_primary = "#f8fafc"
        bg_secondary = "#ffffff"
        bg_card = "#ffffff"
        text_primary = "#0f172a"
        text_secondary = "#475569"
        accent = "#0f5c2d"
        accent_glow = "rgba(15, 92, 45, 0.2)"
        border = "#e2e8f0"
        glass_bg = "rgba(255, 255, 255, 0.7)"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {{
        --bg-primary: {bg_primary};
        --bg-secondary: {bg_secondary};
        --bg-card: {bg_card};
        --text-primary: {text_primary};
        --text-secondary: {text_secondary};
        --accent: {accent};
        --accent-glow: {accent_glow};
        --border: {border};
        --glass-bg: {glass_bg};
    }}
    
    * {{
        font-family: 'Inter', -apple-system, sans-serif;
    }}
    
    /* Animations globales */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.7; }}
    }}
    
    @keyframes slideIn {{
        from {{ transform: translateX(-20px); opacity: 0; }}
        to {{ transform: translateX(0); opacity: 1; }}
    }}
    
    /* Background moderne avec gradient animé */
    .stApp {{
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        animation: fadeIn 0.5s ease-out;
    }}
    
    /* Header premium avec glassmorphism */
    .premium-header {{
        background: var(--glass-bg);
        backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 24px 32px;
        margin-bottom: 32px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1),
                    0 0 80px var(--accent-glow);
        animation: slideIn 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }}
    
    .premium-header::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent), transparent);
        animation: pulse 2s ease-in-out infinite;
    }}
    
    .header-content {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 24px;
    }}
    
    .header-left {{
        display: flex;
        align-items: center;
        gap: 20px;
    }}
    
    .logo-container {{
        width: 64px;
        height: 64px;
        background: linear-gradient(135deg, var(--accent), #059669);
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 32px;
        box-shadow: 0 8px 32px var(--accent-glow);
        animation: pulse 3s ease-in-out infinite;
    }}
    
    .header-text {{
        display: flex;
        flex-direction: column;
        gap: 4px;
    }}
    
    .header-title {{
        font-size: 28px;
        font-weight: 800;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, var(--accent), #059669);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    .header-subtitle {{
        font-size: 14px;
        color: var(--text-secondary);
        font-weight: 500;
        margin: 0;
    }}
    
    .status-badge {{
        display: flex;
        align-items: center;
        gap: 8px;
        background: var(--accent);
        color: white;
        padding: 10px 20px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 13px;
        box-shadow: 0 4px 20px var(--accent-glow);
        animation: slideIn 0.8s ease-out;
    }}
    
    .status-dot {{
        width: 8px;
        height: 8px;
        background: #fff;
        border-radius: 50%;
        animation: pulse 2s ease-in-out infinite;
    }}
    
    /* Chat container moderne */
    .chat-container {{
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 20px;
    }}
    
    /* Messages avec design premium */
    [data-testid="stChatMessage"] {{
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        animation: fadeIn 0.4s ease-out;
        transition: all 0.3s ease;
    }}
    
    [data-testid="stChatMessage"]:hover {{
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }}
    
    /* Message assistant (gauche) */
    [data-testid="stChatMessage"]:has(img[alt*="assistant"]) {{
        background: linear-gradient(135deg, var(--glass-bg), var(--bg-card));
        border-left: 4px solid var(--accent);
        margin-right: 20%;
    }}
    
    /* Message utilisateur (droite) */
    [data-testid="stChatMessage"]:has(img[alt*="user"]) {{
        background: linear-gradient(135deg, var(--accent-glow), var(--glass-bg));
        border-right: 4px solid var(--accent);
        margin-left: 20%;
    }}
    
    /* Avatars premium avec glow */
    [data-testid="stChatMessage"] img {{
        border-radius: 50%;
        border: 3px solid var(--accent);
        box-shadow: 0 0 20px var(--accent-glow),
                    0 4px 12px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }}
    
    [data-testid="stChatMessage"] img:hover {{
        transform: scale(1.1);
        box-shadow: 0 0 30px var(--accent-glow);
    }}
    
    /* Texte des messages */
    .stChatMessage .stMarkdown p {{
        color: var(--text-primary);
        line-height: 1.7;
        font-size: 15px;
        font-weight: 400;
    }}
    
    /* Sidebar premium */
    [data-testid="stSidebar"] {{
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--border);
    }}
    
    [data-testid="stSidebar"] .stMarkdown {{
        color: var(--text-primary);
    }}
    
    /* Inputs modernes */
    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox select,
    .stNumberInput input {{
        background: var(--bg-card) !important;
        border: 2px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        padding: 12px 16px !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
    }}
    
    .stTextInput input:focus,
    .stTextArea textarea:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 4px var(--accent-glow) !important;
        outline: none !important;
    }}
    
    /* Boutons premium */
    .stButton>button {{
        background: linear-gradient(135deg, var(--accent), #059669) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px var(--accent-glow) !important;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 30px var(--accent-glow) !important;
    }}
    
    /* Sliders modernes */
    .stSlider {{
        padding: 16px 0;
    }}
    
    .stSlider [data-baseweb="slider"] {{
        background: var(--border);
    }}
    
    .stSlider [role="slider"] {{
        background: var(--accent) !important;
        box-shadow: 0 0 20px var(--accent-glow);
    }}
    
    /* Toggle switch */
    .stCheckbox {{
        padding: 8px 0;
    }}
    
    /* Métriques premium */
    .metric-card {{
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
    }}
    
    .metric-value {{
        font-size: 32px;
        font-weight: 800;
        color: var(--accent);
        margin: 8px 0;
    }}
    
    .metric-label {{
        font-size: 13px;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    /* Chat input premium */
    .stChatInput {{
        position: sticky;
        bottom: 0;
        background: var(--bg-primary);
        padding: 20px 0;
        z-index: 100;
    }}
    
    .stChatInput input {{
        background: var(--bg-card) !important;
        border: 2px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 16px 24px !important;
        font-size: 15px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05) !important;
    }}
    
    .stChatInput input:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 4px var(--accent-glow),
                    0 4px 20px rgba(0, 0, 0, 0.1) !important;
    }}
    
    /* Caption moderne */
    .stCaption {{
        color: var(--text-secondary) !important;
        font-size: 12px !important;
        font-weight: 500 !important;
    }}
    
    /* Links */
    a {{
        color: var(--accent) !important;
        text-decoration: none !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }}
    
    a:hover {{
        text-decoration: underline !important;
        text-underline-offset: 4px !important;
    }}
    
    /* Code blocks premium */
    code, pre {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 2px 8px !important;
        font-family: 'Monaco', 'Courier New', monospace !important;
        font-size: 13px !important;
    }}
    
    /* Expander premium */
    .streamlit-expanderHeader {{
        background: var(--glass-bg) !important;
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
    }}
    
    /* Scrollbar personnalisée */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: var(--bg-primary);
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: var(--accent);
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: #059669;
    }}
    
    /* Divider stylé */
    hr {{
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border), transparent);
        margin: 24px 0;
    }}
    
    /* Radio buttons modernes */
    .stRadio > label {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 12px;
        transition: all 0.3s ease;
    }}
    
    .stRadio > label:hover {{
        border-color: var(--accent);
        box-shadow: 0 0 20px var(--accent-glow);
    }}
    
    /* File uploader premium */
    [data-testid="stFileUploader"] {{
        background: var(--glass-bg);
        border: 2px dashed var(--border);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s ease;
    }}
    
    [data-testid="stFileUploader"]:hover {{
        border-color: var(--accent);
        background: var(--accent-glow);
    }}
    
    /* Success/Error messages */
    .stSuccess {{
        background: rgba(16, 185, 129, 0.1) !important;
        border-left: 4px solid var(--accent) !important;
        border-radius: 12px !important;
    }}
    
    .stError {{
        background: rgba(239, 68, 68, 0.1) !important;
        border-left: 4px solid #ef4444 !important;
        border-radius: 12px !important;
    }}
    
    /* Loading spinner */
    .stSpinner > div {{
        border-color: var(--accent) transparent transparent transparent !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# ==============================
# SECRETS & CONSTANTES
# ==============================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ **GROQ_API_KEY manquant.** Configurez-le dans Streamlit > Settings > Secrets.")
    st.stop()

client = groq.Client(api_key=GROQ_API_KEY)

# Chemins possibles pour l'index FAISS
INDEX_PATHS = [
    "faiss-compliance-banking-multilingual-index",  # Dossier complet
    ".",  # Racine (si fichiers index.faiss et index.pkl sont à la racine)
    os.path.dirname(__file__)  # Dossier du script
]

# Trouver le bon chemin de l'index
INDEX_PATH = None
for path in INDEX_PATHS:
    faiss_file = os.path.join(path, "index.faiss") if path != "." else "index.faiss"
    if os.path.exists(faiss_file):
        INDEX_PATH = path
        break

# Si aucun index trouvé, utiliser le chemin par défaut
if INDEX_PATH is None:
    INDEX_PATH = os.getenv("INDEX_PATH", ".")
DRIVE_LINK = os.getenv("DRIVE_LINK", "https://drive.google.com/drive/folders/TON_LIEN_FIXE_ICI")
LOGO_URL = os.getenv("LOGO_URL", "")
DEFAULT_MODEL = "llama-3.1-8b-instant"
MAX_SOURCES = 5

# Modèles disponibles
AVAILABLE_MODELS = {
    "llama-3.1-8b-instant": {"name": "Llama 3.1 8B (Rapide)", "speed": "⚡ Ultra-rapide", "quality": "🎯 Précis"},
    "llama-3.3-70b-versatile": {"name": "Llama 3.3 70B (Puissant)", "speed": "🚀 Rapide", "quality": "💎 Excellent"},
    "mixtral-8x7b-32768": {"name": "Mixtral 8x7B", "speed": "⚡ Très rapide", "quality": "🎯 Très bon"},
}

# ==============================
# PROMPTS SYSTÈME
# ==============================
SYSTEM_PROMPT_FR = """Tu es Lexi, assistant IA premium du département Conformité de BANK OF AFRICA (BOA).

**Ton rôle:**
- Répondre en français de manière professionnelle, claire et précise
- Utiliser le contexte documentaire fourni comme source principale
- Structurer les réponses avec des puces et des paragraphes courts
- Être concis mais complet

**Instructions:**
- Si des extraits documentaires sont fournis: base-toi STRICTEMENT dessus
- Si aucun contexte n'est disponible: donne une réponse générale de bonnes pratiques, en mentionnant qu'une vérification dans les documents officiels est recommandée
- Pour les salutations: réponds chaleureusement et naturellement
- N'invente jamais de détails non justifiés par le contexte
- Utilise des émojis avec parcimonie pour rendre la réponse plus engageante"""

SYSTEM_PROMPT_EN = """You are Lexi, premium AI assistant for BANK OF AFRICA's Compliance department.

**Your role:**
- Answer in English professionally, clearly and precisely
- Use provided documentary context as primary source
- Structure responses with bullets and short paragraphs
- Be concise yet comprehensive

**Instructions:**
- If context excerpts are provided: rely STRICTLY on them
- If no context is available: give a general best-practice answer, mentioning verification with official documents is recommended
- For greetings: respond warmly and naturally
- Never fabricate details unsupported by context
- Use emojis sparingly to make responses more engaging"""

# ==============================
# FONCTIONS UTILITAIRES
# ==============================
def detect_lang_simple(text: str) -> str:
    """Détection simple de la langue"""
    if not text:
        return "fr"
    t = text.lower()
    fr_hits = sum(w in t for w in [
        "bonjour","salut","svp","pièce","identité","contrôles","obligatoires",
        "compte","kyc","conformité","client","pourquoi","comment","quels","quelle",
        "lcb-ft","blanchiment","sanctions","rgpd","procédure","audit","merci"
    ])
    en_hits = sum(w in t for w in [
        "hello","hi","please","identity","controls","mandatory","account","kyc",
        "compliance","customer","why","how","what","which","aml","sanctions",
        "gdpr","procedure","audit","thank","thanks"
    ])
    return "en" if en_hits > fr_hits else "fr"

def classify_smalltalk(text: str, lang: str) -> Optional[str]:
    """Classification des small talk"""
    t = text.strip().lower()
    if lang == "fr":
        if re.search(r"\b(bonjour|salut|coucou|bonsoir|hey)\b", t): return "greet"
        if re.search(r"\b(merci|merci beaucoup|je te remercie)\b", t): return "thanks"
        if re.search(r"\b(ça va|ca va|comment ça va)\b", t): return "howare"
        if re.search(r"\b(au revoir|à bientôt|bye)\b", t): return "bye"
        if re.search(r"^\s*(ok|d'accord|parfait|super|cool)\s*$", t): return "ack"
    else:
        if re.search(r"\b(hi|hello|hey|howdy)\b", t): return "greet"
        if re.search(r"\b(thank you|thanks|thx)\b", t): return "thanks"
        if re.search(r"\b(how are you|how's it going)\b", t): return "howare"
        if re.search(r"\b(bye|goodbye|see you)\b", t): return "bye"
        if re.search(r"^\s*(ok|okay|sounds good|great)\s*$", t): return "ack"
    return None

def smalltalk_reply(kind: str, lang: str) -> str:
    """Réponses aux small talk"""
    if lang == "fr":
        return {
            "greet": "👋 Bonjour ! Je suis **Lexi**, votre assistant conformité. Comment puis-je vous aider aujourd'hui ?",
            "thanks": "✨ Avec grand plaisir ! N'hésitez pas si vous avez d'autres questions.",
            "howare": "🌟 Je vais très bien, merci ! Prêt à vous assister. Quel sujet conformité vous intéresse ?",
            "bye": "👋 Au revoir ! À très bientôt pour vos questions conformité.",
            "ack": "👍 Parfait ! Je reste à votre disposition."
        }.get(kind, "👋 Bonjour ! Comment puis-je vous aider ?")
    else:
        return {
            "greet": "👋 Hello! I'm **Lexi**, your compliance assistant. How can I help you today?",
            "thanks": "✨ You're very welcome! Feel free to ask more questions.",
            "howare": "🌟 I'm doing great, thanks! Ready to assist. What compliance topic interests you?",
            "bye": "👋 Goodbye! See you soon for your compliance questions.",
            "ack": "👍 Perfect! I'm here if you need anything."
        }.get(kind, "👋 Hi there! How can I help?")

# ==============================
# EMBEDDINGS / RETRIEVER
# ==============================
@st.cache_resource(show_spinner="🔧 Chargement des embeddings...")
def load_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={'device': device, 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource(show_spinner="📂 Chargement de l'index FAISS...")
def load_retriever(index_path: str):
    """Charge le retriever FAISS depuis le chemin spécifié"""
    try:
        # Vérifier les fichiers requis
        if index_path == ".":
            faiss_file = "index.faiss"
            pkl_file = "index.pkl"
        else:
            faiss_file = os.path.join(index_path, "index.faiss")
            pkl_file = os.path.join(index_path, "index.pkl")
        
        if not os.path.exists(faiss_file):
            st.warning(f"⚠️ Fichier {faiss_file} introuvable")
            return None, 0
        
        if not os.path.exists(pkl_file):
            st.warning(f"⚠️ Fichier {pkl_file} introuvable")
            return None, 0
        
        # Charger l'index
        vs = FAISS.load_local(
            index_path,
            embeddings=load_embeddings(),
            allow_dangerous_deserialization=True
        )
        
        # Créer le retriever
        retr = vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 50, "lambda_mult": 0.3}
        )
        
        ntotal = vs.index.ntotal
        st.success(f"✅ Index chargé : {ntotal} vecteurs trouvés")
        
        return retr, ntotal
        
    except Exception as e:
        st.error(f"❌ Erreur de chargement : {str(e)}")
        return None, 0

def e5_query(text: str) -> str:
    t = text.strip()
    return t if t.lower().startswith("query:") else f"query: {t}"

def build_context(docs, max_chars=4000) -> Tuple[str, List[str]]:
    if not docs:
        return "", []
    seen, parts, sources = set(), [], []
    for d in docs:
        meta = d.metadata or {}
        sf = meta.get("source_file", meta.get("source", "inconnu"))
        pg = meta.get("page_index")
        key = (sf, pg if pg is not None else -1)
        if key in seen: continue
        seen.add(key)
        snippet = (d.page_content or "").strip()
        if not snippet: continue
        parts.append(snippet)
        label = f"{sf}" + (f" (page {pg+1})" if pg is not None else "")
        sources.append(label)
        if sum(len(p) for p in parts) > max_chars: break
    if len(sources) > MAX_SOURCES: sources = sources[:MAX_SOURCES]
    return "\n---\n".join(parts), sources

def ask_groq(system_prompt: str, user_question: str, context_text: str,
             lang: str, used_context: bool, temperature: float, max_tokens: int, model: str) -> str:
    lang_instruction = "Réponds en français." if lang == "fr" else "Answer in English."
    concision = ("- Réponds de façon concise et structurée.\n"
                 if lang == "fr" else
                 "- Answer concisely with clear structure.\n")
    if used_context:
        instruction = "- Base your answer ONLY on the context above.\n- If insufficient, say so explicitly.\n"
    else:
        instruction = "- No context found. Provide general best-practice compliance answer.\n- Be conservative and helpful.\n"

    user_block = (
        f"{lang_instruction}\n\n"
        "Context excerpts:\n"
        f"{context_text if used_context else '(no context found)'}\n\n"
        f"Question: {user_question}\n\n"
        "Instructions:\n"
        f"{instruction}{concision}"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_block},
        ],
        temperature=float(max(0.0, min(1.0, temperature))),
        max_tokens=int(max_tokens),
    )
    return resp.choices[0].message.content.strip()

def add_footer(answer: str, sources: List[str], lang: str, used_context: bool, mode_rag: bool) -> str:
    if used_context:
        if sources:
            answer += ("\n\n📚 **Sources :**\n" if lang == "fr" else "\n\n📚 **Sources:**\n") + "\n".join(f"- {s}" for s in sources)
            if DRIVE_LINK:
                answer += (f"\n\n📂 **Accéder aux documents complets :**\n[➡️ Ouvrir le dossier]({DRIVE_LINK})"
                           if lang == "fr" else
                           f"\n\n📂 **Access full documents:**\n[➡️ Open folder]({DRIVE_LINK})")
    else:
        if mode_rag:
            disclaimer = ("\n\n⚠️ **Aucun document pertinent trouvé** dans la base indexée."
                          if lang == "fr" else
                          "\n\n⚠️ **No relevant document found** in indexed knowledge base.")
        else:
            disclaimer = ("\n\nℹ️ **Mode LLM seul** : Réponse générale sans base documentaire."
                          if lang == "fr" else
                          "\n\nℹ️ **LLM-only mode**: General answer without knowledge base.")
        answer += disclaimer
        if DRIVE_LINK:
            answer += (f"\n\n📂 [Consulter la documentation]({DRIVE_LINK})"
                       if lang == "fr" else
                       f"\n\n📂 [Browse documentation]({DRIVE_LINK})")
    return answer

# ==============================
# STATE MANAGEMENT
# ==============================
def init_state():
    if "dark_mode" not in st.session_state: 
        st.session_state.dark_mode = False
    if "convos" not in st.session_state: 
        st.session_state.convos = {}
    if "active_id" not in st.session_state:
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        st.session_state.active_id = f"conv-{ts}"
        st.session_state.convos[st.session_state.active_id] = [
            {"role": "assistant", "content": "👋 Bonjour ! Je suis **Lexi**, votre assistant conformité IA. Posez-moi vos questions en français ou en anglais !"}
        ]
    if "conv_titles" not in st.session_state:
        st.session_state.conv_titles = {st.session_state.active_id: "✨ Nouvelle discussion"}
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = DEFAULT_MODEL
    if "total_messages" not in st.session_state:
        st.session_state.total_messages = 0
    if "session_start" not in st.session_state:
        st.session_state.session_start = datetime.utcnow()

init_state()
inject_premium_css(st.session_state.dark_mode)

# ==============================
# HEADER PREMIUM
# ==============================
st.markdown(f"""
<div class="premium-header">
    <div class="header-content">
        <div class="header-left">
            <div class="logo-container">
                🛡️
            </div>
            <div class="header-text">
                <h1 class="header-title">Lexi AI</h1>
                <p class="header-subtitle">Assistant Conformité BOA Group • Propulsé par Groq & RAG</p>
            </div>
        </div>
        <div class="status-badge">
            <div class="status-dot"></div>
            En ligne
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================
# SIDEBAR PREMIUM
# ==============================
with st.sidebar:
    st.markdown("### ⚙️ Paramètres")
    
    # Mode sombre
    st.toggle(
        "🌙 Mode sombre",
        key="dark_mode",
        value=st.session_state.dark_mode,
        on_change=lambda: inject_premium_css(st.session_state.dark_mode),
        help="Activer le thème sombre pour réduire la fatigue visuelle"
    )
    
    st.markdown("---")
    
    # Mode RAG
    st.markdown("#### 🎯 Mode de réponse")
    mode_label = st.radio(
        "Choisir la source de connaissance",
        ["🔍 RAG (Base documentaire)", "🧠 LLM seul"],
        index=0,
        help="RAG utilise votre base documentaire indexée. LLM seul répond avec ses connaissances générales."
    )
    MODE_RAG = (mode_label == "🔍 RAG (Base documentaire)")
    
    st.markdown("---")
    
    # Sélection du modèle
    st.markdown("#### 🤖 Modèle IA")
    model_options = list(AVAILABLE_MODELS.keys())
    model_labels = [f"{AVAILABLE_MODELS[m]['name']}" for m in model_options]
    
    selected_model_idx = model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
    
    selected_model_label = st.selectbox(
        "Choisir le modèle",
        options=model_labels,
        index=selected_model_idx,
        help="Chaque modèle offre un équilibre différent entre vitesse et qualité"
    )
    
    # Retrouver le model key
    st.session_state.selected_model = model_options[model_labels.index(selected_model_label)]
    
    # Afficher les specs du modèle
    model_info = AVAILABLE_MODELS[st.session_state.selected_model]
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"{model_info['speed']}")
    with col2:
        st.caption(f"{model_info['quality']}")
    
    st.markdown("---")
    
    # Paramètres avancés
    with st.expander("🔧 Paramètres avancés", expanded=False):
        temperature = st.slider(
            "🎯 Créativité (Température)",
            0.0, 1.0, 0.2, 0.05,
            help="Plus bas = réponses plus précises et déterministes. Plus haut = plus créatif mais moins prévisible."
        )
        
        max_tokens = st.slider(
            "📏 Longueur maximale (tokens)",
            256, 2048, 800, 64,
            help="Nombre maximum de tokens dans la réponse (≈ 750 mots pour 1000 tokens)"
        )
        
        st.markdown("**Paramètres RAG**" if MODE_RAG else "**Paramètres désactivés** (LLM seul)")
        
        k_docs = st.slider(
            "📚 Nombre de documents",
            3, 10, 6, 1,
            help="Nombre de documents à récupérer de la base vectorielle",
            disabled=not MODE_RAG
        )
        
        lambda_mult = st.slider(
            "🎭 Diversité (MMR λ)",
            0.0, 1.0, 0.3, 0.1,
            help="0 = maximum de diversité, 1 = maximum de pertinence",
            disabled=not MODE_RAG
        )
    
    st.markdown("---")
    
    # Statistiques de session
    st.markdown("#### 📊 Statistiques")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.total_messages}</div>
            <div class="metric-label">Messages</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Gestion sécurisée du chargement de l'index
        try:
            if MODE_RAG and INDEX_PATH:
                retriever, ntotal = load_retriever(INDEX_PATH)
                if retriever is None:
                    ntotal = 0
            else:
                ntotal = 0
        except Exception as e:
            ntotal = 0
            st.caption(f"⚠️ Erreur index")
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{ntotal}</div>
            <div class="metric-label">Vecteurs</div>
        </div>
        """, unsafe_allow_html=True)
    
    session_duration = (datetime.utcnow() - st.session_state.session_start).seconds // 60
    st.caption(f"⏱️ Session : {session_duration} min • Modèle : {st.session_state.selected_model.split('-')[0]}")
    
    st.markdown("---")
    
    # Gestion des conversations
    st.markdown("#### 💬 Conversations")
    
    conv_ids = list(st.session_state.convos.keys())
    current_idx = conv_ids.index(st.session_state.active_id) if st.session_state.active_id in conv_ids else 0
    
    selected = st.selectbox(
        "Sélectionner une conversation",
        options=conv_ids,
        index=current_idx,
        format_func=lambda cid: st.session_state.conv_titles.get(cid, cid),
        label_visibility="collapsed"
    )
    
    if selected != st.session_state.active_id:
        st.session_state.active_id = selected
    
    # Actions sur les conversations
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🆕 Nouveau", use_container_width=True):
            ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            new_id = f"conv-{ts}"
            st.session_state.convos[new_id] = [
                {"role": "assistant", "content": "👋 Nouvelle conversation ! Comment puis-je vous aider ?"}
            ]
            st.session_state.conv_titles[new_id] = "✨ Nouvelle discussion"
            st.session_state.active_id = new_id
            st.rerun()
    
    with col2:
        if st.button("🗑️ Supprimer", use_container_width=True):
            if len(st.session_state.convos) > 1:
                st.session_state.convos.pop(st.session_state.active_id, None)
                st.session_state.conv_titles.pop(st.session_state.active_id, None)
                st.session_state.active_id = list(st.session_state.convos.keys())[0]
                st.rerun()
            else:
                st.warning("⚠️ Impossible de supprimer la dernière conversation")
    
    with col3:
        if st.button("🧹 Vider", use_container_width=True):
            st.session_state.convos[st.session_state.active_id] = [
                {"role": "assistant", "content": "🔄 Conversation vidée."}
            ]
            st.rerun()
    
    # Renommer
    new_name = st.text_input(
        "Renommer",
        st.session_state.conv_titles.get(st.session_state.active_id, "Nouvelle discussion"),
        label_visibility="collapsed",
        placeholder="Nom de la conversation..."
    )
    if st.button("💾 Renommer", use_container_width=True):
        st.session_state.conv_titles[st.session_state.active_id] = new_name or st.session_state.active_id
        st.success("✅ Conversation renommée")
    
    st.markdown("---")
    
    # Export / Import
    st.markdown("#### 💾 Sauvegarde")
    
    if st.button("⬇️ Exporter (JSON)", use_container_width=True):
        payload = {
            "titles": st.session_state.conv_titles,
            "convos": st.session_state.convos,
            "exported_at": datetime.utcnow().isoformat(),
            "model": st.session_state.selected_model
        }
        b64 = base64.b64encode(json.dumps(payload, ensure_ascii=False, indent=2).encode()).decode()
        st.markdown(
            f'<a href="data:application/json;base64,{b64}" download="lexi_conversations.json" style="display:block; text-align:center; padding:12px; background:var(--accent); color:white; border-radius:8px; text-decoration:none;">📥 Télécharger conversations.json</a>',
            unsafe_allow_html=True
        )
    
    uploaded = st.file_uploader("⬆️ Importer JSON", type=["json"], label_visibility="collapsed")
    if uploaded:
        try:
            data = json.load(uploaded)
            if isinstance(data.get("convos"), dict):
                st.session_state.convos.update(data["convos"])
                if isinstance(data.get("titles"), dict):
                    st.session_state.conv_titles.update(data["titles"])
                st.success("✅ Historique importé avec succès")
        except Exception as e:
            st.error(f"❌ Erreur d'import : {e}")
    
    st.markdown("---")
    
    # Liens utiles
    st.markdown("#### 🔗 Liens rapides")
    if DRIVE_LINK:
        st.link_button("📂 Documents sources", url=DRIVE_LINK, use_container_width=True)
    st.link_button("🤖 Groq API", url="https://console.groq.com", use_container_width=True)
    st.link_button("📚 Documentation BOA", url="https://www.boa.africa", use_container_width=True)
    
    st.markdown("---")
    
    # Diagnostic de l'index
    with st.expander("🔍 Diagnostic système", expanded=False):
        st.caption("**Chemin de l'index :**")
        st.code(INDEX_PATH if INDEX_PATH else "Non défini")
        
        st.caption("**Fichiers FAISS détectés :**")
        files_found = []
        if INDEX_PATH:
            faiss_file = "index.faiss" if INDEX_PATH == "." else os.path.join(INDEX_PATH, "index.faiss")
            pkl_file = "index.pkl" if INDEX_PATH == "." else os.path.join(INDEX_PATH, "index.pkl")
            
            if os.path.exists(faiss_file):
                files_found.append(f"✅ {faiss_file}")
            else:
                files_found.append(f"❌ {faiss_file}")
            
            if os.path.exists(pkl_file):
                files_found.append(f"✅ {pkl_file}")
            else:
                files_found.append(f"❌ {pkl_file}")
        
        for f in files_found:
            st.caption(f)
        
        st.caption(f"**Répertoire de travail :** `{os.getcwd()}`")
    
    st.markdown("---")
    st.caption("🛡️ **Lexi AI** v2.0 Premium\nPowered by Groq • FAISS • Streamlit")

# ==============================
# CHAT INTERFACE
# ==============================
assistant_avatar = "🛡️"
user_avatar = "👤"

active_msgs = st.session_state.convos[st.session_state.active_id]

for m in active_msgs:
    if m["role"] == "assistant":
        with st.chat_message("assistant", avatar=assistant_avatar):
            st.markdown(m["content"])
    else:
        with st.chat_message("user", avatar=user_avatar):
            st.markdown(m["content"])

# ==============================
# CHAT INPUT & PROCESSING
# ==============================
prompt = st.chat_input("💬 Posez votre question sur la conformité...")

if prompt:
    # Incrémenter compteur
    st.session_state.total_messages += 1
    
    # Ajouter message utilisateur
    st.session_state.convos[st.session_state.active_id].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)
    
    # Détection langue et small talk
    lang = detect_lang_simple(prompt)
    kind = classify_smalltalk(prompt, lang)
    
    if kind is not None and len(prompt.strip()) <= 120:
        reply = smalltalk_reply(kind, lang)
        with st.chat_message("assistant", avatar=assistant_avatar):
            st.markdown(reply)
        st.session_state.convos[st.session_state.active_id].append({"role": "assistant", "content": reply})
        st.stop()
    
    # RAG Processing
    system_prompt = SYSTEM_PROMPT_FR if lang == "fr" else SYSTEM_PROMPT_EN
    used_context = False
    context_text, sources = "", []
    start_retr = time.time()
    
    if MODE_RAG:
        try:
            # Charger le retriever seulement si nécessaire
            if INDEX_PATH:
                retriever, ntotal = load_retriever(INDEX_PATH)
                if retriever is not None:
                    # Mettre à jour les paramètres de recherche
                    retriever.search_kwargs = {"k": k_docs, "fetch_k": k_docs * 8, "lambda_mult": lambda_mult}
                    q = e5_query(prompt)
                    docs = retriever.invoke(q)
                    context_text, sources = build_context(docs, max_chars=5000)
                    used_context = bool(context_text.strip())
                else:
                    st.warning("⚠️ Retriever non disponible. Mode LLM seul activé.")
            else:
                st.warning("⚠️ INDEX_PATH non défini. Mode LLM seul activé.")
        except Exception as e:
            st.error(f"⚠️ Erreur RAG : {str(e)}\nMode LLM seul activé.")
    
    retr_ms = (time.time() - start_retr) * 1000 if MODE_RAG else 0
    
    # LLM Call
    try:
        start_llm = time.time()
        answer = ask_groq(
            system_prompt,
            prompt,
            context_text,
            lang,
            used_context,
            temperature=temperature,
            max_tokens=max_tokens,
            model=st.session_state.selected_model
        )
        answer = add_footer(answer, sources, lang, used_context, MODE_RAG)
        gen_ms = (time.time() - start_llm) * 1000
        
        with st.chat_message("assistant", avatar=assistant_avatar):
            st.markdown(answer)
        
        st.session_state.convos[st.session_state.active_id].append({"role": "assistant", "content": answer})
        
        # Performance caption
        perf_parts = []
        if MODE_RAG:
            perf_parts.append(f"🔎 Retrieval: {retr_ms:.0f}ms")
        perf_parts.append(f"🧠 Génération: {gen_ms:.0f}ms")
        perf_parts.append(f"🤖 {AVAILABLE_MODELS[st.session_state.selected_model]['name']}")
        perf_parts.append(f"🌡️ T={temperature}")
        
        st.caption(" • ".join(perf_parts))
        
    except Exception as e:
        with st.chat_message("assistant", avatar=assistant_avatar):
            st.error(f"❌ **Erreur API Groq:**\n\n{str(e)}\n\nVeuillez réessayer ou changer de modèle.")
        st.session_state.convos[st.session_state.active_id].append({
            "role": "assistant",
            "content": f"❌ Erreur: {str(e)}"
        })
