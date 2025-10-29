# app.py
# -*- coding: utf-8 -*-
import os
import re
import json
import time
import base64
from io import StringIO
from datetime import datetime
from typing import List, Tuple, Optional

import streamlit as st
import torch
from langchain_community.vectorstores import FAISS
# Si dispo et pour éviter les warnings:
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import groq

# ==============================
# PAGE & THEME
# ==============================
st.set_page_config(page_title="Chatbot Conformité — BOA Group", page_icon="✅", layout="wide")

def inject_css(dark: bool):
    # ----- Couleurs / variables (vert sombre + accents demandés)
    accent = "#0f5c2d"                     # vert profond
    accentShadow = "rgba(15, 92, 45, 0.30)"  # intensité d’ombre ~3
    accent25 = "rgba(15, 92, 45, 0.25)"      # voile 25%

    if dark:
        bg = "#0f172a"
        panel = "#111827"
        card = "#1f2937"
        text = "#e5e7eb"
        sub = "#9ca3af"
        border = "#374151"
    else:
        bg = "#ffffff"       # fond blanc demandé
        panel = "#f8fafc"
        card = "#ffffff"
        text = "#0f172a"
        sub = "#475569"
        border = "#e5e7eb"

    st.markdown(f"""
    <style>
      :root {{
        --boa-accent: {accent};
        --boa-accent-25: {accent25};
        --boa-accent-shadow: {accentShadow};
        --boa-text: {text};
        --boa-sub: {sub};
        --boa-border: {border};
        --boa-card: {card};
        --boa-panel: {panel};
        --boa-bg: {bg};
      }}

      .stApp {{
        background: linear-gradient(180deg, var(--boa-panel) 0%, var(--boa-bg) 100%) !important;
      }}

      /* Header */
      .boa-header {{
        border: 1px solid var(--boa-border);
        background: var(--boa-card);
        border-radius: 16px;
        padding: 14px 18px;
        display: flex; align-items: center; gap: 14px;
        box-shadow: 0 10px 30px var(--boa-accent-shadow);
      }}
      .boa-title {{ font-weight: 700; letter-spacing: .3px; color: var(--boa-text); font-size: 20px; margin: 0; }}
      .boa-sub {{ font-size: 13px; color: var(--boa-sub); margin: 0; }}
      .boa-badge {{
        color: white; background: var(--boa-accent);
        padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 700;
        box-shadow: 0 4px 14px var(--boa-accent-shadow);
      }}

      /* Chat bubbles */
      .stChatMessage[data-testid="stChatMessage"] {{
        background: var(--boa-card);
        border: 1px solid var(--boa-border);
        border-radius: 18px;
        padding: 14px 16px;
        box-shadow: 0 6px 22px rgba(0,0,0,{0.22 if dark else 0.06});
      }}
      .stChatMessage .stMarkdown p {{ color: var(--boa-text); line-height: 1.55; }}
      .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{ color: var(--boa-text); }}

      /* Inputs & contrôles – accents 25% */
      .stTextInput input, .stTextArea textarea, .stSelectbox, .stSlider, .stNumberInput input {{
        border-radius: 10px !important;
      }}
      .stTextInput input:focus, .stTextArea textarea:focus {{
        outline: 2px solid var(--boa-accent-25) !important;
        box-shadow: 0 0 0 4px var(--boa-accent-25) !important;
        border-color: var(--boa-accent) !important;
      }}
      .stButton>button {{
        border-radius: 10px !important;
        border: 1px solid var(--boa-border) !important;
      }}
      .stButton>button:hover {{
        box-shadow: 0 6px 18px var(--boa-accent-25) !important;
        border-color: var(--boa-accent) !important;
      }}

      /* Liens & code */
      a, .stMarkdown a {{ color: var(--boa-accent) !important; text-decoration: none; }}
      a:hover, .stMarkdown a:hover {{ text-decoration: underline; text-underline-offset: 3px; }}
      code, pre {{
        font-size: 12.5px !important;
        background: {"#0b1220" if dark else "#f5f7fb"} !important;
        border: 1px solid var(--boa-border) !important;
        border-radius: 10px !important;
      }}
    </style>
    """, unsafe_allow_html=True)

# ==============================
# SECRETS & CONSTANTES
# ==============================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY manquant. Ajoute-le dans Streamlit > Settings > Secrets.")
    st.stop()

client = groq.Client(api_key=GROQ_API_KEY)

INDEX_PATH = os.getenv("INDEX_PATH", "faiss-compliance-banking-multilingual-index")
DRIVE_LINK = os.getenv("DRIVE_LINK", "https://drive.google.com/drive/folders/TON_LIEN_FIXE_ICI")
LOGO_URL = os.getenv("LOGO_URL", "")
MAX_SOURCES = 5
DEFAULT_MODEL = "llama-3.1-8b-instant"

# ==============================
# PROMPTS SYSTÈME
# ==============================
SYSTEM_PROMPT_FR = """Tu es Lexi, assistant IA du département Conformité de BANK OF AFRICA (BOA).
Réponds en français, poliment, clairement et factuellement.
Si la question est une salutation (ex. "Bonjour", "Merci"), réponds de manière chaleureuse et naturelle.
Si des extraits documentaires sont fournis, base-toi STRICTEMENT dessus.
Si aucun extrait n’est disponible, donne une réponse générale de bonnes pratiques conformité de manière prudente et générique, en précisant qu'il est préférable de vérifier dans les documents officiels pour confirmation.
Structure en puces si utile. N’invente pas de détails non justifiés par le contexte."""
SYSTEM_PROMPT_EN = """You are Lexi, an AI assistant for BANK OF AFRICA's Compliance department.
Answer in English, politely, clearly, and factually.
If the user input is a greeting, respond warmly and naturally.
If context excerpts are provided, rely STRICTLY on them.
If no context is provided, give a general best-practice compliance answer, conservatively and generically, and mention it’s recommended to verify with official documents.
Use bullets if helpful. Do not fabricate details unsupported by the context."""

# ==============================
# SMALL TALK
# ==============================
def detect_lang_simple(text: str) -> str:
    if not text:
        return "fr"
    t = text.lower()
    fr_hits = sum(w in t for w in [
        "bonjour","salut","svp","pièce d'identité","contrôles","obligatoires","nouveau",
        "compte","kyc","conformité","client","pourquoi","comment","quels","quelle",
        "quelles","lcb-ft","blanchiment","sanctions","rgpd","procédure","audit","ça va","ca va"
    ])
    en_hits = sum(w in t for w in [
        "hello","hi","please","identity","controls","mandatory","new","account",
        "kyc","compliance","customer","why","how","what","which","aml","sanctions",
        "gdpr","procedure","audit","how are you","how's it going"
    ])
    return "en" if en_hits > fr_hits else "fr"

def classify_smalltalk(text: str, lang: str) -> Optional[str]:
    t = text.strip().lower()
    if lang == "fr":
        if re.search(r"\b(bonjour|salut|coucou|bonsoir|re\s?bonjour)\b", t): return "greet"
        if re.search(r"\b(merci|merci beaucoup|je te remercie|je vous remercie)\b", t): return "thanks"
        if re.search(r"\b(ça va|ca va|comment\s?(ça|ca)\s?va|comment allez[- ]vous|comment vas[- ]tu)\b", t): return "howare"
        if re.search(r"\b(au revoir|à bientôt|a bientôt|à plus|a plus|à la prochaine|bonne nuit)\b", t): return "bye"
        if re.search(r"^\s*(ok|d(’|')accord|parfait|super|top|cool|merci!?)\s*$", t): return "ack"
    else:
        if re.search(r"\b(hi|hello|hey|hiya|howdy)\b", t): return "greet"
        if re.search(r"\b(thank you|thanks|thx|much appreciated)\b", t): return "thanks"
        if re.search(r"\b(how are you|how's it going|how do you do)\b", t): return "howare"
        if re.search(r"\b(bye|goodbye|see you|see ya|take care|later)\b", t): return "bye"
        if re.search(r"^\s*(ok|okay|sounds good|great|awesome|cool|thanks!?)\s*$", t): return "ack"
    return None

def smalltalk_reply(kind: str, lang: str) -> str:
    if lang == "fr":
        return {
            "greet": "Bonjour 👋 Comment puis-je vous aider en conformité aujourd’hui ?",
            "thanks": "Avec plaisir ! N’hésitez pas si vous avez une autre question conformité.",
            "howare": "Ça va très bien, merci ! Et vous ? Un sujet conformité à explorer ?",
            "bye": "Au revoir ! Bonne journée et à bientôt.",
            "ack": "Parfait 👍 Dites-moi ce dont vous avez besoin."
        }.get(kind, "Bonjour ! Que puis-je faire pour vous ?")
    else:
        return {
            "greet": "Hello 👋 How can I help with compliance today?",
            "thanks": "You're welcome! Feel free to ask any other compliance questions.",
            "howare": "I'm doing great, thanks! And you? Any compliance topic to check?",
            "bye": "Goodbye! Have a great day and see you soon.",
            "ack": "Great 👍 Tell me what you need."
        }.get(kind, "Hi there! How can I help?")

# ==============================
# EMBEDDINGS / RETRIEVER
# ==============================
@st.cache_resource(show_spinner="🔧 Chargement des embeddings…")
def load_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={'device': device, 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource(show_spinner="📂 Chargement de l’index FAISS…")
def load_retriever(index_path: str):
    if os.path.exists(index_path):
        vs = FAISS.load_local(
            index_path,
            embeddings=load_embeddings(),
            allow_dangerous_deserialization=True
        )
        retr = vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 50, "lambda_mult": 0.3}
        )
        return retr, vs.index.ntotal
    return None, 0

def e5_query(text: str) -> str:
    t = text.strip()
    return t if t.lower().startswith("query:") else f"query: {t}"

def build_context(docs, max_chars=4000) -> Tuple[str, List[str]]:
    if not docs:
        return "", []
    seen = set()
    parts, sources = [], []
    for d in docs:
        meta = d.metadata or {}
        sf = meta.get("source_file", meta.get("source", "inconnu"))
        sd = meta.get("source_folder", meta.get("path", ""))
        pg = meta.get("page_index")
        key = (sf, pg if pg is not None else -1)
        if key in seen: continue
        seen.add(key)
        snippet = (d.page_content or "").strip()
        if not snippet: continue
        parts.append(snippet)
        label = f"{sf}" + (f" (page {pg+1})" if pg is not None else "")
        if sd: label += f" | dossier: {sd}"
        sources.append(label)
        if sum(len(p) for p in parts) > max_chars: break
    if len(sources) > MAX_SOURCES: sources = sources[:MAX_SOURCES]
    return "\n---\n".join(parts), sources

def ask_groq(system_prompt: str, user_question: str, context_text: str,
             lang: str, used_context: bool, temperature: float, max_tokens: int) -> str:
    lang_instruction = "Réponds en français." if lang == "fr" else "Answer in English."
    concision = ("- Réponds de façon concise et structurée (phrases courtes, puces si utile).\n"
                 if lang == "fr" else
                 "- Answer concisely with clear structure (short sentences, bullets if helpful).\n")
    if used_context:
        instruction = (
            "- Base your answer ONLY on the context above.\n"
            "- If the context is insufficient, say so explicitly.\n"
        )
    else:
        instruction = (
            "- No context was found. Provide a general, best-practice compliance answer.\n"
            "- Keep it conservative and avoid jurisdiction-specific claims.\n"
            "- Be polite and helpful, but do not fabricate facts.\n"
        )

    user_block = (
        f"{lang_instruction}\n\n"
        "Context excerpts:\n"
        f"{context_text if used_context else '(no context found)'}\n\n"
        f"Question: {user_question}\n\n"
        "Instructions:\n"
        f"{instruction}{concision}"
    )

    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
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
            answer += ("\n\nSources :\n" if lang == "fr" else "\n\nSources:\n") + "\n".join(f"- {s}" for s in sources)
            if DRIVE_LINK:
                answer += (
                    f"\n\n📂 Dossier source :\n➡️ {DRIVE_LINK} (Accéder aux documents)"
                    if lang == "fr" else
                    f"\n\n📂 Source folder:\n➡️ {DRIVE_LINK} (Access documents)"
                )
    else:
        # LLM seul OU pas de contexte trouvé
        if mode_rag:
            # Mode RAG mais aucun extrait utile
            disclaimer = (
                "\n\n⚠️ Aucun extrait pertinent n’a été trouvé dans la base indexée."
                if lang == "fr" else
                "\n\n⚠️ No relevant excerpt was found in the indexed knowledge base."
            )
        else:
            # Mode LLM seul explicitement activé
            disclaimer = (
                "\n\nℹ️ Mode **LLM seul** activé : réponse générale sans base documentaire."
                if lang == "fr" else
                "\n\nℹ️ **LLM-only** mode enabled: general answer without the knowledge base."
            )
        answer += disclaimer
        if DRIVE_LINK:
            answer += (
                f"\n\n📂 Documentation :\n➡️ {DRIVE_LINK} (Accéder aux documents)"
                if lang == "fr" else
                f"\n\n📂 Documentation:\n➡️ {DRIVE_LINK} (Access documents)"
            )
    return answer

# ==============================
# STATE: Conversations & UI
# ==============================
def init_state():
    if "dark_mode" not in st.session_state: st.session_state.dark_mode = False  # fond blanc par défaut
    if "convos" not in st.session_state: st.session_state.convos = {}
    if "active_id" not in st.session_state:
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        st.session_state.active_id = f"conv-{ts}"
        st.session_state.convos[st.session_state.active_id] = [
            {"role": "assistant", "content": "Bonjour 👋 Je suis **Lexi**. Posez-moi une question conformité (FR/EN)."}
        ]
    if "conv_titles" not in st.session_state:
        st.session_state.conv_titles = {st.session_state.active_id: "Nouvelle discussion"}

init_state()
inject_css(st.session_state.dark_mode)

# ==============================
# HEADER
# ==============================
colH1, colH2 = st.columns([1,6], vertical_alignment="center")
with colH1:
    if LOGO_URL:
        st.image(LOGO_URL, use_column_width=True)
with colH2:
    st.markdown(
        f"""
        <div class="boa-header">
          <div style="display:flex; flex-direction:column;">
            <p class="boa-title">Chatbot Conformité — BOA Group</p>
            <p class="boa-sub">RAG · FAISS · Multilingue (FR/EN) · Groq {DEFAULT_MODEL}</p>
          </div>
          <div style="flex:1"></div>
          <span class="boa-badge">Lexi</span>
        </div>
        """, unsafe_allow_html=True
    )
st.write("")

# ==============================
# SIDEBAR — Contrôles Pro
# ==============================
with st.sidebar:
    st.subheader("⚙️ Paramètres")
    st.toggle("🌙 Mode sombre", key="dark_mode", value=st.session_state.dark_mode, on_change=lambda: inject_css(st.session_state.dark_mode))
    # ---- Sélecteur de mode : RAG vs LLM seul
    mode_label = st.radio(
        "Mode de réponse",
        options=["RAG (Base documentaire)", "LLM seul"],
        index=0,
        help="Choisissez si le chatbot doit utiliser l’index vectorisé (RAG) ou répondre sans retrieval."
    )
    MODE_RAG = (mode_label == "RAG (Base documentaire)")
    st.write("---")
    temperature = st.slider("🎯 Température (précision)", 0.0, 0.7, 0.15, 0.01,
                            help="Plus bas = plus déterministe, plus concis.")
    max_tokens = st.slider("🧾 Longueur max (tokens)", 256, 1200, 700, 16)
    st.write("---")

    # Gestion historique
    st.subheader("🗂️ Conversations")
    conv_ids = list(st.session_state.convos.keys())
    current_idx = conv_ids.index(st.session_state.active_id) if st.session_state.active_id in conv_ids else 0
    selected = st.selectbox("Sélectionner", options=conv_ids, index=current_idx,
                            format_func=lambda cid: st.session_state.conv_titles.get(cid, cid))
    if selected != st.session_state.active_id:
        st.session_state.active_id = selected

    new_name = st.text_input("Renommer la conversation", st.session_state.conv_titles.get(st.session_state.active_id, "Nouvelle discussion"))
    if st.button("💾 Renommer"):
        st.session_state.conv_titles[st.session_state.active_id] = new_name or st.session_state.active_id

    cols = st.columns(3)
    with cols[0]:
        if st.button("🆕 Nouveau"):
            ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            new_id = f"conv-{ts}"
            st.session_state.convos[new_id] = [{"role": "assistant", "content": "Nouvelle discussion. Comment puis-je aider ?"}]
            st.session_state.conv_titles[new_id] = "Nouvelle discussion"
            st.session_state.active_id = new_id
    with cols[1]:
        if st.button("🧹 Effacer"):
            st.session_state.convos[st.session_state.active_id] = [{"role": "assistant", "content": "Conversation réinitialisée."}]
    with cols[2]:
        if st.button("🗑️ Supprimer"):
            if len(st.session_state.convos) > 1:
                st.session_state.convos.pop(st.session_state.active_id, None)
                st.session_state.conv_titles.pop(st.session_state.active_id, None)
                st.session_state.active_id = list(st.session_state.convos.keys())[0]

    # Export / Import JSON
    st.write("---")
    if st.button("⬇️ Exporter l’historique (JSON)"):
        payload = {
            "titles": st.session_state.conv_titles,
            "convos": st.session_state.convos,
            "exported_at": datetime.utcnow().isoformat()
        }
        b64 = base64.b64encode(json.dumps(payload, ensure_ascii=False, indent=2).encode()).decode()
        st.markdown(f"[Télécharger conversations](data:application/json;base64,{b64})", unsafe_allow_html=True)

    uploaded = st.file_uploader("⬆️ Importer un historique JSON", type=["json"])
    if uploaded:
        try:
            data = json.load(uploaded)
            if isinstance(data.get("convos"), dict):
                st.session_state.convos.update(data["convos"])
                if isinstance(data.get("titles"), dict):
                    st.session_state.conv_titles.update(data["titles"])
                st.success("Historique importé.")
        except Exception as e:
            st.error(f"Import impossible : {e}")

    st.write("---")
    # Infos index + mode courant
    retriever, ntotal = load_retriever(INDEX_PATH)
    if MODE_RAG:
        st.caption(f"📁 Mode: **RAG** • Index: `{INDEX_PATH}` • Vecteurs: **{ntotal}**" if ntotal else "📁 Mode: **RAG** • ⚠️ Index introuvable → fallback sans contexte")
    else:
        st.caption("🧠 Mode: **LLM seul** (pas de retrieval)")
    if DRIVE_LINK:
        st.link_button("📂 Dossier sources", url=DRIVE_LINK, use_container_width=True)

# ==============================
# CHAT — Affichage historique actif
# ==============================
active_msgs = st.session_state.convos[st.session_state.active_id]
for m in active_msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ==============================
# LOOP — Entrée utilisateur
# ==============================
prompt = st.chat_input("Écrivez votre question… (FR/EN)")
if prompt:
    # push user msg
    st.session_state.convos[st.session_state.active_id].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Langue + small talk
    lang = detect_lang_simple(prompt)
    kind = classify_smalltalk(prompt, lang)
    if kind is not None and len(prompt.strip()) <= 120:
        reply = smalltalk_reply(kind, lang)
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.convos[st.session_state.active_id].append({"role": "assistant", "content": reply})
        st.stop()

    # RAG conditionnel selon le mode
    system_prompt = SYSTEM_PROMPT_FR if lang == "fr" else SYSTEM_PROMPT_EN
    used_context = False
    context_text, sources = "", []

    start_retr = time.time()
    if (MODE_RAG and retriever is not None):
        q = e5_query(prompt)
        docs = retriever.invoke(q)
        context_text, sources = build_context(docs, max_chars=4000)
        used_context = bool(context_text.strip())
    retr_ms = (time.time() - start_retr) * 1000 if MODE_RAG else 0

    # LLM
    try:
        start_llm = time.time()
        answer = ask_groq(system_prompt, prompt, context_text, lang, used_context,
                          temperature=st.session_state.get("temperature", 0.15) if "temperature" in st.session_state else 0.15,
                          max_tokens=st.session_state.get("max_tokens", 700) if "max_tokens" in st.session_state else 700)
        # Utiliser les sliders live
        answer = ask_groq(system_prompt, prompt, context_text, lang, used_context,
                          temperature=st.session_state.get("temperature_ui", None) or st.session_state.get("temperature", 0.15) if False else st.sidebar.session_state.get("🎯 Température (précision)", 0.15),
                          max_tokens=st.sidebar.session_state.get("🧾 Longueur max (tokens)", 700))
        # Ajout footer selon mode
        answer = add_footer(answer, sources, lang, used_context, MODE_RAG)
        gen_ms = (time.time() - start_llm) * 1000

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.convos[st.session_state.active_id].append({"role": "assistant", "content": answer})

        # Perf caption
        if MODE_RAG:
            st.caption(f"🔎 Retrieval: {retr_ms:.0f} ms • 🧠 Génération: {gen_ms:.0f} ms • Mode: RAG")
        else:
            st.caption(f"🧠 Génération: {gen_ms:.0f} ms • Mode: LLM seul")
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"⚠️ Erreur API Groq : {e}")
