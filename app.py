# -*- coding: utf-8 -*-
"""
Lexi — RAG Conformité (BOA) sur Streamlit
- UI pro (fond blanc, texte noir, accents vert foncé)
- FR/EN auto selon la question
- RAG via FAISS local: ./faiss-compliance-banking-multilingual-index
- Si pas d'info FAISS -> réponse générale + avertissement + lien doc
- Si info FAISS -> réponse basée contexte + sources + lien doc
- Small talk autorisé (pas de patterns fixes de salutations)
"""

import os
import time
import string
import unicodedata
from typing import List, Tuple

import streamlit as st
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import groq

# ==============================
# CONFIG GÉNÉRALE
# ==============================
INDEX_DIR = "faiss-compliance-banking-multilingual-index"  # ↓ ton dossier local
DRIVE_LINK = os.getenv("DOC_LINK", "")  # lien global vers ta doc (optionnel)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-large")
MAX_SOURCES = 5

# ==============================
# STYLES (blanc/vert BOA)
# ==============================
PRIMARY_GREEN = "#0E7A30"         # vert BOA
DARKER_GREEN = "#0B5E26"          # ~25% plus sombre
TEXT_BLACK = "#111111"
BG_WHITE = "#FFFFFF"
BORDER = "#E5E7EB"

st.set_page_config(
    page_title="Lexi — Conformité BOA",
    page_icon=None,
    layout="wide"
)

st.markdown(f"""
<style>
:root {{
  --boa-green: {PRIMARY_GREEN};
  --boa-dark-green: {DARKER_GREEN};
  --txt: {TEXT_BLACK};
  --bg: {BG_WHITE};
  --border: {BORDER};
}}
html, body, .stApp {{
  background: var(--bg) !important;
  color: var(--txt) !important;
}}
a, .stMarkdown a {{
  color: var(--boa-green) !important;
  text-decoration: none;
}}
a:hover {{ opacity: .9; }}
.sidebar .sidebar-content {{
  background: #f8fafb !important;
}}
.block-container {{
  padding-top: 1.5rem;
}}
.chat-bubble {{
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 14px;
  margin-bottom: 10px;
  background: #fff;
}}
.chat-user .chat-bubble {{
  border-left: 3px solid var(--boa-green);
}}
.chat-bot .chat-bubble {{
  border-left: 3px solid var(--boa-dark-green);
}}
.chat-meta {{
  font-size: 12px; color: #6b7280; margin-top: 6px;
}}
</style>
""", unsafe_allow_html=True)

# ==============================
# UTILS
# ==============================
def normalize(txt: str) -> str:
    if not txt:
        return ""
    txt = ''.join(c for c in unicodedata.normalize('NFD', txt) if unicodedata.category(c) != 'Mn')
    txt = txt.lower()
    for ch in "’" + string.punctuation:
        txt = txt.replace(ch, " ")
    return " ".join(txt.split())

def detect_lang_simple(text: str) -> str:
    """Heuristique FR/EN légère (small talk + conformité)."""
    if not text:
        return "fr"
    t = text.lower()
    fr_hits = sum(w in t for w in [
        "bonjour","bonsoir","salut","svp","pièce","contrôles","obligatoires","nouveau",
        "compte","kyc","conformité","client","pourquoi","comment","quels","quelle",
        "quelles","lcb-ft","blanchiment","sanctions","rgpd","procédure","audit","merci"
    ])
    en_hits = sum(w in t for w in [
        "hello","hi","please","identity","controls","mandatory","new","account",
        "kyc","compliance","customer","why","how","what","which","aml","sanctions",
        "gdpr","procedure","audit","thanks","thank you","hey"
    ])
    return "en" if en_hits > fr_hits else "fr"

def e5_query(text: str) -> str:
    t = text.strip()
    return t if t.lower().startswith("query:") else f"query: {t}"

def build_context(docs, max_chars=4000) -> Tuple[str, List[str]]:
    """Compacte les extraits + liste de sources dédupliquées."""
    if not docs:
        return "", []
    seen = set()
    parts, sources = [], []
    total = 0
    for d in docs:
        sf = d.metadata.get("source_file", "inconnu")
        sd = d.metadata.get("source_folder", "")
        pg = d.metadata.get("page_index")
        key = (sf, pg if pg is not None else -1)
        if key in seen:
            continue
        seen.add(key)

        snippet = (d.page_content or "").strip()
        if not snippet:
            continue
        parts.append(snippet)
        total += len(snippet)

        label = f"{sf}" + (f" (page {pg+1})" if pg is not None else "")
        if sd:
            label += f" | dossier: {sd}"
        sources.append(label)

        if total > max_chars:
            break

    if len(sources) > MAX_SOURCES:
        sources = sources[:MAX_SOURCES]
    return "\n---\n".join(parts), sources

# ==============================
# LLM (Groq)
# ==============================
SYSTEM_PROMPT_FR = """Vous êtes Lexi, assistant IA du département Conformité de BANK OF AFRICA (BOA).
Répondez en français, de manière professionnelle et factuelle.
Si des extraits documentaires sont fournis, appuyez-vous strictement dessus.
S’ils manquent, donnez une réponse générale et prudente (bonnes pratiques de conformité).
N’inventez pas de faits non justifiés. Utilisez des listes si utile."""
SYSTEM_PROMPT_EN = """You are Lexi, an AI assistant for BANK OF AFRICA's Compliance department.
Answer in English, professionally and factually.
If documentary context is provided, rely strictly on it.
If none is available, provide a conservative, general best-practice compliance answer.
Do not fabricate unsupported facts. Use lists where helpful."""

@st.cache_resource(show_spinner=False)
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY non défini (Settings → Variables → Secrets).")
    return groq.Client(api_key=api_key)

def ask_groq(user_question: str, context_text: str, lang: str, used_context: bool, temperature: float) -> str:
    client = get_groq_client()
    system_prompt = SYSTEM_PROMPT_FR if lang == "fr" else SYSTEM_PROMPT_EN
    lang_instruction = "Répondez en français." if lang == "fr" else "Answer in English."

    if used_context:
        instruction = (
            "- Base your answer ONLY on the context above.\n"
            "- If the context is insufficient, say so explicitly.\n"
        )
    else:
        instruction = (
            "- No context was found. Provide a general, best-practice compliance answer.\n"
            "- Keep it conservative and avoid jurisdiction-specific claims.\n"
            "- Be polite and helpful (small talk allowed), but do not fabricate facts.\n"
        )

    user_block = (
        f"{lang_instruction}\n\n"
        "Context excerpts:\n"
        f"{context_text if used_context else '(no context found)'}\n\n"
        f"Question:\n{user_question}\n\n"
        "Instructions:\n"
        f"{instruction}"
        "- Keep the answer concise and structured.\n"
    )

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_block},
        ],
        temperature=float(temperature),
        max_tokens=700
    )
    return resp.choices[0].message.content.strip()

# ==============================
# FAISS (local au repo)
# ==============================
@st.cache_resource(show_spinner=True)
def load_retriever_local(index_dir: str, k=6, fetch_k=50, lambda_mult=0.3):
    """Charge FAISS depuis un dossier local du repo (non zippé)."""
    if not os.path.isdir(index_dir):
        return None, 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={'device': device, 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )
    vs = FAISS.load_local(index_dir, embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult})
    return retriever, vs.index.ntotal

# ==============================
# SIDEBAR (paramètres)
# ==============================
with st.sidebar:
    st.markdown(f"## <span style='color:{PRIMARY_GREEN}'>Lexi — Conformité BOA</span>", unsafe_allow_html=True)
    st.write("Réponses sourcées si disponibles, sinon meilleures pratiques (FR/EN).")
    use_rag = st.checkbox("Activer la recherche documentaire (RAG)", value=True)
    k = st.slider("k (résultats)", 3, 12, 6)
    fetch_k = st.slider("fetch_k (pool MMR)", 20, 200, 50, step=10)
    lambda_mult = st.slider("lambda_mult (diversité)", 0.0, 1.0, 0.3, step=0.1)
    temperature = st.slider("Température LLM", 0.0, 1.0, 0.15, step=0.05)
    doc_link = st.text_input("Lien documentaire (optionnel)", value=DRIVE_LINK)
    st.markdown("---")
    st.caption(f"Index local : `{INDEX_DIR}`")

# ==============================
# CHARGEMENT INDEX
# ==============================
retriever, ntotal = (None, 0)
if use_rag:
    with st.spinner("Chargement de l’index FAISS..."):
        retriever, ntotal = load_retriever_local(INDEX_DIR, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
st.caption(f"Vecteurs FAISS chargés : {ntotal:,}" if ntotal else "Aucun index chargé (mode général).")

# ==============================
# ÉTAT DE CONVERSATION
# ==============================
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown(f"### <span style='color:{DARKER_GREEN}'>Assistant Conformité</span>", unsafe_allow_html=True)

# Zone conversation
for role, content in st.session_state.history:
    css_class = "chat-user" if role == "user" else "chat-bot"
    st.markdown(f"<div class='{css_class}'><div class='chat-bubble'>{content}</div></div>", unsafe_allow_html=True)

# Champ utilisateur
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("Votre message", height=120, placeholder="Ex. : Quels sont les contrôles KYC obligatoires pour un nouveau client ?")
    submitted = st.form_submit_button("Envoyer", use_container_width=True)

if submitted and user_input.strip():
    st.session_state.history.append(("user", user_input.strip()))
    lang = detect_lang_simple(user_input)

    # Retrieval
    context_text, sources, used_context = "", [], False
    if retriever:
        try:
            q = e5_query(user_input)
            t0 = time.time()
            docs = retriever.invoke(q)
            context_text, sources = build_context(docs, max_chars=4000)
            used_context = bool(context_text.strip())
            st.caption(f"Retrieval: {len(docs)} doc(s), contexte={'oui' if used_context else 'non'} | {time.time()-t0:.2f}s")
        except Exception:
            used_context = False

    # LLM
    try:
        answer = ask_groq(user_input, context_text, lang, used_context, temperature)
        # Footer
        if used_context and sources:
            footer = "\n\nSources :\n" if lang == "fr" else "\n\nSources:\n"
            footer += "\n".join(f"- {s}" for s in sources[:MAX_SOURCES])
            if doc_link:
                footer += ("\n\nDossier source :\n" + doc_link) if lang == "fr" else ("\n\nSource folder:\n" + doc_link)
            answer += footer
        elif not used_context:
            disclaimer = (
                "\n\nCette réponse ne provient pas de la base documentaire indexée. Merci de vérifier auprès d’un expert conformité."
                if lang == "fr" else
                "\n\nThis answer is not sourced from the indexed knowledge base. Please verify with a compliance expert."
            )
            if doc_link:
                disclaimer += ("\n\nDocumentation :\n" + doc_link) if lang == "fr" else ("\n\nDocumentation:\n" + doc_link)
            answer += disclaimer

        st.session_state.history.append(("assistant", answer))
    except Exception as e:
        st.session_state.history.append(("assistant", f"Erreur de génération : {e}"))

    st.rerun()
