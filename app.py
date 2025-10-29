# app.py
# -*- coding: utf-8 -*-
import os
import re
import time
import string
import unicodedata
from datetime import datetime
from typing import List, Tuple, Optional

import streamlit as st
import torch
from langchain_community.vectorstores import FAISS
# Recommandé: from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import groq

# ==============================
# CONFIG PAGE
# ==============================
st.set_page_config(page_title="Lexi – RAG Conformité", page_icon="✅")
st.title("Lexi · RAG Conformité (Groq + FAISS)")

# ==============================
# SECRETS & CONSTANTES
# ==============================
# Sur Streamlit Cloud, mets GROQ_API_KEY / DRIVE_LINK / INDEX_PATH dans Settings → Secrets.
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY manquant. Ajoute-le dans Streamlit > Settings > Secrets.")
    st.stop()

client = groq.Client(api_key=GROQ_API_KEY)

# ⚠️ ICI: chemin de ton index FAISS dans le repo
INDEX_PATH = os.getenv("INDEX_PATH", "faiss-compliance-banking-multilingual-index")
DRIVE_LINK = os.getenv("DRIVE_LINK", "https://drive.google.com/drive/folders/TON_LIEN_FIXE_ICI")
MAX_SOURCES = 5

# ==============================
# PROMPTS SYSTÈME
# ==============================
SYSTEM_PROMPT_FR = """Tu es Lexi, assistant IA du département Conformité de BANK OF AFRICA (BOA).
Réponds en français, poliment, clairement et factuellement.
Si la question est une salutation (ex. "Bonjour", "Merci"), réponds de manière chaleureuse et naturelle.
Si des extraits documentaires sont fournis, base-toi STRICTEMENT dessus.
Si aucun extrait n’est disponible, donne une réponse générale de bonnes pratiques conformité, de manière prudente et générique, en précisant qu'il est préférable de vérifier dans les documents officiels pour confirmation.
Structure en puces si utile. N’invente pas de détails non justifiés par le contexte."""
SYSTEM_PROMPT_EN = """You are Lexi, an AI assistant for BANK OF AFRICA's Compliance department.
Answer in English, politely, clearly, and factually.
If the user input is a greeting (e.g., "Hello", "Thank you"), respond warmly and naturally.
If context excerpts are provided, rely STRICTLY on them.
If no context is provided, give a general best-practice compliance answer, conservatively and generically, and mention it’s recommended to verify with official documents for confirmation.
Use bullets if helpful. Do not fabricate details unsupported by the context."""

# ==============================
# SMALL TALK / SALUTATIONS
# ==============================
GREETING_PATTERNS_FR = [
    r"\b(bonjour|salut|coucou|bonsoir|re\s?bonjour)\b",
    r"\bbonne\s?(journée|matinée|après[- ]?midi|soirée)\b",
    r"\bmerci\b|\bmerci beaucoup\b|\bje te remercie\b|\bje vous remercie\b|\bthanks?\b",
    r"\b(ça va|ca va|comment\s?(ça|ca)\s?va|comment allez[- ]vous|comment vas[- ]tu)\b",
    r"\b(enchanté|rav(i|ie) de vous rencontrer)\b",
    r"\b(au revoir|à bientôt|a bientôt|à plus|a plus|à la prochaine|bonne nuit)\b",
    r"^\s*(ok|d(’|')accord|parfait|super|top|cool|merci!?)\s*$",
]
GREETING_PATTERNS_EN = [
    r"\b(hi|hello|hey|yo|hiya|howdy|sup)\b",
    r"\b(good\s?(morning|afternoon|evening|night))\b",
    r"\b(thank you|thanks|thx|much appreciated)\b",
    r"\b(how are you|how's it going|hows it going|how are u|how r u|how do you do)\b",
    r"\b(nice to meet you|pleased to meet you)\b",
    r"\b(bye|goodbye|see you|see ya|take care|later)\b",
    r"^\s*(ok|okay|sounds good|great|awesome|cool|thanks!?)\s*$",
]

def detect_lang_simple(text: str) -> str:
    """Heuristique légère FR/EN."""
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
    """
    Retourne l'une des classes:
    - 'greet'   : bonjour/hello
    - 'thanks'  : merci/thanks
    - 'howare'  : comment ça va / how are you
    - 'bye'     : au revoir/bye
    - 'ack'     : ok/super/etc.
    Sinon None.
    """
    t = text.strip().lower()

    # FR
    if lang == "fr":
        if re.search(r"\b(bonjour|salut|coucou|bonsoir|re\s?bonjour)\b", t):
            return "greet"
        if re.search(r"\bmerci\b|\bmerci beaucoup\b|\bje te remercie\b|\bje vous remercie\b", t):
            return "thanks"
        if re.search(r"\b(ça va|ca va|comment\s?(ça|ca)\s?va|comment allez[- ]vous|comment vas[- ]tu)\b", t):
            return "howare"
        if re.search(r"\b(au revoir|à bientôt|a bientôt|à plus|a plus|à la prochaine|bonne nuit)\b", t):
            return "bye"
        if re.search(r"^\s*(ok|d(’|')accord|parfait|super|top|cool|merci!?)\s*$", t):
            return "ack"
    # EN
    else:
        if re.search(r"\b(hi|hello|hey|hiya|howdy)\b", t):
            return "greet"
        if re.search(r"\b(thank you|thanks|thx|much appreciated)\b", t):
            return "thanks"
        if re.search(r"\b(how are you|how's it going|how do you do)\b", t):
            return "howare"
        if re.search(r"\b(bye|goodbye|see you|see ya|take care|later)\b", t):
            return "bye"
        if re.search(r"^\s*(ok|okay|sounds good|great|awesome|cool|thanks!?)\s*$", t):
            return "ack"

    # si c'est un court message composé uniquement de saluts communs
    patterns = GREETING_PATTERNS_FR if lang == "fr" else GREETING_PATTERNS_EN
    for p in patterns:
        if re.search(p, t):
            return "greet"
    return None

def smalltalk_reply(kind: str, lang: str) -> str:
    now = datetime.utcnow()
    # on peut customiser selon l'heure si besoin
    if lang == "fr":
        if kind == "greet":
            return "Bonjour 👋 Comment puis-je vous aider en conformité aujourd’hui ?"
        if kind == "thanks":
            return "Avec plaisir ! N’hésitez pas si vous avez une autre question conformité."
        if kind == "howare":
            return "Ça va très bien, merci ! Et vous ? Souhaitez-vous que l’on regarde un point conformité en particulier ?"
        if kind == "bye":
            return "Au revoir ! Bonne journée et à bientôt."
        if kind == "ack":
            return "Parfait 👍 Dites-moi ce dont vous avez besoin."
        return "Bonjour ! Que puis-je faire pour vous ?"
    else:
        if kind == "greet":
            return "Hello 👋 How can I help with compliance today?"
        if kind == "thanks":
            return "You're welcome! Feel free to ask any other compliance questions."
        if kind == "howare":
            return "I'm doing great, thanks! And you? Would you like me to check anything compliance-related?"
        if kind == "bye":
            return "Goodbye! Have a great day and see you soon."
        if kind == "ack":
            return "Great 👍 Tell me what you need."
        return "Hi there! How can I help?"

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
    """Compacte les extraits + prépare une liste de sources dédupliquées."""
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
        if key in seen:
            continue
        seen.add(key)

        snippet = (d.page_content or "").strip()
        if not snippet:
            continue
        parts.append(snippet)

        label = f"{sf}" + (f" (page {pg+1})" if pg is not None else "")
        if sd:
            label += f" | dossier: {sd}"
        sources.append(label)

        if sum(len(p) for p in parts) > max_chars:
            break

    if len(sources) > MAX_SOURCES:
        sources = sources[:MAX_SOURCES]

    return "\n---\n".join(parts), sources

def ask_groq(system_prompt: str, user_question: str, context_text: str, lang: str, used_context: bool) -> str:
    """Appel Groq. Si pas de contexte, autorise une réponse générale (polie)."""
    lang_instruction = "Réponds en français." if lang == "fr" else "Answer in English."
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
        f"{instruction}"
        "- Keep the answer concise and structured.\n"
    )

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_block},
        ],
        temperature=0.15 if used_context else 0.2,
        max_tokens=700,
    )
    return resp.choices[0].message.content.strip()

def add_footer(answer: str, sources: List[str], lang: str, used_context: bool) -> str:
    """
    - Si used_context=True et sources: ajouter sources + lien doc.
    - Si used_context=False: ajouter avertissement + lien doc.
    """
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
        disclaimer = (
            "\n\n⚠️ Cette réponse ne provient pas de la base documentaire indexée. "
            "Merci de vous rapprocher d’un expert en conformité pour validation."
            if lang == "fr" else
            "\n\n⚠️ This answer does not come from the indexed knowledge base. "
            "Please consult a compliance expert for validation."
        )
        answer += disclaimer
        if DRIVE_LINK:
            answer += (
                f"\n\n📂 Documentation :\n➡️ {DRIVE_LINK} (Accéder aux documents)"
                if lang == "fr" else
                f"\n\n📂 Documentation:\n➡️ {DRIVE_LINK} (Access documents)"
            )
    return answer

retriever, ntotal = load_retriever(INDEX_PATH)

with st.sidebar:
    st.subheader("⚙️ Configuration")
    st.write(f"Index FAISS : `{INDEX_PATH}`")
    st.write(f"Vecteurs chargés : **{ntotal}**" if ntotal else "Mode **dégradé** (sans index)")
    st.write("Modèle Groq : `llama-3.1-8b-instant`")
    st.link_button("📂 Dossier sources", url=DRIVE_LINK, use_container_width=True)

# ==============================
# CHAT UI
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour 👋 Je suis **Lexi**. Posez-moi une question conformité (FR/EN)."}
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Écrivez votre question…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Détection langue + small talk
    lang = detect_lang_simple(prompt)
    kind = classify_smalltalk(prompt, lang)

    # Si c'est uniquement une salutation / small talk -> on répond sans LLM/RAG
    # Heuristique: court message ET classé small talk
    if kind is not None and len(prompt.strip()) <= 120:
        reply = smalltalk_reply(kind, lang)
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    else:
        # Sinon, RAG normal
        system_prompt = SYSTEM_PROMPT_FR if lang == "fr" else SYSTEM_PROMPT_EN

        # Retrieval
        used_context = False
        context_text, sources = "", []
        if retriever is not None:
            q = e5_query(prompt)
            t0 = time.time()
            docs = retriever.invoke(q)
            latency = time.time() - t0
            context_text, sources = build_context(docs, max_chars=4000)
            used_context = bool(context_text.strip())
            with st.sidebar:
                st.caption(f"🔎 Retrieval: {len(docs)} doc(s) • contexte={'oui' if used_context else 'non'} • ⏱ {latency:.2f}s")
        else:
            with st.sidebar:
                st.caption("🔎 Mode dégradé (sans FAISS) : réponse générale.")

        # Réponse LLM
        try:
            t0 = time.time()
            answer = ask_groq(system_prompt, prompt, context_text, lang, used_context)
            answer = add_footer(answer, sources, lang, used_context)
            gen_latency = time.time() - t0

            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})

            with st.sidebar:
                st.caption(f"🧠 Généré en {gen_latency:.2f}s")
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"⚠️ Erreur API Groq : {e}")
