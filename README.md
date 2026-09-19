# 🤖 RAG Chatbot — Agentic, Multilingual Document QA with RLHF

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Llama 4](https://img.shields.io/badge/Llama_4_Scout-17B-purple)](https://groq.com)
[![Qwen](https://img.shields.io/badge/Qwen_3-32B-blue)](https://groq.com)
[![Whisper](https://img.shields.io/badge/Whisper-Large_v3_Turbo-orange)](https://groq.com)
[![Groq](https://img.shields.io/badge/Groq-Free_API-orange)](https://console.groq.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-red)](https://faiss.ai)
[![Multilingual](https://img.shields.io/badge/Multilingual-50%2B_Languages-06b6d4)](https://www.sbert.net)
[![RLHF](https://img.shields.io/badge/RLHF-PPO_Style-cyan)](https://github.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B)](https://penguin-ai-eqxw9u846dnak3drizxjmt.streamlit.app)
[![Project Page](https://img.shields.io/badge/Project%20Page-GitHub%20Pages-222?logo=github)](https://rajneeshbabu.github.io/rag-chatbot/)

A production-quality multi-mode AI chatbot with **RLHF (Reinforcement Learning from Human Feedback)**, **Multilingual Production RAG**, **Agentic RAG**, **voice-to-text**, animated neural network UI, and support for the latest models via the **Groq free API** — no GPU needed.

🚀 **[Launch Streamlit App →](https://penguin-ai-eqxw9u846dnak3drizxjmt.streamlit.app)**

🌐 **[View Project Page →](https://rajneeshbabu.github.io/rag-chatbot/)**

---

## Features

- **4 Chat Modes** — General Chat, Advanced RAG, Agentic RAG, Domain Expert
- **6 Models** — Llama 4 Scout, Qwen 3 32B, Llama 3.3 70B, Llama 3.1 8B, Mixtral 8×7B, Llama 3 70B
- **🌐 Multilingual RAG** — Ask questions in Hindi, French, German, Spanish, Chinese, Japanese (50+ languages) — shared embedding space maps cross-lingual queries to documents natively
- **Auto Language Detection** — `langdetect` identifies query language; LLM responds in the same language; language badge shown in UI
- **Multilingual Embeddings** — `paraphrase-multilingual-MiniLM-L12-v2` (50+ languages, 471 MB, CPU)
- **Multilingual Cross-Encoder** — `mmarco-mMiniLMv2-L12-H384-v1` reranker across 13 languages
- **Voice-to-Text** — Mic button in chat (Whisper Large v3 Turbo via Groq + Web Speech API fallback)
- **Whisper Multilingual** — Audio file → Text transcription with auto language detection (99 languages)
- **Llama Guard 3 20B** — Safety content filter toggle
- **RLHF System** — PPO-style reward model, value baseline, advantage computation, policy adaptation
- **Production RAG Pipeline** — 6 layers: semantic chunking → SHA-256 dedup → BM25+FAISS hybrid → RRF fusion → multilingual cross-encoder reranking → citation grounding
- **Agentic RAG** — ReAct loop with multi-hop retrieval (up to 3 hops)
- **Animated UI** — Neural network canvas background, glowing chat bubbles, live token speed counter

---

## Models

### Chat Models (selectable in sidebar)

| Model | Groq ID | Context | Best For |
|---|---|---|---|
| ⚡ Llama 4 Scout 17B | `meta-llama/llama-4-scout-17b-16e-instruct` | 128K | Latest Meta — fast & accurate |
| 🌐 Qwen 3 32B | `qwen/qwen3-32b` | 32K | Multilingual tasks |
| 🧠 Llama 3.3 70B | `llama-3.3-70b-versatile` | 128K | Best overall quality (default) |
| 🔥 Llama 3.1 8B | `llama-3.1-8b-instant` | 128K | Speed-critical tasks |
| 🔀 Mixtral 8×7B | `mixtral-8x7b-32768` | 32K | Long documents |
| 💎 Llama 3 70B | `llama3-70b-8192` | 8K | General use |

### Audio & Safety Models

| Model | Groq ID | Purpose |
|---|---|---|
| 🎙️ Whisper Large v3 Turbo | `whisper-large-v3-turbo` | Speech → Text |
| 🛡️ Llama Guard 3 20B | `meta-llama/llama-guard-3-20b` | Content safety filter |

---

## RLHF System

RAG Chatbot implements a full **PPO-style RLHF loop** that runs in-session without any GPU:

1. **Reward Model** — Combines human 👍/👎 (60%) + length quality (20%) + diversity (10%) + specificity (10%) → reward in `[-1, +1]`
2. **Value Baseline** — Exponential moving average of past rewards (EMA critic)
3. **Advantage** — `reward - baseline` tells the policy if a response is better/worse than average
4. **Policy Update** — Clips temperature shift to ±0.20 (prevents policy collapse). High advantage → lower temp (exploit). Low advantage → higher temp (explore)
5. **In-context Steering** — Top-3 high-advantage responses injected into system prompt so the model learns preferred style without retraining
6. **Replay Buffer Export** — Download `rlhf_replay_buffer.jsonl` for offline fine-tuning with HuggingFace TRL

---

## RAG Pipeline

```
Document Upload
    │
    ├── Semantic Chunking (paragraph/sentence boundaries, not fixed tokens)
    ├── SHA-256 Deduplication (cuts index by 40%+)
    ├── Metadata Enrichment (source, page, section, hash)
    │
    ├── FAISS Dense Index (MiniLM-L6-v2 embeddings)
    ├── BM25 Sparse Index (keyword retrieval)
    │
    ├── Hybrid Retrieval → RRF Fusion (merges dense + sparse ranked lists)
    ├── Cross-Encoder Reranking (top-20 → top-5, ms-marco-MiniLM-L-6-v2)
    ├── MMR Deduplication + Context Compression
    ├── Citation Grounding (source, page, score per chunk)
    ├── LRU Query Cache (1hr TTL, saves 30-60% inference cost)
    └── Faithfulness Scoring (answer checked back against its retrieved context)
```

Every answer is scored for **faithfulness** against the context it was actually given, so an
answer that drifts away from the retrieved passages is visible rather than silently trusted.

---

## Agentic RAG (ReAct Loop)

```
User Question
    │
    └── LLM: Thought → Action[tool] → Observation → repeat (max 3 hops)
            Tools: search_document | summarize | verify_fact
        └── Final Answer
```

---

## Live Demo

The `index.html` file is deployed as a GitHub Pages static demo — no server or API key needed.

**View it here → [rajneeshbabu.github.io/rag-chatbot](https://rajneeshbabu.github.io/rag-chatbot/)**

It showcases the animated UI, auto-playing chat demo, RLHF flow diagram, RAG pipeline, model table, and quick-start steps. To enable GitHub Pages: go to your repo → **Settings → Pages → Source: Deploy from branch → Branch: main → Folder: / (root)**.

---

## Project Structure

```
rag-chatbot/
├── app.py                      # Main Streamlit app (RLHF + RAG + Voice + all modes)
├── chatbot_pipeline.ipynb      # LangChain pipeline walkthrough notebook
├── requirements.txt            # All dependencies
├── .env.example                # API key template
├── .gitignore                  # Excludes .env, checkpoints, replay buffer
├── rlhf_replay_buffer.jsonl    # Generated at runtime — RLHF training data (gitignored)
├── README.md
└── index.html                  # GitHub Pages static demo (live at link above)
```

---

## Quick Start

### Step 1 — Clone the repo

```bash
git clone https://github.com/rajneeshbabu/rag-chatbot.git
cd rag-chatbot
```

### Step 2 — Install dependencies

> **Important:** Use the full path to your Python's pip to avoid environment conflicts.

```bash
# For Anaconda users (recommended):
/opt/anaconda3/bin/pip install -r requirements.txt

# For standard Python:
pip install -r requirements.txt
```

This installs everything including `streamlit-mic-recorder` for the in-chat voice-to-text mic button. If you skip it, the app still works — the mic falls back to the browser's built-in Web Speech API (Chrome only).

### Step 3 — Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up / log in (free)
3. Click **API Keys** → **Create API Key**
4. Copy the key (starts with `gsk_`)

### Step 4 — Set your API key

```bash
# Create the .env file:
echo "GROQ_API_KEY=gsk_your_key_here" > .env
```

Or copy the template and edit it:

```bash
cp .env.example .env
# Open .env in any text editor and paste your key
```

### Step 5 — Run the app

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

---

## Running the Notebook

```bash
# Install Jupyter if needed:
/opt/anaconda3/bin/pip install jupyter

# Launch:
jupyter notebook chatbot_pipeline.ipynb
```

Run cells top to bottom. Cell 1 installs all packages automatically.

---

## Dependencies

```
groq>=0.9.0
langchain==0.2.17
langchain-community>=0.2.0
langchain-groq>=0.1.0
langchain-huggingface>=0.0.3
langchain-core>=0.2.0
faiss-cpu>=1.8.0
sentence-transformers>=3.0.0
rank-bm25>=0.2.2
pypdf>=4.0.0
pdfplumber>=0.11.0
docx2txt>=0.8
streamlit>=1.35.0
python-dotenv>=1.0.0
tiktoken>=0.7.0
```

Install all at once:
```bash
/opt/anaconda3/bin/pip install groq langchain==0.2.17 langchain-community langchain-groq langchain-huggingface langchain-core faiss-cpu sentence-transformers rank-bm25 pypdf pdfplumber docx2txt streamlit python-dotenv tiktoken
```

---

## Usage Guide

### General Chat
Type any question. The RLHF policy adapts the model's temperature based on your 👍/👎 feedback over time.

### Document Q&A (Advanced RAG)
1. Select **📄 Advanced RAG** in sidebar
2. Upload a PDF, DOCX, or TXT file
3. Ask questions — answers are grounded in the document with citations

### Agentic RAG
1. Select **🕵️ Agentic RAG** in sidebar
2. Upload a document
3. Ask complex multi-part questions — the agent reasons step by step and retrieves across multiple hops

### Voice-to-Text (in-chat mic)
Two modes — whichever is available runs automatically:

**Mode A — Whisper (best quality, requires `streamlit-mic-recorder`):**
1. Click the **🎙️** mic button above the chat input
2. Speak your question
3. Click stop — Whisper Large v3 Turbo transcribes it instantly
4. The transcript is sent as your message automatically

**Mode B — Web Speech API (no install, Chrome only):**
1. Click the **🎙️** mic button above the chat input
2. Speak — words appear in real time
3. Click **✉️ Use this** — fills the chat box
4. Press Enter to send

### Audio File Transcription
1. Scroll to **🎙️ Whisper Audio → Text** in sidebar
2. Upload mp3, wav, m4a, or webm
3. Click **▶ Transcribe**

### Safety Guard
Toggle **🛡️ Safety Guard** in sidebar — Llama Guard 3 20B checks every response and blocks unsafe content.

### RLHF Feedback
- Click **👍** after a good response → reward +1, model steers toward this style
- Click **👎** after a bad response → reward -1, model explores alternatives
- Sidebar shows live cumulative reward curve
- Click **⬇️ Export RLHF Buffer** to download training data for offline fine-tuning

---

## Troubleshooting

| Error | Fix |
|---|---|
| `No module named 'groq'` | Run: `/opt/anaconda3/bin/pip install groq` |
| `No module named 'langchain.memory'` | Run: `/opt/anaconda3/bin/pip install langchain==0.2.17` |
| `No module named 'langdetect'` | Run: `pip install langdetect` — or ignore, language detection gracefully falls back to English |
| `No module named 'streamlit_mic_recorder'` | Run: `pip install streamlit-mic-recorder` — or ignore, Web Speech API fallback activates |
| Mic button not working | Use Chrome; allow microphone access when browser asks |
| `Invalid API key` | Check `.env` file — no spaces around `=` sign |
| Model not found error | Switch to **Llama 3.3 70B** (always available on Groq free tier) |
| Port already in use | Run: `streamlit run app.py --server.port 8503` |
| Whisper file too large | Groq supports audio up to ~25MB — compress/trim before uploading |
| First load slow (multilingual model) | `paraphrase-multilingual-MiniLM-L12-v2` (~471 MB) downloads once on first run, then cached |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 4 Scout / Qwen 3 32B / Llama 3.3 70B (via Groq) |
| Inference | Groq Cloud (~500–900 tok/s) |
| Audio | Whisper Large v3 Turbo — multilingual auto-detect (Groq, 99 langs) |
| Safety | Llama Guard 3 20B (Groq) |
| RLHF | PPO-style reward model + value baseline + advantage |
| Pipeline | LangChain 0.2 + custom production layers |
| **Multilingual Embeddings** | **paraphrase-multilingual-MiniLM-L12-v2 (50+ languages, shared embedding space)** |
| Vector Store | FAISS (local, in-memory) |
| Sparse Retrieval | BM25 (rank-bm25) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| **Multilingual Reranking** | **mmarco-mMiniLMv2-L12-H384-v1 cross-encoder (13 languages)** |
| Language Detection | langdetect — auto-detects query language, badge shown in UI |
| UI | Streamlit + animated neural canvas (JavaScript) |

---

## License

MIT License — free to use, modify, and distribute.

---

⚠️ **Disclaimer:** Domain expert responses (Medical, Legal, Finance) are for educational purposes only. Always consult qualified professionals for actual medical, legal, or financial decisions.

---

*Built with Llama 4 · Qwen 3 · Whisper · Groq · LangChain · FAISS · RLHF*
