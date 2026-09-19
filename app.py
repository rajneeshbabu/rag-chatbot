"""
RAG Chatbot — Production RAG + Agentic RAG + Multilingual RAG
Pipeline based on: RAG at Scale (Production Architecture Guide)

Layers implemented:
  1. Semantic chunking + SHA-256 dedup + metadata enrichment
  2. Multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2, 50+ langs) + FAISS
  3. Hybrid retrieval: BM25 (sparse) + FAISS (dense) → RRF fusion
  4. Multilingual cross-encoder reranking (mmarco-mMiniLMv2, top-20 → top-5)
  5. MMR dedup + context compression + citation grounding
  6. LRU query cache + RAGAS-style faithfulness eval

Multilingual RAG:
  - Query & documents in any of 50+ languages — same shared embedding space
  - langdetect auto-detects query language → language badge in UI
  - Whisper auto language detection (no longer English-only)
  - LLM responds in the same language as the user query

Agentic RAG:
  ReAct loop — LLM decides when/what to retrieve, supports multi-hop
"""

import streamlit as st
import streamlit.components.v1 as components
import os, re, json as _json, hashlib, time
from functools import lru_cache
from dotenv import load_dotenv

# ── Language detection (graceful fallback if not installed) ────────────────────
try:
    from langdetect import detect as _langdetect, LangDetectException
    def detect_language(text: str) -> str:
        """Returns ISO-639-1 language code, e.g. 'en', 'hi', 'fr'. Falls back to 'en'."""
        try:
            return _langdetect(text[:400]) if len(text.strip()) > 10 else "en"
        except LangDetectException:
            return "en"
    LANGDETECT_AVAILABLE = True
except ImportError:
    def detect_language(text: str) -> str:
        return "en"
    LANGDETECT_AVAILABLE = False

LANG_NAMES = {
    "en":"English","hi":"Hindi","fr":"French","de":"German","es":"Spanish",
    "zh-cn":"Chinese","ja":"Japanese","ko":"Korean","ar":"Arabic","pt":"Portuguese",
    "ru":"Russian","it":"Italian","nl":"Dutch","tr":"Turkish","pl":"Polish",
    "sv":"Swedish","da":"Danish","fi":"Finnish","no":"Norwegian","bn":"Bengali",
    "ta":"Tamil","te":"Telugu","mr":"Marathi","gu":"Gujarati","ur":"Urdu",
}

load_dotenv()

# ── API key: check st.secrets (Streamlit Cloud) first, then .env (local) ──────
def _get_server_api_key():
    """Returns key from server config (secrets/env). Never exposed to the UI."""
    try:
        return st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    except Exception:
        return os.getenv("GROQ_API_KEY", "")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot — Agentic Multilingual RAG",
    page_icon="🐧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":        "https://github.com/rajneeshbabu/rag-chatbot",
        "Report a bug":    "https://github.com/rajneeshbabu/rag-chatbot/issues",
        "About":           "**RAG Chatbot** — Production RAG + Agentic RAG chatbot\nBuilt by [Rajneesh](https://github.com/rajneeshbabu)",
    }
)

# ── CSS + Animated Background ──────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Keep Streamlit header + three-dot menu visible ── */
header[data-testid="stHeader"]{background:rgba(5,5,15,0.85)!important;backdrop-filter:blur(8px);border-bottom:1px solid rgba(124,58,237,.15)}
footer{visibility:hidden}

/* ── Base ── */
.stApp{background:#05050f;min-height:100vh;position:relative;overflow-x:hidden}
.block-container{padding:2.5rem 2rem 2rem;max-width:1260px;position:relative;z-index:2}

/* ── AI Circuit-board Canvas Background ── */
#ai-bg{position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none}

/* ── Animated title ── */
@keyframes titlePulse{0%,100%{filter:brightness(1) drop-shadow(0 0 8px #7c3aed66)}
  50%{filter:brightness(1.2) drop-shadow(0 0 22px #a78bfa99)}}
.main-title{font-size:2.6rem;font-weight:900;text-align:center;
  background:linear-gradient(135deg,#a78bfa,#7c3aed,#4f46e5,#06b6d4);
  background-size:300% 300%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;
  animation:titlePulse 3s ease-in-out infinite,gradShift 6s ease infinite;margin-bottom:.1rem}
@keyframes gradShift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
.sub-title{text-align:center;color:#6677aa;font-size:.9rem;margin-bottom:1rem;letter-spacing:.04em}

/* ── Badges ── */
.pipeline-badge{display:inline-block;padding:.18rem .7rem;border-radius:12px;font-size:.72rem;
  font-weight:700;margin:.15rem;background:rgba(124,58,237,.15);
  border:1px solid rgba(124,58,237,.35);color:#a78bfa}
.layer-badge{background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.3);color:#34d399}
.agent-badge{background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.3);color:#fbbf24}

/* ── Chat bubbles with glow ── */
@keyframes msgIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes userGlow{0%,100%{box-shadow:0 0 8px rgba(124,58,237,.3)}50%{box-shadow:0 0 20px rgba(124,58,237,.6)}}
@keyframes botGlow{0%,100%{box-shadow:0 0 6px rgba(6,182,212,.15)}50%{box-shadow:0 0 18px rgba(6,182,212,.35)}}

.user-msg{background:linear-gradient(135deg,rgba(124,58,237,.25),rgba(79,70,229,.18));
  border:1px solid rgba(124,58,237,.4);border-radius:18px 18px 4px 18px;
  padding:.9rem 1.2rem;margin:.5rem 0;max-width:82%;margin-left:auto;
  color:#e8e0ff;font-size:.91rem;line-height:1.7;
  animation:msgIn .35s ease,userGlow 4s ease-in-out infinite}
.bot-msg{background:linear-gradient(135deg,rgba(6,182,212,.06),rgba(255,255,255,.03));
  border:1px solid rgba(6,182,212,.18);border-radius:18px 18px 18px 4px;
  padding:.9rem 1.2rem;margin:.5rem 0;max-width:93%;
  color:#d4e8f0;font-size:.91rem;line-height:1.7;
  animation:msgIn .35s ease,botGlow 4s ease-in-out infinite}
.agent-msg{background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.2);
  border-radius:12px;padding:.7rem 1rem;margin:.3rem 0;font-size:.85rem;color:#d0c090}
.msg-lbl{font-size:.68rem;color:#4455aa;margin-bottom:.25rem;font-weight:700;letter-spacing:.08em;
  text-transform:uppercase}

/* ── Typing pulse ── */
@keyframes pulse{0%,80%,100%{transform:scale(0);opacity:.4}40%{transform:scale(1);opacity:1}}
.typing-dot{display:inline-block;width:8px;height:8px;margin:0 3px;border-radius:50%;
  background:#7c3aed;animation:pulse 1.4s ease-in-out infinite}
.typing-dot:nth-child(2){animation-delay:.2s}
.typing-dot:nth-child(3){animation-delay:.4s}
.typing-wrap{padding:.6rem 1rem;background:rgba(124,58,237,.08);border-radius:12px;
  display:inline-block;border:1px solid rgba(124,58,237,.2)}

/* ── Source citation card ── */
.cite-card{background:rgba(16,185,129,.06);border:1px solid rgba(16,185,129,.18);
  border-radius:10px;padding:.55rem .85rem;margin:.25rem 0;font-size:.8rem;color:#aaa}
.cite-title{color:#34d399;font-weight:700;font-size:.78rem;margin-bottom:.15rem}
.cite-score{font-size:.7rem;color:#666;float:right}

/* ── Metric cards ── */
.m-card{background:linear-gradient(135deg,rgba(124,58,237,.08),rgba(6,182,212,.05));
  border:1px solid rgba(124,58,237,.2);border-radius:14px;padding:.8rem;text-align:center;
  transition:border-color .3s}
.m-card:hover{border-color:rgba(124,58,237,.5)}
.m-val{font-size:1.3rem;font-weight:900;
  background:linear-gradient(135deg,#a78bfa,#06b6d4);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.m-lbl{font-size:.68rem;color:#556;margin-top:.15rem;letter-spacing:.05em}

/* ── RL reward card ── */
.rl-card{background:linear-gradient(135deg,rgba(16,185,129,.1),rgba(6,182,212,.05));
  border:1px solid rgba(16,185,129,.25);border-radius:14px;padding:.8rem;
  text-align:center;margin:.3rem 0}
.rl-val{font-size:1.4rem;font-weight:900;
  background:linear-gradient(135deg,#34d399,#06b6d4);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.rl-bar-wrap{height:6px;background:rgba(255,255,255,.07);border-radius:3px;margin:.4rem 0}
.rl-bar{height:6px;border-radius:3px;
  background:linear-gradient(90deg,#7c3aed,#06b6d4);transition:width .5s ease}

/* ── Token speed badge ── */
@keyframes speedPulse{0%,100%{opacity:.7}50%{opacity:1}}
.speed-badge{font-size:.68rem;color:#06b6d4;display:inline-block;
  animation:speedPulse 1.5s ease-in-out infinite;margin-left:.5rem}

/* ── Feedback buttons ── */
.fb-wrap{margin:.3rem 0;display:flex;gap:.4rem;align-items:center}

/* ── Pipeline step ── */
@keyframes stepGlow{0%,100%{border-color:rgba(124,58,237,.25)}50%{border-color:rgba(124,58,237,.6)}}
.pipe-step{background:rgba(124,58,237,.1);border:1px solid rgba(124,58,237,.25);
  border-radius:8px;padding:.4rem .75rem;font-size:.73rem;color:#c4b5fd;
  text-align:center;display:inline-block;margin:.1rem;animation:stepGlow 3s ease-in-out infinite}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,rgba(4,4,20,.95) 0%,rgba(7,7,28,.95) 100%)!important;
  border-right:1px solid rgba(124,58,237,.15)}
section[data-testid="stSidebar"] label{color:#99aacc!important}
.stButton>button{background:linear-gradient(135deg,#7c3aed,#4f46e5);color:#fff;
  border:none;border-radius:10px;font-weight:700;padding:.48rem 1.4rem;
  transition:all .2s;letter-spacing:.03em}
.stButton>button:hover{transform:translateY(-1px);box-shadow:0 4px 15px rgba(124,58,237,.4)}
div[data-testid="stSelectbox"] label,div[data-testid="stSlider"] label{color:#99aacc!important}
[data-testid="stFileUploader"]{background:rgba(255,255,255,.02);border-radius:12px;
  border:1px dashed rgba(124,58,237,.3)}

/* ── Policy update flash ── */
@keyframes policyFlash{0%{opacity:0;transform:scale(.9)}20%{opacity:1;transform:scale(1.02)}
  80%{opacity:1}100%{opacity:0;transform:scale(.98)}}
.policy-update{background:linear-gradient(135deg,rgba(16,185,129,.15),rgba(6,182,212,.1));
  border:1px solid #34d399;border-radius:10px;padding:.5rem 1rem;font-size:.8rem;
  color:#34d399;text-align:center;animation:policyFlash 2.5s ease forwards}
</style>

<script>
// KEY FIX: create canvas in JS and append directly to document.body
// so it escapes Streamlit's component container
(function initBG(){
  if(document.getElementById('penguin-bg')) return;
  var c=document.createElement('canvas');
  c.id='penguin-bg';
  c.style.cssText='position:fixed;top:0;left:0;width:100vw;height:100vh;'
    +'z-index:0;pointer-events:none;opacity:1;';
  document.body.appendChild(c);

  var ctx=c.getContext('2d'), t=0, W, H;
  var GRID=60, junctions=[], GLYPHS='01アイ∑∇λπθ∈⊕≈∞◆▲●'.split('');
  var cols=[], COL_W=22, NODES=40, nodes=[], bubbles=[];

  function setup(){
    W=c.width=window.innerWidth; H=c.height=window.innerHeight;
    junctions=[];
    for(var x=0;x<W;x+=GRID) for(var y=0;y<H;y+=GRID)
      if(Math.random()<.5) junctions.push({x:x,y:y,p:Math.random()*6.28,active:Math.random()<.3});
    cols=[];
    for(var x=0;x<W;x+=COL_W){
      var len=8+Math.floor(Math.random()*12);
      cols.push({x:x,y:Math.random()*H,spd:.4+Math.random()*1.1,len:len,
        ch:Array.from({length:len},function(){return GLYPHS[Math.floor(Math.random()*GLYPHS.length)];})});
    }
    nodes=[];
    for(var i=0;i<NODES;i++) nodes.push({
      x:Math.random()*W, y:Math.random()*H,
      vx:(Math.random()-.5)*.35, vy:(Math.random()-.5)*.35,
      r:1.5+Math.random()*2, p:Math.random()*6.28, tp:i%3
    });
  }
  setup();
  window.addEventListener('resize',setup);
  setInterval(function(){
    if(bubbles.length<14) bubbles.push({
      x:Math.random()*W*.8+W*.1, y:H+20,
      vy:-(0.4+Math.random()*.55), r:10+Math.random()*16, a:.07+Math.random()*.1
    });
  },1400);

  function draw(){
    t+=.012; W=c.width; H=c.height;
    ctx.clearRect(0,0,W,H);
    // 1 circuit grid
    ctx.strokeStyle='rgba(124,58,237,.055)'; ctx.lineWidth=.5;
    for(var x=0;x<W;x+=GRID){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
    for(var y=0;y<H;y+=GRID){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
    // 2 junctions
    junctions.forEach(function(j){
      var s=Math.sin(t*1.8+j.p);
      ctx.beginPath(); ctx.arc(j.x,j.y,2,0,6.28);
      ctx.fillStyle='rgba(124,58,237,'+(0.15+s*.1)+')'; ctx.fill();
      if(j.active){ctx.beginPath();ctx.arc(j.x,j.y,6+s*3,0,6.28);
        ctx.strokeStyle='rgba(124,58,237,.07)';ctx.lineWidth=1;ctx.stroke();}
    });
    // 3 data streams
    ctx.font='10px monospace';
    cols.forEach(function(col){
      col.y+=col.spd; if(col.y>H+col.len*14) col.y=-col.len*14;
      col.ch.forEach(function(ch,i){
        var a=i===col.ch.length-1?.32:.02+i/col.ch.length*.06;
        ctx.fillStyle=i===col.ch.length-1?'rgba(6,182,212,'+a+')':'rgba(100,60,200,'+a+')';
        ctx.fillText(ch,col.x,col.y+i*14);
      });
    });
    // 4 node connections + signal packets
    for(var i=0;i<NODES;i++) for(var j=i+1;j<NODES;j++){
      var dx=nodes[i].x-nodes[j].x, dy=nodes[i].y-nodes[j].y, d=Math.sqrt(dx*dx+dy*dy);
      if(d<140){
        var a=(1-d/140)*.13;
        var g=ctx.createLinearGradient(nodes[i].x,nodes[i].y,nodes[j].x,nodes[j].y);
        g.addColorStop(0,'rgba(124,58,237,'+a+')');
        g.addColorStop(.5,'rgba(6,182,212,'+(a*.8)+')');
        g.addColorStop(1,'rgba(124,58,237,'+a+')');
        ctx.strokeStyle=g; ctx.lineWidth=.7;
        ctx.beginPath(); ctx.moveTo(nodes[i].x,nodes[i].y); ctx.lineTo(nodes[j].x,nodes[j].y); ctx.stroke();
        if(d<90&&Math.sin(t*3+i*j*.1)>.88){
          var f=(t*.7)%1;
          ctx.fillStyle='rgba(6,182,212,.8)';
          ctx.beginPath(); ctx.arc(nodes[i].x-dx*f,nodes[i].y-dy*f,1.8,0,6.28); ctx.fill();
        }
      }
    }
    // 5 nodes
    nodes.forEach(function(n){
      n.x+=n.vx; n.y+=n.vy;
      if(n.x<0||n.x>W)n.vx*=-1; if(n.y<0||n.y>H)n.vy*=-1;
      var s=Math.sin(t*1.6+n.p);
      var gw=ctx.createRadialGradient(n.x,n.y,0,n.x,n.y,n.r*5);
      gw.addColorStop(0,'rgba(167,139,250,.38)'); gw.addColorStop(1,'rgba(0,0,0,0)');
      ctx.fillStyle=gw; ctx.beginPath(); ctx.arc(n.x,n.y,n.r*5,0,6.28); ctx.fill();
      ctx.fillStyle='rgba(167,139,250,'+(0.48+s*.2)+')';
      if(n.tp===0){ctx.beginPath();ctx.arc(n.x,n.y,n.r,0,6.28);ctx.fill();}
      else if(n.tp===1){ctx.fillRect(n.x-n.r,n.y-n.r,n.r*2,n.r*2);}
      else{ctx.beginPath();ctx.moveTo(n.x,n.y-n.r*1.4);ctx.lineTo(n.x+n.r*1.4,n.y);
           ctx.lineTo(n.x,n.y+n.r*1.4);ctx.lineTo(n.x-n.r*1.4,n.y);ctx.closePath();ctx.fill();}
    });
    // 6 floating chat bubbles
    bubbles=bubbles.filter(function(b){return b.y>-60;});
    bubbles.forEach(function(b){
      b.y+=b.vy;
      ctx.save(); ctx.globalAlpha=b.a*(Math.min(1,(H-b.y)/200));
      ctx.strokeStyle='rgba(6,182,212,.55)'; ctx.lineWidth=1;
      ctx.beginPath();
      ctx.roundRect?ctx.roundRect(b.x,b.y,b.r*2.8,b.r,.8*b.r):ctx.rect(b.x,b.y,b.r*2.8,b.r);
      ctx.stroke(); ctx.restore();
    });
    requestAnimationFrame(draw);
  }
  draw();
  // Re-run setup after Streamlit finishes rendering (it may shift layout)
  setTimeout(setup, 800);
})();
</script>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — SEMANTIC CHUNKING + DEDUPLICATION + METADATA
# ══════════════════════════════════════════════════════════════════════════════
def semantic_chunk(text: str, max_chunk: int = 500, overlap: int = 60) -> list[str]:
    """
    Semantic chunking: split on paragraph/sentence boundaries, not fixed tokens.
    Payoff: 30-50% better retrieval precision (from PDF).
    """
    # Split on paragraph breaks first
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks, current, current_len = [], [], 0
    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para)
        for sent in sentences:
            wc = len(sent.split())
            if current_len + wc > max_chunk and current:
                chunks.append(' '.join(current))
                # Overlap: keep last few sentences
                overlap_words = ' '.join(current).split()[-overlap:]
                current = [' '.join(overlap_words)] if overlap_words else []
                current_len = len(current[0].split()) if current else 0
            current.append(sent)
            current_len += wc
    if current:
        chunks.append(' '.join(current))
    return [c for c in chunks if len(c.strip()) > 30]


def sha256_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_and_index_document(uploaded_file):
    """
    Full Layer 1+2 ingestion pipeline:
    Parse → Semantic chunk → SHA-256 dedup → Metadata enrich → FAISS + BM25
    """
    import tempfile, os
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.schema import Document
    from rank_bm25 import BM25Okapi

    suffix = "." + uploaded_file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix in (".docx", ".doc"):
            loader = Docx2txtLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
        raw_docs = loader.load()
    finally:
        os.unlink(tmp_path)

    # ── Semantic chunking per page/doc ─────────────────────────────────────
    all_chunks, seen_hashes = [], set()
    for page_num, doc in enumerate(raw_docs):
        page_chunks = semantic_chunk(doc.page_content)
        for i, chunk_text in enumerate(page_chunks):
            chunk_hash = sha256_hash(chunk_text)
            # SHA-256 deduplication — cuts index size by 40%+
            if chunk_hash in seen_hashes:
                continue
            seen_hashes.add(chunk_hash)
            # Detect section from first line
            lines = chunk_text.strip().split('\n')
            section = lines[0][:60] if lines else "body"
            # Metadata enrichment
            metadata = {
                "source":     uploaded_file.name,
                "page":       page_num + 1,
                "chunk_idx":  i,
                "section":    section,
                "chunk_hash": chunk_hash,
                "char_count": len(chunk_text),
                "doc_type":   suffix.lstrip('.'),
            }
            all_chunks.append(Document(page_content=chunk_text, metadata=metadata))

    if not all_chunks:
        return None, None, 0, 0

    # ── Layer 2: Embeddings + FAISS ────────────────────────────────────────
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(all_chunks, embeddings)

    # ── BM25 sparse index ─────────────────────────────────────────────────
    bm25_corpus = [c.page_content.lower().split() for c in all_chunks]
    bm25_index  = BM25Okapi(bm25_corpus)

    orig_count = sum(len(semantic_chunk(d.page_content)) for d in raw_docs)
    dedup_saved = max(0, orig_count - len(all_chunks))
    return vectorstore, (bm25_index, all_chunks), len(all_chunks), dedup_saved


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — HYBRID RETRIEVAL: BM25 + FAISS + RRF FUSION
# ══════════════════════════════════════════════════════════════════════════════
def bm25_search(bm25_tuple, query: str, top_k: int = 20):
    bm25_index, chunks = bm25_tuple
    tokens = query.lower().split()
    scores = bm25_index.get_scores(tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(chunks[i], float(scores[i])) for i in top_idx if scores[i] > 0]


def reciprocal_rank_fusion(dense_docs, sparse_results, k: int = 60) -> list:
    """
    RRF fusion: merge dense (FAISS) + sparse (BM25) ranked lists.
    Payoff: 10× smaller search space, better recall on rare keywords.
    """
    rrf_scores: dict = {}
    doc_map: dict    = {}

    for rank, doc in enumerate(dense_docs):
        key = doc.page_content[:120]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rank + k)
        doc_map[key] = doc

    for rank, (doc, _) in enumerate(sparse_results):
        key = doc.page_content[:120]
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rank + k)
        doc_map[key] = doc

    sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]


def hybrid_retrieve(vectorstore, bm25_tuple, query: str, top_k: int = 20):
    """Full hybrid retrieval pipeline (Layer 3)."""
    # Dense ANN search
    dense_docs = vectorstore.similarity_search(query, k=top_k)
    # Sparse BM25 search
    sparse_docs = bm25_search(bm25_tuple, query, top_k=top_k)
    # RRF fusion
    fused = reciprocal_rank_fusion(dense_docs, sparse_docs)
    return fused[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — CROSS-ENCODER RERANKING (top-20 → top-5)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_cross_encoder():
    from sentence_transformers import CrossEncoder
    # mmarco-mMiniLMv2: multilingual cross-encoder trained on MS MARCO (13 languages)
    # Scores query–passage relevance across language boundaries
    return CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", max_length=512)


def rerank_chunks(query: str, candidates: list, top_k: int = 5) -> list[tuple]:
    """
    Cross-encoder reranking: top-20 fused → top-5 by relevance.
    Payoff: +20-40% top-5 accuracy.
    """
    if not candidates:
        return []
    try:
        reranker = get_cross_encoder()
        pairs    = [(query, doc.page_content) for doc in candidates]
        scores   = reranker.predict(pairs)
        ranked   = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
    except Exception:
        # Fallback: no reranking, return top_k
        return [(doc, 0.0) for doc in candidates[:top_k]]


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — CONTEXT ASSEMBLY: MMR + COMPRESSION + CITATIONS
# ══════════════════════════════════════════════════════════════════════════════
def apply_mmr(vectorstore, query: str, k: int = 5, fetch_k: int = 20) -> list:
    """Max Marginal Relevance — removes redundant chunks."""
    return vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)


def compress_chunk(text: str, max_tokens: int = 300) -> str:
    """Extractive compression: keep first N words if chunk is too long."""
    words = text.split()
    if len(words) <= max_tokens:
        return text
    # Keep first 70% of max_tokens (most important sentences are first)
    keep = int(max_tokens * 0.7)
    return ' '.join(words[:keep]) + " …"


def assemble_context(ranked_chunks: list[tuple]) -> tuple[str, list[dict]]:
    """
    Build context string with citation metadata.
    Payoff: 25% fewer hallucinations via citation grounding.
    """
    context_parts, citations = [], []
    for i, (doc, score) in enumerate(ranked_chunks):
        compressed = compress_chunk(doc.page_content, max_tokens=300)
        m = doc.metadata
        citation = {
            "idx":      i + 1,
            "source":   m.get("source", "document"),
            "page":     m.get("page", "?"),
            "section":  m.get("section", "")[:50],
            "score":    round(float(score), 3),
            "preview":  compressed[:180],
        }
        citations.append(citation)
        context_parts.append(
            f"[Chunk {i+1} | {m.get('source','doc')} p.{m.get('page','?')}]:\n{compressed}"
        )
    return "\n\n---\n\n".join(context_parts), citations


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 6 — LRU QUERY CACHE
# ══════════════════════════════════════════════════════════════════════════════
_query_cache: dict = {}   # {cache_key: (citations, context, timestamp)}
CACHE_TTL = 3600          # 1 hour

def cache_key(query: str, doc_name: str) -> str:
    return sha256_hash(f"{doc_name}::{query.lower().strip()}")

def cached_retrieve(query, vectorstore, bm25_tuple, doc_name, top_k=5):
    """Check LRU cache before running full retrieval pipeline."""
    key = cache_key(query, doc_name)
    if key in _query_cache:
        result, ts = _query_cache[key]
        if time.time() - ts < CACHE_TTL:
            return result, True   # (result, from_cache)
    # Full pipeline
    fused   = hybrid_retrieve(vectorstore, bm25_tuple, query, top_k=20)
    ranked  = rerank_chunks(query, fused, top_k=top_k)
    context, citations = assemble_context(ranked)
    result = (context, citations, ranked)
    _query_cache[key] = (result, time.time())
    return result, False


# ══════════════════════════════════════════════════════════════════════════════
# AGENTIC RAG — ReAct Loop (Reason → Act → Observe → Repeat)
# ══════════════════════════════════════════════════════════════════════════════
AGENT_SYSTEM = """You are an Agentic RAG assistant. Reason step-by-step and use tools to retrieve information from the document before answering.

Available tools:
  search[query]      — Search the document for relevant chunks
  summarize[topic]   — Get a focused summary on a specific topic
  verify[claim]      — Check if a specific claim is in the document

Use EXACTLY this format (no deviations):

Thought: what do I need to find out?
Action: search[what to search for]
Observation: [tool results inserted here]
... (repeat Thought/Action/Observation up to 3 times if needed)
Final Answer: [comprehensive, grounded answer with citations like [p.X]]

If the document does not contain the answer, say so clearly in Final Answer.

Question: {question}"""


def parse_agent_action(text: str) -> tuple[str | None, str | None]:
    """Extract Action: tool[query] from LLM output."""
    match = re.search(r'Action:\s*(search|summarize|verify)\[(.+?)\]', text, re.IGNORECASE)
    if match:
        return match.group(1).lower(), match.group(2).strip()
    return None, None


def agent_tool_call(tool: str, query: str, vectorstore, bm25_tuple, top_k=4) -> str:
    """Execute a tool and return formatted observation."""
    fused  = hybrid_retrieve(vectorstore, bm25_tuple, query, top_k=12)
    ranked = rerank_chunks(query, fused, top_k=top_k)
    if not ranked:
        return "No relevant content found in the document."
    parts = []
    for doc, score in ranked:
        m = doc.metadata
        parts.append(
            f"[p.{m.get('page','?')} | score:{score:.2f}] {compress_chunk(doc.page_content, 180)}"
        )
    if tool == "summarize":
        return f"Summary on '{query}':\n" + "\n".join(parts)
    elif tool == "verify":
        return f"Verification evidence for '{query}':\n" + "\n".join(parts)
    return "\n".join(parts)


def run_agentic_rag(client, question: str, vectorstore, bm25_tuple,
                    model: str, temperature: float, max_hops: int = 3):
    """
    Full ReAct agentic loop:
    1. LLM reasons and picks a tool
    2. Tool runs hybrid retrieval + reranking
    3. Observation fed back to LLM
    4. Repeats until Final Answer or max_hops
    """
    messages = [
        {"role": "system", "content": "You are an Agentic RAG assistant that uses tools to retrieve and reason over documents."},
        {"role": "user",   "content": AGENT_SYSTEM.format(question=question)},
    ]
    agent_trace = []
    final_answer = None

    for hop in range(max_hops):
        response = client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=800, stream=False
        )
        llm_output = response.choices[0].message.content.strip()
        agent_trace.append({"hop": hop+1, "llm": llm_output})

        # Check for Final Answer
        if "Final Answer:" in llm_output:
            final_answer = llm_output.split("Final Answer:")[-1].strip()
            break

        # Parse tool call
        tool, tool_query = parse_agent_action(llm_output)
        if tool and tool_query:
            observation = agent_tool_call(tool, tool_query, vectorstore, bm25_tuple)
            agent_trace[-1]["tool"]        = tool
            agent_trace[-1]["tool_query"]  = tool_query
            agent_trace[-1]["observation"] = observation
            # Append to message history
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({"role": "user",      "content": f"Observation: {observation}\n\nContinue your reasoning."})
        else:
            # No tool call — treat as final answer
            final_answer = llm_output
            break

    if not final_answer:
        final_answer = "I was unable to find a confident answer in the document after searching."

    return final_answer, agent_trace


# ══════════════════════════════════════════════════════════════════════════════
# RAGAS-STYLE FAITHFULNESS EVAL
# ══════════════════════════════════════════════════════════════════════════════
def faithfulness_score(answer: str, context: str) -> float:
    """
    Lightweight faithfulness check: what fraction of answer sentences
    have at least one supporting word in the context?
    """
    ctx_words = set(context.lower().split())
    sentences = re.split(r'[.!?]', answer)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return 1.0
    supported = 0
    for sent in sentences:
        words = set(sent.lower().split())
        # At least 3 non-stopword words overlap
        meaningful = words - {'the','a','an','is','are','was','were','in','on',
                              'of','to','and','or','it','this','that','with'}
        if len(meaningful & ctx_words) >= 3:
            supported += 1
    return round(supported / len(sentences), 2)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED RESOURCES
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    # paraphrase-multilingual-MiniLM-L12-v2:
    #   - 50+ languages in a shared embedding space
    #   - Hindi/French/German queries match English documents natively
    #   - 471 MB, runs on CPU, drop-in replacement for all-MiniLM-L6-v2
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource(show_spinner=False)
def get_groq_client(api_key: str):
    from groq import Groq
    return Groq(api_key=api_key)

def run_safety_guard(client, text: str) -> dict:
    """
    Llama Guard 3 20B — content safety check.
    Returns {"safe": bool, "category": str, "raw": str}
    """
    try:
        resp = client.chat.completions.create(
            model="meta-llama/llama-guard-3-20b",
            messages=[{"role": "user", "content": text}],
            max_tokens=50,
            temperature=0,
        )
        raw = resp.choices[0].message.content.strip()
        safe = raw.lower().startswith("safe")
        category = raw if not safe else "safe"
        return {"safe": safe, "category": category, "raw": raw}
    except Exception as e:
        # Guard model unavailable — fail open (don't block response)
        return {"safe": True, "category": "guard_unavailable", "raw": str(e)}

def transcribe_audio(client, audio_bytes: bytes, filename: str) -> str:
    """Whisper Large v3 Turbo — multilingual speech to text via Groq.
    No language= param → Whisper auto-detects from audio (supports 99 languages).
    """
    try:
        transcription = client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model="whisper-large-v3-turbo",
            response_format="text",
            # language omitted → auto-detect (Hindi, French, German, etc.)
        )
        return transcription
    except Exception as e:
        return f"⚠️ Transcription error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# RLHF — REINFORCEMENT LEARNING FROM HUMAN FEEDBACK
#
# Architecture (session-level, no GPU needed):
#   1. Reward Model   — scores each response from human 👍/👎 + auto heuristics
#   2. Value Baseline — running mean reward (reduces variance like PPO critic)
#   3. Advantage      — reward - baseline (positive = better than average)
#   4. Policy Update  — adjust temperature & system prompt via advantage signal
#   5. Replay Buffer  — store (prompt, response, reward) for in-context steering
# ══════════════════════════════════════════════════════════════════════════════
import math as _math

RLHF_LOG_PATH = os.path.join(os.getcwd(), "rlhf_replay_buffer.jsonl")

# ── Reward Model ──────────────────────────────────────────────────────────────
def reward_model_score(response: str, human_signal: int) -> float:
    """
    Composite reward:
      human_signal  : +1 (👍) or -1 (👎)  — weight 0.6
      length_score  : penalise too short (<50w) or too long (>600w) — weight 0.2
      diversity     : unique-word ratio (avoids repetitive responses) — weight 0.1
      specificity   : presence of numbers/code/bullets — weight 0.1
    Returns reward in [-1, +1]
    """
    words = response.split()
    n     = max(len(words), 1)

    # Length score: peak at 150 words, soft penalty outside [50, 600]
    ideal   = 150
    length_score = _math.exp(-((n - ideal) ** 2) / (2 * 120 ** 2))

    # Diversity: unique word ratio
    diversity = len(set(w.lower() for w in words)) / n

    # Specificity: numeric/code/bullet signals
    has_num    = any(c.isdigit() for c in response)
    has_code   = "```" in response or "    " in response
    has_bullet = any(response.count(m) > 1 for m in ["•", "-", "*", "\n-"])
    specificity = (int(has_num) + int(has_code) + int(has_bullet)) / 3

    auto = 0.2 * length_score + 0.1 * diversity + 0.1 * specificity
    r    = 0.6 * human_signal + auto
    return round(max(-1.0, min(1.0, r)), 4)

# ── Value Baseline (PPO-style critic) ────────────────────────────────────────
def compute_value_baseline() -> float:
    """Exponential moving average of past rewards — the 'critic' baseline."""
    rewards = st.session_state.rl_rewards
    if not rewards:
        return 0.0
    alpha = 0.3   # EMA decay
    ema = rewards[0]
    for r in rewards[1:]:
        ema = alpha * r + (1 - alpha) * ema
    return round(ema, 4)

def compute_advantage(reward: float, baseline: float) -> float:
    """Advantage = reward - baseline. Positive → better than average."""
    return round(reward - baseline, 4)

# ── Replay Buffer ────────────────────────────────────────────────────────────
def rlhf_save_to_buffer(prompt: str, response: str, reward: float, advantage: float):
    record = {"prompt": prompt, "response": response,
              "reward": reward, "advantage": advantage}
    try:
        with open(RLHF_LOG_PATH, "a") as f:
            f.write(_json.dumps(record) + "\n")
    except Exception:
        pass

def rlhf_load_buffer() -> list[dict]:
    try:
        with open(RLHF_LOG_PATH) as f:
            return [_json.loads(l) for l in f if l.strip()]
    except Exception:
        return []

# ── In-context Policy Steering ───────────────────────────────────────────────
def rlhf_build_system_addon() -> str:
    """
    Inject top-k high-advantage responses as few-shot examples into the system
    prompt — this steers the model toward human-preferred outputs without retraining.
    """
    buf = rlhf_load_buffer()
    # Sort by advantage descending, take top 3
    top = sorted(buf, key=lambda x: x["advantage"], reverse=True)[:3]
    if not top:
        return ""
    examples = "\n\n".join(
        f"[High-reward example | advantage={e['advantage']:+.2f}]\n"
        f"Q: {e['prompt'][:100]}\nA: {e['response'][:220]}"
        for e in top
    )
    return (
        "\n\n--- RLHF Steering: Preferred Response Examples ---\n"
        + examples
        + "\n--- End RLHF Examples ---"
    )

# ── Policy Update (temperature adaptation) ───────────────────────────────────
def rl_adapt_temperature(base_temp: float) -> tuple[float, str]:
    """
    PPO-inspired clipped temperature update:
      • Positive advantage → lower temp (exploit — stay close to what worked)
      • Negative advantage → higher temp (explore — try different responses)
      • Clip ratio: max shift ±0.20 per update (prevents policy collapse)
    """
    rewards = st.session_state.rl_rewards
    if not rewards:
        return base_temp, ""

    baseline  = compute_value_baseline()
    last_r    = rewards[-1]
    advantage = compute_advantage(last_r, baseline)

    # PPO clip: limit policy step size
    clip      = 0.20
    delta     = max(-clip, min(clip, -advantage * 0.18))   # negative: high adv → lower temp
    eff       = round(min(1.0, max(0.05, base_temp + delta)), 3)

    n_buf     = len(rlhf_load_buffer())
    direction = "↓ exploiting" if delta < 0 else "↑ exploring"
    msg = (
        f"🧠 RLHF policy update {direction}\n"
        f"Reward={last_r:+.3f} · Baseline={baseline:+.3f} · Adv={advantage:+.3f}\n"
        f"Δtemp={delta:+.3f} → eff_temp={eff} · Buffer={n_buf}"
    )
    return eff, msg

def rl_cumulative_reward() -> float:
    r = st.session_state.rl_rewards
    return round(sum(r), 3) if r else 0.0

DOMAIN_PROMPTS = {
    "🩺 Medical Advisor":    "You are an expert medical AI. Provide evidence-based information. Always advise consulting a physician for diagnosis or treatment.",
    "⚖️ Legal Advisor":      "You are an expert legal AI. Provide clear legal analysis. Always recommend consulting a licensed attorney for specific advice.",
    "💰 Finance Advisor":    "You are an expert financial AI. Provide financial education and analysis. Note this is not personalized financial advice.",
    "💻 Code Assistant":     "You are an expert software engineer. Help with code, algorithms, debugging, and best practices. Write clean, well-commented code.",
    "🧘 Fitness & Wellness": "You are an expert fitness coach. Provide evidence-based advice on exercise, nutrition, and mental health.",
    "🔬 Research Assistant": "You are an expert research assistant. Help with literature reviews, methodology, and academic writing.",
}
GENERAL_PROMPT = "You are a highly capable and friendly AI assistant. Be helpful, concise when needed, and detailed when depth is required."


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "messages":       [], "vectorstore": None, "bm25_tuple": None,
        "doc_name":       None, "doc_chunks": 0, "dedup_saved": 0,
        "msg_count":      0, "total_tokens": 0, "cache_hits": 0,
        "faith_scores":   [],
        # RL state
        "rl_rewards":     [],          # list of +1/-1 per rated message
        "rl_temperature": None,        # None = use slider value
        "rl_episode":     0,           # number of feedback events
        "rl_feedback":    {},          # {msg_index: "up"/"down"}
        "rl_policy_msg":  "",          # notification text
        "token_speeds":   [],          # tokens/sec per response
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v
init_state()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🤖 RAG Chatbot")
    st.markdown("<span class='pipeline-badge'>Production RAG</span> <span class='pipeline-badge agent-badge'>Agentic RAG</span> <span class='pipeline-badge' style='background:rgba(6,182,212,.15);border-color:rgba(6,182,212,.35);color:#67e8f9'>🌐 Multilingual</span>", unsafe_allow_html=True)
    st.markdown("---")

    _server_key = _get_server_api_key()
    if _server_key:
        api_key = _server_key
    else:
        api_key = st.text_input("Groq API Key", type="password",
            value="", placeholder="gsk_...", help="Free at console.groq.com")
        if not api_key:
            st.warning("⚠️ Enter Groq API key to start", icon="🔑")

    st.markdown("---")
    mode = st.radio("**Chat Mode**", [
        "🤖 General Chat",
        "📄 Advanced RAG",
        "🕵️ Agentic RAG",
        "🧠 Domain Expert",
    ])

    st.markdown("---")

    # ── Model selector ──────────────────────────────────────────────────────────
    MODELS = {
        "⚡ Llama 4 Scout 17B  [NEW]":      "meta-llama/llama-4-scout-17b-16e-instruct",
        "🧠 Llama 3.3 70B (default)":       "llama-3.3-70b-versatile",
        "🌐 Qwen 3 32B  [NEW]":             "qwen/qwen3-32b",
        "🔥 Llama 3.1 8B Instant":          "llama-3.1-8b-instant",
        "🔀 Mixtral 8×7B 32K":              "mixtral-8x7b-32768",
        "💎 Llama 3 70B 8K":                "llama3-70b-8192",
    }
    model_label = st.selectbox("**Chat Model**", list(MODELS.keys()))
    model = MODELS[model_label]

    # Safety guard toggle (Llama Guard 3 20B)
    use_safety_guard = st.toggle(
        "🛡️ Safety Guard (Llama Guard 3 20B)",
        value=False,
        help="Runs Llama Guard 3 20B on every response to flag unsafe content before displaying it."
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.05)
    max_tokens  = st.slider("Max tokens", 256, 4096, 1024, 128)

    st.markdown("---")
    if mode == "🧠 Domain Expert":
        domain = st.selectbox("**Expert Persona**", list(DOMAIN_PROMPTS.keys()))
    else:
        domain = None

    # Document upload for RAG modes
    if mode in ("📄 Advanced RAG", "🕵️ Agentic RAG"):
        st.markdown("**Upload Document**")
        uploaded = st.file_uploader(
            "PDF, TXT, DOCX", type=["pdf","txt","docx","md"],
            label_visibility="collapsed"
        )
        if uploaded and uploaded.name != st.session_state.doc_name:
            progress = st.progress(0, text="Parsing document...")
            progress.progress(20, text="Semantic chunking...")
            vs, bm25_t, n_chunks, dedup = load_and_index_document(uploaded)
            progress.progress(70, text="Building FAISS + BM25 index...")
            time.sleep(0.3)
            progress.progress(100, text="Done!")
            if vs:
                st.session_state.vectorstore = vs
                st.session_state.bm25_tuple  = bm25_t
                st.session_state.doc_name    = uploaded.name
                st.session_state.doc_chunks  = n_chunks
                st.session_state.dedup_saved = dedup
                st.success(f"✅ {n_chunks} chunks | 🗑️ {dedup} dupes removed")
            progress.empty()
        elif st.session_state.doc_name:
            st.info(f"📄 {st.session_state.doc_name}\n{st.session_state.doc_chunks} chunks")
    else:
        uploaded = None

    # ── Whisper audio upload (always visible) ───────────────────────────────
    st.markdown("---")
    st.markdown("**🎙️ Whisper Audio → Text**")
    audio_file = st.file_uploader(
        "Upload audio (mp3, wav, m4a, webm)", type=["mp3","wav","m4a","webm","ogg","flac"],
        label_visibility="collapsed", key="audio_upload"
    )
    if audio_file and api_key:
        if st.button("▶ Transcribe", use_container_width=True):
            with st.spinner("Whisper Large v3 Turbo transcribing…"):
                _client = get_groq_client(api_key)
                transcript = transcribe_audio(_client, audio_file.read(), audio_file.name)
            st.text_area("Transcript", transcript, height=120)
            st.success("✅ Transcription complete — copy text above")

    if mode in ("📄 Advanced RAG", "🕵️ Agentic RAG"):
        st.markdown("---")
        st.markdown("**RAG Pipeline**")
        for step in ["Semantic chunk","SHA-256 dedup","BM25+FAISS","RRF fusion","mMiniLM rerank","MMR+cite","LRU cache"]:
            st.markdown(f"<span class='pipeline-badge layer-badge' style='display:block;margin:.1rem 0'>✓ {step}</span>", unsafe_allow_html=True)
        st.markdown("<span class='pipeline-badge' style='display:block;margin:.1rem 0;background:rgba(6,182,212,.12);border-color:rgba(6,182,212,.3);color:#67e8f9'>🌐 50+ language embeddings</span>", unsafe_allow_html=True)

    # ── Live stats ──────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1: st.markdown(f"<div class='m-card'><div class='m-val'>{st.session_state.msg_count}</div><div class='m-lbl'>Messages</div></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='m-card'><div class='m-val'>{st.session_state.cache_hits}</div><div class='m-lbl'>Cache hits</div></div>", unsafe_allow_html=True)
    avg_faith = round(sum(st.session_state.faith_scores)/len(st.session_state.faith_scores), 2) if st.session_state.faith_scores else "-"
    avg_speed = round(sum(st.session_state.token_speeds)/len(st.session_state.token_speeds), 0) if st.session_state.token_speeds else "-"
    c3, c4 = st.columns(2)
    with c3: st.markdown(f"<div class='m-card' style='margin-top:.4rem'><div class='m-val'>{avg_faith}</div><div class='m-lbl'>Faithfulness</div></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='m-card' style='margin-top:.4rem'><div class='m-val'>{avg_speed}</div><div class='m-lbl'>tok/s</div></div>", unsafe_allow_html=True)

    # ── RLHF Dashboard ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**🧠 RLHF Policy**")
    rewards  = st.session_state.rl_rewards
    n_fb     = len(rewards)
    cum_r    = rl_cumulative_reward()
    baseline = compute_value_baseline()
    eff_t, _ = rl_adapt_temperature(0.5)
    n_buf    = len(rlhf_load_buffer())
    pos_r    = sum(1 for r in rewards if r > 0)
    neg_r    = sum(1 for r in rewards if r <= 0)
    bar_w    = int(pos_r / max(n_fb, 1) * 100)
    st.markdown(
        f"<div class='rl-card'>"
        f"<div class='rl-val'>{cum_r:+.2f}</div>"
        f"<div style='font-size:.68rem;color:#6677aa;margin-bottom:.25rem'>Cumulative Reward · {n_fb} ratings</div>"
        f"<div class='rl-bar-wrap'><div class='rl-bar' style='width:{bar_w}%'></div></div>"
        f"<div style='font-size:.65rem;color:#6677aa;margin-top:.25rem'>"
        f"👍 {pos_r} &nbsp;·&nbsp; 👎 {neg_r} &nbsp;·&nbsp; baseline={baseline:+.3f}</div>"
        f"<div style='font-size:.65rem;color:#34d399;margin-top:.15rem'>"
        f"🔄 Replay buffer: {n_buf} &nbsp;·&nbsp; eff_temp≈{eff_t:.2f}</div>"
        f"</div>", unsafe_allow_html=True
    )
    if rewards:
        import pandas as pd
        rewards_df = pd.DataFrame({
            "episode":    list(range(1, n_fb + 1)),
            "cum_reward": [round(sum(rewards[:i+1]), 3) for i in range(n_fb)],
        }).set_index("episode")
        st.line_chart(rewards_df, height=95, use_container_width=True)

    if st.session_state.rl_policy_msg:
        st.markdown(f"<div class='policy-update'>{st.session_state.rl_policy_msg}</div>", unsafe_allow_html=True)

    # Export RLHF replay buffer
    if n_buf > 0:
        buf_data = "\n".join(_json.dumps(r) for r in rlhf_load_buffer())
        st.download_button(
            "⬇️ Export RLHF Buffer",
            data=buf_data,
            file_name="rlhf_replay_buffer.jsonl",
            mime="application/jsonl",
            use_container_width=True,
            help=f"Download {n_buf} (prompt, response, reward, advantage) tuples for offline RLHF fine-tuning"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.msg_count     = 0
        st.session_state.total_tokens  = 0
        st.session_state.faith_scores  = []
        st.session_state.rl_rewards    = []
        st.session_state.rl_feedback   = {}
        st.session_state.rl_episode    = 0
        st.session_state.rl_policy_msg = ""
        st.session_state.token_speeds  = []
        st.rerun()

    st.markdown("---")
    st.caption("Llama 4 Scout · Qwen 3 · Groq · BM25 · FAISS · RRF · Whisper")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='main-title'>🐧 RAG Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Production RAG · Agentic RAG · Llama 4 Scout · Qwen 3 32B · Whisper · Groq</div>", unsafe_allow_html=True)

# Pipeline ribbon
st.markdown(
    "<div style='text-align:center;margin-bottom:.8rem'>" +
    "".join([f"<span class='pipe-step'>{s}</span>" for s in
        ["Query","→ Meta filter","→ BM25+ANN","→ RRF merge","→ Rerank","→ Compress+Cite","→ LLM"]]) +
    "</div>", unsafe_allow_html=True
)

# Mode indicator
mode_info = {
    "🤖 General Chat":   ("#a78bfa","General conversational AI with memory"),
    "📄 Advanced RAG":   ("#34d399","Hybrid retrieval · Cross-encoder rerank · Citations · Cache"),
    "🕵️ Agentic RAG":   ("#fbbf24","ReAct agent · Multi-hop retrieval · Tool use · Self-reasoning"),
    "🧠 Domain Expert":  ("#60a5fa",f"Expert persona: {domain or 'select in sidebar'}"),
}
mc, mdesc = mode_info[mode]
st.markdown(
    f"<div style='text-align:center;margin-bottom:1.2rem'>"
    f"<span style='border:1px solid {mc}55;border-radius:20px;padding:.3rem 1rem;"
    f"color:{mc};font-size:.82rem;font-weight:600;background:rgba(0,0,0,.3)'>"
    f"● {mode} &nbsp;·&nbsp; {mdesc}</span></div>",
    unsafe_allow_html=True
)

# ── Welcome cards ──────────────────────────────────────────────────────────────
if not st.session_state.messages:
    cols = st.columns(4)
    cards = [
        ("🤖","General Chat","Conversational AI with 20-turn sliding memory window"),
        ("📄","Advanced RAG","BM25+FAISS hybrid → RRF → Cross-encoder → MMR → Citations"),
        ("🕵️","Agentic RAG","ReAct loop: LLM reasons, picks tools, multi-hop retrieval"),
        ("🧠","Domain Expert","Medical · Legal · Finance · Code · Fitness · Research"),
    ]
    for col,(icon,title,desc) in zip(cols,cards):
        with col:
            st.markdown(f"<div class='m-card' style='padding:1.1rem'>"
                f"<div style='font-size:2rem'>{icon}</div>"
                f"<div style='color:#a78bfa;font-weight:700;margin:.4rem 0;font-size:.95rem'>{title}</div>"
                f"<div style='color:#666;font-size:.78rem;line-height:1.5'>{desc}</div>"
                f"</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ── Chat history ───────────────────────────────────────────────────────────────
bot_msg_indices = [i for i,m in enumerate(st.session_state.messages) if m["role"]=="assistant"]

for msg_idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(f"<div class='msg-lbl'>YOU</div><div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        lbl = msg.get("mode","RAG CHATBOT")
        st.markdown(
            f"<div class='msg-lbl'>🐧 {lbl.upper()}</div>"
            f"<div class='bot-msg'>{msg['content']}</div>",
            unsafe_allow_html=True
        )
        # Speed badge shown separately below the bubble
        if msg.get("tok_speed"):
            st.markdown(
                f"<span class='speed-badge' style='font-size:.67rem;color:#06b6d4;margin-left:.3rem'>"
                f"⚡ {msg['tok_speed']:.0f} tok/s</span>",
                unsafe_allow_html=True
            )

        # Faithfulness badge
        if "faithfulness" in msg:
            color = "#34d399" if msg["faithfulness"] >= 0.7 else "#fbbf24" if msg["faithfulness"] >= 0.4 else "#e74c3c"
            st.markdown(f"<span style='font-size:.7rem;color:{color}'>⚡ Faithfulness: {msg['faithfulness']}</span>", unsafe_allow_html=True)

        # ── RL Feedback buttons ────────────────────────────────────────────
        fb_key  = f"fb_{msg_idx}"
        given   = st.session_state.rl_feedback.get(msg_idx)
        if given:
            emoji = "👍" if given == "up" else "👎"
            st.markdown(
                f"<span style='font-size:.72rem;color:#445'>{emoji} Feedback recorded"
                f" &nbsp;·&nbsp; reward: {'+1' if given=='up' else '-1'}</span>",
                unsafe_allow_html=True
            )
        else:
            col_up, col_dn, col_sp = st.columns([1,1,8])
            with col_up:
                if st.button("👍", key=f"up_{msg_idx}", help="Good response — RLHF reward +1"):
                    prev_user = ""
                    if msg_idx > 0 and st.session_state.messages[msg_idx-1]["role"] == "user":
                        prev_user = st.session_state.messages[msg_idx-1]["content"]
                    # Reward model: human +1 + auto heuristics
                    reward    = reward_model_score(msg["content"], human_signal=1)
                    baseline  = compute_value_baseline()
                    advantage = compute_advantage(reward, baseline)
                    st.session_state.rl_rewards.append(reward)
                    st.session_state.rl_feedback[msg_idx] = "up"
                    st.session_state.rl_episode += 1
                    rlhf_save_to_buffer(prev_user, msg["content"], reward, advantage)
                    _, policy_msg = rl_adapt_temperature(temperature)
                    st.session_state.rl_policy_msg = policy_msg
                    st.rerun()
            with col_dn:
                if st.button("👎", key=f"dn_{msg_idx}", help="Bad response — RLHF reward -1"):
                    prev_user = ""
                    if msg_idx > 0 and st.session_state.messages[msg_idx-1]["role"] == "user":
                        prev_user = st.session_state.messages[msg_idx-1]["content"]
                    # Reward model: human -1 + auto heuristics
                    reward    = reward_model_score(msg["content"], human_signal=-1)
                    baseline  = compute_value_baseline()
                    advantage = compute_advantage(reward, baseline)
                    st.session_state.rl_rewards.append(reward)
                    st.session_state.rl_feedback[msg_idx] = "down"
                    st.session_state.rl_episode += 1
                    rlhf_save_to_buffer(prev_user, msg["content"], reward, advantage)
                    _, policy_msg = rl_adapt_temperature(temperature)
                    st.session_state.rl_policy_msg = policy_msg
                    st.rerun()

        # Citations
        if msg.get("citations"):
            with st.expander(f"📚 {len(msg['citations'])} source chunks (reranked)", expanded=False):
                for c in msg["citations"]:
                    st.markdown(
                        f"<div class='cite-card'>"
                        f"<span class='cite-title'>Chunk {c['idx']} · {c['source']} · p.{c['page']}</span>"
                        f"<span class='cite-score'>score: {c['score']}</span>"
                        f"<div style='margin-top:.3rem'>{c['preview']}…</div>"
                        f"</div>", unsafe_allow_html=True
                    )
        # Agent trace
        if msg.get("agent_trace"):
            with st.expander(f"🕵️ Agent reasoning trace ({len(msg['agent_trace'])} hops)", expanded=False):
                for step in msg["agent_trace"]:
                    st.markdown(f"**Hop {step['hop']}**")
                    if "tool" in step:
                        st.markdown(f"<div class='agent-msg'>🔧 Tool: <b>{step['tool']}[{step['tool_query']}]</b></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='agent-msg' style='font-size:.8rem'>{step['llm'][:500]}</div>", unsafe_allow_html=True)

# ── Voice-to-Text (mic-recorder + Whisper) ────────────────────────────────────
if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""
if "voice_pending" not in st.session_state:
    st.session_state.voice_pending = False

# Try streamlit-mic-recorder (records audio → Whisper transcription via Groq)
_mic_loaded = False
try:
    from streamlit_mic_recorder import mic_recorder as _mic_recorder
    _mic_loaded = True
except ImportError:
    pass

_voice_col, _voice_label_col = st.columns([1, 8])
with _voice_col:
    if _mic_loaded:
        _audio_data = _mic_recorder(
            start_prompt="🎙️",
            stop_prompt="⏹️",
            just_once=True,
            use_container_width=True,
            format="wav",
            key="mic_recorder_main",
        )
        if _audio_data and _audio_data.get("bytes") and api_key:
            with st.spinner("Transcribing…"):
                try:
                    _tmp_client = get_groq_client(api_key)
                    _transcript = transcribe_audio(_tmp_client, _audio_data["bytes"], "voice.wav")
                    if _transcript:
                        st.session_state.voice_transcript = _transcript.strip()
                        st.session_state.voice_pending = True
                        st.rerun()
                except Exception as _e:
                    st.warning(f"Transcription failed: {_e}")
    else:
        # Fallback: Web Speech API mic button (browser-native, no Whisper needed)
        _mic_html = """
<style>
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:transparent;}
  #mw{display:flex;align-items:center;gap:8px;font-family:'Inter',sans-serif;}
  #mb{background:linear-gradient(135deg,#1e1b4b,#312e81);border:1.5px solid #4c1d95;
      border-radius:50%;width:42px;height:42px;cursor:pointer;display:flex;
      align-items:center;justify-content:center;font-size:18px;color:#a78bfa;
      transition:all .2s;flex-shrink:0;}
  #mb.on{background:linear-gradient(135deg,#4c1d95,#7c3aed);border-color:#7c3aed;
         box-shadow:0 0 0 6px rgba(124,58,237,.25),0 0 18px rgba(124,58,237,.5);
         animation:pr 1.2s ease-in-out infinite;color:#fff;}
  @keyframes pr{0%{box-shadow:0 0 0 0 rgba(124,58,237,.5)}
                70%{box-shadow:0 0 0 10px rgba(124,58,237,0)}
                100%{box-shadow:0 0 0 0 rgba(124,58,237,0)}}
  #ms{font-size:.75rem;color:#7c3aed;min-width:120px;}
  #mt{font-size:.8rem;color:#c4b5fd;background:rgba(30,27,75,.8);border:1px solid #4c1d95;
      border-radius:8px;padding:5px 9px;max-width:380px;word-break:break-word;display:none;}
  #msb{background:linear-gradient(135deg,#7c3aed,#4c1d95);border:none;border-radius:8px;
       color:#fff;font-size:.75rem;padding:4px 10px;cursor:pointer;display:none;}
</style>
<div id="mw">
  <button id="mb">🎙️</button>
  <span id="ms">Click to speak</span>
  <div id="mt"></div>
  <button id="msb">✉️ Use this</button>
</div>
<script>
(function(){
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  const btn=document.getElementById('mb'),st=document.getElementById('ms'),
        tr=document.getElementById('mt'),sb=document.getElementById('msb');
  if(!SR){st.textContent='⚠️ Use Chrome for mic';btn.style.opacity='.4';btn.disabled=true;return;}
  const rec=new SR();rec.continuous=false;rec.interimResults=true;rec.lang='en-US';
  let on=false,final='';
  btn.onclick=()=>{on?rec.stop():rec.start();};
  rec.onstart=()=>{on=true;btn.classList.add('on');btn.textContent='⏹️';st.textContent='🔴 Listening…';final='';tr.style.display='none';sb.style.display='none';};
  rec.onresult=(e)=>{let interim='';
    for(let i=e.resultIndex;i<e.results.length;i++){
      const t=e.results[i][0].transcript;
      if(e.results[i].isFinal)final+=t+' ';else interim+=t;}
    tr.style.display='block';tr.textContent=(final+interim).trim()||'…';};
  rec.onend=()=>{on=false;btn.classList.remove('on');btn.textContent='🎙️';
    if(final.trim()){st.textContent='✅ Click "Use this"';tr.textContent=final.trim();tr.style.display='block';sb.style.display='inline-block';}
    else{st.textContent='Click to speak';}};
  rec.onerror=(e)=>{on=false;btn.classList.remove('on');btn.textContent='🎙️';
    const m={'not-allowed':'❌ Allow mic in browser','no-speech':'🔇 Nothing heard','aborted':'Click to speak'};
    st.textContent=m[e.error]||'Error: '+e.error;};
  sb.onclick=()=>{
    const txt=final.trim();if(!txt)return;
    try{window.parent.sessionStorage.setItem('penguin_voice',txt);}catch(e){}
    try{
      const inp=window.parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
      if(inp){
        const nativeInputValueSetter=Object.getOwnPropertyDescriptor(window.parent.HTMLTextAreaElement.prototype,'value').set;
        nativeInputValueSetter.call(inp,txt);
        inp.dispatchEvent(new Event('input',{bubbles:true}));
        inp.focus();
      }
    }catch(e){}
    st.textContent='✅ Filled in chat box below!';sb.style.display='none';};
})();
</script>"""
        components.html(_mic_html, height=58, scrolling=False)

with _voice_label_col:
    st.caption("🎙️ **Voice mic**")

# Show captured voice transcript
if st.session_state.get("voice_pending") and st.session_state.voice_transcript:
    st.success(f"🎙️ **Voice:** {st.session_state.voice_transcript}")

# ── Input ──────────────────────────────────────────────────────────────────────
ph = {
    "🤖 General Chat":  "Ask me anything...",
    "📄 Advanced RAG":  "Ask a question about your document (hybrid retrieval + reranking)...",
    "🕵️ Agentic RAG":  "Ask a complex question — the agent will reason and multi-hop retrieve...",
    "🧠 Domain Expert": f"Ask your {(domain or '').split(' ',1)[-1]} question...",
}

# Use voice transcript as default if available
_chat_placeholder = ph.get(mode, "Ask anything...")
user_input = st.chat_input(_chat_placeholder)

# Also accept voice input if nothing typed but voice is pending
if not user_input and st.session_state.get("voice_pending") and st.session_state.voice_transcript:
    user_input = st.session_state.voice_transcript
    st.session_state.voice_transcript = ""
    st.session_state.voice_pending = False

if user_input:
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()
    if mode in ("📄 Advanced RAG","🕵️ Agentic RAG") and not st.session_state.vectorstore:
        st.warning("Upload a document first.")
        st.stop()

    # Clear any pending voice state
    st.session_state.voice_transcript = ""
    st.session_state.voice_pending = False

    st.session_state.messages.append({"role":"user","content":user_input})
    st.session_state.msg_count += 1
    st.markdown(f"<div class='msg-lbl'>YOU</div><div class='user-msg'>{user_input}</div>", unsafe_allow_html=True)

    client = get_groq_client(api_key)
    full_response = ""
    citations     = None
    agent_trace   = None
    faith_score   = None
    from_cache    = False
    tok_speed     = None

    # ── RL: adapt temperature based on feedback history ────────────────────
    eff_temp, rl_msg = rl_adapt_temperature(temperature)
    if rl_msg and st.session_state.rl_episode > 0:
        st.session_state.rl_policy_msg = rl_msg

    def stream_with_metrics(stream_obj, placeholder):
        """Stream tokens, update placeholder. Returns (full_text, tok/s)."""
        text = ""
        t0 = time.time()
        tok_count = 0
        for chunk in stream_obj:
            delta = chunk.choices[0].delta.content
            if delta:
                text += delta
                tok_count += len(delta.split())
                elapsed = max(time.time() - t0, 0.001)
                speed = tok_count / elapsed
                placeholder.markdown(
                    f"<div class='bot-msg'>{text}"
                    f"<span style='color:#7c3aed'>▌</span></div>"
                    f"<span class='speed-badge'>⚡ {speed:.0f} tok/s</span>",
                    unsafe_allow_html=True
                )
        elapsed = max(time.time() - t0, 0.001)
        return text, max(tok_count / elapsed, 0.1)

    try:
        # ── Typing indicator ───────────────────────────────────────────────
        thinking = st.empty()
        thinking.markdown(
            "<div class='typing-wrap'>"
            "<span class='typing-dot'></span><span class='typing-dot'></span><span class='typing-dot'></span>"
            "&nbsp;<span style='font-size:.75rem;color:#6677aa'>Thinking…</span></div>",
            unsafe_allow_html=True
        )

        # ── AGENTIC RAG ────────────────────────────────────────────────────
        if mode == "🕵️ Agentic RAG":
            thinking.empty()
            detected_lang = detect_language(user_input)
            lang_name = LANG_NAMES.get(detected_lang, detected_lang.upper())
            st.markdown(
                f"<span style='font-size:.7rem;color:#67e8f9'>🌐 Detected language: <b>{lang_name}</b></span>",
                unsafe_allow_html=True
            )
            st.markdown("<div class='msg-lbl'>🕵️ AGENTIC RAG</div>", unsafe_allow_html=True)
            with st.spinner("Agent reasoning & retrieving…"):
                t0 = time.time()
                full_response, agent_trace = run_agentic_rag(
                    client, user_input,
                    st.session_state.vectorstore,
                    st.session_state.bm25_tuple,
                    model, eff_temp, max_hops=3
                )
                tok_speed = len(full_response.split()) / max(time.time()-t0, 0.001)
            st.markdown(f"<div class='bot-msg'>{full_response}</div>", unsafe_allow_html=True)

        # ── ADVANCED RAG ───────────────────────────────────────────────────
        elif mode == "📄 Advanced RAG":
            thinking.empty()
            # ── Language detection ─────────────────────────────────────────
            detected_lang = detect_language(user_input)
            lang_name = LANG_NAMES.get(detected_lang, detected_lang.upper())
            st.markdown(
                f"<span style='font-size:.7rem;color:#67e8f9'>🌐 Detected language: <b>{lang_name}</b></span>",
                unsafe_allow_html=True
            )
            with st.spinner("🔍 Hybrid retrieval → Reranking → Assembling…"):
                result, from_cache = cached_retrieve(
                    user_input,
                    st.session_state.vectorstore,
                    st.session_state.bm25_tuple,
                    st.session_state.doc_name,
                    top_k=5
                )
            context, citations, ranked = result
            if from_cache:
                st.session_state.cache_hits += 1
                st.toast("⚡ Cache hit!", icon="⚡")

            system_prompt = (
                "You are a RAG assistant in Advanced mode. Answer ONLY from the provided context. "
                "Cite chunks as [p.X] or [Chunk N]. If unsure, say so. "
                f"IMPORTANT: The user is writing in {lang_name} — respond in the same language.\n\n"
                f"RETRIEVED CONTEXT:\n{context}"
            )
            history = st.session_state.messages[-20:]
            messages = [{"role":"system","content":system_prompt}]
            for m in history:
                messages.append({"role":m["role"],"content":m["content"]})

            st.markdown("<div class='msg-lbl'>🐧 ADVANCED RAG · HYBRID+RERANK</div>", unsafe_allow_html=True)
            placeholder = st.empty()
            stream = client.chat.completions.create(
                model=model, messages=messages, temperature=eff_temp,
                max_tokens=max_tokens, stream=True
            )
            full_response, tok_speed = stream_with_metrics(stream, placeholder)
            placeholder.markdown(f"<div class='bot-msg'>{full_response}</div>", unsafe_allow_html=True)
            faith_score = faithfulness_score(full_response, context)
            st.session_state.faith_scores.append(faith_score)
            color = "#34d399" if faith_score >= 0.7 else "#fbbf24"
            st.markdown(f"<span style='font-size:.7rem;color:{color}'>⚡ Faithfulness: {faith_score} &nbsp;|&nbsp; {'⚡ Cache' if from_cache else '🔍 Live'} &nbsp;|&nbsp; {len(citations)} chunks &nbsp;|&nbsp; temp={eff_temp:.2f}</span>", unsafe_allow_html=True)

        # ── GENERAL CHAT ───────────────────────────────────────────────────
        elif mode == "🤖 General Chat":
            thinking.empty()
            history = st.session_state.messages[-40:]
            rlhf_addon = rlhf_build_system_addon()
            messages = [{"role":"system","content":GENERAL_PROMPT + rlhf_addon}]
            for m in history:
                messages.append({"role":m["role"],"content":m["content"]})
            st.markdown("<div class='msg-lbl'>🤖 RAG CHATBOT</div>", unsafe_allow_html=True)
            placeholder = st.empty()
            stream = client.chat.completions.create(
                model=model, messages=messages, temperature=eff_temp,
                max_tokens=max_tokens, stream=True
            )
            full_response, tok_speed = stream_with_metrics(stream, placeholder)
            placeholder.markdown(f"<div class='bot-msg'>{full_response}</div>", unsafe_allow_html=True)

        # ── DOMAIN EXPERT ──────────────────────────────────────────────────
        else:
            thinking.empty()
            sys_prompt = DOMAIN_PROMPTS.get(domain, GENERAL_PROMPT)
            history = st.session_state.messages[-40:]
            messages = [{"role":"system","content":sys_prompt}]
            for m in history:
                messages.append({"role":m["role"],"content":m["content"]})
            lbl = domain or "Expert"
            st.markdown(f"<div class='msg-lbl'>🧠 {lbl.upper()}</div>", unsafe_allow_html=True)
            placeholder = st.empty()
            stream = client.chat.completions.create(
                model=model, messages=messages, temperature=eff_temp,
                max_tokens=max_tokens, stream=True
            )
            full_response, tok_speed = stream_with_metrics(stream, placeholder)
            placeholder.markdown(f"<div class='bot-msg'>{full_response}</div>", unsafe_allow_html=True)

        if tok_speed:
            st.session_state.token_speeds.append(tok_speed)

        # ── Safety Guard (Llama Guard 3 20B) ──────────────────────────────
        if use_safety_guard and full_response:
            with st.spinner("🛡️ Safety Guard checking…"):
                guard = run_safety_guard(client, full_response)
            if not guard["safe"]:
                st.warning(f"🚨 **Safety Guard flagged this response** — Category: `{guard['category']}`\n\nThe response has been hidden. Please rephrase your question.")
                full_response = f"⚠️ [Response blocked by Llama Guard 3 20B — {guard['category']}]"
            else:
                st.markdown("<span style='font-size:.7rem;color:#34d399'>✅ Safety Guard: Safe</span>", unsafe_allow_html=True)

        # Store message
        bot_msg = {
            "role":      "assistant",
            "content":   full_response,
            "mode":      mode,
            "tok_speed": tok_speed,
        }
        if citations:               bot_msg["citations"]    = citations
        if agent_trace:             bot_msg["agent_trace"]  = agent_trace
        if faith_score is not None: bot_msg["faithfulness"] = faith_score
        st.session_state.messages.append(bot_msg)
        st.session_state.total_tokens += len(user_input.split()) + len(full_response.split())

    except Exception as e:
        try: thinking.empty()
        except: pass
        st.error(f"❌ {e}")
        if "api_key" in str(e).lower() or "auth" in str(e).lower():
            st.info("Get a free key at console.groq.com")
        if "model" in str(e).lower() or "not found" in str(e).lower():
            st.info("💡 This model may not be available yet on your Groq plan. Try Llama 3.3 70B instead.")
