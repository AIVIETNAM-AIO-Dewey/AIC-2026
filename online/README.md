# 🎬 AIC-2026 Online Multimodal Video Retrieval Engine

High-performance, low-latency (< 80ms) multimodal video retrieval engine built for the **Ho Chi Minh City AI Challenge (AIC)** and **Video Browser Showdown (VBS)**.

---

## 🏛️ Architecture Overview

The system operates as a **two-stage cascaded hybrid retrieval funnel**:

```
                       Raw User Query (Vietnamese / English)
                                        │
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │ 1. Query Decomposer & Task Classifier               │
             │    (Gemini Flash API + Local Offline Fallback)      │
             └──────────────────────────┬──────────────────────────┘
                                        │
     ┌──────────────────┬───────────────┴──────────────┬──────────────────┐
     ▼                  ▼                              ▼                  ▼
┌──────────────┐ ┌──────────────┐             ┌────────────────┐ ┌────────────────┐
│ Channel 1:   │ │ Channel 2:   │             │ Channel 3:     │ │ Channel 4:     │
│ Visual Scene │ │ DAM Objects  │             │ Audio Speech   │ │ Screen Text    │
│ (SigLIP-2)   │ │ (BGE-M3 Dense│             │ (BGE-M3 + BM25)│ │ (BM25-OCR)     │
│ 768-d Vector │ │ 1024-d Vector│             │ 1024-d Vector  │ │ Text Match     │
└──────┬───────┘ └──────┬───────┘             └────────┬───────┘ └────────┬───────┘
       │                │                              │                  │
       └────────────────┴──────────────┬───────────────┴──────────────────┘
                                       ▼
             ┌─────────────────────────────────────────────────────┐
             │ Stage 1 Funnel: Weighted Reciprocal Rank Fusion     │
             │ (177,321 Master Keyframes ──► Top-50 Pool in ~25ms) │
             └─────────────────────────┬───────────────────────────┘
                                       │ Top-50 Candidates
                                       ▼
             ┌─────────────────────────────────────────────────────┐
             │ Stage 2: Precision Cross-Attention & Reasoner Layer │
             │ • KIS:   bge-reranker-v2-m3 Cross-Encoder (~20ms)   │
             │ • TRAKE: Temporal Monotonicity Filter (t1 < t2)     │
             │ • VQA:   Extractive LLM Question Answering          │
             └─────────────────────────┬───────────────────────────┘
                                       ▼
               Output: Ranked Keyframes + Bounding Boxes + AIC Submission Code
```

---

## 📂 Directory Layout & Data Assumptions

The retrieval pipeline assumes the following directory structure:

```
AIC_HCM/
├── map-keyframes/               <── 873 CSVs mapping n -> pts_time -> frame_idx (177,321 rows)
├── artifacts/
│   ├── dam_descriptions/       <── 873 JSONLs with 435,713 50-word DAM object captions
│   ├── asr_segments/           <── 873 JSONLs with 55,168 speech transcripts & keyframe links
│   └── ocr_transcripts/        <── (Optional) On-screen text JSONLs (Default w_ocr=0.0)
└── qdrant_db/                   <── Embedded on-disk Qdrant vector database

[AIC2026] Scene Embeddings/
└── *.f16.npy, *.jsonl          <── 873 NumPy matrices with 177,321 768-d SigLIP-2 visual vectors

keyframes/                       <── (Optional local images) keyframes/<VIDEO_ID>/<00n>.jpg
```

---

## 🚀 Quickstart & Installation

### 1. Python Environment Setup
```bash
# Activate environment or create virtual env
conda activate speech_to_text
# or: python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r online/requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(If no API key is provided, the engine automatically uses the fast local rule-based query parser and offline VQA heuristic extractor).*

---

## 🛠️ Step 1: One-Time Qdrant Indexing

To index the entire dataset into the local embedded Qdrant store:

```bash
# Index all 873 videos into Qdrant (Runs on Mac MPS GPU / CUDA)
python -m online.src.index.run_indexing

# Or index a small subset for quick testing:
python -m online.src.index.run_indexing --videos L21_V001 L21_V002
```

---

## 🌐 Step 2: Launch the Interactive Web Application

Start the FastAPI web server:

```bash
uvicorn online.src.ui.app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at **`http://localhost:8000`**.

### 🌟 Web UI Features:
1. **Task Switcher**: Tabs for **KIS** (Known-Item Search), **TRAKE** (Temporal Sequences), and **VQA** (Question Answering).
2. **Editable Query Inspector (Human-in-the-Loop)**: Click **"Parse Sub-Queries"** to view and manually adjust the decomposed visual scene, DAM objects, speech keywords, and channel weight sliders before searching!
3. **Canvas Bounding Box Overlays**: Detected DAM objects are rendered with **translucent colored highlights and entity tags** directly on the keyframe images.
4. **Collapsible Explainability Drawer ("Inspect Evidence")**:
   - 🖼️ SigLIP-2 Visual Cosine Similarity
   - 🔍 Detailed DAM Object Captions & Bounding Box coordinates
   - 🎙️ Spoken ASR Audio Transcript with exact playback timestamp interval
   - 🧠 BGE-Reranker Cross-Attention Score
5. **1-Click Submission Code Copy**: Click to copy `<VIDEO_ID>, <FRAME_IDX>` directly to your clipboard.

---

## 🧪 Testing & Benchmarks

Run the automated test suite:

```bash
# 1. Test Qdrant indexing & vector search
python -m unittest online/tests/test_qdrant_indexing.py

# 2. Test End-to-End retrieval across KIS and VQA
python -m unittest online/tests/test_retrieval_e2e.py

# 3. Benchmark latency (Budget: < 80ms)
python -m online.tests.benchmark_latency
```
