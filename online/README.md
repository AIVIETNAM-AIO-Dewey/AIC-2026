# AIC-2026 Multimodal Online Retrieval Studio

A high-speed, client-server multimodal video retrieval platform for the **Ho Chi Minh City AI Challenge (AIC 2026)**.

Features a **FastAPI GPU backend** with warm PyTorch models and a **modern dark-mode Web Studio UI** with instant weight re-tuning, bounding box overlays, and filmstrip timeline navigation.

---

## 🚀 Key Highlights

1. **Client-Server Architecture**:
   - Heavy AI models (`SigLIP-2`, `BGE-M3`, `BGE-Reranker-v2-m3`) and memory-mapped matrices (177k keyframes, 435k DAM objects) stay warm in GPU VRAM.
   - No cold-start overhead per query.

2. **Instant (< 6ms) CPU Re-Fusion (Branch Caching)**:
   - When a query is run, raw retrieval hits from all 4 branches (`vis`, `dam`, `asr`, `ocr`) are cached in-memory.
   - Dragging modality weight sliders and clicking **`[ ⚡ Re-Fuse Pool ]`** re-scores and re-ranks 300 candidates on CPU in **~5ms** without re-embedding!

3. **3 Query Modes**:
   - **Auto-Run**: Natural language query $\rightarrow$ LLM parses $\rightarrow$ runs search immediately.
   - **Edit-then-Send**: Natural language query $\rightarrow$ LLM decomposes to JSON $\rightarrow$ inspect & modify JSON/weights $\rightarrow$ execute.
   - **Direct JSON**: Paste structured JSON directly (e.g., from web Gemini chat) $\rightarrow$ execute (skips LLM step).

4. **Multi-LLM Query Decomposer with Graceful Fallback**:
   - **Gemini 3.6 Flash / 3.7**: High-precision cloud parser with a UI toggle button.
   - **Local Qwen 2.5 7B (via Ollama)**: Offline competition mode on Mac Metal/GPU.
   - **Rule-based Fallback**: Zero-dependency offline backup if APIs are unreachable.

5. **Interactive Frame Inspector**:
   - Large keyframe preview with canvas overlay for **DAM object bounding boxes**.
   - Explainability panel: Video ID, Frame Number, PTS timestamp, score breakdown, full ASR dialogue, and DAM description.
   - **Filmstrip Timeline Slider**: Scrollable chronological filmstrip of all keyframes in the video with `[←]` / `[→]` keyboard navigation.
   - **Sticky Submission Bar**: One-click `[ 📋 Copy Submission ]` in official format (`<VIDEO_ID>, <FRAME_IDX>`).

---

## 🛠️ Quick Start Guide

### 1. Create Environment & Install Dependencies
Navigate to the repository root:

**Using Python `venv` (Zero install needed)**:
```bash
python3 -m venv aic
source aic/bin/activate
pip install --upgrade pip
pip install -r online/requirements.txt
```

**Or Using `conda`**:
```bash
conda create -n aic python=3.10 -y
conda activate aic
pip install -r online/requirements.txt
```

*(Optional for Gemini Cloud Parser)*: Create a `.env` file in the repo root with your Gemini API key:
```bash
echo "GEMINI_API_KEY=your_key_here" >> .env
```

*(Optional for local Qwen parser)*: Make sure Ollama is installed and running:
```bash
ollama run qwen2.5:7b
```

---

### 2. Configure Dataset Directories

Open [`online/configs/server_config.yaml`](online/configs/server_config.yaml) and adjust the paths to match your local machine:

```yaml
paths:
  # 1. Root directory containing keyframe images (e.g. L21_V001/001.jpg)
  keyframes_root: "/Users/khoale/Downloads/AIC_Challenger/data/keyframes"

  # 2. Unified index directory (npy vectors + jsonl metadata)
  unified_index: "/Users/khoale/Downloads/AIC_HCM/unified_index"

  # 3. ASR segments directory (per-video JSONL transcripts)
  asr_segments: "/Users/khoale/Downloads/AIC_HCM/artifacts/asr_segments"

  # 4. DAM descriptions directory (per-video object descriptions)
  dam_descriptions: "/Users/khoale/Downloads/AIC_HCM/artifacts/dam_descriptions"

  # 5. Keyframes CSV mapping directory
  map_keyframes: "/Users/khoale/Downloads/AIC_HCM/map-keyframes"
```

---

### 3. Launch the Server

Navigate to the repository root and start the server:

```bash
cd /Users/khoale/Downloads/AIC_Challenger/shared_repo/AIC-2026
python -m online.server
```

*(Or specify your conda python path)*:
```bash
/opt/anaconda3/envs/speech_to_text/bin/python -m online.server
```

When ready, the server will output:
```
2026-08-21 13:13:33 [INFO] ✅ Server fully warmed and ready in 23.7s!
INFO: Uvicorn running on http://127.0.0.1:8890 (Press CTRL+C to quit)
```

---

### 4. Open the Web Studio UI

Open your browser and navigate to:
👉 **[http://localhost:8890](http://localhost:8890)**

---

## 📁 Data Structure Reference

For full details on the dataset directory layout and metadata schemas, refer to [`online/DATA_STRUCTURE.md`](DATA_STRUCTURE.md).

```
<keyframes_root>/
  ├── L21_V001/
  │   ├── 001.jpg
  │   ├── 002.jpg
  │   └── ...
  └── L30_V020/

<unified_index>/
  ├── keyframes_visual_vectors.f16.npy    # (177321, 768) float16 (SigLIP-2)
  ├── keyframes_speech_vectors.f16.npy    # (177321, 1024) float16 (BGE-M3 Speech)
  ├── dam_vectors.f16.npy                 # (435713, 1024) float16 (DAM Objects)
  ├── keyframes_metadata.jsonl            # 177,321 keyframe metadata records
  └── dam_metadata.jsonl                  # 435,713 localized object records
```

---

## ⌨️ Useful Keyboard Shortcuts (in Frame Inspector)

| Key | Action |
|---|---|
| <kbd>←</kbd> | Previous keyframe in video timeline |
| <kbd>→</kbd> | Next keyframe in video timeline |
| <kbd>Enter</kbd> | Toggle keyframe in/out of current submission |
| <kbd>Esc</kbd> | Close Frame Inspector modal |
| <kbd>Ctrl</kbd> + <kbd>Enter</kbd> | Execute search from query / JSON textarea |
