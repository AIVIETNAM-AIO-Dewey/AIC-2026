#!/usr/bin/env python3
"""
Generate an interactive, high-performance HTML inspection gallery for an entire video.
Allows smooth scrubbing across all frames, real-time search across captions/OCR,
and dynamic toggling of DAM bounding boxes and EasyOCR text overlays.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full interactive HTML gallery for a video.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("/Users/khoale/Downloads/AIC_Challenger/downloaded_artifacts"),
        help="Path to directory containing downloaded artifact streams.",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default="L21_V001",
        help="Video ID to inspect (e.g. L21_V001).",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("/Users/khoale/Downloads/AIC_Challenger/video_gallery_L21_V001.html"),
        help="Output HTML file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    video_id = args.video_id
    output_html = args.output_html

    unified_file = artifacts_dir / "unified_metadata" / f"{video_id}.jsonl"
    desc_file = artifacts_dir / "descriptions" / f"{video_id}.jsonl"
    ocr_file = artifacts_dir / "ocr_transcripts" / f"{video_id}.jsonl"
    safetensors_file = artifacts_dir / "scene_embeddings" / f"{video_id}.safetensors"
    zip_file = artifacts_dir / "keyframes_zips" / f"{video_id}.zip"

    # Verify paths
    for p, name in [
        (unified_file, "unified_metadata"),
        (desc_file, "descriptions"),
        (ocr_file, "ocr_transcripts"),
        (safetensors_file, "scene_embeddings"),
        (zip_file, "keyframes_zips"),
    ]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing {name} at {p}")

    print(f"Loading metadata for {video_id}...")
    with open(unified_file, "r", encoding="utf-8") as f:
        unified_records = [json.loads(line) for line in f]

    with open(desc_file, "r", encoding="utf-8") as f:
        desc_records = [json.loads(line) for line in f]

    with open(ocr_file, "r", encoding="utf-8") as f:
        ocr_records = [json.loads(line) for line in f]

    mat_dict = load_file(str(safetensors_file))
    embeddings = mat_dict["embeddings"]  # Shape: (N, 768)

    # Extract keyframe images to gallery directory next to HTML
    assets_dir = output_html.parent / f"{video_id}_frames"
    assets_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {len(unified_records)} keyframes to {assets_dir}...")
    with zipfile.ZipFile(zip_file, "r") as zf:
        for member in zf.namelist():
            if member.endswith(".jpg"):
                filename = Path(member).name
                target_path = assets_dir / filename
                if not target_path.exists():
                    target_path.write_bytes(zf.read(member))

    # Build bundled payload for client-side viewer
    frames_data = []
    for i, meta in enumerate(unified_records):
        desc = desc_records[i] if i < len(desc_records) else {}
        ocr = ocr_records[i] if i < len(ocr_records) else {}
        vec = embeddings[meta["embedding_row"]].astype(np.float32)
        norm = float(np.linalg.norm(vec))
        
        filename = Path(meta["image_relpath"]).name
        img_rel = f"{video_id}_frames/{filename}"
        
        # Format regions
        regions = []
        for r_idx, reg in enumerate(desc.get("regions", [])):
            det = reg.get("detector", {})
            label = det.get("class_name") or det.get("class_entity") or f"Object {r_idx+1}"
            score = det.get("score")
            caption = reg.get("caption", {}).get("description_en", "")
            
            bbox = reg.get("bbox_xyxy_px") or reg.get("bbox_yxyx_norm")
            coord_type = "px" if reg.get("bbox_xyxy_px") else "norm"
            regions.append({
                "label": label,
                "score": score,
                "caption": caption,
                "bbox": bbox,
                "coord_type": coord_type
            })

        # Format OCR spans
        ocr_spans = []
        for span in ocr.get("spans", []):
            poly = span.get("polygon_norm") or span.get("polygon")
            text = span.get("normalized_text") or span.get("raw_text", "")
            conf = span.get("confidence")
            ocr_spans.append({
                "text": text,
                "confidence": conf,
                "polygon": poly
            })

        frames_data.append({
            "idx": i,
            "uid": meta["frame_uid"],
            "point_id": meta["point_id"],
            "pts": meta["pts_time_s"],
            "img_src": img_rel,
            "dam_summary": meta.get("dam_summary_en", ""),
            "ocr_text": meta.get("ocr_text", ""),
            "regions": regions,
            "ocr_spans": ocr_spans,
            "l2_norm": norm,
        })

    json_payload = json.dumps(frames_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AIC-2026 Multi-Modal Video Inspector: {video_id}</title>
<style>
  :root {{
    --bg: #121316;
    --panel-bg: #1a1c22;
    --border: #2c2f38;
    --accent-dam: #00ffcc;
    --accent-ocr: #ffea00;
    --text: #f0f2f5;
    --text-muted: #9ba1b0;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  body {{ background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}
  
  /* Header */
  header {{ background: var(--panel-bg); border-bottom: 1px solid var(--border); padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; gap: 20px; }}
  .brand {{ display: flex; align-items: center; gap: 12px; font-weight: 700; font-size: 1.1rem; }}
  .badge {{ background: #2a2e39; color: var(--accent-dam); padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-family: monospace; }}
  
  /* Controls */
  .controls {{ display: flex; align-items: center; gap: 16px; flex: 1; max-width: 600px; }}
  .search-box {{ flex: 1; background: #0f1013; border: 1px solid var(--border); padding: 8px 14px; border-radius: 8px; color: var(--text); font-size: 0.9rem; }}
  .search-box:focus {{ outline: none; border-color: var(--accent-dam); }}
  .toggles {{ display: flex; align-items: center; gap: 14px; font-size: 0.85rem; }}
  .toggle-label {{ display: flex; align-items: center; gap: 6px; cursor: pointer; }}
  
  /* Main Container */
  .main-content {{ flex: 1; display: grid; grid-template-columns: 1.4fr 1fr; gap: 16px; padding: 16px; overflow: hidden; }}
  
  /* Canvas Viewer */
  .viewer-panel {{ background: #000; border-radius: 12px; border: 1px solid var(--border); display: flex; flex-direction: column; position: relative; overflow: hidden; }}
  .canvas-wrapper {{ flex: 1; display: flex; align-items: center; justify-content: center; position: relative; overflow: hidden; }}
  canvas {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
  
  /* Playback Bar */
  .playback-bar {{ background: var(--panel-bg); border-top: 1px solid var(--border); padding: 12px 18px; display: flex; align-items: center; gap: 16px; }}
  .btn {{ background: #2c2f38; border: none; color: var(--text); padding: 8px 14px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 0.9rem; }}
  .btn:hover {{ background: #3a3f4d; }}
  .scrubber {{ flex: 1; accent-color: var(--accent-dam); cursor: pointer; height: 6px; }}
  .time-display {{ font-family: monospace; font-size: 0.9rem; color: var(--accent-dam); min-width: 90px; }}
  
  /* Metadata Panel */
  .meta-panel {{ background: var(--panel-bg); border-radius: 12px; border: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }}
  .meta-header {{ padding: 14px 18px; border-bottom: 1px solid var(--border); font-weight: 600; font-size: 1rem; display: flex; justify-content: space-between; }}
  .meta-body {{ flex: 1; overflow-y: auto; padding: 18px; display: flex; flex-direction: column; gap: 18px; }}
  
  .card {{ background: #131418; border: 1px solid var(--border); border-radius: 8px; padding: 14px; }}
  .card-title {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 8px; font-weight: 700; }}
  
  .dam-region {{ margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid #22252e; }}
  .dam-region:last-child {{ margin-bottom: 0; padding-bottom: 0; border-bottom: none; }}
  .region-tag {{ display: inline-flex; align-items: center; gap: 6px; font-size: 0.78rem; font-weight: 700; padding: 3px 8px; border-radius: 4px; background: rgba(0,255,204,0.15); color: var(--accent-dam); margin-bottom: 6px; }}
  .region-desc {{ font-size: 0.9rem; line-height: 1.45; color: #e1e4ea; }}
  
  .ocr-content {{ font-size: 0.95rem; color: var(--accent-ocr); line-height: 1.4; font-weight: 500; }}
  .vector-stat {{ font-family: monospace; font-size: 0.85rem; color: #73e6b5; }}
</style>
</head>
<body>

<header>
  <div class="brand">
    <span>🎬 AIC-2026 Frame Inspector</span>
    <span class="badge">{video_id} ({len(unified_records)} frames)</span>
  </div>
  
  <div class="controls">
    <input type="text" id="searchInput" class="search-box" placeholder="🔍 Search captions or OCR across all frames (e.g. xe buýt, skyline, logo)...">
    <div class="toggles">
      <label class="toggle-label"><input type="checkbox" id="toggleDAM" checked> <span style="color:var(--accent-dam)">DAM BBoxes</span></label>
      <label class="toggle-label"><input type="checkbox" id="toggleOCR" checked> <span style="color:var(--accent-ocr)">OCR Spans</span></label>
    </div>
  </div>
</header>

<div class="main-content">
  <!-- Left Panel: Canvas Viewer -->
  <div class="viewer-panel">
    <div class="canvas-wrapper">
      <canvas id="viewCanvas"></canvas>
    </div>
    
    <div class="playback-bar">
      <button class="btn" id="prevBtn">◀ Prev</button>
      <button class="btn" id="playBtn">▶ Play</button>
      <button class="btn" id="nextBtn">Next ▶</button>
      <input type="range" id="frameSlider" class="scrubber" min="0" max="{len(unified_records)-1}" value="0">
      <div class="time-display" id="timeDisplay">00:00.00</div>
    </div>
  </div>

  <!-- Right Panel: Multi-Modal Metadata -->
  <div class="meta-panel">
    <div class="meta-header">
      <span id="frameTitle">Frame 1 / {len(unified_records)}</span>
      <span class="badge" id="pointIdBadge">Point ID: 1</span>
    </div>
    
    <div class="meta-body">
      <div class="card">
        <div class="card-title">📌 Frame Info</div>
        <div style="font-size:0.85rem; color:var(--text-muted); display:flex; gap:16px;">
          <span>PTS: <b style="color:#fff;" id="ptsVal">0.00s</b></span>
          <span>Frame ID: <b style="color:#fff;" id="uidVal">L21_V001:4</b></span>
          <span>SigLIP2 Norm: <b class="vector-stat" id="normVal">1.0000</b></span>
        </div>
      </div>

      <div class="card">
        <div class="card-title" style="color:var(--accent-dam);">🎯 DAM-3B Focal Descriptions</div>
        <div id="damContainer"></div>
      </div>

      <div class="card">
        <div class="card-title" style="color:var(--accent-ocr);">🔤 EasyOCR Vietnamese Transcript</div>
        <div id="ocrContainer" class="ocr-content"></div>
      </div>
    </div>
  </div>
</div>

<script>
const frames = {json_payload};
let currentIdx = 0;
let isPlaying = false;
let playInterval = null;

const canvas = document.getElementById("viewCanvas");
const ctx = canvas.getContext("2d");
const slider = document.getElementById("frameSlider");
const timeDisplay = document.getElementById("timeDisplay");
const frameTitle = document.getElementById("frameTitle");
const pointIdBadge = document.getElementById("pointIdBadge");
const ptsVal = document.getElementById("ptsVal");
const uidVal = document.getElementById("uidVal");
const normVal = document.getElementById("normVal");
const damContainer = document.getElementById("damContainer");
const ocrContainer = document.getElementById("ocrContainer");
const searchInput = document.getElementById("searchInput");

const toggleDAM = document.getElementById("toggleDAM");
const toggleOCR = document.getElementById("toggleOCR");

const imgCache = new Map();

function preloadImage(idx) {{
  if (!frames[idx]) return Promise.resolve(null);
  if (imgCache.has(idx)) return Promise.resolve(imgCache.get(idx));
  return new Promise((resolve) => {{
    const img = new Image();
    img.src = frames[idx].img_src;
    img.onload = () => {{
      imgCache.set(idx, img);
      resolve(img);
    }};
  }});
}}

async function renderFrame(idx) {{
  if (idx < 0 || idx >= frames.length) return;
  currentIdx = idx;
  slider.value = idx;
  const f = frames[idx];

  // Update Metadata Text
  frameTitle.textContent = `Frame ${{idx + 1}} / ${{frames.length}}`;
  pointIdBadge.textContent = `Point ID: ${{f.point_id}}`;
  ptsVal.textContent = `${{f.pts.toFixed(2)}}s`;
  uidVal.textContent = f.uid;
  normVal.textContent = f.l2_norm.toFixed(4);

  const mins = Math.floor(f.pts / 60);
  const secs = (f.pts % 60).toFixed(2);
  timeDisplay.textContent = `${{mins.toString().padStart(2, '0')}}:${{secs.padStart(5, '0')}}`;

  // Render DAM Regions
  if (f.regions && f.regions.length > 0) {{
    damContainer.innerHTML = f.regions.map((r, r_i) => `
      <div class="dam-region">
        <div class="region-tag">#${{r_i + 1}} ${{r.label.toUpperCase()}} ${{r.score ? '(' + r.score.toFixed(2) + ')' : ''}}</div>
        <div class="region-desc">${{r.caption || 'N/A'}}</div>
      </div>
    `).join('');
  }} else {{
    damContainer.innerHTML = `<div class="region-desc">${{f.dam_summary || 'No object regions detected.'}}</div>`;
  }}

  // Render OCR Transcript
  ocrContainer.textContent = f.ocr_text ? `"${{f.ocr_text}}"` : 'None';

  // Load and Draw Image on Canvas
  const img = await preloadImage(idx);
  if (!img) return;

  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);

  const W = canvas.width;
  const H = canvas.height;

  // Draw DAM Bounding Boxes
  if (toggleDAM.checked && f.regions) {{
    const colors = ["#00ffcc", "#ff007f", "#00bfff", "#39ff14", "#ff6600"];
    f.regions.forEach((r, r_i) => {{
      if (!r.bbox) return;
      let [y1, x1, y2, x2] = [0, 0, 0, 0];
      if (r.coord_type === 'px') {{
        [x1, y1, x2, y2] = r.bbox;
      }} else {{
        [y1, x1, y2, x2] = r.bbox;
        x1 *= W; y1 *= H; x2 *= W; y2 *= H;
      }}
      const color = colors[r_i % colors.length];
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // Label background
      ctx.fillStyle = color;
      ctx.font = "bold 14px sans-serif";
      const tag = `[DAM #${{r_i+1}}] ${{r.label}}`;
      const textWidth = ctx.measureText(tag).width;
      ctx.fillRect(x1, Math.max(0, y1 - 22), textWidth + 8, 22);

      ctx.fillStyle = "#000";
      ctx.fillText(tag, x1 + 4, Math.max(16, y1 - 6));
    }});
  }}

  // Draw OCR Polygons
  if (toggleOCR.checked && f.ocr_spans) {{
    f.ocr_spans.forEach(span => {{
      if (!span.polygon || span.polygon.length < 3) return;
      ctx.beginPath();
      span.polygon.forEach((p, p_i) => {{
        const px = p[0] <= 1.0 ? p[0] * W : p[0];
        const py = p[1] <= 1.0 ? p[1] * H : p[1];
        if (p_i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }});
      ctx.closePath();
      ctx.fillStyle = "rgba(255, 234, 0, 0.22)";
      ctx.fill();
      ctx.strokeStyle = "#ffea00";
      ctx.lineWidth = 2;
      ctx.stroke();

      // OCR Tag
      const p0 = span.polygon[0];
      const p0x = p0[0] <= 1.0 ? p0[0] * W : p0[0];
      const p0y = p0[1] <= 1.0 ? p0[1] * H : p0[1];
      ctx.fillStyle = "#ffea00";
      ctx.font = "bold 13px sans-serif";
      const tag = `[OCR] ${{span.text}}`;
      const textWidth = ctx.measureText(tag).width;
      ctx.fillRect(p0x, Math.max(0, p0y - 18), textWidth + 6, 18);
      ctx.fillStyle = "#000";
      ctx.fillText(tag, p0x + 3, Math.max(14, p0y - 4));
    }});
  }}

  // Preload neighboring frames for instantaneous scrubbing
  preloadImage(idx + 1);
  preloadImage(idx + 2);
  preloadImage(idx - 1);
}}

// Slider scrubbing
slider.addEventListener("input", (e) => renderFrame(parseInt(e.target.value)));

// Prev / Next
document.getElementById("prevBtn").addEventListener("click", () => renderFrame(currentIdx - 1));
document.getElementById("nextBtn").addEventListener("click", () => renderFrame(currentIdx + 1));

// Keyboard Arrows
window.addEventListener("keydown", (e) => {{
  if (e.key === "ArrowLeft") renderFrame(currentIdx - 1);
  if (e.key === "ArrowRight") renderFrame(currentIdx + 1);
  if (e.key === " ") {{
    e.preventDefault();
    togglePlay();
  }}
}});

// Toggle re-render
toggleDAM.addEventListener("change", () => renderFrame(currentIdx));
toggleOCR.addEventListener("change", () => renderFrame(currentIdx));

// Play / Pause
function togglePlay() {{
  isPlaying = !isPlaying;
  document.getElementById("playBtn").textContent = isPlaying ? "⏸ Pause" : "▶ Play";
  if (isPlaying) {{
    playInterval = setInterval(() => {{
      if (currentIdx < frames.length - 1) {{
        renderFrame(currentIdx + 1);
      }} else {{
        togglePlay();
      }}
    }}, 400);
  }} else {{
    clearInterval(playInterval);
  }}
}}
document.getElementById("playBtn").addEventListener("click", togglePlay);

// Search Query Filter
searchInput.addEventListener("keydown", (e) => {{
  if (e.key === "Enter") {{
    const query = searchInput.value.trim().toLowerCase();
    if (!query) return;
    const matchIdx = frames.findIndex((f, idx) => {{
      if (idx <= currentIdx) return false;
      const damMatch = (f.dam_summary || "").toLowerCase().includes(query) ||
                       (f.regions || []).some(r => (r.caption || "").toLowerCase().includes(query) || (r.label || "").toLowerCase().includes(query));
      const ocrMatch = (f.ocr_text || "").toLowerCase().includes(query);
      return damMatch || ocrMatch;
    }});
    if (matchIdx !== -1) {{
      renderFrame(matchIdx);
    }} else {{
      // Wrap around search
      const wrapIdx = frames.findIndex((f) => {{
        const damMatch = (f.dam_summary || "").toLowerCase().includes(query) ||
                         (f.regions || []).some(r => (r.caption || "").toLowerCase().includes(query) || (r.label || "").toLowerCase().includes(query));
        const ocrMatch = (f.ocr_text || "").toLowerCase().includes(query);
        return damMatch || ocrMatch;
      }});
      if (wrapIdx !== -1) renderFrame(wrapIdx);
      else alert(`No matching frame found for "${{query}}"`);
    }}
  }}
}});

// Initial Render
renderFrame(0);
</script>
</body>
</html>
"""
    output_html.write_text(html_content, encoding="utf-8")
    print(f"\n🎉 Interactive Gallery Generated Successfully at:")
    print(f"👉 {output_html}")
    print(f"Open it in your browser with: open \"{output_html}\"")


if __name__ == "__main__":
    main()
