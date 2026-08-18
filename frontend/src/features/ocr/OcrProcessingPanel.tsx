import { useEffect, useState } from "react";

import {
  getOcrJobs,
  indexOcrJob,
  runOcrJob,
  type OcrJobs,
} from "../../api/client";

export function OcrProcessingPanel({ onIndexed }: { onIndexed: () => void }) {
  const [jobs, setJobs] = useState<OcrJobs>();
  const [selected, setSelected] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  const refresh = () =>
    getOcrJobs()
      .then((next) => {
        setJobs(next);
        setSelected((current) => current || next.datasets[0]?.manifest_id || "");
      })
      .catch((cause) => setMessage(cause instanceof Error ? cause.message : "Không đọc được OCR job"));

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    if (!jobs?.active_manifest_id) return;
    const timer = window.setInterval(refresh, 1500);
    return () => window.clearInterval(timer);
  }, [jobs?.active_manifest_id]);

  const dataset = jobs?.datasets.find((item) => item.manifest_id === selected);

  const execute = async (action: "run" | "index") => {
    if (!selected) return;
    setBusy(true);
    setMessage("");
    try {
      if (action === "run") {
        setJobs(await runOcrJob(selected));
        setMessage(dataset?.status === "interrupted" ? "Đã resume OCR." : "Đã bắt đầu OCR.");
      } else {
        await indexOcrJob(selected);
        setMessage("Đã index và activate collection OCR mới.");
        onIndexed();
      }
    } catch (cause) {
      setMessage(cause instanceof Error ? cause.message : "OCR job thất bại");
    } finally {
      setBusy(false);
      refresh();
    }
  };

  return (
    <details className="ocr-processing">
      <summary>Process / resume OCR dataset</summary>
      {!jobs && <p role="status">Đang đọc OCR artifacts…</p>}
      {jobs && !jobs.enabled && (
        <p className="operator-note">
          Runner đang khóa. Operator phải bật <code>AIC_OCR_JOBS_ENABLED=true</code>; model vẫn
          chạy offline, không download hoặc fallback.
        </p>
      )}
      {jobs && jobs.datasets.length === 0 && <p>Chưa có frame manifest để chạy OCR.</p>}
      {jobs && jobs.datasets.length > 0 && (
        <div className="ocr-job-controls">
          <label>
            Frame manifest
            <select value={selected} onChange={(event) => setSelected(event.target.value)}>
              {jobs.datasets.map((item) => (
                <option key={item.manifest_id} value={item.manifest_id}>
                  {item.manifest_id} · {item.processed_frames}/{item.total_frames}
                </option>
              ))}
            </select>
          </label>
          {dataset && (
            <>
              <progress value={dataset.processed_frames} max={dataset.total_frames || 1} />
              <small>
                {dataset.status} · {dataset.processed_frames}/{dataset.total_frames} frame · còn {dataset.remaining_frames}
              </small>
              <div>
                <button
                  type="button"
                  disabled={busy || !jobs.enabled || dataset.status === "running" || dataset.status === "completed"}
                  onClick={() => execute("run")}
                >
                  {dataset.status === "interrupted" ? "Resume OCR" : "Run OCR"}
                </button>
                <button
                  type="button"
                  disabled={busy || !jobs.enabled || dataset.status !== "completed"}
                  onClick={() => execute("index")}
                >
                  Index để tìm kiếm
                </button>
              </div>
            </>
          )}
        </div>
      )}
      {message && <p role="status">{message}</p>}
    </details>
  );
}
