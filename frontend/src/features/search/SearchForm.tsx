import type { TaskCapability, TaskType } from "../../api/client";

type Props = {
  task: TaskType;
  setTask: (task: TaskType) => void;
  onSearch: (query: string) => void;
  loading: boolean;
  capabilities: Record<TaskType, TaskCapability>;
  fuzzy: boolean;
  setFuzzy: (enabled: boolean) => void;
};

export function SearchForm({
  task,
  setTask,
  onSearch,
  loading,
  capabilities,
  fuzzy,
  setFuzzy,
}: Props) {
  const activeCapability = capabilities[task] ?? { ready: false, missing: ["backend"] };

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        const query = new FormData(event.currentTarget).get("query")?.toString().trim();
        if (query && activeCapability.ready) onSearch(query);
      }}
    >
      <div role="tablist" aria-label="Loại truy vấn">
        {(["kis", "qa", "trake", "ocr"] as TaskType[]).map((mode) => {
          const capability = capabilities[mode] ?? { ready: false, missing: ["backend"] };
          return (
            <button
              key={mode}
              type="button"
              role="tab"
              aria-selected={task === mode}
              disabled={!capability.ready}
              title={capability.missing.join(", ")}
              onClick={() => setTask(mode)}
            >
              {mode.toUpperCase()}
            </button>
          );
        })}
      </div>
      <input
        name="query"
        aria-label="Vietnamese query"
        placeholder={task === "ocr" ? "Nhập chữ cần tìm trong frame" : "Nhập truy vấn tiếng Việt"}
        disabled={!activeCapability.ready}
        required
      />
      <button disabled={loading || !activeCapability.ready}>
        {loading ? "Đang tìm…" : "Tìm kiếm"}
      </button>
      {task === "ocr" && (
        <label className="fuzzy-toggle">
          <input
            type="checkbox"
            checked={fuzzy}
            onChange={(event) => setFuzzy(event.target.checked)}
          />
          <span>
            <strong>Fuzzy OCR</strong>
            <small>Levenshtein rerank cho chữ OCR sai; accent folding và trigram luôn bật.</small>
          </span>
        </label>
      )}
      {!activeCapability.ready && (
        <small className="form-hint" role="status">
          {activeCapability.missing.length > 0
            ? `Chưa thể chạy ${task.toUpperCase()}: thiếu ${activeCapability.missing.join(", ")}.`
            : `Chưa thể chạy ${task.toUpperCase()}.`}
        </small>
      )}
    </form>
  );
}
