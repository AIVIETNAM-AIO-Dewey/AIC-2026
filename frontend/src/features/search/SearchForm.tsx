import { useState } from "react";

import type {
  StructuredQuery,
  TaskCapability,
  TaskType,
} from "../../api/client";

type Props = {
  task: TaskType;
  setTask: (task: TaskType) => void;
  onSearch: (query: StructuredQuery) => void;
  loading: boolean;
  capabilities: Record<TaskType, TaskCapability>;
};

const examples: Record<TaskType, StructuredQuery> = {
  kis: {
    schema_version: "aic26.query.v1",
    task_type: "kis",
    raw_query_vi: "Tìm một người mặc áo đỏ đang phát biểu ngoài trời.",
    scene_en: "a person wearing a red shirt speaking outdoors",
    objects_en: ["a person wearing a red shirt"],
    ocr_vi: [],
    audio_vi: [],
    audio_events_en: [],
    answer_sources: [],
    events: null,
  },
  qa: {
    schema_version: "aic26.query.v1",
    task_type: "qa",
    raw_query_vi: "Có bao nhiêu người đứng trên sân khấu?",
    scene_en: "people standing on a stage",
    objects_en: ["people standing on a stage"],
    ocr_vi: [],
    audio_vi: [],
    audio_events_en: [],
    question_vi: "Có bao nhiêu người đứng trên sân khấu?",
    question_en: "How many people are standing on the stage?",
    answer_sources: ["visual"],
    events: null,
  },
  trake: {
    schema_version: "aic26.query.v1",
    task_type: "trake",
    raw_query_vi: "Vận động viên giậm nhảy rồi tiếp đất.",
    scene_en: "an athlete completing a jump",
    objects_en: ["an athlete"],
    ocr_vi: [],
    audio_vi: [],
    audio_events_en: [],
    answer_sources: [],
    events: [
      {
        label: "take-off",
        scene_en: "an athlete taking off from the ground",
        objects_en: ["an athlete leaving the ground"],
        temporal_operator: "onset",
      },
      {
        label: "landing",
        scene_en: "the athlete landing after the jump",
        objects_en: ["an athlete touching the landing surface"],
        temporal_operator: "onset",
      },
    ],
  },
};

function parseQuery(value: string, task: TaskType): StructuredQuery {
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch {
    throw new Error("JSON không hợp lệ.");
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Query phải là một JSON object.");
  }
  const query = parsed as Partial<StructuredQuery>;
  if (query.schema_version !== "aic26.query.v1") {
    throw new Error('schema_version phải là "aic26.query.v1".');
  }
  if (query.task_type !== task) {
    throw new Error(`task_type trong JSON phải là "${task}".`);
  }
  if (!query.raw_query_vi?.trim() || !query.scene_en?.trim()) {
    throw new Error("raw_query_vi và scene_en không được để trống.");
  }
  return query as StructuredQuery;
}

export function SearchForm({ task, setTask, onSearch, loading, capabilities }: Props) {
  const activeCapability = capabilities[task];
  const [validationError, setValidationError] = useState("");

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        const value = new FormData(event.currentTarget).get("query")?.toString().trim();
        if (!value || !activeCapability.ready) return;
        try {
          const query = parseQuery(value, task);
          setValidationError("");
          onSearch(query);
        } catch (error) {
          setValidationError(error instanceof Error ? error.message : "Query không hợp lệ.");
        }
      }}
    >
      <div role="tablist" aria-label="Loại truy vấn">
        {(["kis", "qa", "trake"] as TaskType[]).map((mode) => (
          <button
            key={mode}
            type="button"
            role="tab"
            aria-selected={task === mode}
            disabled={!capabilities[mode].ready}
            title={capabilities[mode].missing.join(", ")}
            onClick={() => {
              setValidationError("");
              setTask(mode);
            }}
          >
            {mode.toUpperCase()}
          </button>
        ))}
      </div>
      <label htmlFor="structured-query">Structured query JSON</label>
      <textarea
        key={task}
        id="structured-query"
        name="query"
        aria-label="Structured query JSON"
        defaultValue={JSON.stringify(examples[task], null, 2)}
        disabled={!activeCapability.ready}
        spellCheck={false}
        required
      />
      <button disabled={loading || !activeCapability.ready}>
        {loading ? "Đang tìm…" : "Tìm kiếm"}
      </button>
      <small className="form-hint">
        Dán JSON aic26.query.v1 do GPT Web/Gemini tạo; backend không dịch lại query.
      </small>
      {validationError && <small className="form-error" role="alert">{validationError}</small>}
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
