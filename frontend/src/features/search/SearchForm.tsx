import type { TaskCapability, TaskType } from "../../api/client";

type Props = {
  task: TaskType;
  setTask: (task: TaskType) => void;
  onSearch: (query: string) => void;
  loading: boolean;
  capabilities: Record<TaskType, TaskCapability>;
};

export function SearchForm({ task, setTask, onSearch, loading, capabilities }: Props) {
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        const query = new FormData(event.currentTarget).get("query")?.toString().trim();
        if (query) onSearch(query);
      }}
    >
      <div role="tablist">
        {(["kis", "qa", "trake"] as TaskType[]).map((mode) => (
          <button
            key={mode}
            type="button"
            role="tab"
            aria-selected={task === mode}
            disabled={!capabilities[mode].ready}
            title={capabilities[mode].missing.join(", ")}
            onClick={() => setTask(mode)}
          >
            {mode.toUpperCase()}
          </button>
        ))}
      </div>
      <input
        name="query"
        aria-label="Vietnamese query"
        placeholder="Nhập truy vấn tiếng Việt"
        required
      />
      <button disabled={loading}>{loading ? "Đang tìm…" : "Tìm kiếm"}</button>
    </form>
  );
}
