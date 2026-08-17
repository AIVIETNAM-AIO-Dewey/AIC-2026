import { useEffect, useState } from "react";

import {
  getCapabilities,
  search,
  type Capabilities,
  type FrameHit,
  type SearchResponse,
  type TaskType,
} from "./api/client";
import { SubmissionBasket } from "./components/SubmissionBasket";
import { AnswerPanel } from "./features/qa/AnswerPanel";
import { ResultsGrid } from "./features/results/ResultsGrid";
import { SearchForm } from "./features/search/SearchForm";
import { Timeline } from "./features/trake/Timeline";
import "./style.css";

const unavailableTasks: Capabilities["tasks"] = {
  kis: { ready: false, missing: ["backend"] },
  qa: { ready: false, missing: ["backend"] },
  trake: { ready: false, missing: ["backend"] },
};

export default function App() {
  const [task, setTask] = useState<TaskType>("kis");
  const [capabilities, setCapabilities] = useState<Capabilities>();
  const [data, setData] = useState<SearchResponse>();
  const [basket, setBasket] = useState<FrameHit[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getCapabilities()
      .then(setCapabilities)
      .catch((cause) => {
        setError(cause instanceof Error ? cause.message : "Backend chưa sẵn sàng");
      });
  }, []);

  const toggle = (hit: FrameHit) =>
    setBasket((items) =>
      items.some(
        (item) => item.video_id === hit.video_id && item.frame_idx === hit.frame_idx,
      )
        ? items.filter(
            (item) => item.video_id !== hit.video_id || item.frame_idx !== hit.frame_idx,
          )
        : [...items, hit],
    );

  const run = async (query: string) => {
    setLoading(true);
    setError("");
    try {
      setData(await search(task, query));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Tìm kiếm thất bại");
    } finally {
      setLoading(false);
    }
  };

  const resultCount = task === "trake" ? data?.sequences.length : data?.results.length;
  const taskCapabilities = capabilities?.tasks ?? unavailableTasks;

  return (
    <main>
      <header className="hero">
        <p className="eyebrow">HCMC AI Challenge 2026</p>
        <h1>Tìm kiếm keyframe đa phương thức</h1>
        <p>Tìm đúng video và frame index bằng mô tả tiếng Việt.</p>
      </header>

      {!capabilities && !error && <p role="status">Đang kiểm tra backend…</p>}

      <SearchForm
        task={task}
        setTask={setTask}
        onSearch={run}
        loading={loading}
        capabilities={taskCapabilities}
      />

      {capabilities && !capabilities.search_ready && (
        <section className="readiness" aria-label="Trạng thái hệ thống">
          <h2>Pipeline tìm kiếm chưa sẵn sàng</h2>
          <p>
            Form được giữ hiển thị để kiểm tra UI, nhưng chỉ hoạt động sau khi ingest
            artifact thật.
          </p>
          {(["kis", "qa", "trake"] as TaskType[]).map((mode) => (
            <p key={mode}>
              <strong>{mode.toUpperCase()}:</strong>{" "}
              {capabilities.tasks[mode].ready
                ? "sẵn sàng"
                : `thiếu ${capabilities.tasks[mode].missing.join(", ")}`}
            </p>
          ))}
        </section>
      )}

      {error && <p role="alert">{error}</p>}
      {data?.degraded && <p>GPT-4o chưa sẵn sàng; KIS đang dùng truy vấn gốc.</p>}
      {data && resultCount === 0 && <p role="status">Không tìm thấy kết quả phù hợp.</p>}

      {task === "trake" ? (
        <Timeline sequences={data?.sequences ?? []} />
      ) : (
        <>
          <section className="results-heading">
            <div>
              <p className="eyebrow">Retrieval results</p>
              <h2>Kết quả keyframe</h2>
            </div>
            <span>{data?.results.length ?? 0} frame</span>
          </section>
          {!data && (
            <p className="empty-state">
              Keyframe phù hợp sẽ xuất hiện tại đây sau khi backend và collection KIS sẵn
              sàng.
            </p>
          )}
          <ResultsGrid hits={data?.results ?? []} basket={basket} toggle={toggle} />
          {task === "qa" && (
            <AnswerPanel answer={data?.answer} confidence={data?.confidence} />
          )}
          <SubmissionBasket frames={basket} />
        </>
      )}
    </main>
  );
}
