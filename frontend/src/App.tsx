import { useState } from "react";
import { search, type FrameHit, type SearchResponse, type TaskType } from "./api/client";
import { SubmissionBasket } from "./components/SubmissionBasket";
import { AnswerPanel } from "./features/qa/AnswerPanel";
import { ResultsGrid } from "./features/results/ResultsGrid";
import { SearchForm } from "./features/search/SearchForm";
import { Timeline } from "./features/trake/Timeline";
import "./style.css";

export default function App() { const [task, setTask] = useState<TaskType>("kis"); const [data, setData] = useState<SearchResponse>(); const [basket, setBasket] = useState<FrameHit[]>([]); const [error, setError] = useState(""); const [loading, setLoading] = useState(false); const toggle = (hit: FrameHit) => setBasket(items => items.some(item => item.video_id === hit.video_id && item.frame_idx === hit.frame_idx) ? items.filter(item => item.video_id !== hit.video_id || item.frame_idx !== hit.frame_idx) : [...items, hit]); const run = async (query: string) => { setLoading(true); setError(""); try { setData(await search(task, query)); } catch (cause) { setError(cause instanceof Error ? cause.message : "Search failed"); } finally { setLoading(false); } }; return <main><h1>AIC 2026 MultiRetrieval</h1><SearchForm task={task} setTask={setTask} onSearch={run} loading={loading}/>{error && <p role="alert">{error}</p>}{data?.degraded && <p>GPT-4o unavailable: KIS raw-query fallback is active.</p>}{task === "trake" ? <Timeline sequences={data?.sequences ?? []}/> : <><ResultsGrid hits={data?.results ?? []} basket={basket} toggle={toggle}/>{task === "qa" && <AnswerPanel answer={data?.answer} confidence={data?.confidence}/>}<SubmissionBasket frames={basket}/></>}</main>; }
