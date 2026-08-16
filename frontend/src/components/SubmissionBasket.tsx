import type { FrameHit } from "../api/client";
export function SubmissionBasket({ frames }: { frames: FrameHit[] }) { return <aside><h2>Submission basket ({frames.length})</h2><ol>{frames.map(hit => <li key={`${hit.video_id}:${hit.frame_idx}`}>{hit.video_id} / frame_idx {hit.frame_idx}</li>)}</ol></aside>; }
