import assert from "node:assert/strict";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { createServer } from "vite";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");

test("mock API serves enough deterministic candidates and valid placeholder images", async (context) => {
  const server = await createServer({
    configFile: join(root, "vite.config.ts"),
    mode: "mock",
    server: {
      host: "127.0.0.1",
      port: 0,
      strictPort: true,
      hmr: false,
    },
  });
  await server.listen();
  context.after(() => server.close());

  const address = server.httpServer?.address();
  assert(address && typeof address === "object");
  const base = `http://127.0.0.1:${address.port}`;

  const searchResponse = await fetch(`${base}/api/search`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "{}",
  });
  assert.equal(searchResponse.status, 200);
  const search = await searchResponse.json();
  const pools = Object.values(search.modality_results);
  assert.deepEqual(
    pools.map((pool) => pool.results.length),
    [100, 100, 100, 100],
  );
  const uniqueCandidates = new Set(
    pools.flatMap((pool) => pool.results.map((item) => `${item.video_id}:${item.frame_idx}`)),
  );
  assert(uniqueCandidates.size >= 120);

  const kisResponse = await fetch(`${base}/api/search/fusion/kis`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "{}",
  });
  assert.equal(kisResponse.status, 200);
  const kis = await kisResponse.json();
  assert.equal(kis.schema_version, "kis.fusion.result.v1");
  assert.equal(kis.result_count, 150);
  assert.equal(kis.results.length, 150);
  assert.equal(kis.results[0].beit3_query_scores.original.language, "en");
  assert.equal(kis.results[0].rerank_formula.beit3_weight, 0.25);
  assert.equal(kis.results[0].rerank_formula.previous_weight, 0.75);
  assert.equal(
    kis.results[0].final_score,
    Number((0.25 * kis.results[0].beit3_normalized + 0.75 * kis.results[0].rrf_normalized).toFixed(6)),
  );
  assert(kis.results.slice(0, 100).some((item) => item.rank_delta !== 0));
  for (const item of kis.results.slice(0, 100)) {
    assert.equal(item.rank_delta, item.pre_rerank_rank - item.rank);
    assert.equal(
      item.rrf_score,
      Number((
        0.40 / (60 + item.pre_rerank_rank)
        + 0.30 / (60 + item.pre_rerank_rank)
        + 0.15 / (60 + item.pre_rerank_rank)
        + 0.15 / (60 + item.pre_rerank_rank)
      ).toFixed(8)),
    );
  }
  assert(kis.results.slice(0, 99).every((item, index) => item.final_score >= kis.results[index + 1].final_score));
  assert.equal("beit3_raw_cosine" in kis.results[100], false);
  assert.equal("rank_delta" in kis.results[100], false);
  assert.equal(kis.results[100].pre_rerank_rank, kis.results[100].rank);
  assert.equal(kis.results[100].score, kis.results[100].rrf_score);
  assert.equal(kis.results[100].score_type, "weighted_rrf");
  assert.equal("asr_transcript" in kis.results[0], false);
  assert.equal("ocr_text" in kis.results[0], false);
  assert.equal("dam_summary" in kis.results[0], false);
  assert.equal(kis.results[0].branch_provenance.branch2.dense_best_query_language, "en");
  assert.equal("dense_query_scores" in kis.results[0].branch_provenance.branch2, false);

  const orderedKisResponse = await fetch(`${base}/api/search/fusion/kis/temporal`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      task_type: "KIS",
      events: [
        { order: 1, description: "first action", en: "first action" },
        { order: 2, description: "second action", en: "second action" },
      ],
      top_k_sequences: 20,
      max_gap_seconds: 30,
    }),
  });
  assert.equal(orderedKisResponse.status, 200);
  const orderedKis = await orderedKisResponse.json();
  assert.equal(orderedKis.operation, "ordered_kis_fusion");
  assert.equal(orderedKis.event_count, 2);
  assert.equal(orderedKis.event_fusion_applied, true);
  assert.equal(orderedKis.cross_modal_fusion_applied, true);
  assert.equal(orderedKis.reranking_applied, true);
  assert.equal(orderedKis.frame_index_base, 0);
  assert.deepEqual(
    orderedKis.sequences[0].matched_events.map((event) => event.event_order),
    [1, 2],
  );
  assert(orderedKis.sequences[0].matched_events.every(
    (event, index, values) => index === 0
      || event.frame_idx > values[index - 1].frame_idx,
  ));

  const planResponse = await fetch(`${base}/api/query/kis/plan`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      query: "A chef prepares shrimp, then grills the shrimp",
      task_type: "TRAKE",
    }),
  });
  assert.equal(planResponse.status, 200);
  const plan = await planResponse.json();
  assert.equal(plan.schema_version, "kis.query-plan.v1");
  assert.equal(plan.event_count, 2);
  assert.equal(plan.query_bundle.schema_version, "branch1.query.v1");
  assert.equal(plan.retrieval_invoked, false);

  const timelineResponse = await fetch(`${base}/api/video/L21_V001/timeline`);
  assert.equal(timelineResponse.status, 200);
  const timeline = await timelineResponse.json();
  assert.equal(timeline.schema_version, "video.frame-timeline.v1");
  assert.equal(timeline.frame_index_base, 0);
  assert.equal(timeline.fps, 30);
  assert.equal(timeline.max_frame_idx, 7199);
  assert.equal(timeline.keyframe_count, 128);
  assert.equal(timeline.keyframes[0].video_id, "L21_V001");

  const exactFrameResponse = await fetch(
    `${base}/api/frame/L21-V001/${kis.results[0].frame_idx}`,
  );
  assert.equal(exactFrameResponse.status, 200);
  const exactFrame = await exactFrameResponse.json();
  assert.equal(exactFrame.schema_version, "frame.identity.v1");
  assert.equal(exactFrame.exact_match, true);
  assert.equal(exactFrame.keyframe.frame_uid, `L21_V001:${kis.results[0].frame_idx}`);
  assert.equal(exactFrame.keyframe.validation, "canonical");
  const missingExactFrame = await fetch(`${base}/api/frame/L21_V001/5`);
  assert.equal(missingExactFrame.status, 404);

  const sourceFrameResponse = await fetch(`${base}/api/video/L21-V001/source-frame/5`);
  assert.equal(sourceFrameResponse.status, 200);
  const sourceFrame = await sourceFrameResponse.json();
  assert.equal(sourceFrame.schema_version, "video.source-frame.v1");
  assert.equal(sourceFrame.exact_match, true);
  assert.equal(sourceFrame.source_frame.frame_uid, "L21_V001:5");
  assert.equal(sourceFrame.source_frame.frame_idx, 5);
  assert.equal(sourceFrame.source_frame.indexed_keyframe, false);
  assert.equal(sourceFrame.source_frame.preview_frame_idx, 4);

  const relatedResponse = await fetch(`${base}/api/submission/related-frames`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      video_id: sourceFrame.source_frame.video_id,
      frame_idx: sourceFrame.source_frame.frame_idx,
      limit: 12,
    }),
  });
  assert.equal(relatedResponse.status, 200);
  const related = await relatedResponse.json();
  assert.equal(related.schema_version, "submission.related-frames.v1");
  assert.equal(related.query_pipeline_invoked, false);
  assert.equal(related.result_count, 12);
  assert.equal(related.requested_seed.frame_idx, 5);
  assert.equal(related.embedding_seed.frame_idx, 4);
  assert.equal(related.results.length, 12);
  assert(related.results.every((item) => item.frame_uid !== exactFrame.keyframe.frame_uid));
  assert.equal(new Set(related.results.map((item) => item.frame_uid)).size, 12);

  const imageBody = new FormData();
  imageBody.append("file", new Blob(["mock"], { type: "image/png" }), "query.png");
  imageBody.append("top_k", "50");
  const imageResponse = await fetch(`${base}/api/search/image`, { method: "POST", body: imageBody });
  assert.equal(imageResponse.status, 200);
  const imageSearch = await imageResponse.json();
  assert.equal(imageSearch.query_modality, "image");
  assert.equal(imageSearch.fusion_applied, false);
  assert.equal(imageSearch.modality_result.results.length, 50);

  const prepareResponse = await fetch(`${base}/api/submission/prepare`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      task_type: "KIS",
      query_id: "Q1",
      target_rows: 100,
      manual_selections: [sourceFrame.source_frame],
      candidate_reservoir: timeline.keyframes,
    }),
  });
  assert.equal(prepareResponse.status, 200);
  const prepared = await prepareResponse.json();
  assert.equal(prepared.complete, true);
  assert.equal(prepared.rows.length, 100);
  assert.equal(prepared.rows[0].frame_idx, 5);
  assert.equal(prepared.valid_for_download, true);
  assert.equal(prepared.official_csv.has_header, false);
  assert.equal(prepared.official_csv.row_count, 100);
  assert.equal(prepared.official_csv.content.split(/\r?\n/)[0], `${timeline.video_id},5`);
  assert(!prepared.official_csv.content.includes("Q1,"));

  const toSequence = (start) => ({
    video_id: timeline.video_id,
    events: timeline.keyframes.slice(start, start + 3).map((item, index) => ({
      event_order: index + 1,
      video_id: item.video_id,
      frame_idx: item.frame_idx,
      pts_time_s: item.pts_time_s,
    })),
  });
  const manualSequence = toSequence(0);
  const trakeResponse = await fetch(`${base}/api/submission/prepare`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      task_type: "TRAKE",
      query_id: "query-4-trake",
      target_rows: 100,
      event_count: 3,
      manual_sequences: [manualSequence],
      candidate_sequences: Array.from({ length: 99 }, (_, index) => toSequence(index + 1)),
    }),
  });
  assert.equal(trakeResponse.status, 200);
  const trake = await trakeResponse.json();
  assert.equal(trake.complete, true);
  assert.equal(trake.valid_for_download, true);
  assert.equal(trake.rows.length, 100);
  assert.deepEqual(trake.rows[0].events.map((event) => event.frame_idx), manualSequence.events.map((event) => event.frame_idx));
  assert.equal(
    trake.official_csv.content.split(/\r?\n/)[0],
    [manualSequence.video_id, ...manualSequence.events.map((event) => event.frame_idx)].join(","),
  );

  const invalidPrepareResponse = await fetch(`${base}/api/submission/prepare`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      task_type: "KIS",
      query_id: "Q-invalid",
      target_rows: 100,
      manual_selections: [{ video_id: "L99_V999", frame_idx: 123456 }],
      candidate_reservoir: timeline.keyframes,
    }),
  });
  assert.equal(invalidPrepareResponse.status, 400);
  const invalidPrepared = await invalidPrepareResponse.json();
  assert.equal(invalidPrepared.complete, false);
  assert.match(invalidPrepared.errors[0], /outside the verified/i);

  const drilldownResponse = await fetch(`${base}/api/video/L21-V001/search/siglip`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "{}",
  });
  assert.equal(drilldownResponse.status, 200);
  const drilldown = await drilldownResponse.json();
  assert.equal(drilldown.operation, "manual_video_drilldown");
  assert.equal(drilldown.video_id, "L21_V001");
  assert.equal(drilldown.evaluated_frames, 128);
  assert.equal(drilldown.fusion_applied, false);
  assert.equal(drilldown.reranking_applied, false);
  assert.equal(drilldown.modality_result.results.length, 100);
  assert(drilldown.modality_result.results.every((item) => item.video_id === "L21_V001"));

  const visualFusionResponse = await fetch(`${base}/api/video/L21-V001/search/visual-fusion`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query: "a cyclist", top_k: 25 }),
  });
  assert.equal(visualFusionResponse.status, 200);
  const visualFusion = await visualFusionResponse.json();
  assert.equal(visualFusion.operation, "visual_video_drilldown");
  assert.deepEqual(visualFusion.models, ["siglip2", "metaclip2", "beit3"]);
  assert.equal(visualFusion.cross_modal_fusion_applied, false);
  assert.equal(visualFusion.modality_result.results.length, 25);
  assert(visualFusion.modality_result.results.every((item) => item.video_id === "L21_V001"));

  const cascadeResponse = await fetch(`${base}/api/discover/dam-to-siglip`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "{}",
  });
  assert.equal(cascadeResponse.status, 200);
  const discovery = await cascadeResponse.json();
  assert.equal(discovery.operation, "dam_to_siglip_discovery_cascade");
  assert.equal(discovery.cross_modal_gating_applied, true);
  assert.equal(discovery.fusion_applied, false);
  assert.equal(discovery.dam_score_used_in_final_rank, false);
  assert.equal(discovery.cascades.length, 2);
  assert(discovery.cascades.every((cascade) => cascade.results.length === 20));
  assert(discovery.cascades.flatMap((cascade) => cascade.results)
    .every((item) => item.scope === "dam_to_siglip_cascade"));

  const temporalResponse = await fetch(`${base}/api/search/temporal-intersection`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "{}",
  });
  assert.equal(temporalResponse.status, 200);
  const temporal = await temporalResponse.json();
  assert.equal(temporal.operation, "ordered_siglip_intersection");
  assert.equal(temporal.same_modality_event_aggregation_applied, true);
  assert.equal(
    temporal.same_modality_event_aggregation,
    "mean_context_anchor_and_minimum_event_then_event_mean",
  );
  assert.equal(temporal.anchor_query_applied, true);
  assert.equal(temporal.cross_modal_fusion_applied, false);
  assert.equal(temporal.fusion_applied, false);
  assert.equal(temporal.reranking_applied, false);
  assert.equal(temporal.sequences.length, 1);
  const matchedEvents = temporal.sequences[0].matched_events;
  assert.equal(
    temporal.sequences[0].minimum_event_score,
    Math.min(...matchedEvents.map((event) => event.score)),
  );
  assert.equal(
    temporal.sequences[0].sequence_score,
    Number(((
      temporal.sequences[0].context_anchor_score
      + temporal.sequences[0].minimum_event_score
    ) / 2).toFixed(6)),
  );
  assert.deepEqual(matchedEvents.map((event) => event.event_order), [1, 2, 3]);
  assert.equal(new Set(matchedEvents.map((event) => event.video_id)).size, 1);
  assert(matchedEvents.every((event, index) => index === 0
    || event.pts_time_s > matchedEvents[index - 1].pts_time_s));
  assert(matchedEvents.every((event) => event.score_type === "cosine"));

  const filmstrip = await (await fetch(`${base}/api/video/L21_V001/keyframes`)).json();
  assert.equal(filmstrip.total_keyframes, 128);

  for (const path of [
    "/keyframes/L21_V001/00000004.jpg",
    "/data/keyframe/L21_V001/00004948.jpg",
  ]) {
    const image = await fetch(`${base}${path}`);
    assert.equal(image.status, 200);
    assert.match(image.headers.get("content-type") ?? "", /^image\//);
    assert((await image.arrayBuffer()).byteLength > 100);
  }
});
