import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  extractHtmlIds,
  extractRequiredIds,
  validateFrontendContract,
} from "../../../scripts/check_frontend.mjs";
import {
  branchPoolCountLabel,
  formatOcrEvidence,
  formatKisFusionCardEvidence,
  formatKisFusionEvidence,
  renderVisibleResultPool,
  resolveWinningOcrText,
  splitResultPool,
} from "../src/result-gates.js";

test("contract parser extracts HTML and JavaScript ids", () => {
  assert.deepEqual(extractHtmlIds('<div id="alpha"></div><button id="beta"></button>'), [
    "alpha",
    "beta",
  ]);
  assert.deepEqual(
    extractRequiredIds(
      'document.getElementById("alpha"); requireElement<HTMLButtonElement>("beta");',
    ),
    ["alpha", "beta"],
  );
});

test("frontend source and HTML keep a valid DOM contract", () => {
  assert.deepEqual(validateFrontendContract(), []);
});

test("Branch 2 defaults to 40% BEiT-3 and 60% DAM hybrid", () => {
  const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
  assert.match(html, /id="branch2-weight-beit"[^>]*value="0\.40"/);
  assert.match(html, /id="branch2-weight-previous"[^>]*value="0\.60"/);
});

test("KIS fusion keeps four positive voters and the fixed final gates", () => {
  const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
  const source = readFileSync(new URL("../app.js", import.meta.url), "utf8");
  assert.match(html, /id="kis-fusion-weight-branch1"[^>]*value="0\.40"/);
  assert.match(html, /id="kis-fusion-weight-branch2"[^>]*value="0\.30"/);
  assert.match(html, /id="kis-fusion-weight-ocr"[^>]*value="0\.15"/);
  assert.match(html, /id="kis-fusion-weight-asr"[^>]*value="0\.15"/);
  assert.match(html, /RRF k=60/);
  assert.match(html, /BEiT-3 COCO cosine rerank top 100/);
  assert.match(html, /Final score: 25% BEiT-3 \+ 75% RRF/);
  assert.match(source, /fetch\("\/api\/search\/fusion\/kis"/);
  assert.match(source, /renderVisibleResultPool\([\s\S]*?kisFusionResultsGrid/);
  assert.match(source, /formatKisFusionEvidence/);
  assert.match(source, /item\.retrieval_modality === "kis_fusion"/);
  assert.doesNotMatch(source, /legacyFusionEvidence/);
  assert.doesNotMatch(source, /legacyFusionInspectorEvidence/);
});

test("KIS owns ordered full-fusion events while in-video search stays visual-only", () => {
  const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
  const source = readFileSync(new URL("../app.js", import.meta.url), "utf8");
  assert.match(html, /id="kis-trake-sequence-panel"/);
  assert.match(html, /id="kis-pinned-query-text"[^>]*textarea|<textarea id="kis-pinned-query-text"/);
  assert.match(html, /id="btn-prepare-kis-query"/);
  assert.match(html, /id="kis-query-plan-status"/);
  assert.match(source, /fetch\("\/api\/query\/kis\/plan"/);
  assert.match(source, /generatedBundleSignature/);
  assert.match(source, /replacesManualWork/);
  assert.match(html, /Every event runs through Branch 1, Branch 2, OCR and ASR/);
  assert.match(html, /Each event inherits only the shared context/);
  assert.match(source, /const endpoint = isKisOrdered[\s\S]*?\/api\/search\/fusion\/kis\/temporal/);
  assert.match(source, /fetch\(endpoint/);
  assert.match(source, /state\.taskType,[\s\S]*?query_bundle: queryBundle,[\s\S]*?events/);
  assert.match(source, /\/search\/visual-fusion/);
  assert.match(source, /"visual_fusion"/);
  assert.match(source, /kisTemporalIntersection: null/);
  assert.match(source, /state\.kisTemporalIntersection = intersectionState/);
  assert.match(source, /sequenceIsComplete/);
  assert.match(html, /Visual trio search/);
  assert.doesNotMatch(
    source.match(/async function searchVideoWithText[\s\S]*?\n}\n/)?.[0] || "",
    /\/search\/siglip/,
  );
});

test("branch workspaces keep full pools but render only the first 150", () => {
  const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
  const source = readFileSync(new URL("../app.js", import.meta.url), "utf8");
  const synthetic = Array.from({ length: 501 }, (_, index) => ({ frame_uid: `L21_V001:${index}` }));
  const split = splitResultPool(synthetic);
  assert.equal(split.all, synthetic);
  assert.equal(split.all.length, 501);
  assert.equal(split.visible.length, 150);
  assert.equal(split.visible[0], synthetic[0]);
  assert.equal(split.visible[149], synthetic[149]);
  assert.equal(split.all[150], synthetic[150]);
  const container = {
    children: [],
    replaceChildren(...nodes) {
      this.children = nodes;
    },
  };
  const rendered = renderVisibleResultPool(
    container,
    synthetic,
    (visible, target) => target.replaceChildren(...visible.map((item) => ({ item }))),
  );
  assert.equal(rendered.all.length, 501);
  assert.equal(rendered.visible.length, 150);
  assert.equal(container.children.length, 150);
  assert.equal(container.children[149].item, synthetic[149]);
  assert.equal(rendered.all[150], synthetic[150]);
  const ocrResult = {
    ocr_text: "món ăn",
    ocr_best_query_role: "entity",
    ocr_best_query_language: "vi",
    bm25_raw: -1.25,
    bm25_relevance: 0.75,
    token_coverage: 0.5,
    ngram_coverage: 0.25,
    pts_time_s: 12.5,
  };
  const ocrContainer = { children: [], replaceChildren(...nodes) { this.children = nodes; } };
  renderVisibleResultPool(
    ocrContainer,
    Array.from({ length: 151 }, () => ocrResult),
    (visible, target) => target.replaceChildren(...visible.map((item) => ({ textContent: formatOcrEvidence(item) }))),
  );
  assert.equal(ocrContainer.children.length, 150);
  assert.match(ocrContainer.children[0].textContent, /entity:vi/);
  assert.match(ocrContainer.children[0].textContent, /BM25/);
  assert.match(ocrContainer.children[0].textContent, /token 0\.500/);
  assert.match(ocrContainer.children[0].textContent, /bigram 0\.250/);
  assert.match(ocrContainer.children[0].textContent, /12\.50s/);
  assert.match(html, /API pool: 1,500/);
  assert.match(html, /API pool: 500/);
  assert.match(source, /renderVisibleResultPool\(/);
  assert.match(source, /branchPoolCountLabel\(payload, returnedCount, 500\)/);
  assert.match(source, /branchPoolCountLabel\(payload, returnedCount, 1500\)/);
  assert.match(source, /final_top_k: 1500/);
  assert.match(source, /final_top_k: 500/);
});

test("KIS inspector formats compact four-branch evidence behaviorally", () => {
  const beit3QueryScores = Object.fromEntries(
    ["original", "entity", "action", "context", "synonym", "keyword"].map((role, index) => [
      role,
      { role, language: "en", cosine: 0.81 - index * 0.01, rank: index + 1 },
    ]),
  );
  const evidence = formatKisFusionEvidence({
    rrf_score: 0.02,
    pre_rerank_rank: 4,
    branch_agreement_count: 3,
    branch_ranks: { branch1: 4, branch2: 9, ocr: 2 },
    rrf_contributions: { branch1: 0.006, branch2: 0.004, ocr: 0.002, asr: 0 },
    beit3_raw_cosine: 0.81,
    rank_delta: 2,
    beit3_query_scores: beit3QueryScores,
    rerank_formula: {
      beit3_weight: 0.25,
      previous_weight: 0.75,
      previous_score_field: "rrf_score",
      expression: "beit3_weight * normalized_beit3 + previous_weight * normalized_rrf",
    },
    branch_provenance: {
      branch1: { final_score: 0.8, best_query_role: "entity", best_query_language: "vi" },
      branch2: {
        hybrid_score: 0.7,
        dense_best_query_role: "entity",
        dense_best_query_language: "en",
        dam_winner: { description_en: "person near a vehicle" },
      },
      ocr: { ocr_text: "TIN TUC", ocr_best_query_role: "keyword", ocr_best_query_language: "vi" },
      asr: {
        asr_transcript: "mot nguoi dang noi",
        asr_best_query_role: "original",
        asr_best_query_language: "vi",
        asr_segment_id: "seg-1",
        asr_start_s: 1,
        asr_end_s: 2,
      },
    },
  });
  assert.match(evidence, /RRF 0\.02000/);
  assert.match(evidence, /BEiT-3 0\.8100/);
  assert.match(evidence, /formula 25% BEiT-3 \+ 75% RRF/);
  assert.match(evidence, /3\/4 branches/);
  assert.match(evidence, /rank delta \+2/);
  assert.match(evidence, /Branch 2: .*\(entity:en\)/);
  assert.match(evidence, /OCR: .*\(keyword:vi\)/);
  assert.match(evidence, /segment seg-1/);
  assert.equal(Object.keys(beit3QueryScores).length, 6);
  assert(Object.values(beit3QueryScores).every((score) => score.language === "en"));
  assert.doesNotMatch(evidence, /\u00c2|\u00c3|\u00e2/);
  const cardEvidence = formatKisFusionCardEvidence({
    rrf_score: 0.02,
    beit3_raw_cosine: 0.81,
    pre_rerank_rank: 4,
    rank_delta: 2,
    branch_agreement_count: 3,
    rerank_formula: {
      beit3_weight: 0.25,
      previous_weight: 0.75,
    },
  });
  assert.match(cardEvidence, /RRF 0\.02000/);
  assert.match(cardEvidence, /BEiT-3 0\.8100/);
  assert.match(cardEvidence, /formula 25% BEiT-3 \+ 75% RRF/);
  assert.match(cardEvidence, /rank delta \+2/);

  const tailEvidence = formatKisFusionEvidence({
    rrf_score: 0.01,
    pre_rerank_rank: 120,
    branch_agreement_count: 1,
    branch_ranks: { branch1: 120 },
    rrf_contributions: { branch1: 0.002, branch2: 0, ocr: 0, asr: 0 },
    branch_provenance: { branch1: { final_score: 0.2, best_query_role: "keyword", best_query_language: "en" } },
  });
  assert.match(tailEvidence, /BEiT-3 not scored/);
  assert.match(tailEvidence, /rank delta -/);
  assert.doesNotMatch(tailEvidence, /formula/);
  assert.doesNotMatch(
    formatKisFusionCardEvidence({
      rrf_score: 0.01,
      pre_rerank_rank: 120,
      branch_agreement_count: 1,
    }),
    /formula|rank delta [+-]\d/,
  );
});

test("KIS display helper keeps all state but passes only 150 cards to the DOM", () => {
  const pool = Array.from({ length: 151 }, (_, index) => ({ frame_uid: `f-${index}` }));
  const container = { rendered: [] };
  let visible;
  const split = renderVisibleResultPool(container, pool, (items, target) => {
    visible = items;
    target.rendered = items;
  });
  assert.equal(split.all, pool);
  assert.equal(split.returnedCount, 151);
  assert.equal(visible.length, 150);
  assert.equal(container.rendered.length, 150);
  assert.equal(split.all[150].frame_uid, "f-150");
});

test("branch count badge uses the actual returned pool when it is below the gate", () => {
  assert.equal(branchPoolCountLabel({ gate_top_k: 500 }, 500, 500), "150 shown of 500");
  assert.equal(branchPoolCountLabel({ gate_top_k: 500 }, 200, 500), "150 shown of 200");
  assert.equal(branchPoolCountLabel({ gate_top_k: 500 }, 80, 500), "80 shown of 80");
  assert.equal(branchPoolCountLabel({ gate_top_k: 500 }, 0, 500), "0 shown of 0");
});

test("Branch 3 keeps winning ASR evidence separate from macro context", () => {
  const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
  const source = readFileSync(new URL("../app.js", import.meta.url), "utf8");
  assert.match(html, /id="insp-asr-evidence"/);
  assert.match(html, /id="insp-asr-context"/);
  assert.match(source, /inspAsrEvidence/);
  assert.match(source, /inspAsrContext/);
  assert.doesNotMatch(source, /inspAsrText\.textContent\s*=\s*data\.macro_audio_transcript/);
});

test("OCR inspector preserves an explicitly empty winning value", () => {
  assert.equal(
    resolveWinningOcrText({ ocr_text: "" }, { ocr_text: "detail fallback" }),
    "",
  );
  assert.equal(
    resolveWinningOcrText({ frame_uid: "L21_V001:1" }, { ocr_text: "detail fallback" }),
    "detail fallback",
  );
  assert.equal(
    resolveWinningOcrText(
      { ocr_text: "", retrieval_modality: "branch1" },
      { ocr_text: "detail fallback" },
      false,
    ),
    "detail fallback",
  );
});

test("video controller reports time during polling, seeking, and pause sampling", () => {
  const source = readFileSync(new URL("../src/youtube-video-view.ts", import.meta.url), "utf8");
  assert.match(source, /options\.onTimeChange\?\.\(currentSeconds\)/);
  assert.match(source, /function samplePlayerTime\(\)/);
  assert.match(source, /if \(!playing\) samplePlayerTime\(\)/);
  assert.match(source, /seekTo\(seconds\)[\s\S]*?updatePlaybackUi\(\)/);
});

test("async workspaces, exact source-frame selection, and edited CSV review guard stale responses", () => {
  const source = readFileSync(new URL("../app.js", import.meta.url), "utf8");
  assert.match(source, /function selectImageQueryFile[\s\S]*?imageSearchRequestId \+= 1/);
  assert.match(source, /async function loadStandaloneVideo[\s\S]*?standaloneScopedRequestId \+= 1/);
  assert.match(source, /currentSnapshot\.contextKey !== startSnapshot\.contextKey[\s\S]*?currentSnapshot\.mode !== startSnapshot\.mode/);
  assert.match(source, /\/api\/frame\/\$\{encodeURIComponent\(videoId\)\}/);
  assert.match(source, /\/api\/video\/\$\{encodeURIComponent\(videoId\)\}\/source-frame/);
  assert.match(source, /async function selectWatchSourceFrame/);
  assert.match(source, /standaloneVideoController\.seekTo\(Number\(sourceFrame\.pts_time_s\)\)/);
  assert.match(source, /state\.watch\.selected/);
  assert.doesNotMatch(source, /addFrameToSubmission\(state\.watch\.nearest\.frame/);
  assert.match(source, /const isCurrentReview = \(\) =>[\s\S]*?reviewGeneration === csvReviewGeneration[\s\S]*?reviewContext === submissionStore\.getSnapshot\(\)\.contextKey/);
  assert.match(source, /ensureClientBackfill\(request\.manual_selections, request\.candidate_reservoir, 100, contextKey\)/);
});

test("KIS is the default workspace and the renovated controls are present", () => {
  const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
  const source = readFileSync(new URL("../app.js", import.meta.url), "utf8");
  assert.match(html, /data-workspace="kis_fusion" aria-selected="true"/);
  assert.match(html, /id="kis-sticky-query"/);
  assert.match(html, /id="btn-close-submission-rail"/);
  assert.match(html, /id="submission-related-note"/);
  assert.match(html, /id="btn-add-filmstrip-selection"/);
  assert.match(html, /id="watch-exact-frame-input"/);
  assert.match(source, /setWorkspace\("kis_fusion"\)/);
  assert.match(source, /fetch\("\/api\/submission\/related-frames"/);
  assert.match(source, /submissionStore\.addFrames\(canonicalFrames/);
  assert.doesNotMatch(html, /class="submission-video-edit"|class="submission-frame-edit"/);
});
