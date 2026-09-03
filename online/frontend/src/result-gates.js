/**
 * Keep the complete API pool in state while limiting only DOM rendering.
 * Branch services own their API gates; this helper is deliberately a display
 * concern and never changes, sorts, or sends the result pool back upstream.
 */
export const BRANCH_DISPLAY_LIMIT = 150;

/**
 * Report the number of cards shown against the actual API pool.  The server
 * gate is intentionally not used as the denominator when fewer candidates
 * were returned; a result set of 80 must read "80 shown of 80", not "80 of
 * 500".
 */
export function branchPoolCountLabel(_payload, returnedCount, _fallbackGate) {
  const returned = Math.max(0, Number(returnedCount) || 0);
  return `${Math.min(BRANCH_DISPLAY_LIMIT, returned).toLocaleString()} shown of ${returned.toLocaleString()}`;
}

export function splitResultPool(results, displayLimit = BRANCH_DISPLAY_LIMIT) {
  const pool = Array.isArray(results) ? results : [];
  const limit = Number.isFinite(Number(displayLimit))
    ? Math.max(0, Math.floor(Number(displayLimit)))
    : BRANCH_DISPLAY_LIMIT;
  return {
    all: pool,
    visible: pool.slice(0, limit),
    returnedCount: pool.length,
  };
}

/**
 * Apply the display gate at the DOM boundary without mutating the API pool.
 * ``renderVisible`` owns card construction so the helper stays independent of
 * the workbench's DOM implementation while still making the state/DOM split
 * executable in a small browser test.
 */
export function renderVisibleResultPool(
  container,
  results,
  renderVisible,
  displayLimit = BRANCH_DISPLAY_LIMIT,
) {
  if (!container || typeof renderVisible !== "function") {
    throw new TypeError("container and renderVisible are required");
  }
  const split = splitResultPool(results, displayLimit);
  renderVisible(split.visible, container);
  return split;
}

/** Format the evidence fields shown on an OCR result card/inspector. */
export function formatOcrEvidence(item = {}) {
  const matches = (item.matched_terms || item.matched_keywords || []).join(", ");
  const role = item.ocr_best_query_role
    ? ` · ${item.ocr_best_query_role}${item.ocr_best_query_language ? `:${item.ocr_best_query_language}` : ""}`
    : "";
  const score = item.ocr_normalized_score == null
    ? ""
    : ` · norm ${Number(item.ocr_normalized_score).toFixed(3)}`;
  const combined = item.ocr_raw_score == null
    ? ""
    : ` · combined ${Number(item.ocr_raw_score).toFixed(3)}`;
  const bm25Raw = item.bm25_raw == null
    ? ""
    : ` · BM25 raw ${Number(item.bm25_raw).toFixed(3)}`;
  const bm25 = item.bm25_relevance == null
    ? ""
    : ` · BM25 ${Number(item.bm25_relevance).toFixed(3)}`;
  const token = item.token_coverage == null
    ? ""
    : ` · token ${Number(item.token_coverage).toFixed(3)}`;
  const ngram = item.ngram_coverage == null
    ? ""
    : ` · bigram ${Number(item.ngram_coverage).toFixed(3)}`;
  const time = item.pts_time_s == null
    ? ""
    : ` · ${Number(item.pts_time_s).toFixed(2)}s`;
  return `${matches ? `Matched: ${matches} · ` : ""}${item.ocr_text || "No OCR text"}${role}${combined}${score}${bm25Raw}${bm25}${token}${ngram}${time}`;
}

/**
 * Keep the text that participated in OCR ranking authoritative in the
 * inspector. Detail metadata may fill the field for a non-OCR result whose
 * base-frame OCR field is only an empty placeholder. An explicitly empty
 * winning value remains authoritative when preserveEmptyWinning is true.
 */
export function resolveWinningOcrText(
  activeItem = {},
  detailItem = {},
  preserveEmptyWinning = true,
) {
  if (
    activeItem
    && Object.prototype.hasOwnProperty.call(activeItem, "ocr_text")
    && (preserveEmptyWinning || String(activeItem.ocr_text ?? ""))
  ) {
    return String(activeItem.ocr_text ?? "");
  }
  return String(detailItem?.ocr_text ?? "");
}

/**
 * Format compact evidence from the final KIS fusion response.  This is kept
 * as a pure helper so the browser inspector and the Node contract tests use
 * the same interpretation of nested branch provenance.
 */
export function formatKisFusionEvidence(item = {}) {
  const branches = item.branch_provenance || {};
  const contributions = item.rrf_contributions || {};
  const ranks = item.branch_ranks || {};
  const agreement = Number(item.branch_agreement_count || 0);
  const rrf = item.rrf_score == null ? "-" : Number(item.rrf_score).toFixed(5);
  const beit = item.beit3_raw_cosine == null
    ? "not scored"
    : Number(item.beit3_raw_cosine).toFixed(4);
  const delta = item.rank_delta == null
    ? "-"
    : (Number(item.rank_delta) > 0 ? `+${item.rank_delta}` : String(item.rank_delta));
  const preRank = item.pre_rerank_rank == null ? "-" : `#${item.pre_rerank_rank}`;
  const formula = item.rerank_formula
    && Number.isFinite(Number(item.rerank_formula.beit3_weight))
    && Number.isFinite(Number(item.rerank_formula.previous_weight))
    ? `formula ${Number(item.rerank_formula.beit3_weight * 100).toFixed(0)}% BEiT-3 + ${Number(item.rerank_formula.previous_weight * 100).toFixed(0)}% RRF`
    : "";
  const branchSummary = [
    ["Branch 1", "branch1", branches.branch1],
    ["Branch 2", "branch2", branches.branch2],
    ["OCR", "ocr", branches.ocr],
    ["ASR", "asr", branches.asr],
  ].map(([label, branchKey, evidence]) => {
    if (!evidence) return `${label}: not observed`;
    const role = evidence.best_query_role
      || evidence.dense_best_query_role
      || evidence.sparse_best_query_role
      || evidence.beit3_best_query_role
      || evidence.ocr_best_query_role
      || evidence.asr_best_query_role
      || "-";
    const language = evidence.best_query_language
      || evidence.dense_best_query_language
      || evidence.sparse_best_query_language
      || evidence.beit3_best_query_language
      || evidence.ocr_best_query_language
      || evidence.asr_best_query_language
      || "-";
    const score = evidence.final_score
      ?? evidence.reranked_score
      ?? evidence.hybrid_score
      ?? evidence.ocr_normalized_score
      ?? evidence.asr_normalized_score;
    const rank = ranks[branchKey] == null ? "-" : `#${ranks[branchKey]}`;
    const contribution = contributions[branchKey] == null
      ? "0"
      : Number(contributions[branchKey]).toFixed(5);
    const evidenceDetail = branchKey === "asr"
      ? [
        evidence.asr_segment_id ? `segment ${evidence.asr_segment_id}` : "",
        evidence.asr_start_s == null
          ? ""
          : `span ${Number(evidence.asr_start_s).toFixed(2)}-${Number(evidence.asr_end_s ?? evidence.asr_start_s).toFixed(2)}s`,
        evidence.asr_transcript ? `\"${evidence.asr_transcript}\"` : "",
      ].filter(Boolean).join(" ")
      : branchKey === "branch2"
        ? (evidence.dam_winner?.description_en || evidence.sparse_winner?.description_en || "")
        : branchKey === "ocr"
          ? (evidence.ocr_text || evidence.full_text || "")
          : "";
    return `${label}: ${score == null ? "-" : Number(score).toFixed(4)} (${role}:${language}) rank ${rank} contrib ${contribution}${evidenceDetail ? ` · ${evidenceDetail}` : ""}`;
  });
  return [
    `RRF ${rrf}`,
    `BEiT-3 ${beit}`,
    formula,
    `${agreement}/4 branches`,
    `pre ${preRank}, rank delta ${delta}`,
    branchSummary.join(" | "),
  ].filter(Boolean).join(" | ");
}

/** Compact card summary derived from the same canonical KIS formatter. */
export function formatKisFusionCardEvidence(item = {}) {
  return formatKisFusionEvidence(item).split(" | ").slice(0, 5).join(" | ");
}
