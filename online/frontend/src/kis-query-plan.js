const KIS_QUERY_ROLES = ["original", "entity", "action", "context", "synonym", "keyword"];

const ALIGNMENT_STOPWORDS = new Set([
  "a", "an", "and", "are", "at", "by", "for", "from", "in", "is", "of", "on", "the", "then", "to", "with",
  "image", "frame", "scene", "video",
  "các", "cảnh", "cho", "có", "của", "đang", "được", "hình", "ảnh", "là", "một", "những", "sau", "thì", "trên", "trong", "và", "với",
].map((token) => token.normalize("NFD").replace(/\p{M}/gu, "").toLocaleLowerCase()));

export function normalizeKisPlanText(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}

function normalizedComparison(value) {
  return normalizeKisPlanText(value).toLocaleLowerCase();
}

function roleMap(bundle) {
  if (!bundle || bundle.schema_version !== "branch1.query.v1" || !Array.isArray(bundle.queries)) {
    return new Map();
  }
  const byRole = new Map(bundle.queries.map((query) => [query?.role, query]));
  if (byRole.size !== KIS_QUERY_ROLES.length || !KIS_QUERY_ROLES.every((role) => byRole.has(role))) {
    return new Map();
  }
  return byRole;
}

export function hasCompleteKisBundle(bundle) {
  const byRole = roleMap(bundle);
  return byRole.size === KIS_QUERY_ROLES.length && KIS_QUERY_ROLES.every((role) => {
    const query = byRole.get(role);
    return normalizeKisPlanText(query?.vi) && normalizeKisPlanText(query?.en);
  });
}

export function formatKisOverallQuery(bundle) {
  const original = roleMap(bundle).get("original");
  const vi = normalizeKisPlanText(original?.vi);
  const en = normalizeKisPlanText(original?.en);
  if (!vi) return en;
  if (!en || normalizedComparison(vi) === normalizedComparison(en)) return vi;
  return `${vi} || ${en}`;
}

export function canonicalKisBundleSignature(bundle) {
  if (!hasCompleteKisBundle(bundle)) return "";
  const byRole = roleMap(bundle);
  return JSON.stringify({
    schema_version: "branch1.query.v1",
    queries: KIS_QUERY_ROLES.map((role) => ({
      role,
      vi: normalizeKisPlanText(byRole.get(role)?.vi),
      en: normalizeKisPlanText(byRole.get(role)?.en),
    })),
  });
}

export function parseOrderedKisEvents(value) {
  return String(value || "")
    .split(/\r?\n/)
    .map((line) => line.trim().replace(/^E\d+\s*[:.)-]\s*/i, ""))
    .filter(Boolean)
    .slice(0, 6)
    .map((line, index) => {
      const separator = line.indexOf("||");
      const vi = normalizeKisPlanText(separator >= 0 ? line.slice(0, separator) : line);
      const en = normalizeKisPlanText(separator >= 0 ? line.slice(separator + 2) : line);
      return {
        order: index + 1,
        description: en || vi,
        vi: vi || en,
        en: en || vi,
      };
    })
    .filter((event) => event.description && event.vi && event.en);
}

export function formatOrderedKisEvents(events) {
  return (events || []).slice(0, 6).map((event, index) => {
    const vi = normalizeKisPlanText(event?.vi || event?.description);
    const en = normalizeKisPlanText(event?.en || event?.description);
    const text = vi && en && normalizedComparison(vi) !== normalizedComparison(en)
      ? `${vi} || ${en}`
      : (en || vi);
    return text ? `E${index + 1}: ${text}` : "";
  }).filter(Boolean).join("\n");
}

export function canonicalKisEventsSignature(value) {
  return JSON.stringify(parseOrderedKisEvents(value).map((event) => ({
    order: event.order,
    vi: normalizedComparison(event.vi),
    en: normalizedComparison(event.en),
  })));
}

function meaningfulTokens(value) {
  const normalized = String(value || "")
    .normalize("NFD")
    .replace(/\p{M}/gu, "")
    .toLocaleLowerCase();
  return new Set((normalized.match(/[\p{L}\p{N}]+/gu) || [])
    .filter((token) => token.length >= 3 && !ALIGNMENT_STOPWORDS.has(token)));
}

function textsAreRelated(left, right) {
  const normalizedLeft = normalizedComparison(left);
  const normalizedRight = normalizedComparison(right);
  if (!normalizedLeft || !normalizedRight) return false;
  if (normalizedLeft.includes(normalizedRight) || normalizedRight.includes(normalizedLeft)) return true;
  const leftTokens = meaningfulTokens(left);
  return [...meaningfulTokens(right)].some((token) => leftTokens.has(token));
}

export function assessKisPlanAlignment(sourceQuery, bundle, events) {
  const byRole = roleMap(bundle);
  const original = byRole.get("original") || {};
  const context = byRole.get("context") || {};
  const sourceParts = String(sourceQuery || "").split("||", 2).map(normalizeKisPlanText);
  const originalTexts = [original.vi, original.en].map(normalizeKisPlanText).filter(Boolean);
  const sourceAligned = sourceParts.filter(Boolean).some((source) =>
    originalTexts.some((originalText) => textsAreRelated(source, originalText))
  );
  const parentContext = [sourceQuery, original.vi, original.en, context.vi, context.en]
    .map(normalizeKisPlanText)
    .filter(Boolean)
    .join(" ");
  const mismatchedEventOrders = (events || [])
    .filter((event) => !textsAreRelated(parentContext, `${event?.vi || ""} ${event?.en || ""}`))
    .map((event) => Number(event?.order))
    .filter(Number.isFinite);
  return {
    sourceAligned,
    eventsAligned: mismatchedEventOrders.length === 0,
    mismatchedEventOrders,
    aligned: sourceAligned && mismatchedEventOrders.length === 0,
  };
}

export { KIS_QUERY_ROLES };
