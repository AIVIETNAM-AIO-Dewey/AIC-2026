import assert from "node:assert/strict";
import test from "node:test";

import {
  assessKisPlanAlignment,
  canonicalKisBundleSignature,
  canonicalKisEventsSignature,
  formatKisOverallQuery,
  formatOrderedKisEvents,
  hasCompleteKisBundle,
  parseOrderedKisEvents,
} from "../src/kis-query-plan.js";

const roles = ["original", "entity", "action", "context", "synonym", "keyword"];

function bundle() {
  return {
    schema_version: "branch1.query.v1",
    queries: roles.map((role) => ({
      role,
      vi: role === "original" ? "Vườn trái cây có sầu riêng rồi măng cụt" : `vườn ${role}`,
      en: role === "original" ? "A fruit orchard shows durian then mangosteen" : `orchard ${role}`,
    })),
  };
}

test("one KIS bundle exposes one bilingual overall query", () => {
  const value = bundle();
  assert.equal(hasCompleteKisBundle(value), true);
  assert.equal(
    formatKisOverallQuery(value),
    "Vườn trái cây có sầu riêng rồi măng cụt || A fruit orchard shows durian then mangosteen",
  );
  assert.equal(canonicalKisBundleSignature(value), canonicalKisBundleSignature({
    ...value,
    queries: [...value.queries].reverse(),
  }));
});

test("ordered event text round-trips bilingual event identity and order", () => {
  const text = [
    "E1: Có trái sầu riêng || Durian fruit is visible",
    "E2: Có trái măng cụt || Mangosteen fruit is visible",
  ].join("\n");
  const events = parseOrderedKisEvents(text);
  assert.deepEqual(events.map((event) => event.order), [1, 2]);
  assert.equal(events[0].vi, "Có trái sầu riêng");
  assert.equal(events[1].en, "Mangosteen fruit is visible");
  assert.equal(canonicalKisEventsSignature(formatOrderedKisEvents(events)), canonicalKisEventsSignature(text));
});

test("alignment is advisory and identifies only unrelated event lines", () => {
  const value = bundle();
  const relatedEvents = parseOrderedKisEvents([
    "Có trái sầu riêng || Durian fruit is visible",
    "Có trái măng cụt || Mangosteen fruit is visible",
  ].join("\n"));
  const unrelatedEvents = [...relatedEvents, {
    order: 3,
    description: "A train arrives",
    vi: "Một đoàn tàu đến ga",
    en: "A train arrives at a station",
  }];

  assert.equal(assessKisPlanAlignment(formatKisOverallQuery(value), value, relatedEvents).aligned, true);
  assert.deepEqual(
    assessKisPlanAlignment(formatKisOverallQuery(value), value, unrelatedEvents).mismatchedEventOrders,
    [3],
  );
});
