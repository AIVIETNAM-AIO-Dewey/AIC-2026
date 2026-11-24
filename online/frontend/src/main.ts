const demoMode = new URLSearchParams(window.location.search).get("demo");

if (demoMode === "video-ui") {
  void import("./video-demo");
} else {
  void import("../app.js");
}
