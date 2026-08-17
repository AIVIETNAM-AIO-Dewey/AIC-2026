export function AnswerPanel({ answer, confidence }: { answer?: string; confidence?: number }) {
  return (
    <aside>
      <h2>Câu trả lời Q&A</h2>
      <textarea
        defaultValue={answer ?? ""}
        aria-label="Editable answer"
        placeholder="Câu trả lời xuất hiện ở đây"
      />
      <p>{confidence === undefined ? "" : `Độ tin cậy: ${(confidence * 100).toFixed(0)}%`}</p>
    </aside>
  );
}
