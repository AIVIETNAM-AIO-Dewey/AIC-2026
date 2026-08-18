from aic2026.contracts.query import QuerySpec

from aic_backend.llm.gpt4o import CapabilityUnavailable
from aic_backend.llm.query_parser import QueryParsingService


class UnavailableGpt:
    def parse(self, *, task_type: str, raw_query_vi: str) -> QuerySpec:
        del task_type, raw_query_vi
        raise CapabilityUnavailable("not configured")


def test_kis_fallback_searches_raw_query_across_available_modalities() -> None:
    parsed = QueryParsingService(UnavailableGpt()).parse(
        task_type="kis", raw_query_vi="non song cung mot dai"
    )

    assert parsed.scene_en == "non song cung mot dai"
    assert parsed.objects_en == ["non song cung mot dai"]
    assert parsed.ocr_vi == ["non song cung mot dai"]
    assert parsed.audio_vi == ["non song cung mot dai"]
    assert parsed.audio_events_en == []
