"""Versioned query-decomposition contract shared by online components."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .models import StrictModel


class TaskType(str, Enum):
    KIS = "kis"
    QA = "qa"
    TRAKE = "trake"


class AnswerSource(str, Enum):
    VISUAL = "visual"
    OCR = "ocr"
    SPEECH = "speech"
    AUDIO_EVENT = "audio_event"


class TemporalOperator(str, Enum):
    STATE = "state"
    ONSET = "onset"
    OFFSET = "offset"
    EXTREMUM = "extremum"


class QueryEvent(StrictModel):
    label: str = Field(min_length=1)
    scene_en: str = Field(min_length=1)
    objects_en: list[str] = Field(default_factory=list)
    ocr_vi: list[str] = Field(default_factory=list)
    audio_vi: list[str] = Field(default_factory=list)
    audio_events_en: list[str] = Field(default_factory=list)
    temporal_operator: Literal["state", "onset", "offset", "extremum"] = "state"

    @field_validator("objects_en", "ocr_vi", "audio_vi", "audio_events_en")
    @classmethod
    def validate_non_empty_signals(cls, values: list[str]) -> list[str]:
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("query signal lists cannot contain empty strings")
        return values


class QuerySpec(StrictModel):
    schema_version: Literal["aic26.query.v1"] = "aic26.query.v1"
    task_type: Literal["kis", "qa", "trake"]
    raw_query_vi: str = Field(min_length=1)
    scene_en: str = Field(min_length=1)
    objects_en: list[str] = Field(default_factory=list)
    ocr_vi: list[str] = Field(default_factory=list)
    audio_vi: list[str] = Field(default_factory=list)
    audio_events_en: list[str] = Field(default_factory=list)
    question_vi: str | None = None
    question_en: str | None = None
    answer_sources: list[Literal["visual", "ocr", "speech", "audio_event"]] = Field(
        default_factory=list
    )
    events: list[QueryEvent] | None = None

    @field_validator("objects_en", "ocr_vi", "audio_vi", "audio_events_en")
    @classmethod
    def validate_non_empty_signals(cls, values: list[str]) -> list[str]:
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("query signal lists cannot contain empty strings")
        return values

    @field_validator("answer_sources")
    @classmethod
    def validate_unique_answer_sources(
        cls, values: list[Literal["visual", "ocr", "speech", "audio_event"]]
    ) -> list[Literal["visual", "ocr", "speech", "audio_event"]]:
        if len(values) != len(set(values)):
            raise ValueError("answer_sources cannot contain duplicates")
        return values

    @model_validator(mode="after")
    def validate_task_shape(self) -> QuerySpec:
        if self.task_type == TaskType.QA:
            if not self.question_vi or not self.question_en:
                raise ValueError("Q&A queries require question_vi and question_en")
            if self.events is not None:
                raise ValueError("Q&A queries cannot contain TRAKE events")
        elif self.task_type == TaskType.TRAKE:
            if not self.events:
                raise ValueError("TRAKE queries require at least one ordered event")
            if self.question_vi is not None or self.question_en is not None:
                raise ValueError("TRAKE queries cannot contain Q&A question fields")
        else:
            if self.events is not None:
                raise ValueError("KIS queries cannot contain TRAKE events")
            if self.question_vi is not None or self.question_en is not None:
                raise ValueError("KIS queries cannot contain Q&A question fields")
        return self
