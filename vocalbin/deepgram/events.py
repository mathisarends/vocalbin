from typing import Annotated, Literal

from pydantic import BaseModel, Field


class Word(BaseModel):
    word: str
    confidence: float
    start: float | None = None
    end: float | None = None


class Connected(BaseModel):
    type: Literal["connected"] = "connected"
    request_id: str
    sequence_id: int


class _TurnEvent(BaseModel):
    request_id: str
    sequence_id: int
    turn_index: int
    audio_window_start: float
    audio_window_end: float
    transcript: str
    words: list[Word] = Field(default_factory=list)
    end_of_turn_confidence: float
    # Only `flux-general-multi` reports which languages it heard and was hinted.
    languages: list[str] | None = None
    languages_hinted: list[str] | None = None


class TurnStart(_TurnEvent):
    type: Literal["turn.start"] = "turn.start"


class TurnUpdate(_TurnEvent):
    type: Literal["turn.update"] = "turn.update"


class TurnEagerEnd(_TurnEvent):
    type: Literal["turn.eager_end"] = "turn.eager_end"


class TurnResume(_TurnEvent):
    type: Literal["turn.resume"] = "turn.resume"


class TurnEnd(_TurnEvent):
    type: Literal["turn.end"] = "turn.end"


type TurnEvent = TurnStart | TurnUpdate | TurnEagerEnd | TurnResume | TurnEnd

type Event = Annotated[
    Connected | TurnEvent,
    Field(discriminator="type"),
]
