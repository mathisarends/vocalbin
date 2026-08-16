from typing import Annotated, Literal

from pydantic import BaseModel, Field


class Connected(BaseModel):
    type: Literal["connected"] = "connected"
    request_id: str


class TurnStart(BaseModel):
    type: Literal["turn.start"] = "turn.start"
    request_id: str


class TurnUpdate(BaseModel):
    type: Literal["turn.update"] = "turn.update"
    request_id: str
    transcript: str


class TurnEagerEnd(BaseModel):
    type: Literal["turn.eager_end"] = "turn.eager_end"
    request_id: str
    transcript: str


class TurnResume(BaseModel):
    type: Literal["turn.resume"] = "turn.resume"
    request_id: str


class TurnEnd(BaseModel):
    type: Literal["turn.end"] = "turn.end"
    request_id: str
    transcript: str


type Event = Annotated[
    Connected | TurnStart | TurnUpdate | TurnEagerEnd | TurnResume | TurnEnd,
    Field(discriminator="type"),
]


__all__ = [
    "Connected",
    "Event",
    "TurnEagerEnd",
    "TurnEnd",
    "TurnResume",
    "TurnStart",
    "TurnUpdate",
]
