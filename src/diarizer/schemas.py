from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class AudioInput(BaseModel):
    path: str
    sample_rate: int = 16000
    duration_s: float
    hash: str  # sha256 of file content, used as cache key


class SpeechRegion(BaseModel):
    start: float  # seconds
    end: float


class SpeechRegions(BaseModel):
    regions: list[SpeechRegion]
    audio_hash: str


class SpeakerFrame(BaseModel):
    """Per-frame speaker activity scores from the segmentation model."""
    timestamp: float
    scores: list[float]  # one score per speaker slot (typically 3 for pyannote)


class SegmentationFrames(BaseModel):
    frames: list[SpeakerFrame]
    frame_shift_s: float  # seconds between frames
    audio_hash: str


class SpeakerSegment(BaseModel):
    segment_id: int
    start: float
    end: float
    embedding: list[float]


class SpeakerEmbeddings(BaseModel):
    segments: list[SpeakerSegment]
    audio_hash: str


class SpeakerLabel(BaseModel):
    segment_id: int
    start: float
    end: float
    speaker_id: str  # e.g. "SPEAKER_00"


class SpeakerLabels(BaseModel):
    labels: list[SpeakerLabel]
    num_speakers: int
    audio_hash: str


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    confidence: Optional[float] = None


class WordTimestamps(BaseModel):
    words: list[WordTimestamp]
    language: str
    audio_hash: str


class AlignedWord(BaseModel):
    word: str
    start: float
    end: float
    speaker_id: str
    confidence: Optional[float] = None


class SpeakerTurn(BaseModel):
    speaker_id: str
    start: float
    end: float
    text: str


class AlignedTranscript(BaseModel):
    words: list[AlignedWord]
    turns: list[SpeakerTurn]  # contiguous speaker turns
    audio_hash: str
