"""Align word timestamps with speaker labels → AlignedTranscript.

Strategy: for each word, find the diarization segment(s) that overlap with it
and assign the speaker with the greatest overlap. Adjacent words with the same
speaker are merged into turns.
"""
from __future__ import annotations

from diarizer.schemas import (
    AlignedTranscript,
    AlignedWord,
    SpeakerLabel,
    SpeakerLabels,
    SpeakerTurn,
    WordTimestamp,
    WordTimestamps,
)

UNKNOWN = "SPEAKER_UNK"


def run_align(words: WordTimestamps, labels: SpeakerLabels) -> AlignedTranscript:
    aligned: list[AlignedWord] = []
    for word in words.words:
        speaker = _assign_speaker(word, labels.labels)
        aligned.append(AlignedWord(
            word=word.word,
            start=word.start,
            end=word.end,
            speaker_id=speaker,
            confidence=word.confidence,
        ))

    turns = _build_turns(aligned)
    return AlignedTranscript(words=aligned, turns=turns, audio_hash=words.audio_hash)


def _assign_speaker(word: WordTimestamp, segs: list[SpeakerLabel]) -> str:
    best_speaker = UNKNOWN
    best_overlap = 0.0
    for seg in segs:
        overlap = min(word.end, seg.end) - max(word.start, seg.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = seg.speaker_id
    return best_speaker


def _build_turns(words: list[AlignedWord]) -> list[SpeakerTurn]:
    if not words:
        return []
    turns: list[SpeakerTurn] = []
    cur_speaker = words[0].speaker_id
    cur_start = words[0].start
    cur_end = words[0].end
    cur_words: list[str] = [words[0].word]

    for w in words[1:]:
        if w.speaker_id == cur_speaker:
            cur_end = w.end
            cur_words.append(w.word)
        else:
            turns.append(SpeakerTurn(
                speaker_id=cur_speaker,
                start=cur_start,
                end=cur_end,
                text=" ".join(cur_words),
            ))
            cur_speaker = w.speaker_id
            cur_start = w.start
            cur_end = w.end
            cur_words = [w.word]

    turns.append(SpeakerTurn(
        speaker_id=cur_speaker,
        start=cur_start,
        end=cur_end,
        text=" ".join(cur_words),
    ))
    return turns
