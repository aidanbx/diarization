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


def run_align(
    words: WordTimestamps,
    labels: SpeakerLabels,
    *,
    snap_threshold_s: float = 0.35,
) -> AlignedTranscript:
    aligned: list[AlignedWord] = []
    for word in words.words:
        speaker = _assign_speaker(word, labels.labels, snap_threshold_s=snap_threshold_s)
        aligned.append(AlignedWord(
            word=word.word,
            start=word.start,
            end=word.end,
            speaker_id=speaker,
            confidence=word.confidence,
        ))

    _repair_unknown_spans(aligned, snap_threshold_s=snap_threshold_s)
    turns = _build_turns(aligned)
    return AlignedTranscript(words=aligned, turns=turns, audio_hash=words.audio_hash)


def _assign_speaker(word: WordTimestamp, segs: list[SpeakerLabel], *, snap_threshold_s: float) -> str:
    if not segs:
        return UNKNOWN

    midpoint = (word.start + word.end) / 2.0

    containing = [seg for seg in segs if seg.start <= midpoint <= seg.end]
    if containing:
        return _best_overlap_speaker(word, containing)

    overlapping = [seg for seg in segs if min(word.end, seg.end) - max(word.start, seg.start) > 0.0]
    if overlapping:
        return _best_overlap_speaker(word, overlapping)

    prev_seg = None
    next_seg = None
    for seg in segs:
        if seg.end <= midpoint and (prev_seg is None or seg.end > prev_seg.end):
            prev_seg = seg
        if seg.start >= midpoint and (next_seg is None or seg.start < next_seg.start):
            next_seg = seg

    if prev_seg and next_seg and prev_seg.speaker_id == next_seg.speaker_id:
        if midpoint - prev_seg.end <= snap_threshold_s and next_seg.start - midpoint <= snap_threshold_s:
            return prev_seg.speaker_id

    nearest_seg = None
    nearest_gap = float("inf")
    for seg in (prev_seg, next_seg):
        if seg is None:
            continue
        gap = _segment_gap(word, seg)
        if gap < nearest_gap:
            nearest_gap = gap
            nearest_seg = seg

    if nearest_seg is not None and nearest_gap <= snap_threshold_s:
        return nearest_seg.speaker_id

    return UNKNOWN


def _best_overlap_speaker(word: WordTimestamp, segs: list[SpeakerLabel]) -> str:
    best_speaker = UNKNOWN
    best_overlap = 0.0
    for seg in segs:
        overlap = min(word.end, seg.end) - max(word.start, seg.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = seg.speaker_id
    return best_speaker


def _segment_gap(word: WordTimestamp, seg: SpeakerLabel) -> float:
    if word.end < seg.start:
        return seg.start - word.end
    if seg.end < word.start:
        return word.start - seg.end
    return 0.0


def _repair_unknown_spans(words: list[AlignedWord], *, snap_threshold_s: float) -> None:
    if not words:
        return

    for i, word in enumerate(words):
        if word.speaker_id != UNKNOWN:
            continue

        prev_word = words[i - 1] if i > 0 else None
        next_word = words[i + 1] if i + 1 < len(words) else None

        prev_speaker = prev_word.speaker_id if prev_word and prev_word.speaker_id != UNKNOWN else None
        next_speaker = next_word.speaker_id if next_word and next_word.speaker_id != UNKNOWN else None

        if prev_speaker and prev_speaker == next_speaker:
            word.speaker_id = prev_speaker
            continue

        if prev_speaker and prev_word and word.start - prev_word.end <= snap_threshold_s:
            word.speaker_id = prev_speaker
            continue

        if next_speaker and next_word and next_word.start - word.end <= snap_threshold_s:
            word.speaker_id = next_speaker


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
