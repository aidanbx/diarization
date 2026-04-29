"""Agglomerative clustering of speaker embeddings → SpeakerLabels.

Uses cosine distance + average linkage. The distance threshold is set in
config.yaml (cluster.threshold). No speaker count is required; the threshold
controls how aggressively segments are merged.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

from diarizer.schemas import SpeakerEmbeddings, SpeakerLabel, SpeakerLabels


def run_cluster(embeddings: SpeakerEmbeddings, threshold: float = 0.7) -> SpeakerLabels:
    if not embeddings.segments:
        return SpeakerLabels(labels=[], num_speakers=0, audio_hash=embeddings.audio_hash)

    X = np.array([s.embedding for s in embeddings.segments], dtype=np.float32)
    X = normalize(X)  # L2-norm → cosine distance = 1 - dot

    if len(X) == 1:
        labels_arr = np.array([0])
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="cosine",
            linkage="average",
        )
        labels_arr = clustering.fit_predict(X)

    num_speakers = int(labels_arr.max()) + 1
    labels: list[SpeakerLabel] = []
    for seg, label in zip(embeddings.segments, labels_arr):
        labels.append(SpeakerLabel(
            segment_id=seg.segment_id,
            start=seg.start,
            end=seg.end,
            speaker_id=f"SPEAKER_{int(label):02d}",
        ))

    return SpeakerLabels(
        labels=sorted(labels, key=lambda x: x.start),
        num_speakers=num_speakers,
        audio_hash=embeddings.audio_hash,
    )
