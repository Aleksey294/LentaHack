from __future__ import annotations

from .schemas import Detection, TrackSample, TrackState
from .utils import iou


class SimpleIOUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 8) -> None:
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self._next_track_id = 1
        self._active_tracks: list[TrackState] = []
        self._finished_tracks: list[TrackState] = []

    def update(self, samples: list[TrackSample]) -> None:
        unmatched_tracks = set(range(len(self._active_tracks)))
        unmatched_samples = set(range(len(samples)))
        matches: list[tuple[int, int]] = []

        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._active_tracks):
            for sample_index, sample in enumerate(samples):
                score = iou(track.last_detection, sample.detection)
                if score >= self.iou_threshold:
                    candidates.append((score, track_index, sample_index))

        for _, track_index, sample_index in sorted(candidates, reverse=True):
            if track_index not in unmatched_tracks or sample_index not in unmatched_samples:
                continue
            matches.append((track_index, sample_index))
            unmatched_tracks.remove(track_index)
            unmatched_samples.remove(sample_index)

        for track_index, sample_index in matches:
            self._active_tracks[track_index].add_sample(samples[sample_index])

        for track_index in sorted(unmatched_tracks):
            self._active_tracks[track_index].misses += 1

        survivors: list[TrackState] = []
        for track in self._active_tracks:
            if track.misses > self.max_misses:
                if track.samples:
                    self._finished_tracks.append(track)
            else:
                survivors.append(track)
        self._active_tracks = survivors

        for sample_index in sorted(unmatched_samples):
            sample = samples[sample_index]
            track = TrackState(track_id=self._next_track_id)
            self._next_track_id += 1
            track.add_sample(sample)
            self._active_tracks.append(track)

    def finish(self) -> list[TrackState]:
        for track in self._active_tracks:
            if track.samples:
                self._finished_tracks.append(track)
        self._active_tracks = []
        return sorted(self._finished_tracks, key=lambda item: item.track_id)

