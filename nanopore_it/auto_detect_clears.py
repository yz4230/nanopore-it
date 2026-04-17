import numpy as np
import numpy.typing as npt

__all__ = ["detect_clear_regions"]


def detect_clear_regions(
    waveform: npt.NDArray[np.float64],
    *,
    baseline: float | None = None,
    baseline_std: float | None = None,
    spike_std_threshold: float = 10.0,
    baseline_window_std: float = 1.0,
    sample_rate_hz: float | None = None,
    extra_relaxation_ms: float = 10.0,
    extra_relaxation_samples: int | None = None,
) -> npt.NDArray[np.int_]:
    """Return cut regions detected from a 1D waveform.

    This extracts the calculation used by the GUI's "Auto Detect Clears"
    feature into a standalone function. The returned regions are inclusive
    start / exclusive end sample indices with shape ``(n_regions, 2)``.

    The default behavior mirrors the existing implementation as closely as
    possible:
    - spikes are detected where ``abs(waveform) > baseline + threshold * std``
    - baseline points are samples where
      ``abs(waveform - baseline) < baseline_window_std * std``
    - the relaxation padding is applied using the index count within the
      baseline-neighborhood samples, matching the current GUI logic

    Parameters
    ----------
    waveform:
        One-dimensional signal to analyze.
    baseline, baseline_std:
        Baseline statistics. If omitted, they are estimated from the waveform
        using ``median`` and ``std`` respectively.
    spike_std_threshold:
        Spike threshold in units of ``baseline_std``.
    baseline_window_std:
        Width of the "near baseline" window in units of ``baseline_std``.
    sample_rate_hz, extra_relaxation_ms, extra_relaxation_samples:
        Relaxation padding after a detected spike. Pass
        ``extra_relaxation_samples`` to control it directly, or pass
        ``sample_rate_hz`` to convert ``extra_relaxation_ms`` to samples.
        When neither is provided, no extra padding is added.
    """

    signal = np.asarray(waveform, dtype=float)
    if signal.ndim != 1:
        msg = "waveform must be one-dimensional"
        raise ValueError(msg)
    if signal.size == 0:
        return np.empty((0, 2), dtype=int)

    resolved_baseline = float(np.median(signal) if baseline is None else baseline)
    resolved_std = float(np.std(signal) if baseline_std is None else baseline_std)
    if resolved_std < 0:
        msg = "baseline_std must be non-negative"
        raise ValueError(msg)
    if resolved_std == 0:
        return np.empty((0, 2), dtype=int)

    relaxation_samples = _resolve_relaxation_samples(
        sample_rate_hz=sample_rate_hz,
        extra_relaxation_ms=extra_relaxation_ms,
        extra_relaxation_samples=extra_relaxation_samples,
    )

    spike_indices = np.flatnonzero(
        np.abs(signal) > resolved_baseline + spike_std_threshold * resolved_std
    )
    baseline_indices = np.flatnonzero(
        np.abs(signal - resolved_baseline) < baseline_window_std * resolved_std
    )
    if spike_indices.size == 0 or baseline_indices.size == 0:
        return np.empty((0, 2), dtype=int)

    regions: list[tuple[int, int]] = []
    baseline_cursor = 0
    current_endpoint = 0
    last_sample = signal.size - 1

    spike_pos = 0
    n_spikes = spike_indices.size
    while spike_pos < n_spikes:
        spike_start = int(spike_indices[spike_pos])
        if spike_start < current_endpoint:
            spike_pos += int(
                np.searchsorted(spike_indices[spike_pos:], current_endpoint)
            )
            continue

        baseline_cursor += int(
            np.searchsorted(
                baseline_indices[baseline_cursor:], spike_start, side="right"
            )
        )

        if baseline_cursor > 0:
            region_start = int(baseline_indices[baseline_cursor - 1])
        else:
            region_start = 0

        endpoint_cursor = min(
            baseline_cursor + relaxation_samples,
            baseline_indices.size - 1,
        )
        region_end = int(baseline_indices[endpoint_cursor])
        if region_end <= region_start:
            region_end = min(max(spike_start + 1, region_start + 1), signal.size)
        else:
            region_end = min(region_end + 1, signal.size)

        current_endpoint = min(region_end, last_sample)
        regions.append((region_start, region_end))
        spike_pos += 1

    if not regions:
        return np.empty((0, 2), dtype=int)
    return np.asarray(regions, dtype=int)


def _resolve_relaxation_samples(
    *,
    sample_rate_hz: float | None,
    extra_relaxation_ms: float,
    extra_relaxation_samples: int | None,
) -> int:
    if extra_relaxation_samples is not None:
        if extra_relaxation_samples < 0:
            msg = "extra_relaxation_samples must be non-negative"
            raise ValueError(msg)
        return extra_relaxation_samples

    if sample_rate_hz is None:
        return 0
    if sample_rate_hz <= 0:
        msg = "sample_rate_hz must be positive"
        raise ValueError(msg)
    if extra_relaxation_ms < 0:
        msg = "extra_relaxation_ms must be non-negative"
        raise ValueError(msg)
    return int(extra_relaxation_ms * sample_rate_hz / 1e3)
