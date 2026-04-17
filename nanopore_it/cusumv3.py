"""
CUSUM (Cumulative Sum) Algorithm for Change Point Detection

This module implements a CUSUM-based algorithm for detecting abrupt changes
in time series data, commonly used in nanopore sensing and ion channel analysis.
"""

from typing import Optional, TypedDict

import numpy as np
import numpy.typing as npt
from scipy.ndimage import median_filter

__all__ = ["detect_cusumv2", "CUSUMResultDict"]


class CUSUMResultDict(TypedDict):
    """Type definition for CUSUM detection result dictionary."""

    nStates: int
    starts: npt.NDArray[np.int64]
    threshold: float
    stepsize: float


def _central_moving_average(
    data: npt.NDArray[np.floating],
    window: int,
) -> Optional[npt.NDArray[np.float64]]:
    """
    Compute central moving average with edge handling.

    Parameters
    ----------
    data : npt.NDArray[np.floating]
        Input data array.
    window : int
        One-sided window size.

    Returns
    -------
    npt.NDArray[np.float64] or None
        Smoothed data, or None if data is too short.
    """
    full_window = 2 * window + 1
    if len(data) <= full_window:
        return None

    # Scale data to int64 to avoid floating point accumulation errors
    int32_max = np.iinfo(np.int32).max
    data_absmax = np.abs(data).max()
    if data_absmax == 0:
        return np.zeros_like(data, dtype=np.float64)

    scaled_data = (data / data_absmax * int32_max).astype(np.int64)
    cumsum = np.cumsum(scaled_data)

    result = np.zeros_like(scaled_data)

    # Left edge: expanding window
    result[: window + 1] = cumsum[window:full_window]

    # Center: full window
    result[window + 1 : -window] = cumsum[full_window:] - cumsum[:-full_window]

    # Right edge: contracting window
    result[-window:] = cumsum[-1] - cumsum[-full_window : -window - 1]

    # Convert back to float and normalize
    result = result.astype(np.float64) / int32_max * data_absmax

    # Adjust for varying window sizes at edges
    left_counts = np.arange(window + 1, full_window + 1)
    right_counts = np.arange(full_window - 1, window, -1)

    result[: window + 1] /= left_counts
    result[window + 1 : -window] /= full_window
    result[-window:] /= right_counts

    return result


def _central_moving_median(
    data: npt.NDArray[np.floating],
    window: int,
) -> Optional[npt.NDArray[np.floating]]:
    """
    Compute central moving median with edge padding.

    Parameters
    ----------
    data : npt.NDArray[np.floating]
        Input data array.
    window : int
        One-sided window size.

    Returns
    -------
    npt.NDArray[np.floating] or None
        Filtered data, or None if data is too short.
    """
    full_window = 2 * window + 1
    if len(data) <= full_window:
        return None

    # Pad data with edge values
    padded = np.empty(len(data) + 2 * window)
    padded[:window] = data[:window]
    padded[window:-window] = data
    padded[-window:] = data[-window:]

    # Apply median filter
    filtered = median_filter(padded, full_window)

    return filtered[window:-window]


def _preprocess_data(
    data: npt.NDArray[np.floating],
    window: int,
) -> Optional[npt.NDArray[np.floating]]:
    """
    Apply moving average and median filtering for noise reduction.

    Parameters
    ----------
    data : npt.NDArray[np.floating]
        Raw input data.
    window : int
        One-sided window size for filtering.

    Returns
    -------
    npt.NDArray[np.floating] or None
        Preprocessed data, or None if data is too short.
    """
    if window <= 0:
        # Keep window=0 usable even though the legacy implementation crashes.
        return data.copy()

    smoothed = _central_moving_average(data, window)
    if smoothed is None:
        return None

    return _central_moving_median(smoothed, window)


def _create_single_state_result(
    data_length: int,
    threshold: float,
    stepsize: float,
) -> CUSUMResultDict:
    """Create result for single-state (no change points) case."""
    return {
        "nStates": 1,
        "starts": np.array([0, data_length], dtype=np.int64),
        "threshold": threshold,
        "stepsize": stepsize,
    }


def _cusum_single_pass(
    data: npt.NDArray[np.floating],
    base_sd: float,
    threshold: float,
    stepsize: float,
    min_length: int,
) -> tuple[int, list[int]]:
    """
    Core CUSUM detection loop with inlined computations for performance.

    All per-sample statistics (Welford variance, log-likelihoods, cumulative
    sums) are computed using local variables to avoid method-call and
    attribute-lookup overhead.  Cumulative-sum arrays are only zeroed at the
    current position on reset instead of across the full length.

    Returns
    -------
    tuple[int, list[int]]
        (n_states, edges) where *edges* is a list of boundary indices
        including the leading ``0`` and the trailing ``len(data)``.
    """
    length = len(data)
    cpos = np.zeros(length, dtype=np.float64)
    cneg = np.zeros(length, dtype=np.float64)
    gpos = np.zeros(length, dtype=np.float64)
    gneg = np.zeros(length, dtype=np.float64)

    base_variance = base_sd * base_sd
    delta = stepsize * base_sd
    delta_half = delta * 0.5

    # Welford online variance – local scalars avoid object/attribute overhead
    w_m = float(data[0])
    w_s = 0.0
    w_count = 1

    edges: list[int] = [0]
    anchor = 0

    for k in range(1, length):
        val = float(data[k])

        # --- Welford update ---
        w_count += 1
        old_m = w_m
        w_m += (val - w_m) / w_count
        w_s += (val - old_m) * (val - w_m)

        variance = w_s / w_count
        if variance <= 0.0:
            variance = base_variance

        # --- log-likelihood for positive / negative jumps ---
        scale = delta / variance
        deviation = val - w_m
        log_pos = scale * (deviation - delta_half)
        log_neg = -scale * (deviation + delta_half)

        # --- cumulative & decision sums ---
        prev = k - 1
        cp = cpos[prev] + log_pos
        cn = cneg[prev] + log_neg
        gp = gpos[prev] + log_pos
        if gp < 0.0:
            gp = 0.0
        gn = gneg[prev] + log_neg
        if gn < 0.0:
            gn = 0.0

        cpos[k] = cp
        cneg[k] = cn
        gpos[k] = gp
        gneg[k] = gn

        # --- threshold check ---
        if gp > threshold or gn > threshold:
            if gp > threshold:
                jump = int(anchor + np.argmin(cpos[anchor : k + 1]))
                if jump - edges[-1] > min_length:
                    edges.append(jump)

            if gn > threshold:
                jump = int(anchor + np.argmin(cneg[anchor : k + 1]))
                if jump - edges[-1] > min_length:
                    edges.append(jump)

            # Reset only the current position (not the full array)
            anchor = k
            cpos[k] = 0.0
            cneg[k] = 0.0
            gpos[k] = 0.0
            gneg[k] = 0.0
            w_m = val
            w_s = 0.0
            w_count = 1

    edges.append(length)
    return len(edges) - 1, edges


def detect_cusumv2(
    data0: npt.NDArray[np.floating],
    basesd: float,
    dt: Optional[float] = None,
    threshhold: float = 10,
    stepsize: float = 3,
    minlength: int = 1000,
    maxstates: int = -1,
    moving_oneside_window: int = 0,
) -> CUSUMResultDict:
    """
    Detect change points in time series data using CUSUM algorithm.

    This function identifies abrupt changes (jumps) in the data, commonly used
    for analyzing nanopore or ion channel recordings.

    Parameters
    ----------
    data0 : npt.NDArray[np.floating]
        Input time series data.
    basesd : float
        Baseline standard deviation (noise level).
    dt : float, optional
        Time step (currently unused, kept for API compatibility).
    threshhold : float, default=10
        Detection threshold. Higher values reduce sensitivity.
    stepsize : float, default=3
        Expected jump size in units of basesd.
    minlength : int, default=1000
        Minimum number of samples between detected changes.
    maxstates : int, default=-1
        Maximum number of detected jumps before the final terminal edge is
        added. This matches the legacy CUSUMV3.detect_cusum behavior.
        Use 0 for single state, -1 for unlimited.
    moving_oneside_window : int, default=0
        Window size for preprocessing filters. Use 0 to skip preprocessing.

    Returns
    -------
    CUSUMResultDict
        Detection results containing:
        - nStates: Number of detected states
        - starts: Array of state boundary indices
        - threshold: Final threshold used
        - stepsize: Final stepsize used
    """
    if maxstates == 0:
        return _create_single_state_result(len(data0), threshhold, stepsize)

    data = _preprocess_data(data0, moving_oneside_window)
    if data is None:
        return _create_single_state_result(len(data0), threshhold, stepsize)

    assert len(data) == len(data0)

    relaxation_factor = 1.1
    current_threshold = float(threshhold)
    current_stepsize = float(stepsize)

    while True:
        n_states, edges = _cusum_single_pass(
            data, basesd, current_threshold, current_stepsize, minlength
        )
        if maxstates < 0 or n_states - 1 <= maxstates:
            break
        current_threshold *= relaxation_factor
        current_stepsize *= relaxation_factor

    return {
        "nStates": n_states,
        "starts": np.array(edges, dtype=np.int64),
        "threshold": current_threshold,
        "stepsize": current_stepsize,
    }
