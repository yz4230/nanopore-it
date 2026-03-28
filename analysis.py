from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import signal

from cusumv3 import CUSUMResultDict, detect_cusumv2

__all__ = [
    "InputConfig",
    "AnalysisConfig",
    "AnalysisTables",
    "analyze_tables",
    "analyze",
]


def _segment_statistics(
    segment: npt.NDArray[np.float64],
) -> tuple[float, float, float, float]:
    """Compute (mean, stdev, skewness, kurtosis) without scipy.stats overhead."""
    mean = float(np.mean(segment))
    d = segment - mean
    d2 = d * d
    m2 = float(np.mean(d2))
    if m2 == 0.0:
        return mean, 0.0, 0.0, 0.0
    m2_sqrt = m2**0.5
    m3 = float(np.mean(d2 * d))
    m4 = float(np.mean(d2 * d2))
    return mean, m2_sqrt, m3 / (m2 * m2_sqrt), m4 / (m2 * m2) - 3.0


@dataclass(kw_only=True, frozen=True)
class InputConfig:
    adc_samplerate_hz: int
    lpf_cutoff_hz: int = 100


@dataclass(kw_only=True, frozen=True)
class AnalysisConfig:
    baseline_a: float = np.nan
    baseline_std_a: float = np.nan
    threshold_a: float = 0.3e-9
    enable_subevent_state_detection: bool = True
    max_states: int = 16
    cusum_stepsize: int = 10
    cusum_threshhold: int = 30
    merge_delta_blockade: float = 0.02
    prefilt_window_us: int = 100
    state_min_duration_us: int = 150


@dataclass(kw_only=True, frozen=True)
class AnalysisTables:
    events: pd.DataFrame
    states: pd.DataFrame
    n_children: npt.NDArray[np.int64]


class MergedCUSUMResult(TypedDict):
    nStates: int
    starts: npt.NDArray[np.int64]


class StateRow(TypedDict):
    parent_index: int
    index: int
    start_point: int
    end_point: int
    delli: float
    frac: float
    dwell: float
    mean: float
    stdev: float
    skewness: float
    kurtosis: float


EVENT_HEADERS = [
    "index",
    "start_point",
    "end_point",
    "delli",
    "frac",
    "dwell",
    "dt",
    "mean",
    "stdev",
    "skewness",
    "kurtosis",
    "offset_to_first_min",
    "stdev_tt",
    "skewness_tt",
    "kurtosis_tt",
    "fft_mean",
]

STATE_HEADERS = [
    "parent_index",
    "index",
    "start_point",
    "end_point",
    "delli",
    "frac",
    "dwell",
    "mean",
    "stdev",
    "skewness",
    "kurtosis",
]


def _empty_event_df() -> pd.DataFrame:
    return pd.DataFrame(columns=pd.Index(EVENT_HEADERS))


def _empty_state_df() -> pd.DataFrame:
    return pd.DataFrame(columns=pd.Index(STATE_HEADERS))


def merge_oversegmentation(
    data: npt.NDArray[np.float64],
    cusum_res: CUSUMResultDict,
    merge_delta_i: float,
) -> MergedCUSUMResult:
    starts = cusum_res["starts"]
    n_states = cusum_res["nStates"]

    mean_i_1 = np.mean(data[starts[0] : starts[1]])
    starts_to_retain = [0]
    for state_index in range(n_states - 1):
        mean_i_0 = mean_i_1
        mean_i_1 = np.mean(data[starts[state_index + 1] : starts[state_index + 2]])
        if abs(mean_i_1 - mean_i_0) > merge_delta_i:
            starts_to_retain.append(state_index + 1)
    starts_to_retain.append(n_states)
    return {
        "nStates": len(starts_to_retain) - 1,
        "starts": np.array([starts[idx] for idx in starts_to_retain]),
    }


def analyze_tables(
    *,
    data: npt.NDArray[np.float64],
    iconf: InputConfig,
    aconf: AnalysisConfig,
) -> AnalysisTables:
    if iconf.adc_samplerate_hz <= 0:
        raise ValueError("adc_samplerate_hz must be positive")

    (below,) = np.where(data < aconf.threshold_a)
    start_and_end = np.diff(below)
    if len(start_and_end) == 0:
        return AnalysisTables(
            events=_empty_event_df(),
            states=_empty_state_df(),
            n_children=np.array([], dtype=np.int64),
        )

    start_points = np.insert(start_and_end, 0, 2)
    end_points = np.insert(start_and_end, -1, 2)
    (start_points,) = np.where(start_points > 1)
    (end_points,) = np.where(end_points > 1)
    start_points = below[start_points]
    end_points = below[end_points]

    if start_points.size == 0:
        return AnalysisTables(
            events=_empty_event_df(),
            states=_empty_state_df(),
            n_children=np.array([], dtype=np.int64),
        )
    if start_points[0] == 0:
        start_points = start_points[1:]
        end_points = end_points[1:]
    if end_points.size == 0:
        return AnalysisTables(
            events=_empty_event_df(),
            states=_empty_state_df(),
            n_children=np.array([], dtype=np.int64),
        )
    if end_points[-1] == data.size - 1:
        start_points = start_points[:-1]
        end_points = end_points[:-1]

    number_of_events = start_points.size
    high_thresh = aconf.baseline_a - aconf.baseline_std_a

    for event_index in range(number_of_events):
        start_point = start_points[event_index]
        while start_point > 0 and data[start_point] < high_thresh:
            start_point -= 1
        start_points[event_index] = start_point

        end_point = end_points[event_index]
        if end_point == data.size - 1:
            start_points[event_index] = 0
            end_points[event_index] = 0
            end_point = 0
            break

        while data[end_point] < high_thresh:
            end_point += 1
            if end_point == data.size - 1:
                start_points[event_index] = 0
                end_points[event_index] = 0
                end_point = 0
                break

            if (
                event_index + 1 < number_of_events
                and end_point > start_points[event_index + 1]
            ):
                start_points[event_index + 1] = 0
                end_points[event_index] = 0
                end_point = 0
                break

        end_points[event_index] = end_point

    valid_mask = (start_points != 0) & (end_points != 0)
    start_points = start_points[valid_mask]
    end_points = end_points[valid_mask]
    number_of_events = start_points.size

    delis = np.zeros(number_of_events)
    dwells = np.zeros(number_of_events)
    first_min_offsets = np.full(number_of_events, -1, dtype=np.int32)

    for event_index in range(number_of_events):
        (relmin,) = signal.argrelmin(
            data[start_points[event_index] : end_points[event_index]]
        )
        mins = np.array(relmin + start_points[event_index])
        cut = (
            aconf.baseline_a
            + np.mean(data[start_points[event_index] : end_points[event_index]])
        ) / 2
        mins = mins[data[mins] < cut]
        if len(mins) == 1:
            delis[event_index] = aconf.baseline_a - min(
                data[start_points[event_index] : end_points[event_index]]
            )
            dwells[event_index] = (
                (end_points[event_index] - start_points[event_index])
                / iconf.adc_samplerate_hz
                * 1e6
            )
            end_points[event_index] = mins[0]
            first_min_offsets[event_index] = -2
        elif len(mins) > 1:
            delis[event_index] = aconf.baseline_a - np.mean(data[mins[0] : mins[-1]])
            end_points[event_index] = mins[-1]
            dwells[event_index] = (
                (end_points[event_index] - start_points[event_index])
                / iconf.adc_samplerate_hz
                * 1e6
            )
            first_min_offsets[event_index] = mins[0] - start_points[event_index]

    valid_events = np.logical_and(delis != 0, dwells != 0)
    start_points = start_points[valid_events]
    end_points = end_points[valid_events]
    first_min_offsets = first_min_offsets[valid_events]
    delis = delis[valid_events]
    dwells = dwells[valid_events]
    fracs = delis / aconf.baseline_a
    number_of_events = start_points.size
    if number_of_events == 0:
        return AnalysisTables(
            events=_empty_event_df(),
            states=_empty_state_df(),
            n_children=np.array([], dtype=np.int64),
        )

    dts = np.empty(number_of_events, dtype=np.float64)
    dts[0] = np.nan
    if number_of_events > 1:
        dts[1:] = np.diff(start_points) / iconf.adc_samplerate_hz

    means = np.empty(number_of_events, dtype=np.float64)
    noise = np.empty(number_of_events, dtype=np.float64)
    skew = np.empty(number_of_events, dtype=np.float64)
    kurt = np.empty(number_of_events, dtype=np.float64)
    stdev_tt = np.full(number_of_events, np.nan)
    skew_tt = np.full(number_of_events, np.nan)
    kurt_tt = np.full(number_of_events, np.nan)
    fft_mean = np.full(number_of_events, np.nan)

    for event_index, (start_point, end_point) in enumerate(
        zip(start_points, end_points)
    ):
        segment = data[start_point:end_point]
        means[event_index], noise[event_index], skew[event_index], kurt[event_index] = (
            _segment_statistics(segment)
        )

        first_min_offset = first_min_offsets[event_index]
        if first_min_offset > 0:
            trough = data[start_point + first_min_offset : end_point]
            if trough.size != 0:
                _, stdev_tt[event_index], skew_tt[event_index], kurt_tt[event_index] = (
                    _segment_statistics(trough)
                )

        # FFT computation (DC component excluded, using rfft for real input)
        if segment.size > 1:
            fft_magnitude = np.abs(np.fft.rfft(segment))
            # Exclude DC component (index 0), average remaining frequencies
            if fft_magnitude.size > 1:
                fft_mean[event_index] = np.mean(fft_magnitude[1:])

    events = pd.DataFrame(
        {
            "index": np.arange(number_of_events),
            "start_point": start_points,
            "end_point": end_points,
            "delli": delis,
            "dwell": dwells,
            "frac": fracs,
            "dt": dts,
            "mean": means,
            "stdev": noise,
            "skewness": skew,
            "kurtosis": kurt,
            "offset_to_first_min": first_min_offsets,
            "stdev_tt": stdev_tt,
            "skewness_tt": skew_tt,
            "kurtosis_tt": kurt_tt,
            "fft_mean": fft_mean,
        },
        columns=pd.Index(EVENT_HEADERS),
    )

    n_children = np.zeros(number_of_events, dtype=np.int64)
    if not aconf.enable_subevent_state_detection:
        return AnalysisTables(
            events=events, states=_empty_state_df(), n_children=n_children
        )

    cusum_min_len = int(aconf.state_min_duration_us * iconf.adc_samplerate_hz * 1e-6)
    prefilt_oneside_window = int(
        aconf.prefilt_window_us / 2 * iconf.adc_samplerate_hz * 1e-6
    )
    cusum_max_states = aconf.max_states - 1
    merge_delta_i = aconf.merge_delta_blockade * aconf.baseline_a

    state_rows: list[StateRow] = []
    for event_index, (start_point, end_point) in enumerate(
        zip(start_points, end_points)
    ):
        first_min_offset = first_min_offsets[event_index]
        if first_min_offset <= 0:
            continue

        trough = data[start_point + first_min_offset : end_point]
        if trough.size < cusum_min_len:
            continue

        cusum_res = detect_cusumv2(
            trough,
            aconf.baseline_std_a,
            minlength=cusum_min_len,
            maxstates=cusum_max_states,
            stepsize=aconf.cusum_stepsize,
            threshhold=aconf.cusum_threshhold,
            moving_oneside_window=prefilt_oneside_window,
        )
        merged_cusum_res = merge_oversegmentation(trough, cusum_res, merge_delta_i)
        n_children[event_index] = merged_cusum_res["nStates"]

        if merged_cusum_res["nStates"] <= 1:
            continue

        for state_index in range(merged_cusum_res["nStates"]):
            state_start = int(merged_cusum_res["starts"][state_index])
            state_end = int(merged_cusum_res["starts"][state_index + 1])
            state_data = trough[state_start:state_end]
            state_mean, state_stdev, state_skew, state_kurt = _segment_statistics(
                state_data
            )
            state_delli = float(aconf.baseline_a - state_mean)
            state_rows.append(
                {
                    "parent_index": event_index,
                    "index": state_index,
                    "start_point": state_start + first_min_offset + start_point,
                    "end_point": state_end + first_min_offset + start_point,
                    "delli": state_delli,
                    "frac": float(state_delli / aconf.baseline_a),
                    "dwell": float(len(state_data) / iconf.adc_samplerate_hz * 1e6),
                    "mean": state_mean,
                    "stdev": state_stdev,
                    "skewness": state_skew,
                    "kurtosis": state_kurt,
                }
            )

    if state_rows:
        states = pd.DataFrame(state_rows, columns=pd.Index(STATE_HEADERS))
    else:
        states = _empty_state_df()

    return AnalysisTables(events=events, states=states, n_children=n_children)


def analyze(
    *,
    data: npt.NDArray[np.float64],
    iconf: InputConfig,
    aconf: AnalysisConfig,
) -> pd.DataFrame:
    tables = analyze_tables(data=data, iconf=iconf, aconf=aconf)
    events = tables.events.copy()
    events.attrs["CUSUMState"] = tables.states
    events.attrs["n_children"] = tables.n_children
    return events
