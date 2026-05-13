from datetime import timedelta
from typing import Final

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import streamlit as st
from streamlit.elements.plotly_chart import PlotlyState

import nanopore_it


LINEAR_LINE: Final[dict[str, str | int]] = {"shape": "linear", "smoothing": 0}
EVENT_HIGHLIGHT_COLOR: Final[str] = "rgba(255, 127, 14, 0.18)"
SELECTED_EVENT_HIGHLIGHT_COLOR: Final[str] = "rgba(44, 160, 44, 0.32)"
SELECTED_EVENT_LINE_COLOR: Final[str] = "rgba(44, 160, 44, 0.95)"

fft = st.cache_data(ttl=timedelta(minutes=5))(nanopore_it.fft)
downsample = st.cache_data(ttl=timedelta(minutes=5))(nanopore_it.downsample)


def compute_frequency_spectrum(
    values: npt.NDArray[np.float64],
    *,
    adc_samplerate: float,
    max_points: int = 10000,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size < 2:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    freqs, spectrum = fft(finite_values, fs=int(adc_samplerate))
    return downsample(freqs, spectrum, max_points=max_points)


def compute_spectrum_difference(
    *,
    event_freqs: npt.NDArray[np.float64],
    event_spectrum: npt.NDArray[np.float64],
    baseline_freqs: npt.NDArray[np.float64],
    baseline_spectrum: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    finite_event = np.isfinite(event_freqs) & np.isfinite(event_spectrum)
    finite_baseline = np.isfinite(baseline_freqs) & np.isfinite(baseline_spectrum)
    if np.count_nonzero(finite_event) < 1 or np.count_nonzero(finite_baseline) < 2:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    event_freqs = event_freqs[finite_event]
    event_spectrum = event_spectrum[finite_event]
    baseline_freqs = baseline_freqs[finite_baseline]
    baseline_spectrum = baseline_spectrum[finite_baseline]

    baseline_order = np.argsort(baseline_freqs)
    baseline_freqs = baseline_freqs[baseline_order]
    baseline_spectrum = baseline_spectrum[baseline_order]

    min_freq = float(baseline_freqs[0])
    max_freq = float(baseline_freqs[-1])
    event_mask = (event_freqs > 0) & (event_freqs >= min_freq) & (event_freqs <= max_freq)
    if not np.any(event_mask):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    difference_freqs = event_freqs[event_mask]
    interpolated_baseline = np.interp(
        difference_freqs,
        baseline_freqs,
        baseline_spectrum,
    )
    return difference_freqs, event_spectrum[event_mask] - interpolated_baseline


@st.cache_data(max_entries=1)
def draw_signal(
    signal: npt.NDArray[np.float64],
    baseline: float,
    adc_samplerate: float,
    downsampling_factor: int,
    event_ranges: tuple[tuple[float, float], ...],
    selected_event_range: tuple[float, float] | None,
) -> go.Figure:
    x = np.arange(len(signal)) / adc_samplerate
    x = x[::downsampling_factor]
    signal = signal[::downsampling_factor]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=signal,
            mode="lines",
            line=LINEAR_LINE,
            name="Signal",
        )
    )
    fig.add_hline(
        y=baseline,
        line_dash="dash",
        line_color="red",
    )
    for start_time, end_time in event_ranges:
        fig.add_vrect(
            x0=start_time,
            x1=end_time,
            fillcolor=EVENT_HIGHLIGHT_COLOR,
            line_width=0,
            layer="below",
        )
    if selected_event_range is not None:
        start_time, end_time = selected_event_range
        fig.add_vrect(
            x0=start_time,
            x1=end_time,
            fillcolor=SELECTED_EVENT_HIGHLIGHT_COLOR,
            line_color=SELECTED_EVENT_LINE_COLOR,
            line_width=1,
            layer="below",
        )
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        xaxis={"showgrid": True},
        yaxis={"showgrid": True},
    )
    return fig


def get_selected_event_index(state: PlotlyState) -> int:
    selection = state.get("selection")
    if selection is None:
        return 0
    point_indices = selection.get("point_indices")
    if point_indices is None:
        return 0
    if len(point_indices) == 0:
        return 0
    return point_indices[0]


def render_event_analysis_tab(
    *,
    signal: npt.NDArray[np.float64],
    baseline: float,
    adc_samplerate: float,
    downsampling_factor: int,
    result: nanopore_it.AnalysisTables,
) -> int:
    highlight_events = st.checkbox("Highlight detected event regions", value=False)
    signal_chart = st.empty()

    if highlight_events and not result.events.empty:
        event_ranges = tuple(
            (
                int(start_point) / adc_samplerate,
                (int(end_point) + 1) / adc_samplerate,
            )
            for start_point, end_point in result.events[
                ["start_point", "end_point"]
            ].itertuples(index=False, name=None)
        )
    else:
        event_ranges = ()

    if result.events.empty:
        signal_chart.plotly_chart(
            draw_signal(
                signal,
                baseline,
                adc_samplerate,
                downsampling_factor,
                event_ranges,
                None,
            ),
            width="stretch",
        )
        st.warning("No events detected.")
        return 0

    with st.container(horizontal=True):
        selected = st.container()
        selector = st.container()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.events["dwell"],
            y=result.events["delli"],
            mode="markers",
            name="Events",
        )
    )
    fig.update_layout(
        title=f"Event Dwell Time vs Delli (n={len(result.events)})",
        xaxis_title="Dwell Time (µs)",
        yaxis_title="Delli (pA)",
        xaxis={"showgrid": True, "type": "log"},
        yaxis={"showgrid": True, "type": "log"},
        clickmode="event+select",
    )
    chart_event = selector.plotly_chart(
        fig,
        width="stretch",
        on_select="rerun",
        selection_mode="points",
    )

    selected_event_index = get_selected_event_index(chart_event)
    selected_event_index = min(selected_event_index, len(result.events) - 1)
    event = result.events.iloc[selected_event_index]
    selected_event_range = (
        int(event.start_point) / adc_samplerate,
        (int(event.end_point) + 1) / adc_samplerate,
    )
    signal_chart.plotly_chart(
        draw_signal(
            signal,
            baseline,
            adc_samplerate,
            downsampling_factor,
            event_ranges,
            selected_event_range,
        ),
        width="stretch",
    )

    fig = go.Figure()
    start, end = int(event.start_point) + 1, int(event.end_point)
    around_samples = (end - start) // 3
    chart_start = max(0, start - around_samples)
    chart_end = min(len(signal), end + around_samples)
    x = np.arange(chart_start, chart_end) / adc_samplerate
    y = signal[chart_start:chart_end]
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=LINEAR_LINE,
            name="Event Signal",
        )
    )
    fig.add_hline(y=baseline, line_dash="dash", line_color="red")
    fig.add_vline(x=(start / adc_samplerate), line_dash="dash", line_color="green")
    fig.add_vline(x=(end / adc_samplerate), line_dash="dash", line_color="green")

    states = result.states
    selected_states = states[states["parent_index"] == selected_event_index]
    subevent_colors = (
        "rgba(30, 144, 255, 0.18)",
        "rgba(138, 43, 226, 0.18)",
    )
    for state in selected_states.itertuples(index=False):
        start, end = int(state.start_point) + 1, int(state.end_point)
        if end <= chart_start or start >= chart_end:
            continue

        x0 = max(start, chart_start) / adc_samplerate
        x1 = min(end, chart_end) / adc_samplerate
        color = subevent_colors[int(state.index) % len(subevent_colors)]
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=color,
            line_color="rgba(65, 105, 225, 0.85)",
            line_width=1,
            layer="below",
        )

    fig.update_layout(
        title=f"Event {selected_event_index}: Dwell={event['dwell']:.3e}s, Delli={event['delli']:.3e}pA",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        xaxis={"showgrid": True},
        yaxis={"showgrid": True},
    )
    selected.plotly_chart(fig, width="stretch")

    without_events = signal.copy()
    for _, ev in result.events.iterrows():
        start = int(ev.start_point)
        end = int(ev.end_point) + 1  # include end point
        without_events[start:end] = np.nan
    without_events = without_events[~np.isnan(without_events)]
    baseline_freqs, baseline_spectrum = compute_frequency_spectrum(
        without_events,
        adc_samplerate=adc_samplerate,
    )
    start, end = int(event.start_point) + 1, int(event.end_point)
    event_signal = signal[start:end]
    event_freqs, event_spectrum = compute_frequency_spectrum(
        event_signal,
        adc_samplerate=adc_samplerate,
    )
    difference_freqs, difference_spectrum = compute_spectrum_difference(
        event_freqs=event_freqs,
        event_spectrum=event_spectrum,
        baseline_freqs=baseline_freqs,
        baseline_spectrum=baseline_spectrum,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=baseline_freqs,
            y=baseline_spectrum,
            mode="lines",
            line=LINEAR_LINE,
            name="Baseline Spectrum",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=event_freqs,
            y=event_spectrum,
            mode="lines",
            line=LINEAR_LINE,
            name="Event Spectrum",
        )
    )
    fig.update_layout(
        title="Frequency Spectrum Comparison",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
        xaxis={"showgrid": True, "type": "log"},
        yaxis={"showgrid": True, "type": "log"},
    )
    st.plotly_chart(fig, width="stretch")

    if difference_freqs.size == 0:
        st.info("Spectrum difference is unavailable for this event.")
        return selected_event_index

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=difference_freqs,
            y=difference_spectrum,
            mode="lines",
            line=LINEAR_LINE,
            name="Event - Baseline",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Frequency Spectrum Difference (Event - Baseline)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude Difference",
        xaxis={"showgrid": True, "type": "log"},
        yaxis={"showgrid": True},
    )
    st.plotly_chart(fig, width="stretch")
    return selected_event_index
