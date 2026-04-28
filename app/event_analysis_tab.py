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

fft = st.cache_data(ttl=timedelta(minutes=5))(nanopore_it.fft)
downsample = st.cache_data(ttl=timedelta(minutes=5))(nanopore_it.downsample)


@st.cache_data(max_entries=1)
def draw_signal(
    signal: npt.NDArray[np.float64],
    baseline: float,
    adc_samplerate: float,
    downsampling_factor: int,
    event_ranges: tuple[tuple[float, float], ...],
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

    st.plotly_chart(
        draw_signal(
            signal,
            baseline,
            adc_samplerate,
            downsampling_factor,
            event_ranges,
        )
    )

    if result.events.empty:
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
    baseline_spectr, baseline_freqs = fft(without_events, fs=int(adc_samplerate))
    baseline_spectr, baseline_freqs = downsample(
        baseline_freqs,
        baseline_spectr,
        max_points=10000,
    )
    start, end = int(event.start_point) + 1, int(event.end_point)
    event_signal = signal[start:end]
    event_spectr, event_freqs = fft(event_signal, fs=int(adc_samplerate))
    event_spectr, event_freqs = downsample(
        event_freqs,
        event_spectr,
        max_points=10000,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=baseline_freqs,
            y=baseline_spectr,
            mode="lines",
            line=LINEAR_LINE,
            name="Baseline Spectrum",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=event_freqs,
            y=event_spectr,
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
    return selected_event_index
