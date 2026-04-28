from datetime import timedelta
import os
import secrets
from typing import Final

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import streamlit as st
from streamlit.elements.plotly_chart import PlotlyState
from streamlit.runtime.uploaded_file_manager import UploadedFile

import nanopore_it


LINEAR_LINE: Final[dict[str, str | int]] = {"shape": "linear", "smoothing": 0}


@st.cache_data(max_entries=1)
def load_data(
    uploaded_file: UploadedFile,
    *,
    lpf_cutoff: float,
    adc_samplerate: float,
    invert: bool,
) -> npt.NDArray[np.float64]:
    buf = uploaded_file.read()
    return nanopore_it.load_opt_file(
        data=buf,
        lpf_cutoff=lpf_cutoff,
        adc_samplerate=adc_samplerate,
        invert=invert,
    )


@st.cache_data(max_entries=1)
def draw_signal(
    signal: npt.NDArray[np.float64],
    baseline: float,
    adc_samplerate: float,
    downsampling_factor: int,
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
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        xaxis={"showgrid": True},
        yaxis={"showgrid": True},
    )
    return fig


detect_clear_regions = st.cache_data(max_entries=1)(nanopore_it.detect_clear_regions)
analyze_tables = st.cache_data(max_entries=1)(nanopore_it.analyze_tables)
fft = st.cache_data(ttl=timedelta(minutes=5))(nanopore_it.fft)
downsample = st.cache_data(ttl=timedelta(minutes=5))(nanopore_it.downsample)


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


def authenticate():
    STATE_KEY: Final[str] = "auth_status"

    auth_status = st.session_state.get(STATE_KEY, False)
    if auth_status:
        return

    with st.form("login_form"):
        input_password = st.text_input("Enter password", type="password")
        submit = st.form_submit_button("Login")

    if submit:
        password = os.getenv("APP_PASSWORD", "")
        assert password, "No password set in environment variable APP_PASSWORD"

        if secrets.compare_digest(input_password, password):
            st.session_state[STATE_KEY] = True
            st.rerun()
            return
        else:
            st.error("Incorrect password")

    st.stop()


def main():
    authenticate()

    st.set_page_config(page_title="Nanopore Analysis", layout="wide")

    uploaded_file = st.sidebar.file_uploader("Upload your data", type=["opt"])
    adc_samplerate = st.sidebar.number_input("ADC Samplerate (kHz)", value=250) * 1e3
    lpf_cutoff = st.sidebar.number_input("LPF Cutoff (kHz)", value=100) * 1e3
    downsampling_factor = st.sidebar.number_input(
        "Downsampling factor", value=1024, step=1024, min_value=1024
    )
    invert = st.sidebar.checkbox("Invert signal", value=True)

    auto_detect_clear = st.sidebar.checkbox("Auto-detect clear signal", value=True)

    if auto_detect_clear:
        spike_std_threshold = st.sidebar.number_input(
            "Spike threshold (std devs)", value=10.0, step=0.1
        )
        baseline_window_std = st.sidebar.number_input(
            "Baseline window width (std devs)", value=1.0, step=0.1
        )
        extra_relaxation_ms = st.sidebar.number_input(
            "Extra relaxation (ms)", value=10.0, step=1.0
        )

    st.sidebar.divider()

    cusum_stepsize = st.sidebar.number_input("CUSUM stepsize", value=1.0, step=1.0)
    cusum_threshold = st.sidebar.number_input("CUSUM threshold", value=30.0, step=1.0)

    if uploaded_file is not None:
        signal = load_data(
            uploaded_file,
            lpf_cutoff=lpf_cutoff,
            adc_samplerate=adc_samplerate,
            invert=invert,
        )
        baseline = float(np.median(signal))
        baseline_std = float(np.std(signal))

        if auto_detect_clear:
            regions = detect_clear_regions(
                signal,
                baseline=baseline,
                baseline_std=baseline_std,
                spike_std_threshold=spike_std_threshold,
                baseline_window_std=baseline_window_std,
                sample_rate_hz=adc_samplerate,
                extra_relaxation_ms=extra_relaxation_ms,
            )
            for start, end in regions:
                signal[start:end] = np.nan
            baseline = float(np.nanmedian(signal))
            baseline_std = float(np.nanstd(signal))

        st.plotly_chart(
            draw_signal(
                signal,
                baseline,
                adc_samplerate,
                downsampling_factor,
            )
        )

        result = nanopore_it.analyze_tables(
            data=signal,
            conf=nanopore_it.AnalysisConfig(
                adc_samplerate_hz=int(adc_samplerate),
                lpf_cutoff_hz=int(lpf_cutoff),
                baseline_a=baseline,
                baseline_std_a=baseline_std,
                cusum_stepsize=int(cusum_stepsize),
                cusum_threshhold=int(cusum_threshold),
            ),
        )

        if result.events.empty:
            st.warning("No events detected.")
            return

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
        subevent_colors = ("rgba(30, 144, 255, 0.18)", "rgba(138, 43, 226, 0.18)")
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
            start, end = int(ev.start_point), int(ev.end_point) + 1  # include end point
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


if __name__ == "__main__":
    main()
