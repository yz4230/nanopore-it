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
    fig.add_trace(go.Scatter(x=x, y=signal, mode="lines", name="Signal"))
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


@st.cache_data(max_entries=1)
def baseline_fft(
    signal: npt.NDArray[np.float64],
    result: nanopore_it.AnalysisTables,
    adc_samplerate: float,
) -> tuple[npt.NDArray, npt.NDArray]:
    signal = signal.copy()

    for ev in result.events.itertuples():
        s, e = int(ev.start_point), int(ev.end_point)  # ty:ignore[unresolved-attribute]
        signal[s:e] = np.nan
    signal = signal[~np.isnan(signal)]
    baseline_spectr = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1 / adc_samplerate)

    max_points: Final[int] = 10000
    if len(baseline_spectr) > max_points:
        factor = len(baseline_spectr) // max_points
        baseline_spectr = baseline_spectr[::factor]
        freqs = freqs[::factor]

    return baseline_spectr, freqs


detect_clear_regions = st.cache_data(max_entries=1)(nanopore_it.detect_clear_regions)
analyze_tables = st.cache_data(max_entries=1)(nanopore_it.analyze_tables)


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

    downsampling_factor = st.sidebar.number_input(
        "Downsampling factor", value=1024, step=1024, min_value=1024
    )

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
                line_shape="linear",
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
        start, end = (int(event["start_point"]), int(event["end_point"]))
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
                line_shape="linear",
                name="Event Signal",
            )
        )
        fig.add_hline(y=baseline, line_dash="dash", line_color="red")
        fig.add_vline(x=(start / adc_samplerate), line_dash="dash", line_color="green")
        fig.add_vline(x=(end / adc_samplerate), line_dash="dash", line_color="green")
        fig.update_layout(
            title=f"Event {selected_event_index}: Dwell={event['dwell']:.3e}s, Delli={event['delli']:.3e}pA",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            xaxis={"showgrid": True},
            yaxis={"showgrid": True},
        )
        selected.plotly_chart(fig, width="stretch")

        baseline_spectr, baseline_freqs = baseline_fft(signal, result, adc_samplerate)
        event_signal = signal[start:end]
        # event_signal -= np.median(event_signal)
        event_spectr = np.abs(np.fft.rfft(event_signal))
        event_freqs = np.fft.rfftfreq(len(event_signal), d=1 / adc_samplerate)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=baseline_freqs,
                y=baseline_spectr,
                mode="lines",
                line_shape="linear",
                name="Baseline Spectrum",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=event_freqs,
                y=event_spectr,
                mode="lines",
                line_shape="linear",
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
