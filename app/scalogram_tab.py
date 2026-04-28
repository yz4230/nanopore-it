from datetime import timedelta
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pywt
import streamlit as st

import nanopore_it


LINEAR_LINE: Final[dict[str, str | int]] = {"shape": "linear", "smoothing": 0}
SCALOGRAM_EVENT_COLOR: Final[str] = "rgba(255, 127, 14, 0.85)"
ScalogramWavelet = Literal[
    "complex_morlet", "morlet", "mexican_hat", "gaussian_derivative"
]
SCALOGRAM_WAVELETS: Final[dict[str, ScalogramWavelet]] = {
    "Complex Morlet": "complex_morlet",
    "Morlet": "morlet",
    "Mexican hat": "mexican_hat",
    "Gaussian derivative": "gaussian_derivative",
}
PYWAVELETS_WAVELETS: Final[dict[ScalogramWavelet, str]] = {
    "complex_morlet": "cmor1.5-1.0",
    "morlet": "morl",
    "mexican_hat": "mexh",
    "gaussian_derivative": "gaus1",
}


def interpolate_missing_values(
    values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    finite = np.isfinite(values)
    if finite.all():
        return values.copy()
    if not finite.any():
        return np.zeros_like(values)

    x = np.arange(values.size)
    filled = values.copy()
    filled[~finite] = np.interp(x[~finite], x[finite], values[finite])
    return filled


@st.cache_data(ttl=timedelta(minutes=5))
def compute_scalogram(
    signal: npt.NDArray[np.float64],
    *,
    adc_samplerate: float,
    start_point: int,
    end_point: int,
    min_freq_hz: float,
    max_freq_hz: float,
    num_freqs: int,
    max_samples: int,
    wavelet_type: ScalogramWavelet,
    normalize_power: bool,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    start_point = max(0, start_point)
    end_point = min(len(signal), end_point)
    if end_point <= start_point:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    segment = signal[start_point:end_point]
    step = max(1, int(np.ceil(segment.size / max_samples)))
    segment = segment[::step]
    effective_samplerate = adc_samplerate / step
    times = (start_point + np.arange(segment.size) * step) / adc_samplerate

    finite_values = segment[np.isfinite(segment)]
    if segment.size < 2 or finite_values.size < 2:
        return (
            times,
            np.array([], dtype=np.float64),
            np.empty((0, segment.size), dtype=np.float64),
            segment,
        )

    nyquist = effective_samplerate / 2
    max_freq_hz = min(max_freq_hz, nyquist)
    min_freq_hz = max(min_freq_hz, effective_samplerate / segment.size)
    if min_freq_hz >= max_freq_hz:
        return (
            times,
            np.array([], dtype=np.float64),
            np.empty((0, segment.size), dtype=np.float64),
            segment,
        )

    freqs = np.geomspace(min_freq_hz, max_freq_hz, num=max(2, num_freqs))
    transform_values = interpolate_missing_values(segment)
    transform_values = transform_values - float(np.mean(transform_values))

    sampling_period = 1 / effective_samplerate
    wavelet_name = PYWAVELETS_WAVELETS[wavelet_type]
    scales = pywt.frequency2scale(wavelet_name, freqs * sampling_period)
    coefficients, cwt_freqs = pywt.cwt(
        transform_values,
        scales,
        wavelet_name,
        sampling_period=sampling_period,
        method="fft",
    )

    freqs = np.asarray(cwt_freqs, dtype=np.float64)
    power = np.abs(coefficients) ** 2
    if normalize_power:
        max_power = float(np.nanmax(power))
        if max_power > 0:
            power = power / max_power

    power = np.log10(power + np.finfo(np.float64).eps)
    return times, freqs, power, segment


@st.cache_data(ttl=timedelta(minutes=5))
def draw_scalogram(
    times: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    power: npt.NDArray[np.float64],
    current: npt.NDArray[np.float64],
    event_ranges: tuple[tuple[float, float], ...],
    baseline: float,
    wavelet_label: str,
    normalize_power: bool,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.28, 0.72],
        vertical_spacing=0.05,
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=current,
            mode="lines",
            line=LINEAR_LINE,
            name="Current",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=times,
            y=freqs,
            z=power,
            colorscale="Viridis",
            colorbar={
                "title": "log10 Relative Power" if normalize_power else "log10 Power"
            },
            name="Scalogram",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=baseline, line_dash="dash", line_color="red", row=1, col=1)

    for start_time, end_time in event_ranges:
        for event_time in (start_time, end_time):
            fig.add_vline(
                x=event_time,
                line_width=1,
                line_dash="dot",
                line_color=SCALOGRAM_EVENT_COLOR,
                row="all",
                col=1,
            )

    fig.update_layout(
        title=f"Current Scalogram ({wavelet_label})",
        xaxis2_title="Time (s)",
        yaxis_title="Amplitude",
        yaxis2_title="Frequency (Hz)",
        yaxis2={"type": "log", "showgrid": True},
        xaxis={"showgrid": True},
        xaxis2={"showgrid": True},
    )
    return fig


def render_scalogram_tab(
    *,
    signal: npt.NDArray[np.float64],
    baseline: float,
    adc_samplerate: float,
    lpf_cutoff: float,
    result: nanopore_it.AnalysisTables,
    selected_event_index: int,
) -> None:
    range_options = ["Time range"]
    if not result.events.empty:
        range_options.insert(0, "Selected event")

    range_mode = st.radio(
        "Scalogram range",
        range_options,
        horizontal=True,
    )

    total_duration_s = len(signal) / adc_samplerate
    default_max_freq = max(1.0, min(float(lpf_cutoff), adc_samplerate / 2))
    control_cols = st.columns(6, vertical_alignment="center")
    min_freq_hz = control_cols[0].number_input(
        "Min frequency (Hz)",
        min_value=1.0,
        value=1000.0,
        step=100.0,
    )
    max_freq_hz = control_cols[1].number_input(
        "Max frequency (Hz)",
        min_value=1.0,
        value=default_max_freq,
        step=1000.0,
    )
    num_freqs = control_cols[2].number_input(
        "Frequency bins",
        min_value=8,
        max_value=256,
        value=96,
        step=8,
    )
    max_samples = control_cols[3].number_input(
        "Max samples",
        min_value=1024,
        max_value=65536,
        value=8192,
        step=1024,
    )
    wavelet_label = control_cols[4].selectbox(
        "Wavelet",
        tuple(SCALOGRAM_WAVELETS),
    )
    wavelet_type = SCALOGRAM_WAVELETS[wavelet_label]
    normalize_power = control_cols[5].checkbox("Normalize power", value=False)

    if range_mode == "Selected event" and not result.events.empty:
        selected_event_index = min(selected_event_index, len(result.events) - 1)
        event = result.events.iloc[selected_event_index]
        context_ms = st.number_input(
            "Event context (ms)",
            min_value=0.0,
            value=1.0,
            step=0.1,
        )
        context_samples = int(context_ms * 1e-3 * adc_samplerate)
        view_start = max(0, int(event.start_point) - context_samples)
        view_end = min(
            len(signal),
            int(event.end_point) + 1 + context_samples,
        )
    else:
        min_duration_s = 1 / adc_samplerate
        default_duration_s = min(0.1, total_duration_s)
        range_cols = st.columns(2)
        start_time_s = range_cols[0].number_input(
            "Start time (s)",
            min_value=0.0,
            max_value=max(0.0, total_duration_s),
            value=0.0,
            step=min(0.01, max(min_duration_s, default_duration_s)),
        )
        duration_s = range_cols[1].number_input(
            "Duration (s)",
            min_value=min_duration_s,
            max_value=max(min_duration_s, total_duration_s),
            value=max(min_duration_s, default_duration_s),
            step=max(min_duration_s, default_duration_s),
        )
        view_start = min(len(signal) - 1, int(start_time_s * adc_samplerate))
        view_end = min(
            len(signal),
            int((start_time_s + duration_s) * adc_samplerate),
        )

    times, freqs, power, current = compute_scalogram(
        signal,
        adc_samplerate=adc_samplerate,
        start_point=view_start,
        end_point=view_end,
        min_freq_hz=min_freq_hz,
        max_freq_hz=max_freq_hz,
        num_freqs=int(num_freqs),
        max_samples=int(max_samples),
        wavelet_type=wavelet_type,
        normalize_power=normalize_power,
    )

    if power.size == 0 or freqs.size == 0:
        st.warning("Scalogram cannot be computed for this range.")
        return

    if result.events.empty:
        event_ranges: tuple[tuple[float, float], ...] = ()
        events_in_view = result.events
    else:
        events_in_view = result.events[
            (result.events["start_point"] < view_end)
            & (result.events["end_point"] > view_start)
        ]
        event_ranges = tuple(
            (
                max(int(ev.start_point), view_start) / adc_samplerate,
                min(int(ev.end_point) + 1, view_end) / adc_samplerate,
            )
            for ev in events_in_view.itertuples(index=False)
        )

    metric_cols = st.columns(3)
    metric_cols[0].metric("Events in range", len(events_in_view))
    metric_cols[1].metric(
        "Current median",
        f"{float(np.nanmedian(current)):.3e}",
    )
    metric_cols[2].metric(
        "Current stdev",
        f"{float(np.nanstd(current)):.3e}",
    )
    st.plotly_chart(
        draw_scalogram(
            times,
            freqs,
            power,
            current,
            event_ranges,
            baseline,
            wavelet_label,
            normalize_power,
        ),
        width="stretch",
    )
