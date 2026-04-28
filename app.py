import os
import secrets
from typing import Final

import numpy as np
import numpy.typing as npt
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from app.event_analysis_tab import render_event_analysis_tab
from app.scalogram_tab import render_scalogram_tab
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


detect_clear_regions = st.cache_data(max_entries=1)(nanopore_it.detect_clear_regions)
analyze_tables = st.cache_data(max_entries=1)(nanopore_it.analyze_tables)


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

        result = analyze_tables(
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

        event_tab, scalogram_tab = st.tabs(["Event analysis", "Scalogram"])

        with event_tab:
            selected_event_index = render_event_analysis_tab(
                signal=signal,
                baseline=baseline,
                adc_samplerate=adc_samplerate,
                downsampling_factor=downsampling_factor,
                result=result,
            )

        with scalogram_tab:
            render_scalogram_tab(
                signal=signal,
                adc_samplerate=adc_samplerate,
                baseline=baseline,
                lpf_cutoff=lpf_cutoff,
                result=result,
                selected_event_index=selected_event_index,
            )


if __name__ == "__main__":
    main()
