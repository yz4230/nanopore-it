import json
import os
import secrets
from pathlib import Path
from typing import Any
from typing import Final
from typing import Literal
from typing import cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, ValidationError
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


PARAMETER_STORE_PATH: Final[Path] = Path(".streamlit/file_parameters.json")
PARAMETER_WIDGET_PREFIX: Final[str] = "analysis_parameter_"
PARAMETER_FILENAME_STATE_KEY: Final[str] = "analysis_parameter_filename"
BASELINE_METHOD_OPTIONS: Final[tuple[str, ...]] = ("Q1", "Q2", "Manual")

BaselineMethod = Literal["Q1", "Q2", "Manual"]


class AnalysisParameters(BaseModel):
    model_config = ConfigDict(extra="ignore")

    adc_samplerate_khz: int = Field(default=250, gt=0)
    lpf_cutoff_khz: int = Field(default=100, gt=0)
    downsampling_factor: int = Field(default=1024, ge=1024)
    invert: bool = True
    baseline_method: BaselineMethod = "Q2"
    manual_baseline_a: float = 0.0
    auto_detect_clear: bool = True
    spike_std_threshold: float = Field(default=10.0, gt=0)
    baseline_window_std: float = Field(default=1.0, ge=0)
    extra_relaxation_ms: float = Field(default=10.0, ge=0)
    cusum_stepsize: float = Field(default=1.0, gt=0)
    cusum_threshold: float = Field(default=30.0, gt=0)


DEFAULT_PARAMETERS: Final[AnalysisParameters] = AnalysisParameters()


def parameter_widget_key(parameter_name: str) -> str:
    return f"{PARAMETER_WIDGET_PREFIX}{parameter_name}"


def load_parameter_store() -> dict[str, AnalysisParameters]:
    if not PARAMETER_STORE_PATH.exists():
        return {}

    try:
        store: Any = json.loads(PARAMETER_STORE_PATH.read_text())
    except OSError, json.JSONDecodeError:
        return {}

    if not isinstance(store, dict):
        return {}

    parameter_store: dict[str, AnalysisParameters] = {}
    for filename, settings in store.items():
        if not isinstance(filename, str) or not isinstance(settings, dict):
            continue

        try:
            parameter_store[filename] = AnalysisParameters.model_validate(settings)
        except ValidationError:
            continue

    return parameter_store


def save_parameter_store(store: dict[str, AnalysisParameters]) -> None:
    PARAMETER_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {filename: settings.model_dump() for filename, settings in store.items()}
    PARAMETER_STORE_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def initialize_parameter_widgets(filename: str | None) -> None:
    for name, default in DEFAULT_PARAMETERS.model_dump().items():
        st.session_state.setdefault(parameter_widget_key(name), default)

    if filename is None:
        st.session_state.pop(PARAMETER_FILENAME_STATE_KEY, None)
        return

    if st.session_state.get(PARAMETER_FILENAME_STATE_KEY) == filename:
        return

    store = load_parameter_store()
    settings = store.get(filename, DEFAULT_PARAMETERS)
    for name, value in settings.model_dump().items():
        st.session_state[parameter_widget_key(name)] = value

    st.session_state[PARAMETER_FILENAME_STATE_KEY] = filename


def collect_parameter_settings() -> AnalysisParameters:
    settings = {
        name: st.session_state.get(parameter_widget_key(name), default)
        for name, default in DEFAULT_PARAMETERS.model_dump().items()
    }
    return AnalysisParameters.model_validate(settings)


def save_parameters_for_filename(filename: str) -> None:
    store = load_parameter_store()
    store[filename] = collect_parameter_settings()
    save_parameter_store(store)


def calculate_baseline(
    signal: npt.NDArray[np.float64],
    *,
    method: BaselineMethod,
    manual_baseline_a: float,
) -> float:
    if method == "Manual":
        return float(manual_baseline_a)

    finite_signal = signal[np.isfinite(signal)]
    if finite_signal.size == 0:
        return np.nan

    if method == "Q1":
        return float(np.percentile(finite_signal, 25))
    if method == "Q2":
        return float(np.percentile(finite_signal, 50))

    raise ValueError(f"Unsupported baseline method: {method}")


def calculate_baseline_std(signal: npt.NDArray[np.float64]) -> float:
    finite_signal = signal[np.isfinite(signal)]
    if finite_signal.size == 0:
        return np.nan
    return float(np.std(finite_signal))


def authenticate():
    STATE_KEY: Final[str] = "auth_status"
    password = os.getenv("APP_PASSWORD")

    if password is None:
        # No password set, allow access without authentication
        return

    auth_status = st.session_state.get(STATE_KEY, False)
    if auth_status:
        return

    with st.form("login_form"):
        input_password = st.text_input("Enter password", type="password")
        submit = st.form_submit_button("Login")

    if submit:
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
    uploaded_filename = uploaded_file.name if uploaded_file is not None else None
    initialize_parameter_widgets(uploaded_filename)

    adc_samplerate = (
        st.sidebar.number_input(
            "ADC Samplerate (kHz)",
            key=parameter_widget_key("adc_samplerate_khz"),
        )
        * 1e3
    )
    lpf_cutoff = (
        st.sidebar.number_input(
            "LPF Cutoff (kHz)",
            key=parameter_widget_key("lpf_cutoff_khz"),
        )
        * 1e3
    )
    downsampling_factor = st.sidebar.number_input(
        "Downsampling factor",
        step=1024,
        min_value=1024,
        key=parameter_widget_key("downsampling_factor"),
    )
    invert = st.sidebar.checkbox("Invert signal", key=parameter_widget_key("invert"))

    baseline_method = st.sidebar.selectbox(
        "Baseline current",
        BASELINE_METHOD_OPTIONS,
        key=parameter_widget_key("baseline_method"),
    )
    baseline_method = cast(BaselineMethod, baseline_method)
    if baseline_method == "Manual":
        st.sidebar.number_input(
            "Manual baseline current (A)",
            step=1e-12,
            format="%.3e",
            key=parameter_widget_key("manual_baseline_a"),
        )
    manual_baseline_a = float(
        st.session_state.get(
            parameter_widget_key("manual_baseline_a"),
            DEFAULT_PARAMETERS.manual_baseline_a,
        )
    )

    auto_detect_clear = st.sidebar.checkbox(
        "Auto-detect clear signal", key=parameter_widget_key("auto_detect_clear")
    )

    if auto_detect_clear:
        spike_std_threshold = st.sidebar.number_input(
            "Spike threshold (std devs)",
            step=0.1,
            key=parameter_widget_key("spike_std_threshold"),
        )
        baseline_window_std = st.sidebar.number_input(
            "Baseline window width (std devs)",
            step=0.1,
            key=parameter_widget_key("baseline_window_std"),
        )
        extra_relaxation_ms = st.sidebar.number_input(
            "Extra relaxation (ms)",
            step=1.0,
            key=parameter_widget_key("extra_relaxation_ms"),
        )

    st.sidebar.divider()

    cusum_stepsize = st.sidebar.number_input(
        "CUSUM stepsize", step=1.0, key=parameter_widget_key("cusum_stepsize")
    )
    cusum_threshold = st.sidebar.number_input(
        "CUSUM threshold", step=1.0, key=parameter_widget_key("cusum_threshold")
    )

    if uploaded_filename is not None:
        if st.sidebar.button("Save parameters for this file"):
            save_parameters_for_filename(uploaded_filename)
            st.sidebar.success(f"Saved parameters for {uploaded_filename}")

        store = load_parameter_store()
        if uploaded_filename in store:
            st.sidebar.caption(f"Using saved parameters for {uploaded_filename}")

    if uploaded_file is not None:
        signal = load_data(
            uploaded_file,
            lpf_cutoff=lpf_cutoff,
            adc_samplerate=adc_samplerate,
            invert=invert,
        )
        baseline = calculate_baseline(
            signal,
            method=baseline_method,
            manual_baseline_a=manual_baseline_a,
        )
        baseline_std = calculate_baseline_std(signal)

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
            baseline = calculate_baseline(
                signal,
                method=baseline_method,
                manual_baseline_a=manual_baseline_a,
            )
            baseline_std = calculate_baseline_std(signal)

        st.sidebar.caption(f"Baseline: {baseline:.3e} A ({baseline_method})")

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
