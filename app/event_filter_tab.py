import json
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from streamlit.elements.plotly_chart import PlotlyState

import nanopore_it


LINEAR_LINE: Final[dict[str, str | int]] = {"shape": "linear", "smoothing": 0}
CURRENT_EVENT_ID_STATE_KEY: Final[str] = "event_filter_current_event_id"
IGNORE_PLOTLY_SELECTION_STATE_KEY: Final[str] = (
    "event_filter_ignore_plotly_selection_once"
)
DELETE_HISTORY_STATE_KEY: Final[str] = "event_filter_delete_history"
DELETE_HISTORY_INDEX_STATE_KEY: Final[str] = "event_filter_delete_history_index"
DELETED_EVENT_IDS_STATE_KEY: Final[str] = "event_filter_deleted_event_ids"
FILTER_SIGNATURE_STATE_KEY: Final[str] = "event_filter_signature"
DWELL_MIN_STATE_KEY: Final[str] = "event_filter_dwell_min"
DWELL_MAX_STATE_KEY: Final[str] = "event_filter_dwell_max"
DELLI_MIN_STATE_KEY: Final[str] = "event_filter_delli_min"
DELLI_MAX_STATE_KEY: Final[str] = "event_filter_delli_max"
FILTER_INPUT_STATE_KEYS: Final[tuple[str, ...]] = (
    DWELL_MIN_STATE_KEY,
    DWELL_MAX_STATE_KEY,
    DELLI_MIN_STATE_KEY,
    DELLI_MAX_STATE_KEY,
)


def _event_signature(result: nanopore_it.AnalysisTables) -> tuple[tuple[int, int], ...]:
    if result.events.empty:
        return ()

    return tuple(
        (int(start_point), int(end_point))
        for start_point, end_point in result.events[
            ["start_point", "end_point"]
        ].itertuples(index=False, name=None)
    )


def _initialize_manual_delete_state(result: nanopore_it.AnalysisTables) -> None:
    signature = _event_signature(result)
    if st.session_state.get(FILTER_SIGNATURE_STATE_KEY) == signature:
        return

    st.session_state[FILTER_SIGNATURE_STATE_KEY] = signature
    for state_key in FILTER_INPUT_STATE_KEYS:
        st.session_state.pop(state_key, None)
    st.session_state.pop(CURRENT_EVENT_ID_STATE_KEY, None)
    st.session_state.pop(IGNORE_PLOTLY_SELECTION_STATE_KEY, None)
    st.session_state[DELETED_EVENT_IDS_STATE_KEY] = set()
    st.session_state[DELETE_HISTORY_STATE_KEY] = [set()]
    st.session_state[DELETE_HISTORY_INDEX_STATE_KEY] = 0


def _deleted_event_ids() -> set[int]:
    deleted = st.session_state.get(DELETED_EVENT_IDS_STATE_KEY)
    if isinstance(deleted, set):
        return {int(event_id) for event_id in deleted}
    return set()


def _set_deleted_event_ids(deleted_event_ids: set[int]) -> None:
    st.session_state[DELETED_EVENT_IDS_STATE_KEY] = set(deleted_event_ids)


def _push_manual_delete_state(deleted_event_ids: set[int]) -> None:
    history = st.session_state.get(DELETE_HISTORY_STATE_KEY, [set()])
    if not isinstance(history, list):
        history = [set()]

    history_index = int(st.session_state.get(DELETE_HISTORY_INDEX_STATE_KEY, 0))
    history = history[: history_index + 1]
    if history and set(history[-1]) == deleted_event_ids:
        return

    history.append(set(deleted_event_ids))
    st.session_state[DELETE_HISTORY_STATE_KEY] = history
    st.session_state[DELETE_HISTORY_INDEX_STATE_KEY] = len(history) - 1
    _set_deleted_event_ids(deleted_event_ids)


def _undo_manual_delete() -> None:
    history = st.session_state.get(DELETE_HISTORY_STATE_KEY, [set()])
    history_index = int(st.session_state.get(DELETE_HISTORY_INDEX_STATE_KEY, 0))
    if history_index <= 0:
        return

    history_index -= 1
    st.session_state[DELETE_HISTORY_INDEX_STATE_KEY] = history_index
    _set_deleted_event_ids(set(history[history_index]))


def _redo_manual_delete() -> None:
    history = st.session_state.get(DELETE_HISTORY_STATE_KEY, [set()])
    history_index = int(st.session_state.get(DELETE_HISTORY_INDEX_STATE_KEY, 0))
    if history_index >= len(history) - 1:
        return

    history_index += 1
    st.session_state[DELETE_HISTORY_INDEX_STATE_KEY] = history_index
    _set_deleted_event_ids(set(history[history_index]))


def _selected_event_ids(state: PlotlyState) -> set[int]:
    selection = state.get("selection")
    if selection is None:
        return set()

    points = selection.get("points")
    if points is None:
        return set()

    event_ids: set[int] = set()
    for point in points:
        customdata = point.get("customdata")
        if customdata is not None:
            event_ids.add(int(customdata))
    return event_ids


def _current_event_id(filtered_event_ids: npt.NDArray[np.int64]) -> int | None:
    if filtered_event_ids.size == 0:
        st.session_state.pop(CURRENT_EVENT_ID_STATE_KEY, None)
        return None

    current = st.session_state.get(CURRENT_EVENT_ID_STATE_KEY)
    if current is not None and int(current) in set(filtered_event_ids):
        return int(current)

    first_event_id = int(filtered_event_ids[0])
    st.session_state[CURRENT_EVENT_ID_STATE_KEY] = first_event_id
    return first_event_id


def _set_current_event_id(event_id: int | None) -> None:
    if event_id is None:
        st.session_state.pop(CURRENT_EVENT_ID_STATE_KEY, None)
    else:
        st.session_state[CURRENT_EVENT_ID_STATE_KEY] = int(event_id)


def _set_current_event_id_from_navigation(event_id: int | None) -> None:
    _set_current_event_id(event_id)
    st.session_state[IGNORE_PLOTLY_SELECTION_STATE_KEY] = True


def _neighbor_event_id(
    filtered_event_ids: npt.NDArray[np.int64],
    *,
    current_event_id: int | None,
    direction: int,
) -> int | None:
    if filtered_event_ids.size == 0:
        return None

    if current_event_id is None:
        return int(filtered_event_ids[0])

    matches = np.where(filtered_event_ids == current_event_id)[0]
    if matches.size == 0:
        return int(filtered_event_ids[0])

    next_position = int(matches[0]) + direction
    next_position = max(0, min(next_position, len(filtered_event_ids) - 1))
    return int(filtered_event_ids[next_position])


def _event_position(
    filtered_event_ids: npt.NDArray[np.int64],
    current_event_id: int | None,
) -> int | None:
    if current_event_id is None:
        return None

    matches = np.where(filtered_event_ids == current_event_id)[0]
    if matches.size == 0:
        return None
    return int(matches[0])


def _install_keyboard_shortcuts() -> None:
    components.html(
        """
        <script>
        (() => {
          const shortcuts = {
            ArrowLeft: "Previous event",
            ArrowRight: "Next event",
            d: "Delete selected",
            D: "Delete selected",
            z: "Undo",
            Z: "Undo",
            y: "Redo",
            Y: "Redo",
            r: "Reset manual deletes",
            R: "Reset manual deletes",
          };

          function shouldIgnore(event) {
            const target = event.target;
            if (!target) return false;
            const tagName = target.tagName;
            return (
              target.isContentEditable ||
              tagName === "INPUT" ||
              tagName === "TEXTAREA" ||
              tagName === "SELECT"
            );
          }

          function clickButton(label) {
            const buttons = Array.from(window.parent.document.querySelectorAll("button"));
            const button = buttons.find((candidate) => candidate.innerText.trim() === label);
            if (!button || button.disabled || button.getAttribute("aria-disabled") === "true") {
              return false;
            }
            button.click();
            return true;
          }

          if (window.parent.__nanoporeEventFilterShortcutsInstalled) return;
          window.parent.__nanoporeEventFilterShortcutsInstalled = true;
          window.parent.document.addEventListener("keydown", (event) => {
            if (shouldIgnore(event)) return;
            const label = shortcuts[event.key];
            if (!label) return;
            if (clickButton(label)) {
              event.preventDefault();
              event.stopPropagation();
            }
          }, true);
        })();
        </script>
        """,
        height=0,
    )


def _range_mask(
    events: pd.DataFrame,
    *,
    dwell_min: float,
    dwell_max: float,
    delli_min: float,
    delli_max: float,
) -> pd.Series:
    if dwell_min > dwell_max or delli_min > delli_max:
        return pd.Series(False, index=events.index)

    return (
        (events["dwell"] >= dwell_min)
        & (events["dwell"] <= dwell_max)
        & (events["delli"] >= delli_min)
        & (events["delli"] <= delli_max)
    )


def filter_analysis_tables(
    result: nanopore_it.AnalysisTables,
    event_mask: pd.Series,
) -> nanopore_it.AnalysisTables:
    if result.events.empty:
        return nanopore_it.AnalysisTables(
            events=result.events.copy(),
            states=result.states.copy(),
            n_children=np.array([], dtype=np.int64),
        )

    selected_events = result.events.loc[event_mask].copy()
    original_indices = selected_events.index.to_numpy(dtype=np.int64)
    index_mapping = {
        int(original_index): filtered_index
        for filtered_index, original_index in enumerate(original_indices)
    }

    selected_events = selected_events.reset_index(drop=True)
    if "index" in selected_events.columns:
        selected_events["index"] = np.arange(len(selected_events), dtype=np.int64)

    n_children = result.n_children[original_indices].astype(np.int64, copy=True)

    if result.states.empty or len(selected_events) == 0:
        selected_states = result.states.iloc[0:0].copy()
    else:
        selected_parent_indices = set(index_mapping)
        parent_indices = result.states["parent_index"].astype(int)
        selected_states = result.states[
            parent_indices.isin(selected_parent_indices)
        ].copy()
        selected_states["parent_index"] = selected_states["parent_index"].map(
            index_mapping
        ).astype(np.int64)
        selected_states = selected_states.reset_index(drop=True)

    return nanopore_it.AnalysisTables(
        events=selected_events,
        states=selected_states,
        n_children=n_children,
    )


def _filter_step(
    values: pd.Series,
    *,
    fallback: float,
) -> float:
    if values.empty:
        return fallback

    value_min = float(values.min())
    value_max = float(values.max())
    return max((value_max - value_min) / 100, fallback)


def _draw_selected_event_currents(
    *,
    signal: npt.NDArray[np.float64],
    events: pd.DataFrame,
    event_ids: set[int],
    adc_samplerate: float,
    max_events: int = 20,
) -> go.Figure:
    fig = go.Figure()
    sorted_event_ids = sorted(event_ids)
    displayed_event_ids = sorted_event_ids[:max_events]

    for event_id in displayed_event_ids:
        event = events.loc[event_id]
        start = max(0, int(event.start_point))
        end = min(len(signal), int(event.end_point) + 1)
        if end <= start:
            continue

        event_samples = end - start
        context_samples = max(1, event_samples // 3)
        view_start = max(0, start - context_samples)
        view_end = min(len(signal), end + context_samples)

        current = signal[view_start:view_end]
        relative_time_us = (np.arange(view_start, view_end) - start) / adc_samplerate
        relative_time_us = relative_time_us * 1e6
        fig.add_trace(
            go.Scatter(
                x=relative_time_us,
                y=current,
                mode="lines",
                line=LINEAR_LINE,
                name=f"Event {event_id}",
            )
        )
        event_end_us = event_samples / adc_samplerate * 1e6
        fig.add_vline(
            x=event_end_us,
            line_dash="dot",
            line_color="rgba(220, 20, 60, 0.95)",
            line_width=3,
        )

    title = "Selected Event Current"
    if len(sorted_event_ids) > 1:
        title = f"Selected Event Currents ({len(displayed_event_ids)}/{len(sorted_event_ids)})"

    if displayed_event_ids:
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="rgba(34, 139, 34, 0.95)",
            line_width=3,
        )

    fig.update_layout(
        title=title,
        xaxis_title="Relative Time from Event Start (µs)",
        yaxis_title="Current (A)",
        xaxis={"showgrid": True},
        yaxis={"showgrid": True},
    )
    return fig


def _export_filename(source_filename: str | None) -> str:
    if source_filename is None:
        return "filtered_events.json"

    stem = Path(source_filename).stem
    if not stem:
        return "filtered_events.json"
    return f"{stem}_filtered_events.json"


@st.cache_data(show_spinner=False)
def _event_export_json(
    *,
    events: pd.DataFrame,
    adc_samplerate: float,
    source_filename: str | None,
) -> bytes:
    export_events: list[dict[str, object]] = []

    event_ids = events.index.to_numpy(dtype=np.int64)
    start_points = events["start_point"].to_numpy(dtype=np.int64)
    end_points = events["end_point"].to_numpy(dtype=np.int64)
    for event_id, start_point, end_point in zip(
        event_ids, start_points, end_points, strict=True
    ):
        start = int(start_point)
        end = int(end_point)
        if end < start:
            continue

        export_events.append(
            {
                "event_index": int(event_id),
                "start_sample": start,
                "end_sample": end,
            }
        )

    payload = {
        "source_filename": source_filename,
        "adc_samplerate_hz": adc_samplerate,
        "event_count": len(export_events),
        "events": export_events,
    }
    return json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()


def render_event_filter_tab(
    *,
    signal: npt.NDArray[np.float64],
    adc_samplerate: float,
    result: nanopore_it.AnalysisTables,
    source_filename: str | None = None,
) -> nanopore_it.AnalysisTables:
    _initialize_manual_delete_state(result)

    if result.events.empty:
        st.warning("No events detected.")
        return filter_analysis_tables(
            result,
            pd.Series(False, index=result.events.index),
        )

    events = result.events
    dwell_ms = events["dwell"] / 1e3
    delli_pa = events["delli"] * 1e12
    default_dwell_min_ms = float(dwell_ms.min())
    default_dwell_max_ms = float(dwell_ms.max())
    default_delli_min_pa = float(delli_pa.min())
    default_delli_max_pa = float(delli_pa.max())

    filter_cols = st.columns(4)
    dwell_min_ms = filter_cols[0].number_input(
        "Dwell min (ms)",
        value=default_dwell_min_ms,
        step=_filter_step(dwell_ms, fallback=0.001),
        format="%.6g",
        key=DWELL_MIN_STATE_KEY,
    )
    dwell_max_ms = filter_cols[1].number_input(
        "Dwell max (ms)",
        value=default_dwell_max_ms,
        step=_filter_step(dwell_ms, fallback=0.001),
        format="%.6g",
        key=DWELL_MAX_STATE_KEY,
    )
    delli_min_pa = filter_cols[2].number_input(
        "Delli min (pA)",
        value=default_delli_min_pa,
        step=_filter_step(delli_pa, fallback=0.001),
        format="%.6g",
        key=DELLI_MIN_STATE_KEY,
    )
    delli_max_pa = filter_cols[3].number_input(
        "Delli max (pA)",
        value=default_delli_max_pa,
        step=_filter_step(delli_pa, fallback=0.001),
        format="%.6g",
        key=DELLI_MAX_STATE_KEY,
    )

    range_mask = _range_mask(
        events,
        dwell_min=float(dwell_min_ms) * 1e3,
        dwell_max=float(dwell_max_ms) * 1e3,
        delli_min=float(delli_min_pa) * 1e-12,
        delli_max=float(delli_max_pa) * 1e-12,
    )
    if float(dwell_min_ms) > float(dwell_max_ms):
        st.warning("Dwell min must be less than or equal to Dwell max.")
    if float(delli_min_pa) > float(delli_max_pa):
        st.warning("Delli min must be less than or equal to Delli max.")

    deleted_event_ids = _deleted_event_ids()
    event_ids = events.index.to_series(index=events.index).astype(int)
    manual_delete_mask = ~event_ids.isin(deleted_event_ids)
    final_mask = range_mask & manual_delete_mask

    filtered_events = events.loc[final_mask]
    range_filtered_events = events.loc[range_mask]
    filtered_event_ids: npt.NDArray[np.int64] = filtered_events.index.to_numpy(
        dtype=np.int64
    )
    current_event_id = _current_event_id(filtered_event_ids)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Total events", len(events))
    metric_cols[1].metric("In Dwell/Delli range", len(range_filtered_events))
    metric_cols[2].metric("Manual deletes", len(deleted_event_ids))
    metric_cols[3].metric("Remaining events", len(filtered_events))

    st.download_button(
        "Export remaining events",
        data=_event_export_json(
            events=filtered_events,
            adc_samplerate=adc_samplerate,
            source_filename=source_filename,
        ),
        file_name=_export_filename(source_filename),
        mime="application/json",
        disabled=filtered_events.empty,
        help="Exports source filename and sample ranges for events remaining after Dwell/Delli filtering and manual deletes.",
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered_events["dwell"] / 1e3,
            y=filtered_events["delli"] * 1e12,
            customdata=filtered_event_ids,
            mode="markers",
            name="Events",
        )
    )
    if current_event_id is not None and current_event_id in filtered_events.index:
        current_event = filtered_events.loc[[current_event_id]]
        fig.add_trace(
            go.Scatter(
                x=current_event["dwell"] / 1e3,
                y=current_event["delli"] * 1e12,
                customdata=np.array([current_event_id], dtype=np.int64),
                mode="markers",
                marker={
                    "color": "rgba(255, 255, 255, 0)",
                    "line": {"color": "rgba(220, 20, 60, 0.95)", "width": 3},
                    "size": 13,
                },
                name="Current",
            )
        )
    fig.update_layout(
        title=f"Event Dwell Time vs Delli (n={len(filtered_events)}/{len(events)})",
        xaxis_title="Dwell Time (ms)",
        yaxis_title="Delli (pA)",
        xaxis={"showgrid": True, "type": "log"},
        yaxis={"showgrid": True, "type": "log"},
        clickmode="event+select",
    )

    if filtered_events.empty:
        st.info("No events match the current filter.")

    chart_event = st.plotly_chart(
        fig,
        width="stretch",
        on_select="rerun",
        selection_mode=("points", "box", "lasso"),
    )
    selected_event_ids = _selected_event_ids(chart_event)
    ignore_plotly_selection = bool(
        st.session_state.pop(IGNORE_PLOTLY_SELECTION_STATE_KEY, False)
    )
    if selected_event_ids and not ignore_plotly_selection:
        _set_current_event_id(min(selected_event_ids))
        current_event_id = _current_event_id(filtered_event_ids)
    if ignore_plotly_selection:
        selected_event_ids = set()

    current_event_position = _event_position(filtered_event_ids, current_event_id)
    if current_event_id is None or current_event_position is None:
        st.caption("Selected event index: -")
    else:
        st.caption(
            f"Selected event index: {current_event_id} "
            f"({current_event_position + 1}/{len(filtered_event_ids)})"
        )

    delete_target_event_ids = selected_event_ids
    if not delete_target_event_ids and current_event_id is not None:
        delete_target_event_ids = {current_event_id}

    previous_event_id = _neighbor_event_id(
        filtered_event_ids,
        current_event_id=current_event_id,
        direction=-1,
    )
    next_event_id = _neighbor_event_id(
        filtered_event_ids,
        current_event_id=current_event_id,
        direction=1,
    )

    _install_keyboard_shortcuts()
    nav_cols = st.columns(2)
    if nav_cols[0].button(
        "Previous event",
        disabled=current_event_id is None or previous_event_id == current_event_id,
        help="Left arrow",
        use_container_width=True,
    ):
        _set_current_event_id_from_navigation(previous_event_id)
        st.rerun()

    if nav_cols[1].button(
        "Next event",
        disabled=current_event_id is None or next_event_id == current_event_id,
        help="Right arrow",
        use_container_width=True,
    ):
        _set_current_event_id_from_navigation(next_event_id)
        st.rerun()

    display_event_ids = selected_event_ids
    if not display_event_ids and current_event_id is not None:
        display_event_ids = {current_event_id}

    if display_event_ids:
        st.plotly_chart(
            _draw_selected_event_currents(
                signal=signal,
                events=events,
                event_ids=display_event_ids,
                adc_samplerate=adc_samplerate,
            ),
            width="stretch",
        )
    else:
        st.info("Select events in the scatter plot to view their current traces.")

    history = st.session_state.get(DELETE_HISTORY_STATE_KEY, [set()])
    history_index = int(st.session_state.get(DELETE_HISTORY_INDEX_STATE_KEY, 0))
    button_cols = st.columns(4)
    if button_cols[0].button(
        "Delete selected",
        disabled=len(delete_target_event_ids) == 0,
        help="D",
        use_container_width=True,
    ):
        _push_manual_delete_state(deleted_event_ids | delete_target_event_ids)
        st.rerun()

    if button_cols[1].button(
        "Undo",
        disabled=history_index <= 0,
        help="Z",
        use_container_width=True,
    ):
        _undo_manual_delete()
        st.rerun()

    if button_cols[2].button(
        "Redo",
        disabled=history_index >= len(history) - 1,
        help="Y",
        use_container_width=True,
    ):
        _redo_manual_delete()
        st.rerun()

    if button_cols[3].button(
        "Reset manual deletes",
        disabled=not deleted_event_ids,
        help="R",
        use_container_width=True,
    ):
        _push_manual_delete_state(set())
        st.rerun()

    return filter_analysis_tables(result, final_mask)
