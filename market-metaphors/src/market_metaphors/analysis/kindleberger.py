"""Kindleberger phase definitions and labeling helpers.

Each crisis event is defined with approximate date boundaries for the
five Minsky-Kindleberger phases: displacement, boom, euphoria, distress, revulsion.
"""

from datetime import date

import pandas as pd

# Phase window definitions for each crisis event.
# Each phase is (start_date, end_date) inclusive.
CRISIS_EVENTS: dict[str, dict[str, tuple[date, date]]] = {
    "eu_debt_2011": {
        "displacement": (date(2010, 5, 1), date(2010, 10, 31)),
        "boom": (date(2010, 11, 1), date(2011, 6, 30)),
        "distress": (date(2011, 7, 1), date(2011, 7, 31)),
        "revulsion": (date(2011, 8, 1), date(2011, 9, 30)),
    },
    "china_2015": {
        "displacement": (date(2015, 6, 1), date(2015, 6, 30)),
        "distress": (date(2015, 7, 1), date(2015, 8, 23)),
        "revulsion": (date(2015, 8, 24), date(2015, 9, 30)),
    },
    "brexit_2016": {
        "displacement": (date(2016, 2, 1), date(2016, 2, 29)),
        "boom": (date(2016, 3, 1), date(2016, 5, 31)),
        "distress": (date(2016, 6, 1), date(2016, 6, 23)),
        "revulsion": (date(2016, 6, 24), date(2016, 6, 30)),
    },
    "rate_hike_2018": {
        "displacement": (date(2018, 1, 1), date(2018, 1, 31)),
        "boom": (date(2018, 2, 1), date(2018, 9, 30)),
        "distress": (date(2018, 10, 1), date(2018, 10, 31)),
        "revulsion": (date(2018, 11, 1), date(2018, 12, 31)),
    },
    "covid_2020": {
        "displacement": (date(2020, 1, 1), date(2020, 1, 31)),
        "distress": (date(2020, 2, 1), date(2020, 2, 29)),
        "revulsion": (date(2020, 3, 1), date(2020, 3, 31)),
    },
}

ALL_PHASES = ["displacement", "boom", "euphoria", "distress", "revulsion"]


def label_phase(row_date: date, event: str | None = None) -> str:
    """Return the Kindleberger phase label for a given date.

    If event is specified, only check that crisis. Otherwise check all.
    Returns 'calm' if no phase matches.
    """
    events = {event: CRISIS_EVENTS[event]} if event else CRISIS_EVENTS
    for _event_name, phases in events.items():
        for phase, (start, end) in phases.items():
            if start <= row_date <= end:
                return phase
    return "calm"


def label_dataframe(
    df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Add 'phase' and 'crisis_event' columns to a DataFrame."""
    df = df.copy()
    df["phase"] = "calm"
    df["crisis_event"] = None

    for event_name, phases in CRISIS_EVENTS.items():
        for phase, (start, end) in phases.items():
            mask = (df[date_col].dt.date >= start) & (df[date_col].dt.date <= end)
            df.loc[mask, "phase"] = phase
            df.loc[mask, "crisis_event"] = event_name

    return df


def get_event_window(event: str) -> tuple[date, date]:
    """Return the full date range (min start, max end) for a crisis event."""
    phases = CRISIS_EVENTS[event]
    starts = [s for s, _ in phases.values()]
    ends = [e for _, e in phases.values()]
    return min(starts), max(ends)
