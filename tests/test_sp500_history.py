from datetime import date

import pytest

from backtests.sp500_history import (
    filter_samples_for_membership,
    intervals_by_ticker,
    parse_membership_history,
    tickers_overlapping,
    yahoo_ticker,
)


def test_snapshot_parser_handles_exit_reentry_and_missing_calendar_days():
    content = """date,tickers
2024-01-05,"A,BRK.B"
2024-01-08,"A"
2024-01-10,"A,BRK.B"
"""

    grouped = intervals_by_ticker(parse_membership_history(content))

    assert [(item.start_date, item.end_date) for item in grouped["A"]] == [
        (date(2024, 1, 5), date(2024, 1, 10))
    ]
    assert [(item.start_date, item.end_date) for item in grouped["BRK.B"]] == [
        (date(2024, 1, 5), date(2024, 1, 5)),
        (date(2024, 1, 10), date(2024, 1, 10)),
    ]
    assert yahoo_ticker("BRK.B") == "BRK-B"


def test_membership_filter_is_point_in_time_and_inclusive():
    content = """date,tickers
2024-01-02,"A,B"
2024-01-03,"A"
2024-01-04,"A"
"""
    grouped = intervals_by_ticker(parse_membership_history(content))
    samples = [
        {"sample_date": "2024-01-02"},
        {"sample_date": "2024-01-03"},
        {"sample_date": "2024-01-04"},
    ]

    assert filter_samples_for_membership(samples, grouped["B"]) == [samples[0]]
    assert tickers_overlapping(grouped, date(2024, 1, 3), date(2024, 1, 4)) == ["A"]


def test_same_day_revision_uses_last_snapshot():
    content = """date,tickers
2024-01-02,"A,B"
2024-01-03,"A,B"
2024-01-03,"A,C"
2024-01-04,"A,C"
"""

    grouped = intervals_by_ticker(parse_membership_history(content))

    assert grouped["B"][0].end_date == date(2024, 1, 2)
    assert grouped["C"][0].start_date == date(2024, 1, 3)


@pytest.mark.parametrize(
    "content",
    [
        "wrong,columns\n2024-01-02,A\n",
        "date,tickers\n",
        "date,tickers\n2024-01-03,A\n2024-01-02,B\n",
    ],
)
def test_invalid_membership_history_fails_loudly(content):
    with pytest.raises(ValueError):
        parse_membership_history(content)
