"""Unit tests for the pure calendar-feature function (requirements.md §3, design.md §2).

No I/O, no real data needed -- calendar_features.compute_calendar_vector is a pure function of a
date string, so every branch (Tet window, month/quarter-end, earnings window) can be exercised
exhaustively with hand-picked dates, per CLAUDE.md Testing quality rules (test real behavior, not
just shapes).

Written BEFORE `code/calendar_features.py` exists (test-first, CLAUDE.md §5 SDD Implement phase)
-- this file is expected to fail on collection/import until that module is created.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[1] / "code"
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

import math
import numpy as np
import pytest

from calendar_features import (
    compute_calendar_vector, CALENDAR_FEATURE_NAMES, TET_DATES, CALENDAR_FEATURE_GROUPS,
)

IDX = {name: i for i, name in enumerate(CALENDAR_FEATURE_NAMES)}


def _v(date_str):
    return compute_calendar_vector(date_str)


class TestShapeAndNames:
    def test_returns_10_features(self):
        vec = _v("2024-06-15")
        assert vec.shape == (10,)
        assert len(CALENDAR_FEATURE_NAMES) == 10

    def test_no_nan_or_inf(self):
        for d in ("2006-11-21", "2015-06-30", "2024-02-10", "2026-06-09"):
            vec = _v(d)
            assert np.all(np.isfinite(vec)), f"non-finite value for {d}: {vec}"

    def test_accepts_datetime_with_time_component(self):
        # dataset date strings sometimes carry a time component (see dataset_dual_news._norm_date)
        vec_plain = _v("2024-02-10")
        vec_ts = _v("2024-02-10 00:00:00")
        np.testing.assert_array_almost_equal(vec_plain, vec_ts)


class TestDayOfWeekCyclical:
    def test_monday_is_zero_angle(self):
        # 2024-01-01 is a Monday (weekday()==0)
        vec = _v("2024-01-01")
        assert vec[IDX["dow_sin"]] == pytest.approx(0.0, abs=1e-6)
        assert vec[IDX["dow_cos"]] == pytest.approx(1.0, abs=1e-6)

    def test_friday_matches_formula(self):
        # 2024-01-05 is a Friday (weekday()==4)
        vec = _v("2024-01-05")
        assert vec[IDX["dow_sin"]] == pytest.approx(math.sin(2 * math.pi * 4 / 5), abs=1e-6)
        assert vec[IDX["dow_cos"]] == pytest.approx(math.cos(2 * math.pi * 4 / 5), abs=1e-6)

    def test_weekend_falls_back_via_modulo(self):
        # 2024-01-06 is a Saturday (weekday()==5) -- real trading data should never contain
        # weekend dates, but the function must not crash; documented fallback is weekday%5,
        # which maps Saturday onto the same angle as Monday (weekday 5 % 5 == 0).
        vec = _v("2024-01-06")
        assert vec[IDX["dow_sin"]] == pytest.approx(0.0, abs=1e-6)
        assert vec[IDX["dow_cos"]] == pytest.approx(1.0, abs=1e-6)


class TestMonthCyclical:
    def test_january_is_zero_angle(self):
        vec = _v("2024-01-15")
        assert vec[IDX["month_sin"]] == pytest.approx(0.0, abs=1e-6)
        assert vec[IDX["month_cos"]] == pytest.approx(1.0, abs=1e-6)

    def test_december_matches_formula(self):
        vec = _v("2024-12-15")
        assert vec[IDX["month_sin"]] == pytest.approx(math.sin(2 * math.pi * 11 / 12), abs=1e-6)
        assert vec[IDX["month_cos"]] == pytest.approx(math.cos(2 * math.pi * 11 / 12), abs=1e-6)


class TestTetProximity:
    def test_on_tet_day_proximity_is_one(self):
        assert TET_DATES[2024] == "2024-02-10"
        vec = _v("2024-02-10")
        assert vec[IDX["tet_proximity"]] == pytest.approx(1.0, abs=1e-6)
        assert vec[IDX["in_tet_window"]] == 1.0

    def test_exactly_10_days_after_is_still_in_window(self):
        vec = _v("2024-02-20")  # 10 days after Tet 2024-02-10
        assert vec[IDX["tet_proximity"]] == pytest.approx(math.exp(-1.0), abs=1e-6)
        assert vec[IDX["in_tet_window"]] == 1.0

    def test_11_days_after_is_outside_window(self):
        vec = _v("2024-02-21")  # 11 days after Tet
        assert vec[IDX["in_tet_window"]] == 0.0

    def test_far_from_tet_proximity_near_zero(self):
        vec = _v("2024-07-15")  # far from any Tet date
        assert vec[IDX["tet_proximity"]] < 0.01
        assert vec[IDX["in_tet_window"]] == 0.0

    def test_year_without_explicit_tet_entry_still_works(self):
        # date range only covers 2005-2027 (requirements.md A1); a date outside that range must
        # not crash -- nearest-neighbor search over the full table still returns a finite answer.
        vec = _v("2004-06-01")
        assert np.all(np.isfinite(vec))


class TestMonthAndQuarterEnd:
    def test_last_3_days_of_31_day_month_is_month_end(self):
        for d in ("2024-01-29", "2024-01-30", "2024-01-31"):
            assert _v(d)[IDX["is_month_end"]] == 1.0, d
        assert _v("2024-01-28")[IDX["is_month_end"]] == 0.0

    def test_leap_february_last_3_days(self):
        # 2024 is a leap year: Feb has 29 days
        for d in ("2024-02-27", "2024-02-28", "2024-02-29"):
            assert _v(d)[IDX["is_month_end"]] == 1.0, d
        assert _v("2024-02-26")[IDX["is_month_end"]] == 0.0

    def test_non_leap_february_last_3_days(self):
        for d in ("2023-02-26", "2023-02-27", "2023-02-28"):
            assert _v(d)[IDX["is_month_end"]] == 1.0, d

    def test_quarter_end_requires_month_end_and_quarter_month(self):
        assert _v("2024-03-30")[IDX["is_quarter_end"]] == 1.0   # month-end + March
        assert _v("2024-03-15")[IDX["is_quarter_end"]] == 0.0   # March but not month-end
        assert _v("2024-04-29")[IDX["is_quarter_end"]] == 0.0   # month-end but April (not a quarter month)

    def test_all_four_quarter_months(self):
        for d in ("2024-03-30", "2024-06-29", "2024-09-29", "2024-12-30"):
            assert _v(d)[IDX["is_quarter_end"]] == 1.0, d


class TestEarningsWindow:
    def test_inside_q1_report_window(self):
        vec = _v("2024-04-10")  # between quarter-end (Mar 31) and deadline (Apr 20)
        assert vec[IDX["in_earnings_window"]] == 1.0

    def test_after_q1_deadline_is_outside_window(self):
        vec = _v("2024-04-25")
        assert vec[IDX["in_earnings_window"]] == 0.0

    def test_midyear_outside_any_window(self):
        vec = _v("2024-05-15")
        assert vec[IDX["in_earnings_window"]] == 0.0

    def test_january_window_references_prior_year_q4(self):
        # Jan 1-20 is the reporting window for the PREVIOUS year's Q4 (quarter-end 12/31)
        vec = _v("2024-01-10")
        assert vec[IDX["in_earnings_window"]] == 1.0

    def test_earnings_proximity_matches_nearest_deadline_distance(self):
        vec = _v("2024-01-05")  # 15 days before the 2024-01-20 deadline
        assert vec[IDX["earnings_proximity"]] == pytest.approx(math.exp(-15.0 / 10.0), abs=1e-6)


class TestCalendarFeatureGroups:
    def test_all_group_matches_full_name_list(self):
        assert CALENDAR_FEATURE_GROUPS["all"] == CALENDAR_FEATURE_NAMES

    def test_every_group_name_is_a_valid_feature(self):
        for group, names in CALENDAR_FEATURE_GROUPS.items():
            for n in names:
                assert n in CALENDAR_FEATURE_NAMES, f"group {group!r} has unknown feature {n!r}"

    def test_groups_are_non_overlapping_except_all(self):
        ablation_groups = {k: v for k, v in CALENDAR_FEATURE_GROUPS.items() if k != "all"}
        seen = set()
        for names in ablation_groups.values():
            overlap = seen & set(names)
            assert not overlap, f"feature(s) {overlap} appear in more than one ablation group"
            seen |= set(names)

    def test_ablation_groups_union_covers_all_features(self):
        ablation_groups = {k: v for k, v in CALENDAR_FEATURE_GROUPS.items() if k != "all"}
        union = set().union(*ablation_groups.values())
        assert union == set(CALENDAR_FEATURE_NAMES)
