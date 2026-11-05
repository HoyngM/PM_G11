"""Activities with a fixed location, and the itemsets that are therefore not permuted.

An activity whose every occurrence is at the same position, counted from the start (e.g. a start
activity, always event 1) or counted from the end (e.g. an end activity, always the last event,
whatever the trace length), cannot have an important location, and moving it would create
unrealistic traces (e.g. "close" in the middle). Itemsets that contain such an activity are left
out of the permutation. ``DataManager`` (tools.py) finds the activities when a log is loaded;
``CrossValidationProcessPermutation`` leaves out the itemsets.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def fixed_location_activities(traces: Iterable[Sequence[str]], activities: Sequence[str]) -> list[str]:
    """The activities of ``activities`` whose location never varies over ``traces``."""
    from_start, from_end = {}, {}
    for trace in traces:
        for position, activity in enumerate(trace, start=1):
            from_start.setdefault(activity, set()).add(position)
            from_end.setdefault(activity, set()).add(len(trace) - position + 1)  # 1 = last event
    return [activity for activity in activities
            if len(from_start[activity]) == 1 or len(from_end[activity]) == 1]


def split_fixed_location_itemsets(itemsets: Sequence[frozenset[str]], fixed: Iterable[str]
                                  ) -> tuple[list[frozenset[str]], list[frozenset[str]]]:
    """Split ``itemsets`` into those without and those with a fixed-location activity."""
    fixed = set(fixed)
    kept = [itemset for itemset in itemsets if not itemset & fixed]
    excluded = [itemset for itemset in itemsets if itemset & fixed]
    return kept, excluded
