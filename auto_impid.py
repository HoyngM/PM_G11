"""Automatic IMPresseD pattern discovery.

Ported from ``AutoStepWise_PPD`` in ``Auto_IMPID.py`` of
https://github.com/MozhganVD/InteractivePatternDetection. The function follows the original line by
line, with the same steps, loops and (PEP 8) names, so that the two can be compared side by side.
How it works, in short:

1. Score every single activity on the three interest functions (see ``imipd.py``).
2. Keep the *Pareto front*: the activities for which no other activity is at least as good on
   all three interests and strictly better on one. No thresholds or weights are needed.
3. Extend each activity on the front by one neighbouring event (``pattern_extension``), score the
   new 2-event patterns and keep their Pareto front.
4. Extend those patterns again (``single_pattern_extender``), up to ``max_extension_step`` steps.

Differences with the original, marked ``Original:`` in the code:

* Inputs: a ``DataManager`` and the training case ids instead of 18 separate arguments. The original
  makes its own train/test split; here ``main.py`` makes it.
* Trace graphs are chains of events (the traces are totally ordered by ``event_nr``), not built from
  timestamps: the BPIC11 timestamps are always at 23:00, so most events of a case share one.
* Output: one row per pattern on the fronts (with its interests), from which
  ``extensions/itemset_selection.py`` takes the activity sets. The original returns the pattern
  counts of the training and test cases (``train_X``, ``test_X``) and saves every pattern on a
  front as a JSON file.
* ``max_extension_step`` 0 (single activities only) and an empty list of new patterns are allowed.
"""

from __future__ import annotations

from typing import Sequence

import networkx as nx
import pandas as pd
from paretoset import paretoset

from imipd import (add_pattern_counts, calculate_pairwise_case_distance, create_pattern_attributes,
                   create_patient_data, describe_pattern, pattern_extension, single_pattern_extender,
                   trace_graph_generator, PatternsDictionary)
from tools import DataManager

PARETO_FEATURES = ["Outcome_Interest", "Frequency_Interest", "Case_Distance_Interest"]
PARETO_SENSE = ["max", "max", "min"]  # the Pareto directions of the three interests


def pareto_rows(paretoset_patterns: pd.DataFrame, length: int, patterns_dict: dict | None = None) -> pd.DataFrame:
    """The patterns on a front as output rows: ``pattern`` (e.g. ``a -> b``), ``length`` and ``itemset``.

    Single activities (``patterns_dict`` None) are their own pattern and itemset. The ordering of a
    longer pattern is dropped in its itemset: ``a -> b`` and ``b => a`` both become ``a + b``.
    """
    ids = list(paretoset_patterns["patterns"])
    if patterns_dict is None:
        pattern, itemset = ids, ids
    else:
        pattern = [describe_pattern(patterns_dict[p]["pattern"]) for p in ids]
        itemset = [" + ".join(sorted({v for _, v in patterns_dict[p]["pattern"].nodes(data="value")})) for p in ids]
    return paretoset_patterns.assign(pattern=pattern, length=length, itemset=itemset)[
        ["pattern", "length", "itemset", *PARETO_FEATURES]]


def auto_stepwise_ppd(max_extension_step: int, max_gap_between_events: int, data_manager: DataManager,
                      train_ids: Sequence[str]) -> pd.DataFrame:
    """Discover patterns step by step; return the patterns on the Pareto front of every step.

    ``max_extension_step``: 1 gives patterns up to length 2, 2 up to length 3, and 0 only single
    activities. Instances are counted in all cases, but the interests and so the Pareto fronts only
    use the training cases ``train_ids``.

    Returns one row per selected pattern: ``pattern`` (e.g. ``a -> b``), ``length``, ``itemset``
    (e.g. ``a + b``) and the three interests.
    """
    data, case_id, activity, outcome = (data_manager.data, data_manager.case_id, data_manager.activity,
                                        data_manager.outcome)
    pareto_features, pareto_sense = PARETO_FEATURES, PARETO_SENSE
    # Original: patient_data and pairwise_distances_array are made by the caller.
    patient_data = create_patient_data(data_manager)
    attributes = data_manager.numeric_attributes + data_manager.categorical_attributes
    pairwise_distances_array = calculate_pairwise_case_distance(data_manager.case_table[attributes],
                                                                data_manager.numeric_attributes)

    # Original: split test and train data here (train_test_split, random_state=42, stratified).
    # Here the training cases are given; the test cases are not needed.
    train_data = patient_data[patient_data[case_id].isin(train_ids)]

    all_extended_patterns = []
    activity_attributes = create_pattern_attributes(train_data, outcome,
                                                    list(data[activity].unique()),
                                                    pairwise_distances_array)

    objectives_attributes = activity_attributes[pareto_features]
    # paretoset returns True for every pattern that no other pattern beats on all three.
    mask = paretoset(objectives_attributes, sense=pareto_sense)
    paretoset_activities = activity_attributes[mask]
    all_extended_patterns.extend(list(paretoset_activities["patterns"]))
    # Original: train_X and test_X (the pattern counts) are built here; we keep the front instead.
    selected = [pareto_rows(paretoset_activities, length=1)]
    if max_extension_step == 0:
        return pd.concat(selected).reset_index(drop=True)

    # Extend every activity on the front by one event.
    # This gives patterns of length 2, e.g. "a -> b", "b -> a", "a => c".
    extended_patterns_at_stage = PatternsDictionary()
    all_extended_patterns_1_list = []
    event_log_graphs = dict()
    for core_activity in all_extended_patterns:
        filtered_cases = data.loc[data[activity] == core_activity, case_id]
        new_patterns_for_core = []
        for case in filtered_cases.unique():
            # Original: Trace_graph_generator builds the graph from the timestamps, where events with
            # the same timestamp are concurrent. The BPIC11 timestamps are always at 23:00, so we use
            # the event order instead (see DataManager._load_df).
            if case not in event_log_graphs.keys():
                event_log_graphs[case] = trace_graph_generator(data_manager.traces[case])
            trace_graph = event_log_graphs[case]

            pattern_extension(case, trace_graph, core_activity, extended_patterns_at_stage,
                              max_gap_between_events, new_patterns_for_core)

        # Original: sets the new columns to 0 and fills in the count of every case in a loop.
        patient_data = add_pattern_counts(patient_data, extended_patterns_at_stage, new_patterns_for_core,
                                          data_manager.case_ids)
        all_extended_patterns_1_list.extend(new_patterns_for_core)
    if not all_extended_patterns_1_list:
        return pd.concat(selected).reset_index(drop=True)  # nothing could be extended

    new_train_data = patient_data[patient_data[case_id].isin(train_data[case_id])]

    pattern_attributes = create_pattern_attributes(new_train_data, outcome,
                                                   all_extended_patterns_1_list,
                                                   pairwise_distances_array)
    objectives_attributes = pattern_attributes[pareto_features]
    mask = paretoset(objectives_attributes, sense=pareto_sense)
    paretoset_patterns = pattern_attributes[mask]
    all_extended_patterns.extend(list(paretoset_patterns["patterns"]))

    # Original: saves all patterns on the front in JSON format and builds train_X and test_X.
    selected.append(pareto_rows(paretoset_patterns, 2, extended_patterns_at_stage))

    all_extended_patterns_dict = extended_patterns_at_stage.copy()
    for ext in range(1, max_extension_step):
        new_patterns_per_extension = []
        for chosen_pattern_id in paretoset_patterns["patterns"]:
            # Patterns with an eventual relation are not extended.
            if any(nx.get_edge_attributes(all_extended_patterns_dict[chosen_pattern_id]["pattern"],
                                          "eventually").values()):
                continue

            all_extended_patterns_dict, extended_patterns_at_stage, patient_data = single_pattern_extender(
                all_extended_patterns_dict,
                chosen_pattern_id,
                patient_data, event_log_graphs,
                max_gap_between_events, data_manager.case_ids)

            new_patterns_per_extension.extend(extended_patterns_at_stage.keys())
        if not new_patterns_per_extension:
            break  # nothing could be extended any more

        train_patient_data = patient_data[patient_data[case_id].isin(train_data[case_id])]

        pattern_attributes = create_pattern_attributes(train_patient_data, outcome,
                                                       new_patterns_per_extension,
                                                       pairwise_distances_array)

        objectives_attributes = pattern_attributes[pareto_features]
        mask = paretoset(objectives_attributes, sense=pareto_sense)
        paretoset_patterns = pattern_attributes[mask]
        all_extended_patterns.extend(list(paretoset_patterns["patterns"]))

        # Original: saves all patterns on the front in JSON format and builds train_X and test_X.
        selected.append(pareto_rows(paretoset_patterns, ext + 2, all_extended_patterns_dict))

    # Original: returns train_X, test_X. We return the union of the fronts of all steps.
    return pd.concat(selected).reset_index(drop=True)
