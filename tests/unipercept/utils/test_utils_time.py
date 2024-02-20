from __future__ import annotations

import time

from unipercept.utils.time import ProfileAccumulator, profile


def test_time_profiler():
    acc = ProfileAccumulator()

    n_test_1 = 10
    t_sleep_1 = 1e-4
    for i in range(n_test_1):
        with profile(acc, "test"):
            time.sleep(i * t_sleep_1)

    n_test_2 = 5
    t_sleep_2 = 1e-3
    for i in range(n_test_2):
        with profile(acc, "test2"):
            time.sleep(i * t_sleep_2)

    n_total = n_test_1 + n_test_2

    assert len(acc.records) == n_total
    assert len(acc.keys()) == len({n_test_1, n_test_2})

    df_profile = acc.to_dataframe()
    assert len(df_profile) == n_total
    print(df_profile.to_markdown(index=False, floatfmt=".3f"))

    df_summary = acc.to_summary()
    assert len(df_summary) == 2

    print(df_summary.to_markdown(index=True, floatfmt=".3f"))
