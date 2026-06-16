# -*- coding: utf-8 -*-

from __future__ import annotations


class KnowledgeBase:
    def __init__(self):
        self.prior_knowledge: dict[str, str] = {}
        self.locator_visual_knowledge: dict[str, str] = {}
        self._initialize_dataset_knowledge()

    def _initialize_dataset_knowledge(self) -> None:
        def add(agent_key: str, dataset_key: str, text: str) -> None:
            self.prior_knowledge[f"{agent_key}_{dataset_key.upper()}"] = text

        def add_locator_visual(dataset_key: str, text: str) -> None:
            self.locator_visual_knowledge[dataset_key.upper()] = text

        kpi_long_knowledge = """
## Domain Knowledge:
- The time series you detect is KPI dataset
### Anomaly type
- IOPS dataset contains anomaly like global-level isolated high few-point spike/ low valle or cluster.
- For one-point spike t, catch the t-1 and t+1 as anomaly. For cluster, catch from the begin from baseline to recovery to baseline.
- Try to use diff_zscore to catch the **edge** of those cluster. Use Value_zscore to detector whether the cluster is extremely higher/lower.
- For gradually rising off-baseline clusters, value_zscore may only flag the top; search around flagged points to find where the rise begins and where recovery ends.
- isolated peaks and dips(which means one point) that never show up in the value-z-score>3 are probably **noise** rather than anomalies; So minimize the false positive!
- Those with sharp deviation to a new pleteau (few point cluster, instead of one point), then recovery might be anomaly. Though they might not be higher/lower than +/- 1, they have huge deviation.
- A mean-level regime shift without recovery is not an anomaly; if it later returns abruptly or clearly recovers, treat the whole shift-then-recovery span as anomalous.
### ToolCall
For KPI dataset, call `value_zscore` (threshold 3~3.5) and `diff_zscore` (threshold ~4);
After gathering evidence, do not just use return diff_zscore/value_zscore index as anomaly, try to catch the **whole cluster** (from start of abrupt to the recovery point)
Do not Use grep for longer than 200 point.
""".strip()

        iops_long_knowledge = """
## Domain Knowledge:
- The time series you detect is IOPS dataset
### Anomaly type
- IOPS dataset contains anomaly like global-level isolated high few-point spike/ low valle or cluster.
- For one-point spike t, catch the t-1 and t+1 as anomaly. For cluster, catch from the begin from baseline to recovery to baseline.
- Try to use diff_zscore to catch the **edge** of those cluster. Use Value_zscore to detector whether the cluster is extremely higher/lower.
- For gradually rising off-baseline clusters, value_zscore may only flag the top; search around flagged points to find where the rise begins and where recovery ends.
- isolated peaks and shallow dips that never show up in the value-z-score>3 are probably **noise** rather than anomalies; So minimize the false positive!
- slow trend-like changes that never show up in the diff_zscore are normal, not anomalous.
- A mean-level regime shift without recovery is not an anomaly; if it later returns abruptly or clearly recovers, treat the whole shift-then-recovery span as anomalous.
### ToolCall
For IOPS dataset, call `value_zscore` (threshold ~3.5) and `diff_zscore` (threshold ~4);
After gathering evidence, do not just use return diff_zscore/value_zscore index as anomaly, try to catch the **whole anomaly cluster** (from start of abrupt to the recovery point).
Use localization results and tool results to grep specious intervals. The length of one grep may around 20-200. (If there are no visualization or zscore evidence, do not grep the whole 1000 point!)
""".strip()

        yahoo_long_knowledge = """
## Domain Knowledge:
- The time series below is from YAHOO.
    - one or few point anomaly is most common, there also rare long-period anomaly.
- Try to minilize to interval to those truely anomaly, instead of their neighbor. (But you should which which is anomaly, which is recovery)
- If two adjacent or nearby points have opposite and large diff Z-scores, check the points between them for significant deviation from surrounding values. If present, flag those intermediate points as anomalies, but exclude the last recovery point.
- In Yahoo ,there may exist global anomaly(detected both from diff-zscore and value-zscore) and context anomaly(dip/spike detected only from diff-z-score)
- Slow change(detected only by zscore instead of diff-zscore ) may not anomaly.
- You need figure out whether it is spike or dip (the other may be recovery point), you can refer to grep value and localization hint.
## Tool Advice
Use `diff_zscore` (e.g. threshold >=3) and `value_zscore` (threshold >=3 as needed). (Do not lower the threshold, which may be not anomaly)
YAHOO is short, use grep to scan specious intervals, if servals anomaly intervals are close, combine into one. Set grep length 20-100.
""".strip()

        wsd_long_knowledge = """
## Domain Knowledge:
There may rare but extremely abrupt anomaly few-point spike/dip.
### Anomaly type
- WSD dataset contains anomaly like global-level isolated high few-point spike/ low valle or cluster.
- For one-point spike t, catch the t-1 and t+1 as anomaly. For cluster, catch from the begin from baseline to recovery to baseline.
- Try to use diff_zscore to catch the **edge** of those cluster. Use Value_zscore to detector whether the cluster is extremely higher/lower.
- For gradually rising off-baseline clusters, value_zscore may only flag the top; search around flagged points to find where the rise begins and where recovery ends.
- isolated peaks and shallow dips that never show up in the value-z-score>3 are probably **noise** rather than anomalies; So minimize the false positive!
- slow trend-like changes that never show up in the diff_zscore are normal, not anomalous.
- A mean-level regime shift without recovery is not an anomaly; only “shift-then-recovery” segments are.
After gathering evidence, do not just use return diff_zscore/value_zscore index as anomaly, try to catch the **whole cluster** (from start of abrupt to the recovery point)

## Tool Advice
Try to use `diff_zscore` (threshold about 5) for abrupt rise/fall and `value_zscore` (threshold about 3.5).
Use localization results and tool results to grep specious intervals. The length of one grep may around 20-100
""".strip()

        add_locator_visual(
            "KPI",
            """
- **Only abrupt / sudden changes**: sharp spikes, dips, or high/low **plateaus** that **break** the local level quickly.
- **Do NOT** flag slow monotonic **ramps** (gentle rise or fall over many points), smooth waves, or gradual regime tilts.
- A mean-level regime shift **without recovery** is not an anomaly; a shift/plateau with a clear sudden later recovery may be anomalous.
- KPI often contains global-level isolated high spikes / low valleys or short high/low clusters.
- The series is minimax by 95% and 5%. Abrupt spikes higher than 1 and valleys lower than -1 are anomalies. Those isolated point with smaller value might be noise.
- Those with sharp deviation to a new pleteau (few point cluster), then recovery might be anomaly. Though they might not be higher/lower than +/- 1, they have huge deviation.
""".strip(),
        )
        add_locator_visual(
            "IOPS",
            """
- **Only abrupt / sudden changes**: sharp spikes, dips, or high/low **plateaus** that **break** the local level quickly.
- **Do NOT** flag slow monotonic **ramps** (gentle rise or fall over many points), smooth waves, or gradual regime tilts.
- A mean-level regime shift **without recovery** is not an anomaly; a shift/plateau with a clear later recovery can be anomalous.
- IOPS often contains global-level isolated high spikes / low valleys or short high/low clusters.
- The series is minimax by 95% and 5%. Abrupt spikes higher than 1 and valleys lower than -1 might be anomalies.
""".strip(),
        )
        add_locator_visual(
            "WSD",
            """
- **Only abrupt / sudden changes**: sharp spikes, dips, or high/low **plateaus** that **break** the local level quickly.
- **Do NOT** flag slow monotonic **ramps** (gentle rise or fall over many points), smooth waves, or gradual regime tilts.
- A mean-level regime shift without recovery is not an anomaly; only “shift-then-recovery” segments are.
- WSD has rare but extremely abrupt few-point spike/dip/cluster anomalies.
- The series is minimax by 95% and 5%. Abrupt spikes higher than 1 and valleys lower than -1 might be anomalies.
""".strip(),
        )
        add_locator_visual(
            "TODS",
            """
- In TODS, there might be local-level spike/dip anomalies that do not necessarily exceed extreme absolute values.
- There may also be 2-5 point plateaus and ~10 point period anomalies.
- Detect point or period-level anomalies that break the normal **sin** curve (cycle length ~100).
- If there are sub-series that don't have a cycle length of 100, catch then as anomaly. If there are trend anomaly, catch them.
- For consecutive sharp spikes instead of plateau/trend changes, label them **separately** instead of as a single whole event.
""".strip(),
        )
        add_locator_visual(
            "YAHOO",
            """
- In Yahoo, there are two common series patterns.
- One is a noisy curve with cycle length ~25; anomalies are local spikes/dips whose change magnitude is larger than the noise.
- The other is a smoother noisy curve with rare global-level one-point or few-point anomalies that clearly break the normal value pattern.
- For single point anomaly, report whether it is spike or valley.
""".strip(),
        )
        add_locator_visual(
            "APP",
            """
- Focus on visually obvious elevated-crash episodes: abrupt rise into a high plateau/peak followed by recovery.
- Do not flag gentle baseline drift or long smooth ramps without a clear anomalous episode.
- Prefer coarse episode localization over isolated noisy points.
""".strip(),
        )
        add_locator_visual(
            "SMD",
            """
- SMD is dominated by a **periodic / repeating smooth waveform**. The main visual question is: does a segment **break that periodic rhythm for a sustained span**?
- The anomalies to look for are simple and high-contrast: a **very strong oscillatory burst** appearing inside an otherwise smooth/flat interval, a **spike cluster**, a very obvious **high plateau**, a **deep valley**, or a **global-level spike** that is much more extreme than normal cycles.
- A **local huge noisy spike**, a periodic **slow rounded peak**, a local **small peak**, or a **small step/jump** inside the normal cycle are usually **not** anomalies by themselves.
- Focus on **bounded abnormal episodes** where the periodic pattern is clearly disrupted for many consecutive points and later recovers: e.g. a flattened valley, clipped plateau, missing oscillation, a strong oscillatory burst, or a globally extreme spike cluster.
- Be conservative with apparent baseline shifts or structural changes: if the plot only shows a mild mean change, or a **slow trend**, or a smooth drift without a sharp return, prefer **no localization**.
""".strip(),
        )

        add(
            "planning_agent",
            "KPI",
            kpi_long_knowledge,
        )
        add(
            "planning_agent",
            "YAHOO",
            yahoo_long_knowledge,
        )
        add(
            "planning_agent",
            "IOPS",
            iops_long_knowledge,
        )
        add(
            "planning_agent",
            "WSD",
            wsd_long_knowledge,
        )
        add(
            "planning_agent",
            "TODS",
            """
## Domain Knowledge:
- TODS is strongly periodic and may contain point anomalies, short plateaus, and short period anomalies.
- Consecutive sharp spikes should often remain separate rather than merged.
## Tool Advice
- Use diff-zscore for point-level events and combine with localization for period-level anomalies.
""".strip(),
        )
        add(
            "planning_agent",
            "APP",
            """
## Domain Knowledge:
- APP focuses on elevated crash-rate episodes versus local baseline.
## Tool Advice
- Use `stats`, `value_zscore`, and `diff_zscore` with conservative, low floors when needed.
""".strip(),
        )
        add(
            "planning_agent",
            "SMD",
            """
## Domain Knowledge:
- SMD is mostly periodic; real anomalies usually break the repeating waveform for a sustained span.
## Tool Advice
- Use localization as the main candidate interval, then use `value_zscore` and `diff_zscore` to confirm or shrink it.
""".strip(),
        )
        add(
            "detector",
            "KPI",
            kpi_long_knowledge,
        )
        add(
            "detector",
            "IOPS",
            iops_long_knowledge,
        )
        add(
            "detector",
            "YAHOO",
            yahoo_long_knowledge,
        )
        add(
            "detector",
            "WSD",
            wsd_long_knowledge,
        )
        add(
            "detector",
            "SMD",
            """
## SMD detector guidance
- Visual localization is the primary evidence candidate, but not every visually unusual shape is a real anomaly.
- SMD is mostly periodic; real anomalies usually break the repeating waveform for a sustained span and then recover.
- Typical valid anomaly shapes include a spike cluster, a high plateau, a deep valley, a missing oscillation window, or another bounded abnormal episode.
- If tool evidence supports only a short local event, do not widen it into a long interval.
- Be careful with visual-only impressions: local huge noise, a small peak, a small step, or periodic variation by itself is often normal.
- Slow trend or mild baseline drift is usually not enough; prefer no anomaly unless the periodic structure is clearly broken.
""".strip(),
        )
        add(
            "detector_final",
            "KPI",
            """
## KPI second-stage evidence priority
- Visual localization is only a coarse hint; tool evidence can override it.
- Dense sustained value-zscore outliers can justify a KPI cluster even when diff-zscore edges are weak.
- Prefer full shift-then-recovery spans over only the highest flagged point when evidence supports the cluster.
""".strip(),
        )

    def get_agent_knowledge(self, agent_key: str, dataset_key: str) -> str:
        key = f"{agent_key}_{str(dataset_key).upper()}"
        return self.prior_knowledge.get(key, "")

    def get_locator_visual_knowledge(self, dataset_key: str) -> str:
        return self.locator_visual_knowledge.get(str(dataset_key).upper(), "")
