# Failure modes

- The paper's PIT T-bill and Treasury sleeves are unavailable in the current qlib store; zero-return cash is an explicitly disclosed diagnostic approximation.
- The controller is intentionally asymmetric and cannot increase risk when behind; this differs from the paper's symmetric target-tracking intuition.
- The opportunity target comes from a prior nested workflow whose constituent breadth inputs remain survivorship-biased until PIT holdings are available.
- Outer 2023-2026 is evaluation-only, but the architecture was conceived after prior results were observed, so promotion requires a newly sealed forward period.
