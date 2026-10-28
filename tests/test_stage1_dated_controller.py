import pandas as pd
import pytest

from src.workflow.stage1_dated_controller import DatedModel, DatedStage1Controllers


def _calendar():
    return pd.DataFrame({"signal_date": pd.to_datetime(["2018-12-31", "2019-01-02"]),
                         "trade_date": pd.to_datetime(["2019-01-02", "2019-01-03"])})


def test_dated_controller_uses_signal_year_at_year_boundary():
    open_models = {2018: DatedModel(object(), "2017-12-31", "open_2018"),
                   2019: DatedModel(object(), "2018-12-31", "open_2019")}
    recovery = {2018: DatedModel(None, "2017-12-31", "no_recovery_2018", explicit_no_model=True),
                2019: DatedModel(None, "2018-12-31", "no_recovery_2019", explicit_no_model=True)}
    schedule = DatedStage1Controllers(calendar=_calendar(), open_models=open_models, recovery_models=recovery)
    assert schedule.audit().open_artifact_id.tolist() == ["open_2018", "open_2019"]
    assert schedule(pd.Timestamp("2019-01-02"))["frozen_open_action_value_model"] is open_models[2018].model


def test_final_b472_model_cannot_be_used_on_early_development_dates():
    frozen = DatedModel(object(), "2022-12-31", "frozen_b472")
    recovery = DatedModel(None, "2017-12-31", "no_recovery", explicit_no_model=True)
    with pytest.raises(ValueError, match="opening model trained on or after"):
        DatedStage1Controllers(calendar=_calendar().iloc[:1], open_models={2018: frozen},
                               recovery_models={2018: recovery})


def test_missing_recovery_vintage_cannot_silently_fallback():
    model = DatedModel(object(), "2017-12-31", "open_2018")
    with pytest.raises(ValueError, match="missing annual opening or recovery"):
        DatedStage1Controllers(calendar=_calendar().iloc[:1], open_models={2018: model},
                               recovery_models={})
