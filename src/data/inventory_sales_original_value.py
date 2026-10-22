"""Original-filing inventory/revenue source audit for four SOXX basket names."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import xml.etree.ElementTree as ET
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from src.core.recorder_sources import load_verified_recorder_objects
from src.workflow.inventory_sales_source_qualification import normalized_notebook


NAME = "inventory_sales_original_value_source_20261022"
NVDA_REVENUE_TAGS = (
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    "us-gaap:Revenues",
    "us-gaap:SalesRevenueNet",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), f"invalid JSON object: {path}")
    return value


def _original_fact(data: bytes, spec: dict, tag_name: str, kind: str) -> tuple[set[int], set[str]]:
    """Return distinct own-period, undimensioned original values and contexts."""
    issuer = f"{spec['cik']:010d}"
    unit = "iso4217:" + spec["unit"]
    expected_end = spec["period_end"]
    expected_start = spec["period_start"]
    values: set[int] = set()
    contexts: set[str] = set()
    if spec["fact_format"] == "original_XML":
        namespaces = {prefix: uri for _, (prefix, uri) in ET.iterparse(io.BytesIO(data), events=["start-ns"])}
        prefix, local = tag_name.split(":", 1)
        _require(prefix in namespaces, f"original XML taxonomy namespace missing: {tag_name}")
        root = ET.fromstring(data)
        base = "{" + root.tag.split("}")[0][1:] + "}"
        context_by_id = {node.get("id"): node for node in root.findall(base + "context")}
        unit_by_id = {node.get("id"): node for node in root.findall(base + "unit")}
        facts = root.findall("{" + namespaces[prefix] + "}" + local)
        for fact in facts:
            context = context_by_id.get(fact.get("contextRef"))
            measure = unit_by_id.get(fact.get("unitRef"))
            if context is None or measure is None:
                continue
            entity = context.find(base + "entity")
            period = context.find(base + "period")
            if entity is None or period is None or entity.find(base + "identifier") is None:
                continue
            if entity.find(base + "identifier").text != issuer or entity.find(base + "segment") is not None:
                continue
            if context.find(base + "scenario") is not None:
                continue
            currency = measure.find(base + "measure")
            if currency is None or currency.text != unit:
                continue
            if kind == "inventory":
                instant = period.find(base + "instant")
                if instant is None or instant.text != expected_end:
                    continue
            else:
                start, end = period.find(base + "startDate"), period.find(base + "endDate")
                if start is None or end is None or start.text != expected_start or end.text != expected_end:
                    continue
            try:
                numeric = Decimal((fact.text or "").strip())
                _require(numeric == numeric.to_integral_value(), "non-integral original XML financial fact")
                values.add(int(numeric))
                contexts.add(fact.get("contextRef"))
            except InvalidOperation:
                continue
    else:
        soup = BeautifulSoup(data, "html.parser")
        context_by_id = {node.get("id"): node for node in soup.find_all(
            lambda node: node.name and node.name.lower().endswith("context") and node.get("id"))}
        unit_by_id = {node.get("id"): node for node in soup.find_all(
            lambda node: node.name and node.name.lower().endswith("unit") and node.get("id"))}
        facts = soup.find_all(lambda node: node.name and node.name.lower().endswith("nonfraction")
                              and node.get("name") == tag_name)
        for fact in facts:
            context = context_by_id.get(fact.get("contextref"))
            measure = unit_by_id.get(fact.get("unitref"))
            if context is None or measure is None:
                continue
            identifier = context.find(lambda node: node.name and node.name.lower().endswith("identifier"))
            if identifier is None or identifier.get_text(" ", strip=True) != issuer:
                continue
            if context.find(lambda node: node.name and (node.name.lower().endswith("segment")
                                                       or node.name.lower().endswith("scenario"))):
                continue
            currency = measure.find(lambda node: node.name and node.name.lower().endswith("measure"))
            if currency is None or currency.get_text(" ", strip=True) != unit:
                continue
            if kind == "inventory":
                instant = context.find(lambda node: node.name and node.name.lower().endswith("instant"))
                if instant is None or instant.get_text(" ", strip=True) != expected_end:
                    continue
            else:
                start = context.find(lambda node: node.name and node.name.lower().endswith("startdate"))
                end = context.find(lambda node: node.name and node.name.lower().endswith("enddate"))
                if start is None or end is None or start.get_text(" ", strip=True) != expected_start or end.get_text(" ", strip=True) != expected_end:
                    continue
            display = fact.get_text(" ", strip=True).replace(",", "").replace(" ", "").replace("−", "-")
            try:
                numeric = Decimal(display) * (Decimal(10) ** int(fact.get("scale") or 0))
                if fact.get("sign") == "-":
                    numeric = -numeric
                _require(numeric == numeric.to_integral_value(), "non-integral original inline financial fact")
                values.add(int(numeric))
                contexts.add(fact.get("contextref"))
            except (InvalidOperation, ValueError):
                continue
    return values, contexts


def _load_archive_bytes(root: Path, plan: dict) -> tuple[dict[str, bytes], dict[str, dict]]:
    spec = plan["source_archives"]
    main_manifest_path = root / spec["main_manifest"]
    main_plan_path = root / spec["main_plan"]
    prior_manifest_path = root / spec["nvda_prior_manifest"]
    _require(_sha(main_manifest_path) == spec["main_manifest_sha256"]
             and _sha(main_plan_path) == spec["main_plan_sha256"]
             and _sha(prior_manifest_path) == spec["nvda_prior_manifest_sha256"],
             "original source archive metadata changed")
    main_archive = Path(spec["main_local_archive"])
    prior_archive = root / spec["nvda_prior_archive"]
    _require(_sha(main_archive) == spec["main_archive_sha256"]
             and _sha(prior_archive) == spec["nvda_prior_archive_sha256"],
             "exact original source archive bytes differ")
    main_entries = {x["member"]: x for x in _json(main_plan_path)["sources"]}
    prior_entries = {x["member"]: x for x in _json(prior_manifest_path)["members"]}
    required = {"main": set(), "nvda_prior": set()}
    for spec_row in plan["issuer_years"]:
        primary_archive = spec_row.get("primary_archive", "main")
        fact_archive = spec_row.get("fact_archive", "main")
        required[primary_archive].add(spec_row["primary_member"])
        required[fact_archive].add(spec_row["fact_member"])
    _require(len(required["main"]) == plan["budget"]["main_archive_fact_or_primary_members"]
             and len(required["nvda_prior"]) == plan["budget"]["nvda_prior_archive_members"],
             "original source member budget differs")
    data: dict[str, bytes] = {}
    metadata: dict[str, dict] = {}
    for archive_name, archive_path, entries in (("main", main_archive, main_entries),
                                                 ("nvda_prior", prior_archive, prior_entries)):
        with tarfile.open(archive_path, "r:gz") as archive:
            for name in sorted(required[archive_name]):
                item = entries.get(name)
                _require(item is not None, f"original source member unregistered: {name}")
                member = archive.getmember(name)
                _require(member.isfile() and member.size == item["bytes"], "original member size differs")
                source = archive.extractfile(member)
                _require(source is not None, "original member unreadable")
                raw = source.read()
                _require(hashlib.sha256(raw).hexdigest() == item["sha256"], "original member hash differs")
                data[archive_name + ":" + name] = raw
                metadata[archive_name + ":" + name] = item
    return data, metadata


def _annual_source_rows(plan: dict, archive_bytes: dict[str, bytes], member_meta: dict[str, dict],
                        calendar: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for spec in plan["issuer_years"]:
        primary = spec.get("primary_archive", "main") + ":" + spec["primary_member"]
        fact = spec.get("fact_archive", "main") + ":" + spec["fact_member"]
        _require(primary in archive_bytes and fact in archive_bytes, "original archive member absent")
        inventory_values, inventory_contexts = _original_fact(archive_bytes[fact], spec,
                                                               spec["inventory_tag"], "inventory")
        revenue_by_tag = {tag: _original_fact(archive_bytes[fact], spec, tag, "revenue")
                          for tag in spec["revenue_tags"]}
        values = [next(iter(value)) for value, _ in revenue_by_tag.values() if len(value) == 1]
        status = "original_own_period_qualified"
        if len(inventory_values) != 1 or len(values) != len(spec["revenue_tags"]):
            status = "missing_or_conflicting_original_fact"
        elif len(set(values)) != 1:
            status = "original_revenue_alias_value_conflict"
        elif len(spec["revenue_tags"]) > 1 and len({frozenset(contexts)
                                                     for _, contexts in revenue_by_tag.values()}) != 1:
            status = "original_revenue_alias_context_conflict"
        elif next(iter(inventory_values)) < 0 or values[0] <= 0:
            status = "invalid_original_financial_sign"
        filed = pd.Timestamp(spec["filed"])
        end = pd.Timestamp(spec["period_end"])
        if filed < end:
            status = "filing_precedes_fiscal_end"
        position = calendar.searchsorted(filed, side="right")
        available = calendar[position] if position < len(calendar) else pd.NaT
        if pd.isna(available):
            status = "filing_after_available_calendar"
        rows.append({"ticker": spec["ticker"], "cik": spec["cik"], "fiscal_year": spec["fiscal_year"],
                     "period_start": pd.Timestamp(spec["period_start"]), "period_end": end,
                     "filed": filed, "available_session": available,
                     "calendar_lower_bound_clipped": filed < calendar.min(),
                     "accession": spec["accession"], "unit": spec["unit"],
                     "inventory": next(iter(inventory_values)) if status == "original_own_period_qualified" else None,
                     "revenue": values[0] if status == "original_own_period_qualified" else None,
                     "inventory_contexts": sorted(inventory_contexts),
                     "revenue_contexts": {tag: sorted(contexts) for tag, (_, contexts) in revenue_by_tag.items()},
                     "primary_sha256": member_meta[primary]["sha256"],
                     "fact_sha256": member_meta[fact]["sha256"],
                     "status": status, "historical_PIT_qualified": False})
    return pd.DataFrame(rows)


def _comparable_pairs(annual: pd.DataFrame, plan: dict) -> pd.DataFrame:
    excluded = {(ticker, int(year)) for ticker, year in plan["source_rules"]["excluded_growth_pairs"]}
    pairs = []
    for ticker, group in annual.groupby("ticker"):
        sequence = group.sort_values("period_end").to_dict("records")
        for prior, current in zip(sequence, sequence[1:]):
            if (ticker, int(current["fiscal_year"])) in excluded:
                continue
            if prior["status"] != "original_own_period_qualified" or current["status"] != "original_own_period_qualified":
                continue
            if ((current["period_start"] - prior["period_end"]).days != 1
                    or not 330 <= (current["period_end"] - prior["period_end"]).days <= 380
                    or prior["unit"] != current["unit"] or prior["inventory"] <= 0
                    or prior["revenue"] <= 0):
                continue
            pairs.append({"ticker": ticker, "cik": current["cik"], "unit": current["unit"],
                          "prior_fiscal_end": prior["period_end"], "fiscal_end": current["period_end"],
                          "prior_accession": prior["accession"], "accession": current["accession"],
                          "prior_inventory": prior["inventory"], "inventory": current["inventory"],
                          "prior_revenue": prior["revenue"], "revenue": current["revenue"],
                          "conditional_available_session": max(prior["available_session"], current["available_session"]),
                          "source_identity_supported": True, "historical_PIT_qualified": False})
    return pd.DataFrame(pairs)


def _monthly_source_support(support: pd.DataFrame, pairs: pd.DataFrame, max_age: int) -> pd.DataFrame:
    names = {"NVDA", "TSM", "UMC", "ASX"}
    selected = support.loc[support.ticker.isin(names)].copy()
    output = []
    for row in selected.itertuples(index=False):
        date = pd.Timestamp(row.datetime)
        eligible = pairs.loc[pairs.ticker.eq(row.ticker) & pairs.conditional_available_session.le(date)]
        latest = eligible.loc[eligible.fiscal_end.eq(eligible.fiscal_end.max())] if len(eligible) else eligible
        reason = ""
        if latest.empty:
            reason = "no_original_comparable_pair_available"
        elif len(latest) != 1:
            reason = "ambiguous_original_latest_pair"
        elif not 0 <= (date - latest.fiscal_end.iloc[0]).days <= max_age:
            reason = "original_pair_stale_or_future_period"
        chosen = latest.iloc[0] if not reason else None
        output.append({"datetime": date, "instrument": row.ticker, "ticker": row.ticker,
                       "basket_accession": row.basket_accession, "weight_pct": float(row.weight_pct),
                       "source_supported_weight_pct": float(row.weight_pct) if chosen is not None else 0.0,
                       "uncovered_reason": reason,
                       "original_pair_accession": None if chosen is None else chosen.accession,
                       "original_pair_fiscal_end": pd.NaT if chosen is None else chosen.fiscal_end,
                       "original_pair_available_session": pd.NaT if chosen is None else chosen.conditional_available_session,
                       "historical_PIT_qualified": False})
    return pd.DataFrame(output)


def run(root: Path) -> dict:
    import qlib
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.workflow import R

    root = Path(root).resolve()
    card = root / "research" / NAME
    plan_path = card / "plan.json"
    plan = _json(plan_path)
    _require(plan["frozen"] and plan["execution_authorized"], "frozen original source plan absent")
    _require(not (card / "run_state.json").exists(), "original source attempt already exists")
    _require(_sha(Path(__file__)) == plan["implementation"]["module_sha256"], "module bytes differ")
    notebook_path = card / f"{NAME}.ipynb"
    _require(hashlib.sha256(normalized_notebook(notebook_path)).hexdigest()
             == plan["implementation"]["notebook_sha256"], "notebook code differs")
    for key in ("source_identity_evidence", "ASX_scope_break", "NVDA_alias_bridge"):
        item = plan[key]
        _require(_sha(root / item["path"]) == item["sha256"], f"bound source evidence differs: {key}")
    join = plan["source_identity_evidence"]
    _require(_sha(root / join["join_path"]) == join["join_sha256"], "dated security join differs")
    _require(_json(root / join["join_path"])["status"] == "dated_security_join_passed_source_only",
             "four security identity source chain not established")
    _require(_sha(root / plan["calendar"]["path"]) == plan["calendar"]["sha256"], "Qlib calendar changed")
    archive_bytes, member_meta = _load_archive_bytes(root, plan)
    source = plan["source_monthly_support"]
    old = root / source["card"]
    _require(_sha(old / "summary.json") == source["summary_sha256"]
             and _sha(old / "verification_manifest.json") == source["verification_manifest_sha256"],
             "monthly N-PORT source metadata changed")
    (card / "run_state.json").write_text(json.dumps({"status": "started", "plan_sha256": _sha(plan_path)}) + "\n")
    qlib.init(provider_uri=str(root / plan["calendar"]["provider_uri"]), region="us", kernels=1,
              expression_cache=None, dataset_cache=None)
    objects, audit = load_verified_recorder_objects(
        old, expected_recorder_id=source["recorder_id"], artifacts=[source["artifact"]],
        expected_artifact_sha256={source["artifact"]: source["artifact_sha256"]},
        expected_manifest_sha256=source["verification_manifest_sha256"])
    calendar = pd.DatetimeIndex(D.calendar(start_time=plan["calendar"]["bounds"][0],
                                           end_time=plan["calendar"]["bounds"][1], freq="day"))
    _require(len(calendar) > 0 and _sha(root / plan["calendar"]["path"]) == plan["calendar"]["sha256"],
             "native Qlib calendar identity differs")
    annual = _annual_source_rows(plan, archive_bytes, member_meta, calendar)
    pairs = _comparable_pairs(annual, plan)
    excluded = {(ticker, year) for ticker, year in plan["source_rules"]["excluded_growth_pairs"]}
    _require(not any((r.ticker, int(r.fiscal_end.year)) in excluded for r in pairs.itertuples()),
             "excluded accounting or merger pair entered")
    support = _monthly_source_support(objects[source["artifact"]], pairs,
                                      plan["source_rules"]["max_pair_age_days"])
    _require(len(support) == 278, "four-security disclosed issuer-month count differs")
    panel = support[["datetime", "instrument", "weight_pct", "source_supported_weight_pct"]].copy()
    panel = panel.set_index(["datetime", "instrument"]).sort_index()
    dataset = DatasetH(handler=DataHandlerLP.from_df(pd.concat({"meta": panel}, axis=1)),
                       segments={"all": (None, None)})
    dataset.config(dump_all=True, recursive=True)
    pd.testing.assert_frame_equal(dataset.prepare("all", col_set="meta"), panel)
    good = annual.status.eq("original_own_period_qualified")
    coverage = support.groupby("datetime").agg(
        four_name_weight_pct=("weight_pct", "sum"),
        original_pair_supported_weight_pct=("source_supported_weight_pct", "sum"))
    tracking = card / "qlib_recorder"
    with R.start(experiment_name=NAME, uri=str(tracking)):
        rec = R.get_recorder()
        rec.save_objects(artifact_path="inputs", plan=plan, source_audit=audit,
                         archive_member_inventory={key: member_meta[key] for key in sorted(member_meta)})
        rec.save_objects(artifact_path="analysis", original_annual_values=annual,
                         original_comparable_pairs=pairs, four_name_monthly_support=support,
                         monthly_coverage=coverage)
        rec.save_objects(artifact_path="metadata", source_dataset=dataset)
        rec.save_objects(artifact_path="sources", module=Path(__file__).read_bytes(),
                         notebook=(card / f"{NAME}.ipynb").read_bytes())
        pd.testing.assert_frame_equal(rec.load_object("metadata/source_dataset").prepare("all", col_set="meta"),
                                      panel)
        rec.log_metrics(original_issuer_years=len(annual), original_qualified_years=int(good.sum()),
                        original_comparable_pairs=len(pairs), historical_PIT_rows=0,
                        model_fits=0, native_accounts=0)
        summary = {"experiment": NAME, "status": "completed_source_only_not_promoted",
                   "decision": "original_four_issuer_annual_source_audited",
                   "original_issuer_years": len(annual), "own_original_value_years": int(good.sum()),
                   "missing_or_conflicting_years": annual.loc[~good, ["ticker", "fiscal_year", "status"]].to_dict("records"),
                   "comparable_annual_pairs": len(pairs), "excluded_growth_pairs": plan["source_rules"]["excluded_growth_pairs"],
                   "four_name_issuer_months": len(support),
                   "four_name_months_with_original_pair": int(support.source_supported_weight_pct.gt(0).sum()),
                   "median_raw_four_name_weight_pct": float(coverage.four_name_weight_pct.median()),
                   "median_raw_original_pair_supported_weight_pct": float(coverage.original_pair_supported_weight_pct.median()),
                   "strict_PIT_qualified_rows": 0, "model_input_qualified": False,
                   "factor_effectiveness_tested": False, "SOXX_excess_established": False,
                   "first_public_value_clock_verified": False,
                   "conservative_original_filing_next_session_available": True,
                   "D_calendar_calls": 1, "D_features_calls": 0, "market_price_or_label_reads": 0,
                   "model_fits": 0, "native_accounts": 0,
                   "source_recorder_object_loads": 1,
                   "original_source_members_read": len(member_meta),
                   "source_recorder_id": source["recorder_id"], "plan_sha256": _sha(plan_path),
                   "module_sha256": _sha(Path(__file__)), "recorder_id": rec.id,
                   "experiment_id": str(rec.experiment_id), "recorder_uri": str(tracking),
                   "recorder_local_dir": str(tracking / str(rec.experiment_id) / rec.id),
                   "artifact_policy": plan["artifact_policy"]}
        rec.save_objects(artifact_path="analysis", summary=summary)
    artifact_root = Path(summary["recorder_local_dir"]) / "artifacts"
    manifest = {"recorder_id": summary["recorder_id"], "entries": [
        {"path": str(path.relative_to(card)), "sha256": _sha(path), "bytes": path.stat().st_size}
        for path in sorted(artifact_root.rglob("*")) if path.is_file()]}
    (card / "verification_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (card / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (card / "run_state.json").write_text(json.dumps({"status": "completed", "recorder_id": rec.id,
                                                     "plan_sha256": summary["plan_sha256"]}) + "\n")
    (card / "conclusion.md").write_text(
        "# 四家原始年度库存/收入：来源限定结果\n\n"
        f"Qlib Recorder：`{rec.id}`。从自身原始申报提取 {int(good.sum())}/{len(annual)} 个发行人年度，"
        f"形成 {len(pairs)} 个限定可比的前后年度配对；四家披露证券的 {len(support)} 个发行人月份中，"
        f"有 {int(support.source_supported_weight_pct.gt(0).sum())} 个在决策前具备未超龄的原始配对。"
        "原始篮子权重没有重新归一化。\n\n"
        "ASX FY2018、FY2019 和 NVDA FY2019 的跨口径增长已排除。"
        "只使用原始申报并保守推迟到 filed 日期后的第一个 Qlib 交易日；"
        "这不声称数值首次公开时间已知。历史全篮子 PIT/模型输入尚未获独立确认，"
        "本卡没有计算收益、因子有效性、费用账户或 SOXX 超额，不能晋升。\n")
    return summary
