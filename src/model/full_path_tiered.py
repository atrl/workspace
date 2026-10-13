"""Qlib allocation Model trained on supervised chronological wealth paths.

Exact raw whole-share forward settlement, own holdings/peak feedback and a
fixed final checkpoint. Integer trade gradients are straight-through; this is
not an RL environment or a claim that integer optimization is convex.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib

import numpy as np
import pandas as pd
import torch
from qlib.model.base import Model
from torch import nn

from src.backtest.ibkr_tiered import TieredFees, settle_tiered_event


PRICE_FIELDS = ("known_soxl", "known_soxx", "soxl_open", "soxx_open", "raw_soxl_open", "raw_soxx_open",
                "close_soxl", "close_soxx")


def tiered_tensor_quote(shares, raw_price, sell, monthly_shares, *, tariff, slippage_bps):
    """Same first-fill assumptions, marginal tiers/minimum/cap as TieredFees."""
    active = shares > 0
    value = shares * raw_price
    remaining, volume, broker, lower = shares, monthly_shares, torch.zeros_like(shares), 0.
    for tier in tariff["tiers"]:
        upper = float(tier["through_shares"]) if tier["through_shares"] is not None else float("inf")
        room = (upper - torch.maximum(volume, volume.new_tensor(lower))).clamp_min(0)
        count = torch.minimum(remaining, room)
        broker = broker + count * tier["per_share"]
        remaining, volume = remaining - count, volume + count
        lower = upper
    before = torch.maximum(broker, broker.new_tensor(tariff["minimum_per_order"]))
    cap = value * tariff["maximum_order_fraction"]
    capped = cap < before
    broker = torch.where(active, torch.minimum(before, cap), torch.zeros_like(broker))
    venue = torch.where(raw_price >= 1, shares * tariff["venue_per_share"], value * tariff["venue_subdollar_fraction"])
    clearing = torch.minimum(shares * tariff["clearing_per_share"], value * tariff["clearing_maximum_fraction"])
    regulatory = (value * tariff["sec_sale_fraction"]
                  + torch.minimum(shares * tariff["taf_per_sold_share"], shares.new_tensor(tariff["taf_maximum_per_trade"]))) * sell
    fees = (broker * (1 + tariff["commission_pass_through_fraction"]) + venue + clearing + regulatory
            + shares * tariff["cat_per_share"] + value * slippage_bps / 10000.)
    counted = torch.where(active & ~capped, shares, torch.zeros_like(shares))
    return fees, monthly_shares + counted


def tiered_tensor_settle(q, cash, target, prices, raw, monthly_shares, *, tariff, slippage_bps):
    nav = (q * prices).sum(-1) + cash
    delta = target[:, :2] * nav[:, None] - q * prices
    continuous = delta.abs() / raw
    rounded = torch.floor(continuous + 1e-10)
    shares = continuous + (rounded - continuous).detach()
    direction = torch.sign(delta.detach())
    held, funded, total = q, cash, torch.zeros_like(cash)
    for selling in (True, False):
        for asset in range(2):
            selected = (direction[:, asset] < 0) if selling else (direction[:, asset] > 0)
            amount = torch.where(selected, shares[:, asset], torch.zeros_like(cash))
            quote, monthly_shares = tiered_tensor_quote(amount, raw[:, asset], selling, monthly_shares,
                tariff=tariff, slippage_bps=slippage_bps)
            sign = -1 if selling else 1
            adjustment = torch.zeros_like(held)
            adjustment[:, asset] = sign * amount * raw[:, asset] / prices[:, asset]
            held = held + adjustment
            funded = funded - sign * amount * raw[:, asset] - quote
            total = total + quote
    return held, funded, monthly_shares, total


class PathPolicy(nn.Module):
    def __init__(self, feature_count, base_count, hidden, *, family, minimum_cash, maximum_soxl):
        super().__init__()
        self.feature_count, self.base_count, self.family = feature_count, base_count, family
        self.minimum_cash, self.maximum_soxl = minimum_cash, maximum_soxl
        self.network = nn.Sequential(nn.Linear(feature_count + 4, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 3))
        with torch.no_grad():
            self.network[0].weight[:, base_count:feature_count].zero_()

    def forward(self, features, q, cash, known_prices, peak):
        nav = (q * known_prices).sum(-1) + cash
        held = torch.cat((q * known_prices, cash[:, None]), dim=-1) / nav[:, None]
        drawdown = (1 - nav / peak).clamp_min(0)
        if self.family == "no_market":
            features = torch.zeros_like(features)
        elif self.family == "daily":
            features = torch.cat((features[:, :self.base_count], torch.zeros_like(features[:, self.base_count:])), dim=-1)
        raw = torch.softmax(self.network(torch.cat((features, held, drawdown[:, None]), dim=-1)), dim=-1)
        risky = (1 - self.minimum_cash) * raw[:, :2].sum(-1)
        soxl = torch.minimum((1 - self.minimum_cash) * raw[:, 0], raw.new_tensor(self.maximum_soxl))
        return torch.stack((soxl, risky - soxl, 1 - risky), dim=-1)


def roll_full_path(network, features, prices, months, *, plan):
    batch, length, _ = features.shape
    q = features.new_zeros((batch, 2))
    cash = features.new_full((batch,), plan["initial_cash"])
    peak, monthly = cash.clone(), cash.new_zeros(batch)
    equity, drawdowns, charges = [], [], []
    for step in range(length):
        if step:
            monthly = torch.where(months[:, step] != months[:, step - 1], torch.zeros_like(monthly), monthly)
        known, executable, raw, closing = prices[:, step, :2], prices[:, step, 2:4], prices[:, step, 4:6], prices[:, step, 6:8]
        known_nav = (q * known).sum(-1) + cash
        peak = torch.maximum(peak, known_nav)
        target = network(features[:, step], q, cash, known, peak)
        drawdowns.append((1 - known_nav / peak).clamp_min(0))
        q, cash, monthly, cost = tiered_tensor_settle(q, cash, target, executable, raw, monthly,
            tariff=plan["tariff"], slippage_bps=plan["primary_slippage_bps"])
        close_nav = (q * closing).sum(-1) + cash
        peak = torch.maximum(peak, close_nav)
        drawdowns.append((1 - close_nav / peak).clamp_min(0))
        equity.append(close_nav)
        charges.append(cost)
    return {"equity": torch.stack(equity, dim=1), "drawdowns": torch.stack(drawdowns, dim=1),
            "charges": torch.stack(charges, dim=1), "ending_q": q, "ending_cash": cash}


class FullPathTieredModel(Model):
    def __init__(self, *, columns, base_count, family, seed, cutoff, plan):
        self.columns, self.base_count, self.family = list(columns), int(base_count), family
        self.seed, self.cutoff, self.plan = int(seed), pd.Timestamp(cutoff), deepcopy(plan)

    def _transform(self, frame):
        x = frame[self.columns].to_numpy(float)
        finite = np.isfinite(x)
        result = np.zeros_like(x)
        standardized = (x - self.mean) / self.scale
        result[finite] = standardized[finite].clip(-8, 8)
        if "minute_core_ready" in frame:
            result[frame.minute_core_ready.ne(1).to_numpy(), self.base_count:] = 0.
        # Canonical unknowns stay NaN; only latent inputs are neutral.
        return result

    def fit(self, dataset, **kwargs):
        data = dataset.prepare("train", col_set=["feature", "price", "meta"])
        maturity = pd.to_datetime(data[("meta", "label_available_at")])
        if not pd.to_datetime(data[("meta", "feature_available_at")]).lt(pd.to_datetime(data[("meta", "decision_at")])).all():
            raise ValueError("blocked: training information is not available at decision")
        valid = maturity.notna() & maturity.lt(self.cutoff)
        if (valid.iloc[1:].to_numpy() & ~valid.iloc[:-1].to_numpy()).any():
            raise ValueError("needs_data: unknown maturity breaks the chronological training prefix")
        if not np.isfinite(data.loc[valid, "price"].loc[:, list(PRICE_FIELDS)]).all().all():
            raise ValueError("needs_data: execution prices missing inside the mature training path")
        train = data.loc[valid].sort_index()
        length = self.plan["block_length"] + self.plan["warmup_sessions"]
        if len(train) < length:
            raise ValueError("needs_data: insufficient complete mature training path")
        numeric = train["feature"].loc[:, self.columns].to_numpy(float)
        self.mean, self.scale = np.nanmean(numeric, axis=0), np.nanstd(numeric, axis=0)
        self.mean = np.nan_to_num(self.mean)
        self.scale = np.where(np.isfinite(self.scale) & (self.scale > 1e-12), self.scale, 1.)
        x = self._transform(train["feature"])
        prices = train["price"].loc[:, list(PRICE_FIELDS)].to_numpy(float)
        dates = pd.DatetimeIndex(train.index.get_level_values("datetime"))
        months = dates.year * 100 + dates.month
        starts = sorted(set([*range(0, len(train) - length + 1, self.plan["block_length"]), len(train) - length]))
        indices = np.asarray(starts)[:, None] + np.arange(length)[None, :]
        tx, tp = torch.tensor(x[indices], dtype=torch.float64), torch.tensor(prices[indices], dtype=torch.float64)
        tm = torch.tensor(np.asarray(months)[indices], dtype=torch.int64)
        benchmarks = []
        for block in prices[indices]:
            fees = TieredFees(self.plan["tariff"], slippage_bps=self.plan["primary_slippage_bps"])
            bought = settle_tiered_event([0., 0.], self.plan["initial_cash"], desired=[0., 1., 0.],
                adjusted_prices=block[0, 2:4], raw_prices=block[0, 4:6], stamp=dates[starts[len(benchmarks)]], fees=fees)
            benchmarks.append(block[:, 6:8] @ bought["quantities"] + bought["cash"])
        benchmark = torch.tensor(np.asarray(benchmarks), dtype=torch.float64)
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.network = PathPolicy(len(self.columns), self.base_count, self.plan["hidden_units"], family=self.family,
                minimum_cash=self.plan["minimum_cash_weight"], maximum_soxl=self.plan["maximum_soxl_weight"]).double()
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.plan["learning_rate"], weight_decay=self.plan["weight_decay"])
        self.history = []
        for epoch in range(self.plan["epochs"]):
            optimizer.zero_grad(set_to_none=True)
            result = roll_full_path(self.network, tx, tp, tm, plan=self.plan)
            warm = self.plan["warmup_sessions"]
            relative = torch.log(result["equity"][:, -1] / result["equity"][:, warm - 1]) - torch.log(benchmark[:, -1] / benchmark[:, warm - 1])
            shortfall = torch.sqrt(torch.relu(-relative).square().mean() + 1e-12)
            maxdd = result["drawdowns"].max(dim=1).values
            violation = torch.relu(maxdd - self.plan["maximum_drawdown"])
            loss = -relative.mean() + self.plan["relative_shortfall_coefficient"] * shortfall + self.plan["drawdown_penalty_coefficient"] * violation.mean()
            if not torch.isfinite(loss):
                raise ValueError("blocked: nonfinite full-path objective")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.plan["gradient_clip"])
            if not torch.isfinite(gradient):
                raise ValueError("blocked: nonfinite full-path gradient")
            optimizer.step()
            self.history.append({"epoch": epoch + 1, "loss": float(loss.detach()), "mean_block_log_excess": float(relative.mean().detach()),
                "maximum_block_drawdown": float(maxdd.max().detach()), "mean_violation": float(violation.mean().detach()),
                "gradient_norm": float(gradient)})
        self.network.eval()
        self.audit = {"family": self.family, "seed": self.seed, "cutoff": str(self.cutoff), "training_rows": len(train),
            "latest_label_available_at": str(maturity.loc[valid].max()), "features": self.columns,
            "block_count": len(starts), "block_starts": starts, "scored_sessions": self.plan["block_length"],
            "warmup_sessions": self.plan["warmup_sessions"], "checkpoint": "fixed final epoch", "epochs": self.plan["epochs"],
            "feature_hash": hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
            "price_hash": hashlib.sha256(np.ascontiguousarray(prices).tobytes()).hexdigest(),
            "gradient_role": "integer straight-through approximation; realized forward fees exact; no RL", "final_training": self.history[-1]}
        return self

    def target(self, row, q, cash, known_prices, peak):
        feature = pd.DataFrame([row.loc[self.columns]], columns=self.columns)
        with torch.no_grad():
            value = self.network(torch.tensor(self._transform(feature), dtype=torch.float64),
                torch.tensor(np.asarray(q)[None, :], dtype=torch.float64), torch.tensor([cash], dtype=torch.float64),
                torch.tensor(np.asarray(known_prices)[None, :], dtype=torch.float64), torch.tensor([peak], dtype=torch.float64))
        return value.numpy()[0]

    def predict(self, dataset, segment="test"):
        data = dataset.prepare(segment, col_set="feature")
        # A standalone score has no account context; native execution must use
        # target() with its own real holdings. This explicit initial-cash view
        # is diagnostic, never an executable fixed-target schedule.
        result = [self.target(row, [0., 0.], self.plan["initial_cash"], [1., 1.], self.plan["initial_cash"])
                  for _, row in data.iterrows()]
        return pd.DataFrame(result, index=data.index, columns=["soxl", "soxx", "cash"])
