"""Synthetic stress scenarios for RQ4.

RQ4 asks how withdrawal-plan quality degrades as liquidity worsens and where it breaks down.
Historical data alone cannot answer that: real crashes are few, and none of them targeted
this exact universe. Parameterized synthetic shocks let us sweep severity continuously and
find the breaking point rather than sampling two or three anecdotes.

Three families, each with a severity knob in [0, 1]:
  ADV_COLLAPSE     -- volume evaporates, prices hold. Pure liquidity shock; isolates the
                      liquidity term from the price term.
  VOLATILITY_SPIKE -- prices gap, volume holds. Flash-crash-like.
  COMPOUND         -- both at once, which is what actually happens: in real stress, volume
                      dries up exactly when volatility spikes. The realistic case, and the
                      one that should hurt most.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import StrEnum


class ScenarioType(StrEnum):
    ADV_COLLAPSE = "adv_collapse"
    VOLATILITY_SPIKE = "volatility_spike"
    COMPOUND = "compound"
    BASELINE = "baseline"       # unstressed control


@dataclass(frozen=True)
class StressScenario:
    name: str
    scenario_type: ScenarioType
    severity: float             # 0 = no stress, 1 = maximum
    adv_multiplier: float       # ADV scaling, e.g. 0.2 = volume down 80%
    volatility_multiplier: float
    price_shock_pct: float
    description: str


# At maximum severity, ADV falls to 5% of normal. Grounded in observed behaviour: in the
# March 2020 and Aug 2024 dislocations, depth in stressed names fell by more than an order
# of magnitude even though headline share volume rose.
MIN_ADV_MULTIPLIER = 0.05
MAX_VOLATILITY_MULTIPLIER = 5.0
MAX_PRICE_SHOCK_PCT = 0.20


def make_scenario(scenario_type: ScenarioType, severity: float) -> StressScenario:
    """Build one scenario.

    Severity maps NON-LINEARLY to the ADV multiplier: real liquidity collapses are convex,
    so depth falls off a cliff at high severity rather than degrading linearly. A linear map
    would make RQ4's degradation curve look far gentler than reality.
    """
    severity = float(min(max(severity, 0.0), 1.0))

    if scenario_type is ScenarioType.BASELINE:
        return StressScenario(
            name="baseline", scenario_type=scenario_type, severity=0.0,
            adv_multiplier=1.0, volatility_multiplier=1.0, price_shock_pct=0.0,
            description="Unstressed control condition.",
        )

    # 1.0 at severity 0, MIN_ADV_MULTIPLIER at severity 1, with the collapse concentrated
    # at HIGH severity: mild stress barely dents depth, extreme stress removes it.
    adv_multiplier = 1.0 - (1.0 - MIN_ADV_MULTIPLIER) * severity**2
    vol_multiplier = 1.0 + (MAX_VOLATILITY_MULTIPLIER - 1.0) * severity
    price_shock = -MAX_PRICE_SHOCK_PCT * severity        # negative: an adverse gap

    if scenario_type is ScenarioType.ADV_COLLAPSE:
        return StressScenario(
            name=f"adv_collapse_{severity:.2f}", scenario_type=scenario_type, severity=severity,
            adv_multiplier=adv_multiplier, volatility_multiplier=1.0, price_shock_pct=0.0,
            description=f"Depth falls to {adv_multiplier:.0%} of normal; prices unaffected.",
        )

    if scenario_type is ScenarioType.VOLATILITY_SPIKE:
        return StressScenario(
            name=f"vol_spike_{severity:.2f}", scenario_type=scenario_type, severity=severity,
            adv_multiplier=1.0, volatility_multiplier=vol_multiplier, price_shock_pct=price_shock,
            description=f"Volatility x{vol_multiplier:.1f}, {price_shock:.1%} gap; depth intact.",
        )

    return StressScenario(
        name=f"compound_{severity:.2f}", scenario_type=ScenarioType.COMPOUND, severity=severity,
        adv_multiplier=adv_multiplier, volatility_multiplier=vol_multiplier,
        price_shock_pct=price_shock,
        description=(
            f"Depth {adv_multiplier:.0%} of normal AND volatility x{vol_multiplier:.1f} "
            f"with a {price_shock:.1%} gap -- the realistic joint shock."
        ),
    )


def severity_sweep(scenario_type: ScenarioType, *, n_steps: int = 10) -> list[StressScenario]:
    """A ladder of increasing severity -- the RQ4 degradation curve's x-axis."""
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    return [make_scenario(scenario_type, i / (n_steps - 1)) for i in range(n_steps)]


def apply_to_holdings(
    holdings: dict[str, dict[str, float]], scenario: StressScenario
) -> dict[str, dict[str, float]]:
    """Return a STRESSED COPY of a portfolio: ADV scaled, volatility scaled, prices shocked.

    Deep-copied rather than mutated, so a sweep cannot accidentally compound shocks across
    severity steps -- which would silently make the degradation curve look far worse than it
    is, and would be nearly invisible in the output.
    """
    stressed = copy.deepcopy(holdings)

    for holding in stressed.values():
        holding["adv_usd"] = float(holding.get("adv_usd", 0.0)) * scenario.adv_multiplier

        base_vol = float(holding.get("daily_volatility", 0.0126))
        holding["daily_volatility"] = base_vol * scenario.volatility_multiplier

        if scenario.price_shock_pct:
            shock = 1.0 + scenario.price_shock_pct
            holding["price"] = float(holding.get("price", 1.0)) * shock
            holding["value"] = float(holding.get("value", 0.0)) * shock

        # The fuzzy layer reads volatility as a percentile in [0, 1], so the stress must be
        # visible there too -- otherwise the FIS would keep reasoning about a calm market.
        holding["volatility_pct"] = min(
            1.0, float(holding.get("volatility_pct", 0.5)) * scenario.volatility_multiplier
        )

    return stressed


def standard_scenario_suite() -> list[StressScenario]:
    """The fixed suite every RQ4 run reports, so results stay comparable across runs.

    Three severities per family plus the control: enough to show the shape of the curve
    without a combinatorial explosion once it is crossed with four liquidation methods.
    """
    suite = [make_scenario(ScenarioType.BASELINE, 0.0)]
    for scenario_type in (ScenarioType.ADV_COLLAPSE, ScenarioType.VOLATILITY_SPIKE, ScenarioType.COMPOUND):
        suite.extend(make_scenario(scenario_type, severity) for severity in (0.33, 0.66, 1.0))
    return suite
