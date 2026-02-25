from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from enum import Enum
import numpy as np


class Asset(str, Enum):
    US_EQ = "US_EQ"
    US_BOND = "US_BOND"
    CA_EQ = "CA_EQ"
    CA_BOND = "CA_BOND"


@dataclass(frozen=True)
class MarketConfig:
    mu: Dict[Asset, float]
    sigma: Dict[Asset, float]
    corr: np.ndarray
    fx_start: float
    fx_mu_log: float
    fx_sigma_log: float
    infl_mu: float
    infl_sigma: float
    cola_mu: float
    cola_sigma: float