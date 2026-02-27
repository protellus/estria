from __future__ import annotations

from enum import Enum
    
# ============================================================
# Jurisdiction
# ============================================================

class Jurisdiction(str, Enum):
    US = "US"
    CA = "CA"

# ============================================================
# Asset & Factor Model
# ============================================================

class Asset(str, Enum):
    """
    Investable portfolio exposures.
    """
    US_EQ = "US_EQ"
    US_BOND = "US_BOND"
    US_RE = "US_RE"

    CA_EQ = "CA_EQ"
    CA_BOND = "CA_BOND"
    CA_RE = "CA_RE"
