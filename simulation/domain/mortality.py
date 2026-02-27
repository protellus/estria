from __future__ import annotations
import numpy as np
from enum import Enum

from .mortality_tables_canada_2022 import QX_CANADA_MALE_2022, QX_CANADA_FEMALE_2022
from .mortality_tables_us_2022 import QX_US_FEMALE_2022, QX_US_MALE_2022

class MortalityTable(Enum):
    CANADA_MALE_2022 = "canada_male_2022"
    CANADA_FEMALE_2022 = "canada_female_2022"
    US_FEMALE_2022 = "us_female_2022"
    US_MALE_2022 = "us_male_2022"
    US_FEMALE = "us_female"
    US_MALE = "us_male"
    CANADA_FEMALE = "canada_female"
    CANADA_MALE = "canada_male"

    def qx(self) -> np.ndarray:
        if self in (MortalityTable.US_FEMALE_2022, MortalityTable.US_FEMALE):
            return QX_US_FEMALE_2022
        if self in (MortalityTable.US_MALE_2022, MortalityTable.US_MALE):
            return QX_US_MALE_2022
        if self in (MortalityTable.CANADA_MALE_2022, MortalityTable.CANADA_MALE):
            return QX_CANADA_MALE_2022
        if self in (MortalityTable.CANADA_FEMALE_2022, MortalityTable.CANADA_FEMALE):
            return QX_CANADA_FEMALE_2022

        raise NotImplementedError(f"No qx defined for {self}")