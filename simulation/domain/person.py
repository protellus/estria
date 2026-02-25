from __future__ import annotations
from datetime import date
import numpy as np

AGES = np.arange(40, 111)


class Person:
    """
    Cached (lazy) death age draw.
    Independent mortality (per your design choice).
    """
    def __init__(self, name: str, birthdate: date, ages=AGES, qx_array=None):
        self.name = name
        self.birthdate = birthdate
        self._ages = ages
        self._qx = qx_array
        self._death_age = None

    def current_age(self, year: int) -> int:
        return year - self.birthdate.year

    def death_age(self, start_year: int, rng: np.random.Generator) -> int:
        if self._qx is None:
            raise ValueError("qx_array not set for Person")
        if self._death_age is not None:
            return self._death_age

        age = self.current_age(start_year)
        while age < self._ages[-1]:
            idx = age - self._ages[0]
            if idx >= len(self._qx):
                break
            if rng.random() < self._qx[idx]:
                self._death_age = age
                return age
            age += 1

        self._death_age = int(self._ages[-1])
        return self._death_age

    def is_alive(self, year: int, start_year: int, rng: np.random.Generator) -> bool:
        return self.current_age(year) < self.death_age(start_year, rng)

    def reset(self) -> None:
        self._death_age = None