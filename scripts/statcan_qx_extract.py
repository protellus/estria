from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QxSeries:
    year: int
    geo: str
    sex: str
    qx: np.ndarray  # index = exact age (0..110), where 110 is "110 years and over"


_QX_ELEMENT = "Death probability between age x and x+1 (qx)"


def _parse_age_group_to_int(age_group: str) -> int:
    s = str(age_group).strip()

    m = re.match(r"^(\d+)\s+year\b", s)
    if m:
        return int(m.group(1))

    m = re.match(r"^(\d+)\s+years\b", s)
    if m:
        return int(m.group(1))

    m = re.match(r"^(\d+)\s+years\s+and\s+over$", s)
    if m:
        return int(m.group(1))

    raise ValueError(f"Unrecognized Age group: {age_group!r}")


def load_statcan_qx_csv(
    csv_path: str | Path,
    *,
    year: int,
    sex: str,
    geo: str = "Canada",
) -> QxSeries:
    """
    Loads StatCan table 13-10-0837-01 'databaseLoadingData' CSV and extracts a single qx vector.
    The returned qx is indexed by exact age (qx[60] = q60).
    """
    df = pd.read_csv(csv_path)

    required = {"REF_DATE", "GEO", "Age group", "Sex", "Element", "VALUE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[
        (df["REF_DATE"] == year)
        & (df["GEO"] == geo)
        & (df["Sex"] == sex)
        & (df["Element"] == _QX_ELEMENT)
    ].copy()

    if df.empty:
        raise ValueError(
            f"No rows found for year={year}, geo={geo!r}, sex={sex!r}, element={_QX_ELEMENT!r} "
            f"in {str(csv_path)!r}"
        )

    df["age"] = df["Age group"].map(_parse_age_group_to_int)
    df.sort_values("age", inplace=True)

    ages = df["age"].to_numpy(dtype=int)
    if ages.min() != 0:
        raise ValueError(f"Expected ages to start at 0, got min age={ages.min()}")

    expected = np.arange(0, ages.max() + 1, dtype=int)
    if not np.array_equal(ages, expected):
        missing_ages = sorted(set(expected) - set(ages))
        raise ValueError(f"Age gaps detected. Missing ages: {missing_ages[:20]}...")

    qx = df["VALUE"].to_numpy(dtype=float)

    return QxSeries(year=year, geo=geo, sex=sex, qx=qx)


def _format_np_constant(name: str, arr: np.ndarray) -> str:
    vals = []
    for v in arr:
        if v >= 1.0:
            vals.append("1")
        else:
            vals.append(f"{v:.5f}".rstrip("0").rstrip("."))

    lines = []
    for i in range(0, len(vals), 10):
        chunk = ", ".join(vals[i : i + 10])
        suffix = "," if (i + 10) < len(vals) else ""
        lines.append(f"    {chunk}{suffix}")

    return f"{name} = np.array([\n" + "\n".join(lines) + "\n], dtype=float)\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--male_csv", required=True)
    p.add_argument("--female_csv", required=False)
    p.add_argument("--year", type=int, default=2022)
    p.add_argument("--out_py", default="mortality_canada_qx.py")
    args = p.parse_args()

    male = load_statcan_qx_csv(args.male_csv, year=args.year, sex="Males")
    parts = [
        "from __future__ import annotations\n",
        "import numpy as np\n\n",
        "# Source: Statistics Canada table 13-10-0837-01 (qx element)\n",
        f"# Extracted year: {args.year}\n\n",
        _format_np_constant(f"QX_CANADA_MALE_{args.year}", male.qx),
    ]

    if args.female_csv:
        female = load_statcan_qx_csv(args.female_csv, year=args.year, sex="Females")
        parts.append("\n")
        parts.append(_format_np_constant(f"QX_CANADA_FEMALE_{args.year}", female.qx))

    Path(args.out_py).write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {args.out_py}")


if __name__ == "__main__":
    main()