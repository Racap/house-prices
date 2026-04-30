from __future__ import annotations

from typing import List


MODEL_FEATURES: List[str] = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
]

TARGET_COLUMN = "SalePrice"
