"""Documented VN30 sector map.

No sector-metadata file exists in the repository, so this map is constructed from
public HOSE / GICS-style classifications of the 33 tickers in the universe. It is a
best-effort economic grouping used only for same-sector vs cross-sector comparison and
graph-community annotation (never for a modelling decision on its own).

Singleton sectors (SSI, BVH, POW, MWG, FPT, VJC) contribute no same-sector pairs; that
is expected and does not bias the same-sector test (dominated by the Banks group).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SECTOR_MAP: dict[str, str] = {
    # Banks (14)
    "ACB": "Banks", "BID": "Banks", "CTG": "Banks", "HDB": "Banks",
    "LPB": "Banks", "MBB": "Banks", "SHB": "Banks", "SSB": "Banks",
    "STB": "Banks", "TCB": "Banks", "TPB": "Banks", "VCB": "Banks",
    "VIB": "Banks", "VPB": "Banks",
    # Securities / brokerage (1)
    "SSI": "Securities",
    # Insurance (1)
    "BVH": "Insurance",
    # Real Estate (6)
    "BCM": "RealEstate", "NVL": "RealEstate", "PDR": "RealEstate",
    "VHM": "RealEstate", "VIC": "RealEstate", "VRE": "RealEstate",
    # Materials (2)
    "GVR": "Materials", "HPG": "Materials",
    # Energy / Oil & Gas (2)
    "GAS": "Energy", "PLX": "Energy",
    # Utilities (1)
    "POW": "Utilities",
    # Consumer Staples (3)
    "MSN": "ConsumerStaples", "SAB": "ConsumerStaples", "VNM": "ConsumerStaples",
    # Consumer Discretionary / retail (1)
    "MWG": "ConsumerDiscretionary",
    # Information Technology (1)
    "FPT": "InformationTechnology",
    # Industrials / airlines (1)
    "VJC": "Industrials",
}


def sector_of(ticker: str) -> str:
    """Return the sector label for a ticker (``Unknown`` if unmapped)."""
    return SECTOR_MAP.get(ticker, "Unknown")


def same_sector_matrix(tickers: list[str]) -> pd.DataFrame:
    """Symmetric 0/1 same-sector indicator matrix (diagonal 0)."""
    sec = np.array([sector_of(t) for t in tickers])
    same = (sec[:, None] == sec[None, :]).astype(int)
    np.fill_diagonal(same, 0)
    return pd.DataFrame(same, index=tickers, columns=tickers)
