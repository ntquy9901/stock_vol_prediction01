"""
VN30 Ticker Configuration

List of all VN30 stock symbols for sentiment analysis.
Updated: 2026-06-27
"""

# VN30 Stock List (2026)
VN30_TICKERS = [
    "ACB",  # Asia Commercial Bank
    "BCM",  # Bank for Investment and Development of Cambodia
    "BID",  # Investment and Development Bank
    "BME",  # Baoviet Insurance
    "BMI",  # Military Commercial Joint Stock Bank
    "BVH",  # Bao Viet Holdings
    "CAA",  # Caa Thu Au
    "CMG",  # China Communications Construction Group
    "CTG",  # Vietnam Electricity
    "DBC",  # Dried Batteries
    "DCM",  # Nam Viet Investment and Construction
    "DPM",  # Fertilizers and Chemicals
    "DRH",  # Duc Vu Phat
    "DXG",  # Vinhomes
    "FRT",  # FPT Retail
    "GAS",  # Petrovietnam Gas
    "GVR",  # Vinacapital
    "HDB",  # Housing Development Bank
    "HHS",  # Hanoi Housing
    "HPG",  # Hoa Phat Group
    "KDH",  # KIDO Group
    "KDC",  # Consumer Goods
    "LCG",  # Logistics
    "MBB",  # Military Bank
    "MSN",  # Masan Group
    "MWG",  # Mobile World Group
    "NLG",  # Nam Long Group
    "PDR",  # Phat Dat Real Estate
    "PGV",  # Petrovietnam
    "PLX",  # Petrovietnam Power
    "PNJ",  # Phu Nhuan Jewelry
    "POW",  # Petrovietnam Oil
    "SAB",  # Sabeco
    "SSI",  # Saigon Securities Inc
    "STB",  # Sacombank
    "TCB",  # Techcombank
    "TPB",  # Tien Phong Bank
    "VCB",  # Vietcombank
    "VIB",  # Vietnam International Bank
    "VHM",  # Vinhomes
    "VIC",  # Vingroup
    "VJC",  # Vietjet Air
    "VNM",  # Vinamilk
    "VRE",  # Vincom Retail
    "VPB",  # Vietnam Prosperity Bank
]

# Additional tickers for future expansion
VN100_TICKERS = [
    # Will include all VN30 plus:
    "AAA", "AST", "BGT", "BRC", "CIG", "CLC", "CMV", "D2D", "DHG",
    "DPR", "DTL", "E1VFVN30", "FPT", "GEG", "HT1", "HVN", "IMP", "KBS",
    "KOS", "KQ", "LC1", "LTV", "MBS", "MHC", "MSN", "NGC", "NKG", "NTP",
    "OCB", "PAN", "PC1", "PHR", "PTB", "PVT", "RDP", "SSP", "SZC", "TBP",
    "TDG", "TCH", "TLH", "TNG", "TPP", "TRG", "VBB", "VCG", "VCF", "VCM",
    "VDL", "VDS", "VGC", "VIB", "VIX", "VND", "VNM", "VNT", "VPG", "VSC",
    "VSF", "VTL", "VTT", "VYC"
]

def get_all_tickers() -> list:
    """Get all VN30 tickers"""
    return VN30_TICKERS

def get_ticker_info(ticker: str) -> dict:
    """Get basic information about a ticker"""
    ticker_info = {
        "ACB": {"name": "Asia Commercial Bank", "sector": "Banking"},
        "VCB": {"name": "Vietcombank", "sector": "Banking"},
        "VHM": {"name": "Vinhomes", "sector": "Real Estate"},
        "VNM": {"name": "Vinamilk", "sector": "Consumer Goods"},
        # Add more as needed
    }

    return ticker_info.get(ticker.upper(), {"name": "Unknown", "sector": "Unknown"})

def validate_ticker(ticker: str) -> bool:
    """Validate if ticker is in VN30 list"""
    return ticker.upper() in VN30_TICKERS
