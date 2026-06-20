"""
Vietnam Stock Data Crawler
Crawl data for VN30, VN100, HNX, UPCoM from Yahoo Finance

Date: 2026-06-19
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import List, Dict
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Stock Lists
VN30_STOCKS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SSI", "STB", "TCB", "TPB",
    "VCB", "VDM", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE", "XYZ"
]

# VN100 stocks (includes all VN30 + 70 more)
VN100_STOCKS = VN30_STOCKS + [
    # Additional 70 stocks from VN100
    "AAM", "ABC", "ADG", "AGM", "AHG", "AMV", "ANV", "APC", "APG", "ASM",
    "BAX", "BBG", "BHS", "BII", "BIT", "BMH", "BMP", "BNS", "BSC", "BTP",
    "CAV", "CBT", "CCM", "CDT", "CGM", "CHP", "CMC", "CMG", "COS", "CSC",
    "D2D", "DC4", "DCM", "DGC", "DHT", "DML", "DPM", "DRH", "DTT", "DVP",
    "E1F", "EIB", "FDT", "FLC", "FRT", "FTS", "GEG", "GGG", "GTN", "HAG",
    "HHS", "HRC", "HT1", "HVX", "HVN", "KBC", "KDC", "KDH", "KOS", "KPF",
    "KSH", "KSN", "LCG", "LCM", "LEA", "LHG", "LIX", "LPB", "MCH", "MDG",
    "MHC", "MKP", "MSN", "NBC", "NDF", "NDN", "NHX", "NKG", "NPI", "NT2",
    "NTP", "OCG", "OGC", "PDR", "PGB", "PGD", "PIV", "PJT", "PLM", "PPG",
    "PRT", "PTB", "PTI", "PVD", "PVT", "RCM", "REE", "RDP", "S13", "SAM",
    "SBT", "SCM", "SDC", "SFI", "SHB", "SHP", "SHZ", "SKH", "SMB", "SMC",
    "SRE", "SSC", "ST8", "SZC", "TCM", "TDC", "TDM", "TNG", "TPC", "TPI",
    "TSE", "TTB", "TTI", "TV2", "TVS", "TXG", "TYA", "UDC", "VAF", "VAS",
    "VBC", "VCC", "VCF", "VCG", "VCM", "VCS", "VDG", "VEC", "VFG", "VGI",
    "VGS", "VHC", "VHL", "VID", "VIG", "VIL", "VIN", "VJC", "VKB", "VKL",
    "VLC", "VLH", "VLS", "VMB", "VMC", "VND", "VNG", "VNH", "VNR", "VNP",
    "VNS", "VOC", "VPG", "VPH", "VPI", "VRC", "VSC", "VSH", "VSJ", "VSI",
    "VST", "VTD", "VTF", "VTI", "VTK", "VTL", "VTT", "TVT", "VUH", "VUO",
    "VXG", "VXG", "VXV", "VXY"
]

# HNX stocks (top stocks from Hanoi Stock Exchange)
HNX_STOCKS = [
    "ACB", "APC", "BCC", "BVS", "CAP", "CCH", "CDC", "CGM", "CHP", "CLC",
    "CMC", "CMS", "CNT", "CTC", "D2D", "DCM", "DHA", "DHT", "DND", "DPR",
    "DTS", "DVP", "HAT", "HAX", "HHS", "HNA", "HPC", "HUT", "HVN", "KHB",
    "KHP", "KLS", "KSM", "KTL", "LAS", "LBM", "LCG", "LDG", "LGL", "LIX",
    "LSS", "LTC", "LU4", "MBB", "MCQ", "MDC", "MMP", "MSN", "NBB", "NCM",
    "NDX", "NHM", "NHP", "NKG", "NTP", "NTZ", "PC1", "PGC", "PGS", "PHL",
    "PIC", "PJT", "PMB", "PMS", "PPC", "PRT", "PSC", "PTB", "PTI", "PVC",
    "PVD", "PVT", "PXG", "PXS", "PYC", "QCC", "QNC", "SCG", "SDC", "SDI",
    "SHB", "SHH", "SHS", "SMB", "SMC", "SRC", "SSL", "STB", "ST8", "STM",
    "SVC", "SVM", "T2M", "TBS", "TC6", "TCC", "TCM", "TDH", "TDC", "TEG",
    "THG", "THT", "TIG", "TLC", "TLH", "TLM", "TMP", "TNA", "TNP", "TNT",
    "TPC", "TPI", "TPT", "TSG", "TSM", "TTC", "TV2", "TVC", "TVI", "TVS",
    "TXG", "TYA", "UCG", "UDC", "UNB", "UOM", "VAC", "VAS", "VCB", "VCC",
    "VCF", "VCG", "VCM", "VCS", "VDG", "VDS", "VEC", "VFG", "VFS", "VGC",
    "VGG", "VGS", "VHC", "VHF", "VHG", "VHL", "VIB", "VIC", "VIE", "VIF",
    "VIG", "VIH", "VIL", "VIT", "VJC", "VKF", "VLC", "VLC", "VLS", "VMB",
    "VMI", "VMH", "VMT", "VND", "VNG", "VNJ", "VNL", "VNM", "VNR", "VNP",
    "VNS", "VOC", "VPC", "VPG", "VPI", "VRC", "VRE", "VSB", "VSC", "VSD",
    "VSE", "VSJ", "VSK", "VSM", "VSS", "VST", "VSU", "VSX", "VTD", "VTF",
    "VTI", "VTI", "VTK", "VTL", "VTT", "TVT", "TVO", "VUH", "VUO", "VXC"
]

# Yahoo Finance suffix for Vietnam stocks
VIETNAM_SUFFIX = ".VN"

# Note: Many VN100 and HNX stocks may not have Yahoo Finance data
# In production, we should use Vietnamese data sources or APIs


class VietnamStockCrawler:
    """Crawl Vietnam stock data from Yahoo Finance"""

    def __init__(self, base_dir: str = "data/raw"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def add_vietnam_suffix(self, symbols: List[str]) -> List[str]:
        """Add .VN suffix for Vietnam stocks"""
        return [f"{symbol}{VIETNAM_SUFFIX}" for symbol in symbols]

    def crawl_stock(self, symbol: str, start_date: str = "2006-01-01") -> pd.DataFrame:
        """Crawl single stock data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, interval="1d")

            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return None

            # Reset index and rename columns
            data = data.reset_index()
            data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'splits']

            # Remove dividends and splits columns
            data = data[['date', 'open', 'high', 'low', 'close', 'volume']]

            # Remove .VN suffix from symbol for filename
            clean_symbol = symbol.replace(VIETNAM_SUFFIX, "")

            logger.info(f"Crawled {len(data)} rows for {clean_symbol}")
            return data

        except Exception as e:
            logger.error(f"Error crawling {symbol}: {e}")
            return None

    def crawl_stocks(
        self,
        symbols: List[str],
        output_dir: str,
        start_date: str = "2006-01-01"
    ) -> Dict[str, int]:
        """Crawl multiple stocks and save to CSV"""
        output_path = self.base_dir / output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        # Add Vietnam suffix
        symbols_with_suffix = self.add_vietnam_suffix(symbols)

        results = {"success": 0, "failed": 0, "total": len(symbols)}

        for symbol in symbols_with_suffix:
            clean_symbol = symbol.replace(VIETNAM_SUFFIX, "")
            data = self.crawl_stock(symbol, start_date)

            if data is not None:
                # Save to CSV
                filename = output_path / f"{clean_symbol}_ohlcv.csv"
                data.to_csv(filename, index=False)
                results["success"] += 1
            else:
                results["failed"] += 1

        logger.info(f"Crawling complete: {results['success']}/{results['total']} successful")
        return results

    def create_summary(self, output_dir: str) -> pd.DataFrame:
        """Create summary file with all stocks info"""
        output_path = self.base_dir / output_dir

        stock_files = list(output_path.glob("*_ohlcv.csv"))
        summary_data = []

        for file in stock_files:
            symbol = file.stem.replace("_ohlcv", "")
            df = pd.read_csv(file)

            summary_data.append({
                "symbol": symbol,
                "start_date": df['date'].min(),
                "end_date": df['date'].max(),
                "total_days": len(df),
                "file": str(file)
            })

        summary_df = pd.DataFrame(summary_data)
        summary_file = output_path / "stock_summary.csv"
        summary_df.to_csv(summary_file, index=False)

        logger.info(f"Summary saved to {summary_file}")
        return summary_df


def main():
    """Main execution"""
    crawler = VietnamStockCrawler()

    print("=" * 60)
    print("Vietnam Stock Data Crawler")
    print("=" * 60)

    print("\n1. Crawling VN30 stocks...")
    vn30_results = crawler.crawl_stocks(
        symbols=VN30_STOCKS,
        output_dir="vn30",
        start_date="2006-01-01"
    )
    crawler.create_summary("vn30")

    print("\n2. Crawling VN100 stocks...")
    vn100_results = crawler.crawl_stocks(
        symbols=VN100_STOCKS,
        output_dir="vn100",
        start_date="2006-01-01"
    )
    crawler.create_summary("vn100")

    print("\n3. Crawling HNX stocks...")
    hnx_results = crawler.crawl_stocks(
        symbols=HNX_STOCKS,
        output_dir="hnx",
        start_date="2006-01-01"
    )
    crawler.create_summary("hnx")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"VN30: {vn30_results['success']}/{vn30_results['total']} successful")
    print(f"VN100: {vn100_results['success']}/{vn100_results['total']} successful")
    print(f"HNX: {hnx_results['success']}/{hnx_results['total']} successful")
    print("=" * 60)

    logger.info("Crawling complete!")


if __name__ == "__main__":
    main()
