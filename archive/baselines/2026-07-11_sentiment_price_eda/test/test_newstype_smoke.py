"""Smoke tests for sentiment_newstype_eda (keyword classifier)."""
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

import sentiment_newstype_eda as nws  # noqa: E402


def test_classify_buy_rating():
    c = nws.classify("FPT: Khuyến nghị MUA với giá mục tiêu 95,900 đồng/cổ phiếu")
    assert c["rating"] == "POS"
    assert c["is_event"] is False


def test_classify_sell_rating():
    c = nws.classify("Báo cáo cập nhật SHB - BÁN")
    assert c["rating"] == "NEG"


def test_classify_hold_neutral():
    c = nws.classify("VCB: Khuyến nghị TRUNG LẬP với giá mục tiêu 117,000 đồng/cp")
    assert c["rating"] == "NEU"


def test_classify_earnings_no_rating():
    # earnings update without a recommendation gate -> rating None, is_earnings True
    c = nws.classify("HPG: Báo cáo cập nhật KQKD Q1/2026")
    assert c["rating"] is None
    assert c["is_earnings"] is True


def test_classify_event_ma():
    c = nws.classify("MBB: Sáp nhập công ty Tài chính Cổ phần Sông Đà")
    assert c["is_event"] is True


def test_rating_requires_gate():
    # 'mua' appears but no recommendation gate -> not a rating
    c = nws.classify("HPG thu hồi khoản phải thu từ công ty con")
    assert c["rating"] is None


def test_compound_ban_hang_does_not_flip_buy_to_sell():
    # "maintain BUY ... sales costs" must be POS, not NEG (regression for 'bán hàng')
    c = nws.classify("VNM [Giữ KN MUA +28%] - Chi phí bán hàng ảnh hưởng lợi nhuận")
    assert c["rating"] == "POS"


def test_compound_ban_le_not_a_rating():
    # "retail revenue growth" has no genuine rating
    c = nws.classify("ACB: doanh số bán lẻ tăng trưởng mạnh")
    assert c["rating"] is None


def test_competition_keyword_not_negative():
    # 'cạnh tranh' (competition) must not match the old 'tranh' avoid-keyword
    c = nws.classify("HPG: Sức ép cạnh tranh ngày càng gay gắt")
    assert c["rating"] is None


def test_earnings_nqyy_format():
    # "2Q20" quarterly format must be detected as earnings (leading-space variant missed it)
    c = nws.classify("MBB - Cập nhật 2Q20 [MUA +35.4%]")
    assert c["is_earnings"] is True

