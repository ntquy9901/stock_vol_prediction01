import base64

from scripts.eda.extract_diag_figs import extract_pngs


def test_extract_pngs_decodes_each_data_uri_in_order():
    raw_a = b"\x89PNG\r\n\x1a\nAAA"
    raw_b = b"\x89PNG\r\n\x1a\nBBB"
    html = (
        f"<img src='data:image/png;base64,{base64.b64encode(raw_a).decode()}'>"
        f"<img src='data:image/png;base64,{base64.b64encode(raw_b).decode()}'>"
    )
    assert extract_pngs(html) == [raw_a, raw_b]


def test_extract_pngs_empty_when_no_data_uri():
    assert extract_pngs("<html><body>no images here</body></html>") == []
