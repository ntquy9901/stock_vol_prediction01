"""Import/contract smoke for the earnings-DM audit script (its main() is a data-driven driver, pragma-no-cover).

Importing the module covers its bootstrap + OWN-8 wiring; the heavy walk-forward lives in main()."""


def test_earnings_dm_module_contract():
    import earnings_dm as E
    assert isinstance(E.OWN, list) and len(E.OWN) == 8 and "rq" not in E.OWN
    assert E.FL > 0
    assert hasattr(E, "main")
