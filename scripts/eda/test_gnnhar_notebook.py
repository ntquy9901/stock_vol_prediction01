"""Guard: every code cell in the GNNHAR Colab notebook must be valid Python.

Regression cover: an earlier build leaked a real newline inside an f-string
(``print(f'`` split across lines), which raised "unterminated f-string literal"
only when Colab executed the cell. Compiling each code cell here catches that at
commit time. Magics are written as ``get_ipython().system(...)`` so cells are
pure Python and compile locally.
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[2] / "notebooks" / "train_gnnhar_sp500_colab.ipynb"


def test_every_code_cell_compiles():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells, "notebook has no code cells"
    for i, cell in enumerate(code_cells):
        src = "".join(cell["source"])
        compile(src, f"<cell{i}>", "exec")
