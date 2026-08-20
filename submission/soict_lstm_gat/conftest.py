"""Make the submission folder importable by bare module name (self-contained package).

Reviewers run `pytest` from inside submission/soict_lstm_gat/ and the modules import each other as
`import metrics`, `import data_utils`, etc. — no external package path needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
