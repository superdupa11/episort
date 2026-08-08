import sys
from pathlib import Path

# identifier.py is a single top-level script, not an installed package — make it
# importable as `import identifier` regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
