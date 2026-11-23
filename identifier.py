#!/usr/bin/env python3
"""
EpiSort command-line entry point.

The engine lives in the `episort` package (episort/identifier.py). This file
keeps `python identifier.py ...` and `import identifier` working as before:
importing it hands back the engine module itself, so code (and tests) that
patch `identifier.some_function` patch the engine everyone calls.
"""
import sys

from episort import identifier as _engine

if __name__ == "__main__":
    _engine.main()
else:
    sys.modules[__name__] = _engine
