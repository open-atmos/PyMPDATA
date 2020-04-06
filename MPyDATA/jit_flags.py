"""
Created at 03.2020
"""

# import os
# os.environ["NUMBA_DISABLE_JIT"] = "1"

jit_flags = {
    "fastmath": True,
    "error_model": 'numpy',
    "cache": False
}
