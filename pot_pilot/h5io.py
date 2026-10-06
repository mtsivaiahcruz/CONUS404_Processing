"""Open .nc and .h5 files for reading.

Uses pyfive, a reader written in pure Python. If pyfive is not installed, h5py is used instead.
The two work the same way for what these scripts do (f["name"][:]), but only the pyfive route
has been run.
"""
try:
    import pyfive as reader
except ImportError:
    import h5py as reader


def open_h5(path):
    return reader.File(str(path), "r")
