"""Data Saver module to save output from different learning mechanism."""
import pickle
from copy import deepcopy
from contextlib import contextmanager


class DataSaver():
    """Save data for analysis."""

    def __init__(self, defaultdest=None, label=None):
        """Initialize the DataSaver."""
        self.data = []
        if label is None:
            label = 'root'
        self.defaultdest = defaultdest
        self.labels = [label]

    def add(self, label=None, **kwargs):
        """Add element in the data saver."""
        if label is None:
            label = self.labels[-1]
        self.data.append((label, deepcopy(kwargs)))

    def write(self, path=None, mode="wb"):
        """Write the data into a file."""
        if path is None and self.defaultdest is not None:
            path = self.defaultdest
        elif path is None and self.defaultdest is None:
            raise ValueError('No destination file given and no default'
                             ' destination set.')
        with open(path, mode) as f:
            pickle.dump(self.data, f)

    def flush(self, path=None, mode="ab"):
        """Write the data into a file then reset the data attribute"""
        self.write(path, mode)
        self.data = []

    @contextmanager
    def set_context(self, label):
        """Change the default label for a context."""
        self.labels.append(label)
        try:
            yield
        finally:
            self.labels.pop()


class QuietDataSaver(DataSaver):
    """DataSaver that does nothing."""

    def add(self, label=None, **kwargs):
        """Do nothing on add."""
        pass

    def write(self, path=None, mode="wb"):
        """Do not write anything."""
        pass

    def flush(self, path=None, mode="ab"):
        """Do not flush anything."""
        pass
