import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# How many checks this run collected. test_self_description compares it with the count both
# READMEs publish, so the claim is measured by the run it describes rather than remembered.
COLLECTED = [0]


def pytest_collection_modifyitems(items):
    COLLECTED[0] = len(items)
