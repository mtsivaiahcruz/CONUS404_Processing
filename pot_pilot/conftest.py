"""Tells pytest to skip the check scripts in this folder.

test_pot.py, test_bayes.py and tests/test_extraction_scripts.py are plain scripts. They run
their checks when you run them (python test_pot.py) and are not written as pytest tests.
Without this file, running pytest from the top of the repository would import them and stop
with an error.
"""
collect_ignore = ["test_pot.py", "test_bayes.py", "tests"]
