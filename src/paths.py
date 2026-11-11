"""Repository-relative data locations.

Defaults: <repo>/instances for the 27 benchmark instances and <repo>/results for all outputs.
Override with the environment variables SLACO_INSTANCES / SLACO_RESULTS (e.g. to run into a scratch folder).
"""
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
INSTANCES = os.path.abspath(os.environ.get("SLACO_INSTANCES", os.path.join(REPO, "instances")))
RESULTS = os.path.abspath(os.environ.get("SLACO_RESULTS", os.path.join(REPO, "results")))
