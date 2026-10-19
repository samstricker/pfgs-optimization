"""This module sets up the logger for the project."""

import logging as log


def set_logger():
    log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    log.getLogger("alembic").setLevel(log.WARNING)
    log.getLogger("mlflow").setLevel(log.WARNING)
    log.getLogger("kaleido").setLevel(log.WARNING)
    log.getLogger("choreographer").setLevel(log.WARNING)
