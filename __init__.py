from __future__ import annotations

import logging

LOGGER = logging.getLogger("HiDream_O1")
__version__ = "0.1.6"
LOGGER.propagate = False
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[HiDream_O1] %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as exc:
    LOGGER.exception("Failed to load HiDream O1 nodes: %s", exc)
    raise

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", "__version__"]
