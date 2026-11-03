"""
SYNORA — API Package
Exports FastAPI app and create_app factory.
"""

from synora.api.app import app, create_app

__all__ = ["app", "create_app"]

