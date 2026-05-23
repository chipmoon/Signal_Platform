"""Asset Provider Plugin System."""

from .base import (
    AssetData,
    AssetInfo,
    AssetProvider,
    ProviderRegistry,
    RealtimeQuote,
    registry,
)

# Auto-import providers so they self-register with the registry
from . import commodity  # noqa: F401
from . import us_stocks  # noqa: F401
from . import vietnam_v2  # noqa: F401  # Enhanced with cache + TvDatafeed
from . import taiwan     # noqa: F401

__all__ = [
    "AssetProvider",
    "AssetInfo",
    "AssetData",
    "ProviderRegistry",
    "registry",
]

