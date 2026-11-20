"""Package version (layer L0)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fleetgym")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled tree
    __version__ = "0+unknown"
