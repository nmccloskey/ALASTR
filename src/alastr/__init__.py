from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("alastr")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
