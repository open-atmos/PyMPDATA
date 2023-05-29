from pkg_resources import get_distribution, DistributionNotFound, VersionConflict
try:
    __version__ = get_distribution(__name__).version
except (DistributionNotFound, VersionConflict):
    # package is not installed
    pass
