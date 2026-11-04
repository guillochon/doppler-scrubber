"""Remove aircraft Doppler tones from recordings without touching anything else."""
from .scrub import ScrubConfig, scrub_array, scrub_file
from .tracks import DopplerTrack, find_tracks

__all__ = ["ScrubConfig", "scrub_array", "scrub_file", "DopplerTrack", "find_tracks"]
__version__ = "0.1.0"
