"""Edge travel-time estimation from sparse multi-camera re-identification."""
from vtts.camera import CameraConfig, Observations, make_observations
from vtts.matching import MatchParams, camera_pairs, match_pair
from vtts.network import Network
from vtts.placement import covered_edges, greedy_placement
from vtts.spanner import dilation_stats, greedy_spanner
from vtts.tomography import estimate_factors

__version__ = "0.1.0"
__all__ = ["CameraConfig", "MatchParams", "Network", "Observations", "camera_pairs",
           "covered_edges", "dilation_stats", "estimate_factors", "greedy_placement",
           "greedy_spanner", "make_observations", "match_pair"]
