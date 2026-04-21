"""Tests that filter_reconstruction_by_angular_error produces the same results
as filter_reconstruction_by_angular_error_colmap."""

import copy
import logging

import numpy as np
import pytest

from gluemap.controllers.augmented_bundle_adjustment import (
    filter_reconstruction_by_angular_error,
    filter_reconstruction_by_angular_error_colmap,
)
from tests.helpers import create_synthetic_reconstruction

logger = logging.getLogger(__name__)


def _perturb_points3D(reconstruction, fraction, noise_std, rng):
    """Add Gaussian noise to a fraction of 3D points in-place."""
    ids = [pid for pid, _ in reconstruction.points3D.items()]
    n_perturb = int(len(ids) * fraction)
    perturb_ids = rng.choice(ids, size=n_perturb, replace=False)
    for pid in perturb_ids:
        reconstruction.points3D[pid].xyz += rng.normal(0, noise_std, size=3)
    return set(perturb_ids)


def _apply_colmap_filter(reconstruction, threshold):
    """Filter using filter_reconstruction_by_angular_error_colmap."""
    filter_reconstruction_by_angular_error_colmap(reconstruction, threshold)


def _apply_custom_filter(reconstruction, threshold):
    """Filter using filter_reconstruction_by_angular_error."""
    filter_reconstruction_by_angular_error(
        reconstruction,
        threshold,
        negative_depth_observations=dict(),
        virtual_point_start=dict(),
    )


def _extract_state(reconstruction):
    """Extract comparable state: {point3D_id: frozenset of (image_id, point2D_idx)}."""
    state = {}
    for pid, pt in reconstruction.points3D.items():
        elems = frozenset(
            (e.image_id, e.point2D_idx) for e in pt.track.elements
        )
        state[pid] = elems
    return state


class TestAngularErrorFilterEquivalence:

    @pytest.mark.parametrize(
        "seed,threshold",
        [
            (0, 1.0),
            (42, 0.5),
            (99, 2.0),
            (200, 5.0),
            (999, 1.0),
        ],
    )
    def test_equivalence_with_perturbation(self, seed, threshold):
        """Both filters must produce identical results on perturbed data."""
        rec = create_synthetic_reconstruction(
            num_frames=8, num_points3D=100, seed=seed
        )
        rng = np.random.default_rng(seed)
        _perturb_points3D(rec, fraction=0.5, noise_std=2.0, rng=rng)
        num_before = len([_ for _ in rec.points3D.items()])

        rec_colmap = copy.deepcopy(rec)
        rec_custom = copy.deepcopy(rec)

        _apply_colmap_filter(rec_colmap, threshold)
        _apply_custom_filter(rec_custom, threshold)

        state_colmap = _extract_state(rec_colmap)
        state_custom = _extract_state(rec_custom)

        # Same surviving point IDs
        assert set(state_colmap.keys()) == set(state_custom.keys()), (
            f"Point ID mismatch: "
            f"only in colmap={set(state_colmap.keys()) - set(state_custom.keys())}, "
            f"only in custom={set(state_custom.keys()) - set(state_colmap.keys())}"
        )

        # Same track elements for each surviving point
        for pid in state_colmap:
            assert state_colmap[pid] == state_custom[pid], (
                f"Track mismatch at point3D {pid}"
            )

        # Verify filtering actually removed something
        num_after = len(state_colmap)
        assert num_after < num_before, (
            f"Expected some points to be filtered, but {num_before} -> {num_after}"
        )

    def test_no_perturbation_keeps_all_points(self):
        """Clean synthetic data: both filters should keep all points."""
        rec = create_synthetic_reconstruction(
            num_frames=8, num_points3D=100, seed=0
        )
        num_before = len([_ for _ in rec.points3D.items()])

        rec_colmap = copy.deepcopy(rec)
        rec_custom = copy.deepcopy(rec)

        _apply_colmap_filter(rec_colmap, 0.5)
        _apply_custom_filter(rec_custom, 0.5)

        state_colmap = _extract_state(rec_colmap)
        state_custom = _extract_state(rec_custom)

        assert set(state_colmap.keys()) == set(state_custom.keys())
        for pid in state_colmap:
            assert state_colmap[pid] == state_custom[pid]
        assert len(state_colmap) == num_before

    def test_large_perturbation(self):
        """Heavy noise on all points: both filters should agree on survivors."""
        rec = create_synthetic_reconstruction(
            num_frames=8, num_points3D=100, seed=0
        )
        rng = np.random.default_rng(0)
        _perturb_points3D(rec, fraction=1.0, noise_std=5.0, rng=rng)

        rec_colmap = copy.deepcopy(rec)
        rec_custom = copy.deepcopy(rec)

        _apply_colmap_filter(rec_colmap, 1.0)
        _apply_custom_filter(rec_custom, 1.0)

        state_colmap = _extract_state(rec_colmap)
        state_custom = _extract_state(rec_custom)

        assert set(state_colmap.keys()) == set(state_custom.keys())
        for pid in state_colmap:
            assert state_colmap[pid] == state_custom[pid]
