"""Tests for bundle adjustment: normal vs virtual residual equivalence.

Validates that the same reprojection residuals produce the same Ceres cost
regardless of whether they are added through the normal (pycolmap built-in)
path or the virtual (manual ``pygluemap.ReprojErrorCost``) path.
"""

import copy
import logging

import numpy as np
import pycolmap

from gluemap.estimators.bundle_adjustment import bundle_adjustment
from tests.helpers import create_synthetic_reconstruction, perturb_points3D

logger = logging.getLogger(__name__)


def split_reconstruction(source, point_ids_to_keep):
    """Deep-copy *source* and keep only the selected 3D points.

    Returns a new ``pycolmap.Reconstruction`` with the same cameras, images,
    and poses but only the 3D points whose IDs are in *point_ids_to_keep*.
    """
    rec = copy.deepcopy(source)
    keep = set(point_ids_to_keep)
    for pid in sorted(rec.point3D_ids()):
        if pid not in keep:
            rec.delete_point3D(pid)
    return rec


class TestBundleAdjustmentEquivalence:
    """Verify that swapping which point subset is *normal* vs *virtual* does not
    change the total Ceres cost — i.e. the two code paths for adding reprojection
    residuals are mathematically equivalent.

    Strategy
    --------
    1. Create a synthetic reconstruction (same cameras, poses, and 3D points).
    2. Split the 3D points into two disjoint halves A and B.
    3. Run BA with A as the real (normal) reconstruction and B as the virtual
       reconstruction, recording the initial Ceres cost.
    4. Swap: B becomes real, A becomes virtual.
    5. Assert the initial costs are identical.
    """

    @staticmethod
    def _make_split(seed=42, num_frames=5, num_points3D=50, noise_std=0.05):
        gt_rec = create_synthetic_reconstruction(
            num_frames=num_frames, num_points3D=num_points3D, seed=seed
        )
        rng = np.random.default_rng(seed)
        perturb_points3D(gt_rec, fraction=1.0, noise_std=noise_std, rng=rng)
        ids = sorted(gt_rec.point3D_ids())
        half = len(ids) // 3
        ids_a = ids[:half]
        ids_b = ids[half:]
        rec_a = split_reconstruction(gt_rec, ids_a)
        rec_b = split_reconstruction(gt_rec, ids_b)
        return rec_a, rec_b

    @staticmethod
    def _run_ba(rec_normal, rec_virtual, loss_type_normal, loss_type_virtual=None):
        """Run BA with ``max_num_iterations=0`` (evaluate only) and return the summary."""
        if loss_type_virtual is None:
            loss_type_virtual = loss_type_normal
        _, _, summary = bundle_adjustment(
            copy.deepcopy(rec_normal),
            copy.deepcopy(rec_virtual),
            negative_depth_observations={},
            max_num_iterations=0,
            loss_type_normal=loss_type_normal,
            loss_type_virtual=loss_type_virtual,
        )
        return summary

    def test_swap_equivalence_trivial_loss(self):
        """With trivial (identity) loss the swap must not change cost."""
        rec_a, rec_b = self._make_split()

        summary_ab = self._run_ba(rec_a, rec_b, "trivial")
        summary_ba = self._run_ba(rec_b, rec_a, "trivial")

        logger.info(
            f"trivial — AB cost={summary_ab.initial_cost:.6e}, "
            f"BA cost={summary_ba.initial_cost:.6e}"
        )
        assert summary_ab.initial_cost > 0, "Expected non-zero cost"
        np.testing.assert_allclose(
            summary_ab.initial_cost,
            summary_ba.initial_cost,
            rtol=1e-6,
            err_msg="Initial costs differ after swapping normal/virtual (trivial loss)",
        )

    def test_swap_equivalence_huber_loss(self):
        """With Huber loss the swap must not change cost."""
        rec_a, rec_b = self._make_split()

        summary_ab = self._run_ba(rec_a, rec_b, "huber")
        summary_ba = self._run_ba(rec_b, rec_a, "huber")

        logger.info(
            f"huber — AB cost={summary_ab.initial_cost:.6e}, "
            f"BA cost={summary_ba.initial_cost:.6e}"
        )
        assert summary_ab.initial_cost > 0, "Expected non-zero cost"
        np.testing.assert_allclose(
            summary_ab.initial_cost,
            summary_ba.initial_cost,
            rtol=1e-6,
            err_msg="Initial costs differ after swapping normal/virtual (huber loss)",
        )

    def test_mismatched_loss_breaks_equivalence(self):
        """When virtual points use arctan instead of trivial, the swap must change cost."""
        rec_a, rec_b = self._make_split()

        summary_ab = self._run_ba(rec_a, rec_b, "trivial", "arctan")
        summary_ba = self._run_ba(rec_b, rec_a, "trivial", "arctan")

        logger.info(
            f"mismatched — AB cost={summary_ab.initial_cost:.6e}, "
            f"BA cost={summary_ba.initial_cost:.6e}"
        )
        assert summary_ab.initial_cost > 0, "Expected non-zero cost"
        assert not np.isclose(
            summary_ab.initial_cost, summary_ba.initial_cost, rtol=1e-3
        ), (
            f"Costs should differ when loss types are mismatched, "
            f"got AB={summary_ab.initial_cost:.6e}, BA={summary_ba.initial_cost:.6e}"
        )


class TestBundleAdjustmentEndToEnd:
    """Verify that BA recovers the original reconstruction after adding noise."""

    @staticmethod
    def _perturb_poses(reconstruction, translation_std, rotation_std_deg, rng):
        """Add noise to camera poses in-place."""
        rotation_std_rad = np.deg2rad(rotation_std_deg)
        for img_id in sorted(reconstruction.reg_image_ids()):
            img = reconstruction.image(img_id)
            pose = img.cam_from_world()
            R = np.array(pose.rotation.matrix())
            t = np.array(pose.translation)

            # Perturb translation
            t += rng.normal(0, translation_std, size=3)

            # Perturb rotation via small-angle axis
            axis_angle = rng.normal(0, rotation_std_rad, size=3)
            angle = np.linalg.norm(axis_angle)
            if angle > 1e-12:
                axis = axis_angle / angle
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0],
                ])
                dR = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
                R = dR @ R

            new_pose = pycolmap.Rigid3d(np.hstack([R, t.reshape(3, 1)]))
            img.frame.set_cam_from_world(img.camera_id, new_pose)

    @staticmethod
    def _perturb_points2D(reconstruction, noise_std, rng):
        """Add noise to 2D observations in-place."""
        for img_id in sorted(reconstruction.reg_image_ids()):
            img = reconstruction.image(img_id)
            for i in range(img.num_points2D()):
                img.points2D[i].xy += rng.normal(0, noise_std, size=2)

    def test_ba_recovers_from_noise(self):
        """After adding small noise to 3D points and poses, BA should
        recover a reconstruction close to the original."""
        gt_rec = create_synthetic_reconstruction(
            num_frames=8, num_points3D=100, seed=0
        )

        rec = copy.deepcopy(gt_rec)
        rng = np.random.default_rng(0)

        # Add noise
        perturb_points3D(rec, fraction=1.0, noise_std=0.02, rng=rng)
        self._perturb_poses(rec, translation_std=0.01, rotation_std_deg=0.5, rng=rng)

        # Run BA (real tracks only, Huber loss, 200 iterations)
        rec, _, summary = bundle_adjustment(
            rec,
            virtual_reconstruction=None,
            negative_depth_observations={},
            max_num_iterations=200,
            loss_type_normal="huber",
        )
        logger.info(
            f"BA: initial_cost={summary.initial_cost:.6e}, "
            f"final_cost={summary.final_cost:.6e}"
        )

        # Compare optimized reconstruction to GT via Sim3 alignment
        result = pycolmap.compare_reconstructions(
            rec, gt_rec,
            alignment_error="proj_center",
            max_proj_center_error=100.0,
        )
        assert result is not None, "Sim3 alignment failed"
        errors = result["errors"]

        max_rot = max(e.rotation_error_deg for e in errors)
        max_center = max(e.proj_center_error for e in errors)
        logger.info(f"max rotation error: {max_rot:.4f} deg")
        logger.info(f"max center error:   {max_center:.6f}")

        assert max_rot < 1.0, f"Max rotation error {max_rot:.4f} deg >= 1.0 deg"
        assert max_center < 0.1, f"Max center error {max_center:.6f} >= 0.1"

    def test_ba_recovers_with_2d_noise(self):
        """After adding noise to 3D points, poses, and 2D observations,
        BA should still recover a reconstruction close to the original."""
        gt_rec = create_synthetic_reconstruction(
            num_frames=8, num_points3D=100, seed=1
        )

        rec = copy.deepcopy(gt_rec)
        rng = np.random.default_rng(1)

        # Add noise to all three: 3D points, poses, and 2D observations
        perturb_points3D(rec, fraction=1.0, noise_std=0.02, rng=rng)
        self._perturb_poses(rec, translation_std=0.01, rotation_std_deg=0.5, rng=rng)
        self._perturb_points2D(rec, noise_std=1.0, rng=rng)

        # Run BA (real tracks only, Huber loss, 200 iterations)
        rec, _, summary = bundle_adjustment(
            rec,
            virtual_reconstruction=None,
            negative_depth_observations={},
            max_num_iterations=200,
            loss_type_normal="huber",
        )
        logger.info(
            f"BA (with 2D noise): initial_cost={summary.initial_cost:.6e}, "
            f"final_cost={summary.final_cost:.6e}"
        )

        # Compare optimized reconstruction to GT via Sim3 alignment.
        # With 2D noise the observations are no longer perfectly consistent,
        # so the recovered poses will not be exact — use looser thresholds.
        result = pycolmap.compare_reconstructions(
            rec, gt_rec,
            alignment_error="proj_center",
            max_proj_center_error=100.0,
        )
        assert result is not None, "Sim3 alignment failed"
        errors = result["errors"]

        max_rot = max(e.rotation_error_deg for e in errors)
        max_center = max(e.proj_center_error for e in errors)
        logger.info(f"max rotation error: {max_rot:.4f} deg")
        logger.info(f"max center error:   {max_center:.6f}")

        assert max_rot < 1.0, f"Max rotation error {max_rot:.4f} deg >= 1.0 deg"
        assert max_center < 0.1, f"Max center error {max_center:.6f} >= 0.1"
