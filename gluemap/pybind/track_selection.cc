#include "pybind_utils.h"

#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

static inline uint64_t CanonicalPairKey(int64_t id1, int64_t id2) {
  const uint64_t lo = std::min((uint64_t)id1, (uint64_t)id2);
  const uint64_t hi = std::max((uint64_t)id1, (uint64_t)id2);
  return (lo << 32) | hi;
}

// Build cumulative offset array from per-track lengths (CSR format).
static std::vector<int64_t>
BuildCSROffsets(const std::vector<int32_t> &track_lengths) {
  const int64_t N = (int64_t)track_lengths.size();
  std::vector<int64_t> offsets(N + 1, 0);
  for (int64_t i = 0; i < N; ++i) {
    offsets[i + 1] = offsets[i] + track_lengths[i];
  }
  return offsets;
}

// For each candidate track (given by index into the CSR arrays), check whether
// any of its image pairs has coverage <= min_num_support_abs.  If so, keep the
// track and increment pair_count for all its pairs; otherwise mark for deletion.
//
// Returns the number of tracks kept.
static size_t
SelectOrRemoveTracks(const std::vector<int64_t> &candidate_idxs,
                     const std::vector<int64_t> &point3d_ids,
                     const std::vector<int64_t> &track_image_ids,
                     const std::vector<int64_t> &offsets,
                     std::unordered_map<uint64_t, int> &pair_count,
                     int min_num_support_abs,
                     std::vector<int64_t> &ids_to_remove) {
  size_t num_selected = 0;

  for (const int64_t idx : candidate_idxs) {
    const int64_t start = offsets[idx], end = offsets[idx + 1];

    bool hit = false;
    for (int64_t i = start; i < end && !hit; ++i) {
      for (int64_t j = i + 1; j < end; ++j) {
        const auto key =
            CanonicalPairKey(track_image_ids[i], track_image_ids[j]);
        const auto it = pair_count.find(key);
        if ((it != pair_count.end() ? it->second : 0) <= min_num_support_abs) {
          hit = true;
          break;
        }
      }
    }

    if (hit) {
      num_selected++;
      for (int64_t i = start; i < end; ++i) {
        for (int64_t j = i + 1; j < end; ++j) {
          pair_count[CanonicalPairKey(track_image_ids[i],
                                      track_image_ids[j])] += 1;
        }
      }
    } else {
      ids_to_remove.push_back(point3d_ids[idx]);
    }
  }

  return num_selected;
}

// ── SelectTracksToDelete ────────────────────────────────────────────────────
// Classifies tracks as SIFT / non-SIFT, seeds pair_count from SIFT tracks,
// then selectively keeps non-SIFT tracks that cover under-supported pairs.
std::pair<std::vector<int64_t>, std::unordered_map<uint64_t, int>>
SelectTracksToDelete(const std::vector<int64_t> &point3d_ids,
                     const std::vector<int64_t> &track_image_ids,
                     const std::vector<int64_t> &track_pt2d_idxs,
                     const std::vector<int32_t> &track_lengths,
                     const std::unordered_map<int64_t, int> &sift_count,
                     int min_num_support_abs) {

  const int64_t N = (int64_t)point3d_ids.size();
  const auto offsets = BuildCSROffsets(track_lengths);

  // ── Classify tracks as SIFT / non-SIFT ──
  std::vector<int64_t> sift_idxs, non_sift_idxs;

  for (int64_t i = 0; i < N; ++i) {
    bool is_sift = true;
    for (int64_t k = offsets[i]; k < offsets[i + 1]; ++k) {
      const int64_t img_id = track_image_ids[k];
      const int64_t pt2d_idx = track_pt2d_idxs[k];

      const auto sc_it = sift_count.find(img_id);
      const int s_count = (sc_it != sift_count.end()) ? sc_it->second : 0;

      if (static_cast<int>(pt2d_idx) >= s_count) {
        is_sift = false;
        break;
      }
    }

    if (is_sift)
      sift_idxs.push_back(i);
    else
      non_sift_idxs.push_back(i);
  }

  std::cout << "Track classification: " << sift_idxs.size() << " SIFT, "
            << non_sift_idxs.size() << " non-SIFT" << std::endl;

  // ── Seed pair coverage from SIFT tracks ──
  std::unordered_map<uint64_t, int> pair_count;
  for (const int64_t idx : sift_idxs) {
    const int64_t start = offsets[idx], end = offsets[idx + 1];
    for (int64_t i = start; i < end; ++i) {
      for (int64_t j = i + 1; j < end; ++j) {
        pair_count[CanonicalPairKey(track_image_ids[i], track_image_ids[j])] +=
            1;
      }
    }
  }

  // ── Shuffle and select non-SIFT tracks ──
  std::mt19937 rng(42);
  std::shuffle(non_sift_idxs.begin(), non_sift_idxs.end(), rng);

  std::vector<int64_t> ids_to_remove;
  size_t num_non_sift_selected = SelectOrRemoveTracks(
      non_sift_idxs, point3d_ids, track_image_ids, offsets, pair_count,
      min_num_support_abs, ids_to_remove);

  std::cout << "SelectTrack: kept " << sift_idxs.size() << " SIFT + "
            << num_non_sift_selected << "/" << non_sift_idxs.size()
            << " non-SIFT, removed " << ids_to_remove.size() << std::endl;

  return {std::move(ids_to_remove), std::move(pair_count)};
}

// ── SelectVirtualTracksToDelete ─────────────────────────────────────────────
// Takes an existing pair_count (e.g. from SelectTracksToDelete) and removes
// any track whose image pairs are ALL already sufficiently covered.
std::pair<std::vector<int64_t>, std::unordered_map<uint64_t, int>>
SelectVirtualTracksToDelete(
    const std::vector<int64_t> &point3d_ids,
    const std::vector<int64_t> &track_image_ids,
    const std::vector<int64_t> &track_pt2d_idxs,
    const std::vector<int32_t> &track_lengths,
    std::unordered_map<uint64_t, int> pair_count,
    int min_num_support_abs) {

  const int64_t N = (int64_t)point3d_ids.size();
  const auto offsets = BuildCSROffsets(track_lengths);

  // Shuffle track indices for randomised selection (deterministic seed)
  std::vector<int64_t> indices(N);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 rng(42);
  std::shuffle(indices.begin(), indices.end(), rng);

  std::vector<int64_t> ids_to_remove;
  size_t num_selected = SelectOrRemoveTracks(
      indices, point3d_ids, track_image_ids, offsets, pair_count,
      min_num_support_abs, ids_to_remove);

  std::cout << "SelectVirtualTrack: kept " << num_selected << "/" << N
            << ", removed " << ids_to_remove.size() << std::endl;

  return {std::move(ids_to_remove), std::move(pair_count)};
}

// ── Numpy wrappers ──────────────────────────────────────────────────────────

struct CSRArrays {
  std::vector<int64_t> point3d_ids;
  std::vector<int64_t> track_image_ids;
  std::vector<int64_t> track_pt2d_idxs;
  std::vector<int32_t> track_lengths;
};

static CSRArrays
UnpackCSR(py::array_t<int64_t, py::array::c_style> point3d_ids,
          py::array_t<int64_t, py::array::c_style> track_image_ids,
          py::array_t<int64_t, py::array::c_style> track_pt2d_idxs,
          py::array_t<int32_t, py::array::c_style> track_lengths) {
  return {
      {point3d_ids.data(), point3d_ids.data() + point3d_ids.size()},
      {track_image_ids.data(), track_image_ids.data() + track_image_ids.size()},
      {track_pt2d_idxs.data(), track_pt2d_idxs.data() + track_pt2d_idxs.size()},
      {track_lengths.data(), track_lengths.data() + track_lengths.size()},
  };
}

py::tuple ComputeTracksToDeleteWrapper(
    py::array_t<int64_t, py::array::c_style> point3d_ids,
    py::array_t<int64_t, py::array::c_style> track_image_ids,
    py::array_t<int64_t, py::array::c_style> track_pt2d_idxs,
    py::array_t<int32_t, py::array::c_style> track_lengths,
    const std::unordered_map<int64_t, int> &sift_count,
    int min_num_support_abs) {

  auto csr = UnpackCSR(point3d_ids, track_image_ids, track_pt2d_idxs,
                        track_lengths);
  auto result = SelectTracksToDelete(csr.point3d_ids, csr.track_image_ids,
                                     csr.track_pt2d_idxs, csr.track_lengths,
                                     sift_count, min_num_support_abs);

  return py::make_tuple(VecToArray1D(std::move(result.first)),
                        std::move(result.second));
}

py::tuple ComputeVirtualTracksToDeleteWrapper(
    py::array_t<int64_t, py::array::c_style> point3d_ids,
    py::array_t<int64_t, py::array::c_style> track_image_ids,
    py::array_t<int64_t, py::array::c_style> track_pt2d_idxs,
    py::array_t<int32_t, py::array::c_style> track_lengths,
    const std::unordered_map<uint64_t, int> &pair_count_in,
    int min_num_support_abs) {

  auto csr = UnpackCSR(point3d_ids, track_image_ids, track_pt2d_idxs,
                        track_lengths);
  auto result = SelectVirtualTracksToDelete(
      csr.point3d_ids, csr.track_image_ids, csr.track_pt2d_idxs,
      csr.track_lengths, pair_count_in, min_num_support_abs);

  return py::make_tuple(VecToArray1D(std::move(result.first)),
                        std::move(result.second));
}
