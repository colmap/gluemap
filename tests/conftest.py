"""Pytest configuration: ensure pygluemap is loaded before pycolmap.

pygluemap (the C++ pybind11 extension) must be imported before pycolmap
to avoid a shared-library ABI conflict that causes segfaults when
iterating ``Reconstruction.points3D``.
"""
import pygluemap  # noqa: F401
