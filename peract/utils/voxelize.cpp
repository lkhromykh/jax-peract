#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "Eigen/Dense"
#include "open3d/Open3D.h"

template<typename T>
using Points = Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>;
using DenseVoxelGrid = Eigen::Matrix<uint8_t, Eigen::Dynamic, 4, Eigen::RowMajor>;

int _ravel_multi_index(const Eigen::Ref<const Eigen::Vector3i>& multi_index, int size)
{
    int flat_idx = 0;
    for (auto& idx: multi_index)
        flat_idx = size * flat_idx + idx;
    return flat_idx;
}

const DenseVoxelGrid
create_dense_voxel_grid_from_points(
    Eigen::Ref<Points<double>> points,
    Eigen::Ref<Points<uint8_t>> colors,
    std::pair<Eigen::Array3d, Eigen::Array3d> scene_bounds,
    int num_voxels
    )
{
    int npts = points.rows();
    auto &[min_bound, max_bound] = scene_bounds;
    points = (points.array().rowwise() - min_bound.transpose()).rowwise() / (max_bound - min_bound).transpose();

    auto pcd = std::make_shared<open3d::geometry::PointCloud>();
    pcd->points_.reserve(npts);
    pcd->colors_.reserve(npts);
    auto double_points = points.cast<double>();
    auto norm_colors = colors.cast<double>() / 255;
    for (int i = 0; i < npts; ++i)
    {
        pcd->points_.push_back(double_points.row(i));
        pcd->colors_.push_back(norm_colors.row(i));
    }
    min_bound.fill(0);
    max_bound.fill(1 - 1e-5);
    pcd = pcd->Crop(open3d::geometry::AxisAlignedBoundingBox(min_bound, max_bound));
    auto vgrid = open3d::geometry::VoxelGrid::CreateFromPointCloudWithinBounds(
            *pcd, 1. / num_voxels, min_bound, max_bound);

    auto dense_vgrid = DenseVoxelGrid(num_voxels * num_voxels * num_voxels, 4);
    dense_vgrid.setZero();
    for (auto &voxel: vgrid->GetVoxels())
    {
        auto occupied = Eigen::Vector<uint8_t, 4>();
        occupied << (255 * voxel.color_).cast<uint8_t>(), 255;
        dense_vgrid.row(_ravel_multi_index(voxel.grid_index_, num_voxels)) = std::move(occupied);
    }
    return dense_vgrid;
}


using namespace pybind11::literals;
PYBIND11_MODULE(voxelize, m)
{
    m.def(
        "create_dense_voxel_grid_from_points",
        &create_dense_voxel_grid_from_points,
        "Process point cloud and colors to a voxel grid",
        "points"_a.noconvert(), "colors"_a.noconvert(), "scene_bounds"_a,  "num_voxels"_a
    );
}
