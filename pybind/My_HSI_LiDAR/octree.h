#ifndef OCTREE_H
#define OCTREE_H

#include <iostream>
#include <memory>
#include <thread>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <string>

#include <Eigen/Core>
#include <cstddef>
#include <memory>
#include <vector>

#include <json/json.h>
#include <Eigen/Dense>
#include <algorithm>
#include <unordered_map>

#include "PointCloud.h"
//#include "IJsonConvertible.h"

namespace PPP {
namespace geometry {

//class PointCloud;

//class VoxelGrid;

/// Design decision: do not store origin and size of a node in OctreeNode
/// OctreeNodeInfo is computed on the fly
class OctreeNodeInfo {
public:
    OctreeNodeInfo() : origin_(0, 0, 0), size_(0), depth_(0), child_index_(0) {}
    OctreeNodeInfo(const Eigen::Vector3d& origin,
                   const double& size,
                   const size_t& depth,
                   const size_t& child_index)
        : origin_(origin),
          size_(size),
          depth_(depth),
          child_index_(child_index) {}
    ~OctreeNodeInfo() {}

public:
    Eigen::Vector3d origin_;
    double size_;
    size_t depth_;
    size_t child_index_;
};

/// OctreeNode class
/// Design decision: do not store origin and size of a node
///     - Good: better space efficiency
///     - Bad: need to recompute origin and size when traversing
class OctreeNode  {
public:
    OctreeNode() {}
    virtual ~OctreeNode() {}

    /// Factory function to construct an OctreeNode by parsing the json value.
//    static std::shared_ptr<OctreeNode> ConstructFromJsonValue(
//            const Json::Value& value);
};

/// Children node ordering conventions are as follows.
///
/// For illustration, assume,
/// - root_node: origin == (0, 0, 0), size == 2
///
/// Then,
/// - children_[0]: origin == (0, 0, 0), size == 1
/// - children_[1]: origin == (1, 0, 0), size == 1, along X-axis next to child 0
/// - children_[2]: origin == (0, 1, 0), size == 1, along Y-axis next to child 0
/// - children_[3]: origin == (1, 1, 0), size == 1, in X-Y plane
/// - children_[4]: origin == (0, 0, 1), size == 1, along Z-axis next to child 0
/// - children_[5]: origin == (1, 0, 1), size == 1, in X-Z plane
/// - children_[6]: origin == (0, 1, 1), size == 1, in Y-Z plane
/// - children_[7]: origin == (1, 1, 1), size == 1, furthest from child 0
class OctreeInternalNode : public OctreeNode {
public:
    OctreeInternalNode() : children_(8) {}
    static std::shared_ptr<OctreeNodeInfo> GetInsertionNodeInfo(
            const std::shared_ptr<OctreeNodeInfo>& node_info,
            const Eigen::Vector3d& point);

//    bool ConvertToJsonValue(Json::Value& value) const override;
//    bool ConvertFromJsonValue(const Json::Value& value) override;

public:
    // Use vector instead of C-array for Pybind11, otherwise, need to define
    // more helper functions
    // https://github.com/pybind/pybind11/issues/546#issuecomment-265707318
    std::vector<std::shared_ptr<OctreeNode>> children_;
};

class OctreeLeafNode : public OctreeNode {
public:
    virtual bool operator==(const OctreeLeafNode& other) const = 0;
    virtual std::shared_ptr<OctreeLeafNode> Clone() const = 0;
};

class OctreeColorLeafNode : public OctreeLeafNode {
public:
    bool operator==(const OctreeLeafNode& other) const override;
    std::shared_ptr<OctreeLeafNode> Clone() const override;
    static std::function<std::shared_ptr<OctreeLeafNode>()> GetInitFunction();
    static std::function<void(std::shared_ptr<OctreeLeafNode>)>
    GetUpdateFunction(const Eigen::Vector3d& color);

//    bool ConvertToJsonValue(Json::Value& value) const override;
//    bool ConvertFromJsonValue(const Json::Value& value) override;
    // TODO: flexible data, with lambda function for handling node
    Eigen::Vector3d color_ = Eigen::Vector3d(0, 0, 0);
};

class Octree {
public:
    Octree():
          origin_(0, 0, 0),
          size_(0),
          max_depth_(0) {}
    Octree(const Octree& src_octree);
    ~Octree() {}

public:
    Octree& Clear() ;
    bool IsEmpty() const ;
    Eigen::Vector3d GetMinBound() const ;
    Eigen::Vector3d GetMaxBound() const ;
    Eigen::Vector3d GetCenter() const ;
//    AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const override;
//    OrientedBoundingBox GetOrientedBoundingBox() const override;
    Octree& Transform(const Eigen::Matrix4d& transformation) ;
    Octree& Translate(const Eigen::Vector3d& translation,
                      bool relative = true) ;
    Octree& Scale(const double scale, bool center = true) ;
    Octree& Rotate(const Eigen::Matrix3d& R, bool center = true) ;
//    bool ConvertToJsonValue(Json::Value& value) const override;
//    bool ConvertFromJsonValue(const Json::Value& value) override;

public:
    void ConvertFromPointCloud(const PointCloud& point_cloud,
                               double size_expand = 0.01);

    /// Root of the octree
    std::shared_ptr<OctreeNode> root_node_ = nullptr;

    /// Global min bound (include). A point is within bound iff
    /// origin_ <= point < origin_ + size_
    Eigen::Vector3d origin_;

    /// Outer bounding box edge size for the whole octree. A point is within
    /// bound iff origin_ <= point < origin_ + size_
    double size_;

    /// Max depth of octree. The depth is defined as the distance from the
    /// deepest leaf node to root. A tree with only the root node has depth 0.
    size_t max_depth_;

    /// Insert point
    void InsertPoint(
            const Eigen::Vector3d& point,
            const std::function<std::shared_ptr<OctreeLeafNode>()>& f_init,
            const std::function<void(std::shared_ptr<OctreeLeafNode>)>&
                    f_update);

    /// DFS traversal of Octree from the root, with callback function called
    /// for each node
    void Traverse(
            const std::function<void(const std::shared_ptr<OctreeNode>&,
                                     const std::shared_ptr<OctreeNodeInfo>&)>&
                    f);

    /// Const version of Traverse. DFS traversal of Octree from the root, with
    /// callback function called for each node
    void Traverse(
            const std::function<void(const std::shared_ptr<OctreeNode>&,
                                     const std::shared_ptr<OctreeNodeInfo>&)>&
                    f) const;

    std::pair<std::shared_ptr<OctreeLeafNode>, std::shared_ptr<OctreeNodeInfo>>
    LocateLeafNode(const Eigen::Vector3d& point) const;

    /// Return true if point within bound, that is,
    /// origin <= point < origin + size
    static bool IsPointInBound(const Eigen::Vector3d& point,
                               const Eigen::Vector3d& origin,
                               const double& size);

    /// Returns true if the Octree is completely the same, used for testing
    bool operator==(const Octree& other) const;

    /// Convert to voxel grid
//    std::shared_ptr<geometry::VoxelGrid> ToVoxelGrid() const;

    /// Convert from voxel grid
//    void CreateFromVoxelGrid(const geometry::VoxelGrid& voxel_grid);

private:
    static void TraverseRecurse(
            const std::shared_ptr<OctreeNode>& node,
            const std::shared_ptr<OctreeNodeInfo>& node_info,
            const std::function<void(const std::shared_ptr<OctreeNode>&,
                                     const std::shared_ptr<OctreeNodeInfo>&)>&
                    f);

    void InsertPointRecurse(
            const std::shared_ptr<OctreeNode>& node,
            const std::shared_ptr<OctreeNodeInfo>& node_info,
            const Eigen::Vector3d& point,
            const std::function<std::shared_ptr<OctreeLeafNode>()>& f_init,
            const std::function<void(std::shared_ptr<OctreeLeafNode>)>&
                    f_update);
};

//=============================================================================================
//=============================================================================================
//=============================================================================================
std::shared_ptr<OctreeNodeInfo> OctreeInternalNode::GetInsertionNodeInfo(
        const std::shared_ptr<OctreeNodeInfo>& node_info,
        const Eigen::Vector3d& point) {
    if (!Octree::IsPointInBound(point, node_info->origin_, node_info->size_)) {
        assert(
                "Internal error: cannot insert to child since point not in "
                "parent node bound.");
    }

    double child_size = node_info->size_ / 2.0;
    size_t x_index = point(0) < node_info->origin_(0) + child_size ? 0 : 1;
    size_t y_index = point(1) < node_info->origin_(1) + child_size ? 0 : 1;
    size_t z_index = point(2) < node_info->origin_(2) + child_size ? 0 : 1;
    size_t child_index = x_index + y_index * 2 + z_index * 4;
    Eigen::Vector3d child_origin =
            node_info->origin_ + Eigen::Vector3d(x_index * child_size,
                                                 y_index * child_size,
                                                 z_index * child_size);
    auto child_node_info = std::make_shared<OctreeNodeInfo>(
            child_origin, child_size, node_info->depth_ + 1, child_index);
    return child_node_info;
}

std::function<std::shared_ptr<OctreeLeafNode>()>
OctreeColorLeafNode::GetInitFunction() {
    return []() -> std::shared_ptr<geometry::OctreeLeafNode> {
        return std::make_shared<geometry::OctreeColorLeafNode>();
    };
}

std::function<void(std::shared_ptr<OctreeLeafNode>)>
OctreeColorLeafNode::GetUpdateFunction(const Eigen::Vector3d& color) {
    // Here the captured "color" cannot be a refernce, must be a copy,
    // otherwise pybind does not have the correct value
    return [color](std::shared_ptr<geometry::OctreeLeafNode> node) -> void {
        if (auto color_leaf_node =
                    std::dynamic_pointer_cast<geometry::OctreeColorLeafNode>(
                            node)) {
            color_leaf_node->color_ = color;
        } else {
           assert(
                    "Internal error: leaf node must be OctreeLeafNode");
        }
    };
}

std::shared_ptr<OctreeLeafNode> OctreeColorLeafNode::Clone() const {
    auto cloned_node = std::make_shared<OctreeColorLeafNode>();
    cloned_node->color_ = color_;
    return cloned_node;
}

bool OctreeColorLeafNode::operator==(const OctreeLeafNode& that) const {
    try {
        const OctreeColorLeafNode& that_color_node =
                dynamic_cast<const OctreeColorLeafNode&>(that);
        return this->color_.isApprox(that_color_node.color_);
    } catch (const std::exception&) {
        return false;
    }
}

Octree::Octree(const Octree& src_octree)
    : origin_(src_octree.origin_),
      size_(src_octree.size_),
      max_depth_(src_octree.max_depth_) {
    // First traversal: clone nodes without edges
    std::unordered_map<std::shared_ptr<OctreeNode>, std::shared_ptr<OctreeNode>>
            map_src_to_dst_node;
    auto f_build_map =
            [&map_src_to_dst_node](
                    const std::shared_ptr<OctreeNode>& src_node,
                    const std::shared_ptr<OctreeNodeInfo>& src_node_info)
            -> void {
        if (auto src_internal_node =
                    std::dynamic_pointer_cast<OctreeInternalNode>(src_node)) {
            auto dst_internal_node = std::make_shared<OctreeInternalNode>();
            map_src_to_dst_node[src_internal_node] = dst_internal_node;
        } else if (auto src_leaf_node =
                           std::dynamic_pointer_cast<OctreeLeafNode>(
                                   src_node)) {
            map_src_to_dst_node[src_leaf_node] = src_leaf_node->Clone();
        } else {
            assert("Internal error: unknown node type");
        }
    };
    src_octree.Traverse(f_build_map);

    // Second traversal: add edges
    auto f_clone_edges =
            [&map_src_to_dst_node](
                    const std::shared_ptr<OctreeNode>& src_node,
                    const std::shared_ptr<OctreeNodeInfo>& src_node_info)
            -> void {
        if (auto src_internal_node =
                    std::dynamic_pointer_cast<OctreeInternalNode>(src_node)) {
            auto dst_internal_node =
                    std::dynamic_pointer_cast<OctreeInternalNode>(
                            map_src_to_dst_node.at(src_internal_node));
            for (size_t child_index = 0; child_index < 8; child_index++) {
                auto src_child_node = src_internal_node->children_[child_index];
                if (src_child_node != nullptr) {
                    auto dst_child_node =
                            map_src_to_dst_node.at(src_child_node);
                    dst_internal_node->children_[child_index] = dst_child_node;
                }
            }
        }
    };
    src_octree.Traverse(f_clone_edges);

    // Save root node to dst_octree (this octree)
    root_node_ = map_src_to_dst_node.at(src_octree.root_node_);
}

bool Octree::operator==(const Octree& that) const {
    // Check basic properties
    bool rc = true;
    rc = rc && origin_.isApprox(that.origin_);
    rc = rc && size_ == that.size_;
    rc = rc && max_depth_ == that.max_depth_;
    if (!rc) {
        return rc;
    }

    // Assign and check node ids
    std::unordered_map<std::shared_ptr<OctreeNode>, size_t> map_node_to_id;
    std::unordered_map<size_t, std::shared_ptr<OctreeNode>> map_id_to_node;
    size_t next_id = 0;
    auto f_assign_node_id =
            [&map_node_to_id, &map_id_to_node, &next_id](
                    const std::shared_ptr<OctreeNode>& node,
                    const std::shared_ptr<OctreeNodeInfo>& node_info) -> void {
        map_node_to_id[node] = next_id;
        map_id_to_node[next_id] = node;
        next_id++;
    };

    map_node_to_id.clear();
    map_id_to_node.clear();
    next_id = 0;
    Traverse(f_assign_node_id);
    std::unordered_map<std::shared_ptr<OctreeNode>, size_t>
            this_map_node_to_id = map_node_to_id;
    std::unordered_map<size_t, std::shared_ptr<OctreeNode>>
            this_map_id_to_node = map_id_to_node;
    size_t num_nodes = next_id;

    map_node_to_id.clear();
    map_id_to_node.clear();
    next_id = 0;
    that.Traverse(f_assign_node_id);
    std::unordered_map<std::shared_ptr<OctreeNode>, size_t>
            that_map_node_to_id = map_node_to_id;
    std::unordered_map<size_t, std::shared_ptr<OctreeNode>>
            that_map_id_to_node = map_id_to_node;

    rc = rc && this_map_node_to_id.size() == num_nodes &&
         that_map_node_to_id.size() == num_nodes &&
         this_map_id_to_node.size() == num_nodes &&
         that_map_id_to_node.size() == num_nodes;
    if (!rc) {
        return rc;
    }

    // Check nodes
    for (size_t id = 0; id < num_nodes; ++id) {
        std::shared_ptr<OctreeNode> this_node = this_map_id_to_node.at(id);
        std::shared_ptr<OctreeNode> that_node = that_map_id_to_node.at(id);
        // Check if internal node
        auto is_same_node_type = false;
        auto this_internal_node =
                std::dynamic_pointer_cast<OctreeInternalNode>(this_node);
        auto that_internal_node =
                std::dynamic_pointer_cast<OctreeInternalNode>(that_node);
        if (this_internal_node != nullptr && that_internal_node != nullptr) {
            is_same_node_type = true;
            for (size_t child_index = 0; child_index < 8; child_index++) {
                const std::shared_ptr<OctreeNode>& this_child =
                        this_internal_node->children_[child_index];
                int this_child_id = -1;
                if (this_child != nullptr) {
                    this_child_id = int(this_map_node_to_id.at(this_child));
                }
                const std::shared_ptr<OctreeNode>& that_child =
                        that_internal_node->children_[child_index];
                int that_child_id = -1;
                if (that_child != nullptr) {
                    that_child_id = int(that_map_node_to_id.at(that_child));
                }
                rc = rc && this_child_id == that_child_id;
            }
        }
        // Check if leaf node
        auto this_leaf_node =
                std::dynamic_pointer_cast<OctreeLeafNode>(this_node);
        auto that_leaf_node =
                std::dynamic_pointer_cast<OctreeLeafNode>(that_node);
        if (this_leaf_node != nullptr && that_leaf_node != nullptr) {
            is_same_node_type = true;
            rc = rc && (*this_leaf_node) == (*that_leaf_node);
        }
        // Handle case where node types are different
        rc = rc && is_same_node_type;
    }

    return rc;
}

Octree& Octree::Clear() {
    // Inherited Clear function
    root_node_ = nullptr;
    origin_.setZero();
    size_ = 0;
    return *this;
}

bool Octree::IsEmpty() const { return root_node_ == nullptr; }

Eigen::Vector3d Octree::GetMinBound() const {
    if (IsEmpty()) {
        return Eigen::Vector3d::Zero();
    } else {
        return origin_;
    }
}

Eigen::Vector3d Octree::GetMaxBound() const {
    if (IsEmpty()) {
        return Eigen::Vector3d::Zero();
    } else {
        return origin_ + Eigen::Vector3d(size_, size_, size_);
    }
}

Eigen::Vector3d Octree::GetCenter() const {
    return origin_ + Eigen::Vector3d(size_, size_, size_) / 2;
}


void Octree::ConvertFromPointCloud(const PointCloud& point_cloud,
                                   double size_expand) {
    if (size_expand > 1 || size_expand < 0) {
        assert("size_expand shall be between 0 and 1");
    }

    // Set bounds
    Clear();
    Eigen::Array3d min_bound = point_cloud.GetMinBound();
    Eigen::Array3d max_bound = point_cloud.GetMaxBound();
    Eigen::Array3d center = (min_bound + max_bound) / 2;
    Eigen::Array3d half_sizes = center - min_bound;
    double max_half_size = half_sizes.maxCoeff();
    origin_ = min_bound.min(center - max_half_size);
    if (max_half_size == 0) {
        size_ = size_expand;
    } else {
        size_ = max_half_size * 2 * (1 + size_expand);
    }

    // Insert points
    for (size_t idx = 0; idx < point_cloud.points_.size(); idx++) {
        InsertPoint(point_cloud.points_[idx],
                    geometry::OctreeColorLeafNode::GetInitFunction(),
                    geometry::OctreeColorLeafNode::GetUpdateFunction(
                            point_cloud.colors_[idx]));
    }
}

void Octree::InsertPoint(
        const Eigen::Vector3d& point,
        const std::function<std::shared_ptr<OctreeLeafNode>()>& f_init,
        const std::function<void(std::shared_ptr<OctreeLeafNode>)>& f_update) {
    if (root_node_ == nullptr) {
        if (max_depth_ == 0) {
            root_node_ = f_init();
        } else {
            root_node_ = std::make_shared<OctreeInternalNode>();
        }
    }
    auto root_node_info =
            std::make_shared<OctreeNodeInfo>(origin_, size_, 0, 0);
    InsertPointRecurse(root_node_, root_node_info, point, f_init, f_update);
}

void Octree::InsertPointRecurse(
        const std::shared_ptr<OctreeNode>& node,
        const std::shared_ptr<OctreeNodeInfo>& node_info,
        const Eigen::Vector3d& point,
        const std::function<std::shared_ptr<OctreeLeafNode>()>& f_init,
        const std::function<void(std::shared_ptr<OctreeLeafNode>)>& f_update) {
    if (!IsPointInBound(point, node_info->origin_, node_info->size_)) {
        return;
    }
    if (node_info->depth_ > max_depth_) {
        return;
    } else if (node_info->depth_ == max_depth_) {
        if (auto leaf_node = std::dynamic_pointer_cast<OctreeLeafNode>(node)) {
            f_update(leaf_node);
        } else {
            assert(
                    "Internal error: leaf node must be OctreeLeafNode");
        }
    } else {
        if (auto internal_node =
                    std::dynamic_pointer_cast<OctreeInternalNode>(node)) {
            // Get child node info
            std::shared_ptr<OctreeNodeInfo> child_node_info =
                    internal_node->GetInsertionNodeInfo(node_info, point);

            // Create child node with factory function
            size_t child_index = child_node_info->child_index_;
            if (internal_node->children_[child_index] == nullptr) {
                if (node_info->depth_ == max_depth_ - 1) {
                    internal_node->children_[child_index] = f_init();
                } else {
                    internal_node->children_[child_index] =
                            std::make_shared<OctreeInternalNode>();
                }
            }

            // Insert to the child
            InsertPointRecurse(internal_node->children_[child_index],
                               child_node_info, point, f_init, f_update);
        } else {
            assert(
                    "Internal error: internal node must be "
                    "OctreeInternalNode");
        }
    }
}

bool Octree::IsPointInBound(const Eigen::Vector3d& point,
                            const Eigen::Vector3d& origin,
                            const double& size) {
    bool rc = (Eigen::Array3d(origin) <= Eigen::Array3d(point)).all() &&
              (Eigen::Array3d(point) < Eigen::Array3d(origin) + size).all();
    return rc;
}

void Octree::Traverse(
        const std::function<void(const std::shared_ptr<OctreeNode>&,
                                 const std::shared_ptr<OctreeNodeInfo>&)>& f) {
    // root_node_'s child index is 0, though it isn't a child node
    TraverseRecurse(root_node_,
                    std::make_shared<OctreeNodeInfo>(origin_, size_, 0, 0), f);
}

void Octree::Traverse(
        const std::function<void(const std::shared_ptr<OctreeNode>&,
                                 const std::shared_ptr<OctreeNodeInfo>&)>& f)
        const {
    // root_node_'s child index is 0, though it isn't a child node
    TraverseRecurse(root_node_,
                    std::make_shared<OctreeNodeInfo>(origin_, size_, 0, 0), f);
}

void Octree::TraverseRecurse(
        const std::shared_ptr<OctreeNode>& node,
        const std::shared_ptr<OctreeNodeInfo>& node_info,
        const std::function<void(const std::shared_ptr<OctreeNode>&,
                                 const std::shared_ptr<OctreeNodeInfo>&)>& f) {
    if (node == nullptr) {
        return;
    } else if (auto internal_node =
                       std::dynamic_pointer_cast<OctreeInternalNode>(node)) {
        f(internal_node, node_info);
        double child_size = node_info->size_ / 2.0;

        for (size_t child_index = 0; child_index < 8; ++child_index) {
            size_t x_index = child_index % 2;
            size_t y_index = (child_index / 2) % 2;
            size_t z_index = (child_index / 4) % 2;

            auto child_node = internal_node->children_[child_index];
            Eigen::Vector3d child_node_origin =
                    node_info->origin_ + Eigen::Vector3d(double(x_index),
                                                         double(y_index),
                                                         double(z_index)) *
                                                 child_size;
            auto child_node_info = std::make_shared<OctreeNodeInfo>(
                    child_node_origin, child_size, node_info->depth_ + 1,
                    child_index);
            TraverseRecurse(child_node, child_node_info, f);
        }
    } else if (auto leaf_node =
                       std::dynamic_pointer_cast<OctreeLeafNode>(node)) {
        f(leaf_node, node_info);
    } else {
        assert("Internal error: unknown node type");
    }
}

std::pair<std::shared_ptr<OctreeLeafNode>, std::shared_ptr<OctreeNodeInfo>>
Octree::LocateLeafNode(const Eigen::Vector3d& point) const {
    // TODO: add early stoping to callback function when the target has been
    //       found, i.e. add return value to callback function.
    // TODO: consider adding node's depth as parameter to the callback
    // function.
    std::shared_ptr<OctreeLeafNode> target_leaf_node = nullptr;
    std::shared_ptr<OctreeNodeInfo> target_leaf_node_info = nullptr;
    auto f_locate_leaf_node =
            [&target_leaf_node, &target_leaf_node_info, &point](
                    const std::shared_ptr<OctreeNode>& node,
                    const std::shared_ptr<OctreeNodeInfo>& node_info) -> void {
        if (IsPointInBound(point, node_info->origin_, node_info->size_)) {
            if (auto leaf_node =
                        std::dynamic_pointer_cast<OctreeLeafNode>(node)) {
                target_leaf_node = leaf_node;
                target_leaf_node_info = node_info;
            }
        }
    };
    Traverse(f_locate_leaf_node);
    return std::make_pair(target_leaf_node, target_leaf_node_info);
}

}  // namespace geometry
}  // namespace PPP

#endif //OCTREE_H
