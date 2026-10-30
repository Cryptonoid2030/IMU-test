//
//  PointCloudMap.swift
//  IMU-test
//
//  Created by JohnQuavious JeanCarlo on 9/28/26.
//

import Foundation
import ARKit
import simd

/// Maintains a persistent, spatially deduplicated 3D feature point cloud map across ARKit frames
class PointCloudMap {
    private(set) var points: [simd_float3] = []

    private var persistentPointsByID: [UInt64: simd_float3] = [:]
    private var pointInsertionOrder: [UInt64] = []
    private var occupiedVoxels: [SIMD3<Int>: UInt64] = [:]

    private let voxelSize: Float
    private let maxPoints: Int

    init(voxelSize: Float = 0.02, maxPoints: Int = 40_000) {
        self.voxelSize = voxelSize
        self.maxPoints = maxPoints
    }

    /// Integrates new ARKit raw feature points into the persistent map.
    /// Returns `true` if the point cloud changed.
    @discardableResult
    func accumulate(_ rawFeatures: ARPointCloud) -> Bool {
        let rawPoints = rawFeatures.points
        let ids = rawFeatures.identifiers
        guard !rawPoints.isEmpty, rawPoints.count == ids.count else { return false }

        var didChange = false

        for i in 0..<rawPoints.count {
            let pt = rawPoints[i]
            let id = ids[i]

            guard pt.x.isFinite && pt.y.isFinite && pt.z.isFinite else { continue }

            let voxel = voxelKey(for: pt)

            if let existingPt = persistentPointsByID[id] {
                // Refine existing landmark position with new triangulation observation
                let refined = 0.7 * existingPt + 0.3 * pt
                persistentPointsByID[id] = refined
                occupiedVoxels[voxel] = id
                didChange = true
            } else if let existingIDAtVoxel = occupiedVoxels[voxel],
                      let existingPt = persistentPointsByID[existingIDAtVoxel] {
                // Merge into existing point in the same 2 cm voxel cell
                persistentPointsByID[existingIDAtVoxel] = 0.8 * existingPt + 0.2 * pt
                didChange = true
            } else {
                // Insert new 3D landmark
                persistentPointsByID[id] = pt
                occupiedVoxels[voxel] = id
                pointInsertionOrder.append(id)
                didChange = true
            }
        }

        // Enforce maximum capacity using FIFO eviction
        if pointInsertionOrder.count > maxPoints {
            let overflow = pointInsertionOrder.count - maxPoints
            for i in 0..<overflow {
                let oldID = pointInsertionOrder[i]
                if let oldPt = persistentPointsByID.removeValue(forKey: oldID) {
                    let oldVoxel = voxelKey(for: oldPt)
                    if occupiedVoxels[oldVoxel] == oldID {
                        occupiedVoxels.removeValue(forKey: oldVoxel)
                    }
                }
            }
            pointInsertionOrder.removeFirst(overflow)
            didChange = true
        }

        if didChange {
            points = Array(persistentPointsByID.values)
        }

        return didChange
    }

    func reset() {
        points.removeAll()
        persistentPointsByID.removeAll()
        pointInsertionOrder.removeAll()
        occupiedVoxels.removeAll()
    }

    private func voxelKey(for point: simd_float3) -> SIMD3<Int> {
        SIMD3<Int>(
            Int(floor(point.x / voxelSize)),
            Int(floor(point.y / voxelSize)),
            Int(floor(point.z / voxelSize))
        )
    }
}
