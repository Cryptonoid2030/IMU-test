//
//  SLAMOptimizer.swift
//  IMU-test
//
//  Created by JohnQuavious JeanCarlo on 9/28/26.
//

import Foundation
import Accelerate
import simd

// MARK: - Accelerate Sparse Cholesky Solver
class SLAMOptimizer {
    /// Solves the normal equation H * dx = b using Sparse Cholesky Decomposition
    func solveGaussNewton(size: Int32, hRows: [Int32], hCols: [Int32], hVals: [Double], b: [Double]) -> [Double]? {
        guard size > 0, !hVals.isEmpty, b.count == Int(size) else { return nil }

        var rowIndices = hRows
        var colIndices = hCols
        var values = hVals

        var attributes = SparseAttributes_t()
        attributes.triangle = SparseLowerTriangle
        attributes.kind = SparseSymmetric

        let H_COO = SparseConvertFromCoordinate(
            size, size,
            values.count,
            1,
            attributes,
            &rowIndices,
            &colIndices,
            &values
        )
        defer { SparseCleanup(H_COO) }

        let factorization = SparseFactor(SparseFactorizationCholesky, H_COO)
        defer { SparseCleanup(factorization) }

        guard factorization.status == SparseStatusOK else {
            print("Accelerate Matrix Solver Failed with status: \(factorization.status)")
            return nil
        }

        var bVector = b
        var dxVector = Array(repeating: 0.0, count: Int(size))

        bVector.withUnsafeMutableBufferPointer { bPtr in
            dxVector.withUnsafeMutableBufferPointer { dxPtr in
                guard let bBase = bPtr.baseAddress, let dxBase = dxPtr.baseAddress else { return }
                let bDense = DenseVector_Double(count: size, data: bBase)
                let dxDense = DenseVector_Double(count: size, data: dxBase)
                SparseSolve(factorization, bDense, dxDense)
            }
        }

        return dxVector
    }
}

// MARK: - Graph SLAM Data Structures

/// A node in the pose graph representing a keyframe pose
public struct PoseGraphNode: Sendable {
    public let id: Int
    public var position: simd_float3
    public var orientation: simd_quatf
    public var isFixed: Bool

    public init(id: Int, position: simd_float3, orientation: simd_quatf, isFixed: Bool = false) {
        self.id = id
        self.position = position
        self.orientation = orientation
        self.isFixed = isFixed
    }
}

/// An edge (factor constraint) connecting two keyframe nodes in the pose graph
public struct PoseGraphEdge: Sendable {
    public let fromId: Int
    public let toId: Int
    public let measuredRelativeTranslation: simd_float3
    public let measuredRelativeRotation: simd_quatf
    public var informationWeight: Float
    public let isLoopClosure: Bool

    public init(
        fromId: Int,
        toId: Int,
        measuredRelativeTranslation: simd_float3,
        measuredRelativeRotation: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1),
        informationWeight: Float = 1.0,
        isLoopClosure: Bool = false
    ) {
        self.fromId = fromId
        self.toId = toId
        self.measuredRelativeTranslation = measuredRelativeTranslation
        self.measuredRelativeRotation = measuredRelativeRotation
        self.informationWeight = informationWeight
        self.isLoopClosure = isLoopClosure
    }
}

/// Configuration and tuning hyperparameters for Graph SLAM
public struct GraphSLAMConfig: Sendable {
    /// Number of Gauss-Newton / IRLS iterations
    public var maxIterations: Int = 3

    /// Relative weight of loop closure constraints relative to odometry (0.01 ... 1.0)
    public var loopWeight: Float = 0.35

    /// Base weight of sequential odometry constraints
    public var odometryWeight: Float = 1.0

    /// Threshold for Huber M-estimator robust loss in meters (downweights large outlier loop residuals)
    public var huberThreshold: Float = 0.35

    /// Maximum displacement allowed on the live tracking pose per loop closure (meters)
    public var maxDriftClamp: Float = 0.50

    /// Damping ratio when propagating graph correction to current live odometry (0.0 ... 1.0)
    public var livePoseDamping: Float = 0.40

    public init(
        maxIterations: Int = 3,
        loopWeight: Float = 0.35,
        odometryWeight: Float = 1.0,
        huberThreshold: Float = 0.35,
        maxDriftClamp: Float = 0.50,
        livePoseDamping: Float = 0.40
    ) {
        self.maxIterations = maxIterations
        self.loopWeight = loopWeight
        self.odometryWeight = odometryWeight
        self.huberThreshold = huberThreshold
        self.maxDriftClamp = maxDriftClamp
        self.livePoseDamping = livePoseDamping
    }
}

// MARK: - Graph SLAM Non-Linear Least Squares Optimizer
/// Full Graph SLAM (Pose Graph Optimizer) utilizing Accelerate Sparse Cholesky Decomposition
public final class GraphSLAMOptimizer: @unchecked Sendable {
    private let solver = SLAMOptimizer()

    public init() {}

    /// Solves the Pose Graph Optimization problem using Gauss-Newton with Huber robust cost on loop edges
    public func optimize(
        nodes: inout [PoseGraphNode],
        edges: [PoseGraphEdge],
        config: GraphSLAMConfig
    ) -> Bool {
        let n = nodes.count
        guard n >= 2, !edges.isEmpty else { return false }

        let stateDim = Int32(3 * n)

        for _ in 0..<config.maxIterations {
            // Aggregate symmetric sparse matrix entries in COO format (lower triangle: row >= col)
            var matrixEntries: [Int64: Double] = [:]
            var bVector = Array(repeating: 0.0, count: Int(stateDim))

            // 1. Anchor node 0 at world origin with strong gauge prior
            let priorWeight: Double = 1e5
            for dim in 0..<3 {
                let key = (Int64(dim) << 32) | Int64(dim)
                matrixEntries[key, default: 0.0] += priorWeight
            }

            // Small Levenberg-Marquardt diagonal damping to guarantee positive-definiteness
            let lambdaDamping: Double = 1e-4
            for i in 0..<Int(stateDim) {
                let key = (Int64(i) << 32) | Int64(i)
                matrixEntries[key, default: 0.0] += lambdaDamping
            }

            // 2. Formulate constraints for all graph edges
            for edge in edges {
                guard edge.fromId < n && edge.toId < n else { continue }
                let u = min(edge.fromId, edge.toId)
                let v = max(edge.fromId, edge.toId)
                guard u != v else { continue }

                let pU = nodes[u].position
                let pV = nodes[v].position

                // Residual error: r = (p_v - p_u) - measuredRelativeTranslation
                // (respecting direction of edge)
                let expectedDiff = (edge.toId > edge.fromId)
                    ? edge.measuredRelativeTranslation
                    : -edge.measuredRelativeTranslation
                let residual = (pV - pU) - expectedDiff
                let resNorm = length(residual)

                // Apply Huber robust cost function to loop closures to prevent outlier pull
                var effectiveWeight = edge.isLoopClosure ? (edge.informationWeight * config.loopWeight) : config.odometryWeight
                if edge.isLoopClosure && resNorm > config.huberThreshold && resNorm > 1e-5 {
                    // Huber weight scale: threshold / ||r||
                    effectiveWeight *= (config.huberThreshold / resNorm)
                }

                let w = Double(effectiveWeight)

                // Edge Jacobian contributions: J_u = -I, J_v = +I
                // H_uu += w * I, H_vv += w * I, H_vu -= w * I (lower triangle: v > u)
                for d in 0..<3 {
                    let uIdx = Int64(3 * u + d)
                    let vIdx = Int64(3 * v + d)

                    // Diagonal blocks
                    let keyUU = (uIdx << 32) | uIdx
                    let keyVV = (vIdx << 32) | vIdx
                    matrixEntries[keyUU, default: 0.0] += w
                    matrixEntries[keyVV, default: 0.0] += w

                    // Off-diagonal block (row = vIdx, col = uIdx, since v > u)
                    let keyVU = (vIdx << 32) | uIdx
                    matrixEntries[keyVU, default: 0.0] -= w

                    // Right hand side: b = -J^T * Omega * r
                    // b_u += w * residual[d], b_v -= w * residual[d]
                    let resD = Double(residual[d])
                    bVector[Int(uIdx)] += w * resD
                    bVector[Int(vIdx)] -= w * resD
                }
            }

            // 3. Flatten sparse matrix entries into COO arrays
            var hRows: [Int32] = []
            var hCols: [Int32] = []
            var hVals: [Double] = []
            hRows.reserveCapacity(matrixEntries.count)
            hCols.reserveCapacity(matrixEntries.count)
            hVals.reserveCapacity(matrixEntries.count)

            for (key, val) in matrixEntries {
                let r = Int32(key >> 32)
                let c = Int32(key & 0xFFFFFFFF)
                hRows.append(r)
                hCols.append(c)
                hVals.append(val)
            }

            // 4. Solve normal equations via Accelerate Sparse Cholesky
            guard let dx = solver.solveGaussNewton(size: stateDim, hRows: hRows, hCols: hCols, hVals: hVals, b: bVector) else {
                return false
            }

            // 5. Update node positions
            for i in 0..<n {
                if nodes[i].isFixed { continue }
                let dxX = Float(dx[3 * i])
                let dxY = Float(dx[3 * i + 1])
                let dxZ = Float(dx[3 * i + 2])
                nodes[i].position += simd_float3(dxX, dxY, dxZ)
            }
        }

        return true
    }
}
