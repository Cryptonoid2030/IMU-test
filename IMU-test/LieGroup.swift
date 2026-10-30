//
//  LieGroup.swift
//  IMU-test
//
//  Created by JohnQuavious JeanCarlo on 9/28/26.
//

import Foundation
import simd

// MARK: - SO(3) Rotation Lie Group & Lie Algebra so(3)
enum SO3 {
    /// Skew-symmetric hat operator [v]_x mapping R^3 -> so(3)
    static func hat(_ v: simd_float3) -> simd_float3x3 {
        simd_float3x3(
            simd_float3(0, v.z, -v.y),
            simd_float3(-v.z, 0, v.x),
            simd_float3(v.y, -v.x, 0)
        )
    }

    /// Safe quaternion normalization on S^3
    static func normalize(_ q: simd_quatf) -> simd_quatf {
        let len = length(q.vector)
        return len > 0 ? simd_quatf(vector: q.vector / len) : simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    }

    /// Exponential map exp: so(3) -> SO(3) (tangent vector phi in R^3 to unit quaternion)
    static func exp(_ phi: simd_float3) -> simd_quatf {
        let angle = length(phi)
        if angle < 1e-5 {
            let halfPhi = 0.5 * phi
            return normalize(simd_quatf(ix: halfPhi.x, iy: halfPhi.y, iz: halfPhi.z, r: 1.0))
        } else {
            return simd_quatf(angle: angle, axis: phi / angle)
        }
    }

    /// Logarithmic map log: SO(3) -> so(3) (unit quaternion to rotation vector in R^3)
    static func log(_ qIn: simd_quatf) -> simd_float3 {
        var q = normalize(qIn)
        // Enforce shortest-path hemisphere (real part >= 0)
        if q.real < 0 {
            q = simd_quatf(vector: -q.vector)
        }
        let sinHalfAngle = length(q.imag)
        if sinHalfAngle < 1e-5 {
            return 2.0 * q.imag
        } else {
            let angle = 2.0 * atan2(sinHalfAngle, q.real)
            return (angle / sinHalfAngle) * q.imag
        }
    }

    /// Relative rotation error on so(3) between a reference and target orientation: log(q_ref^{-1} * q_target)
    static func betweenError(from reference: simd_quatf, to target: simd_quatf) -> simd_float3 {
        log(reference.inverse * target)
    }
}

// MARK: - SE(3) Rigid Body Pose Lie Group & Lie Algebra se(3)
enum SE3 {
    /// Composes an SE(3) 4x4 homogeneous matrix from an SO(3) quaternion and R^3 translation
    static func compose(rotation: simd_quatf, translation: simd_float3) -> simd_float4x4 {
        var matrix = simd_matrix4x4(SO3.normalize(rotation))
        matrix.columns.3 = simd_float4(translation.x, translation.y, translation.z, 1.0)
        return matrix
    }

    /// Extracts the SO(3) quaternion and R^3 translation from an SE(3) 4x4 matrix
    static func decompose(_ matrix: simd_float4x4) -> (rotation: simd_quatf, translation: simd_float3) {
        let translation = simd_float3(matrix.columns.3.x, matrix.columns.3.y, matrix.columns.3.z)
        let rotation = SO3.normalize(simd_quatf(matrix))
        return (rotation, translation)
    }
}

// MARK: - simd_float4x4 Manifold Perturbation Extension
extension simd_float4x4 {
    /// Applies a 6-DOF Lie Algebra perturbation to the SE(3) matrix
    /// dx = [rho_x, rho_y, rho_z, phi_x, phi_y, phi_z] (Translation, Rotation)
    mutating func updateManifold(dx: [Double], startIndex: Int) {
        let rho = simd_float3(Float(dx[startIndex]), Float(dx[startIndex + 1]), Float(dx[startIndex + 2]))
        let phi = simd_float3(Float(dx[startIndex + 3]), Float(dx[startIndex + 4]), Float(dx[startIndex + 5]))

        let (currentRotation, currentTranslation) = SE3.decompose(self)
        let updatedRotation = SO3.normalize(currentRotation * SO3.exp(phi))
        let updatedTranslation = currentTranslation + rho

        self = SE3.compose(rotation: updatedRotation, translation: updatedTranslation)
    }
}
