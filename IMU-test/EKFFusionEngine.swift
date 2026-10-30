//
//  EKFFusionEngine.swift
//  IMU-test
//
//  Created by JohnQuavious JeanCarlo on 9/28/26.
//

import Foundation
import simd

/// Error-State Extended Kalman Filter (ES-EKF) fusing 100 Hz IMU with 6-DOF Visual Odometry
class EKFFusionEngine {
    private let optimizer = SLAMOptimizer()

    // --- Nominal State ---
    private(set) var pureSE3Transform: simd_float4x4 = matrix_identity_float4x4
    private(set) var position = simd_float3(0, 0, 0)
    private(set) var velocity = simd_float3(0, 0, 0)
    private(set) var orientation = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    private(set) var accelBias = simd_float3(0, 0, 0)
    private(set) var gyroBias = simd_float3(0, 0, 0)

    // --- Error-State Covariance Blocks (P) ---
    private(set) var pPos = simd_float3(repeating: 0.01)
    private(set) var pVel = simd_float3(repeating: 0.05)
    private(set) var pPosVel = simd_float3(repeating: 0.0)
    private(set) var pRot = simd_float3(repeating: 0.01)
    private(set) var pBiasA = simd_float3(repeating: 0.001)
    private(set) var pBiasG = simd_float3(repeating: 0.0005)

    // --- Process Noise Hyperparameters (Q) ---
    private(set) var qPos = simd_float3(repeating: 0.02)
    private(set) var qVel = simd_float3(repeating: 0.25)
    private(set) var qRot = simd_float3(repeating: 0.005)
    private(set) var qBiasA = simd_float3(repeating: 1e-4)
    private(set) var qBiasG = simd_float3(repeating: 1e-5)

    /// Initializes the EKF pose state from the first valid visual observation
    func initializePose(position: simd_float3, orientation: simd_quatf) {
        self.position = position
        self.orientation = SO3.normalize(orientation)
        self.pureSE3Transform = SE3.compose(rotation: self.orientation, translation: self.position)
    }

    /// Updates EKF biases, initial covariances P, and process noise Q from stationary calibration statistics
    func applyCalibration(_ cal: IMUCalibrationResult) {
        let n = Float(max(cal.sampleCount, 1))

        // 1. Seed initial gyroscope bias directly (userAcceleration in CMDeviceMotion already removes gravity,
        //    so we only take small residual XY accelerometer offsets and keep Z near 0)
        self.gyroBias = cal.gyroBiasRadPerSec
        self.accelBias = simd_float3(cal.accelBiasMetersPerSec2.x * 0.2, cal.accelBiasMetersPerSec2.y * 0.2, 0.0)

        // Clamp measured variances to sensible physical bounds so extreme table bumps don't break the filter
        let clampedVarA = simd_clamp(cal.accelVarianceMetersPerSec2, simd_float3(repeating: 1e-5), simd_float3(repeating: 2.0))
        let clampedVarG = simd_clamp(cal.gyroVarianceRadPerSec, simd_float3(repeating: 1e-7), simd_float3(repeating: 0.1))

        // 2. Process Noise Q: scale measured stationary white-noise variance for dynamic hand motion
        self.qVel = simd_max(clampedVarA * 10.0, simd_float3(repeating: 0.01))
        self.qRot = simd_max(clampedVarG * 5.0, simd_float3(repeating: 1e-4))
        self.qBiasA = clampedVarA * 0.05
        self.qBiasG = clampedVarG * 0.05

        // 3. Initial Bias Covariance P_bias = s^2 / N (Standard Error of the Sample Mean)
        self.pBiasA = simd_max(clampedVarA / n, simd_float3(repeating: 1e-6))
        self.pBiasG = simd_max(clampedVarG / n, simd_float3(repeating: 1e-7))

        // Since the phone is resting on the table during calibration, velocity is known to be ~0
        self.velocity = .zero
        self.pVel = simd_max(clampedVarA * 0.01, simd_float3(repeating: 1e-4))
    }

    // MARK: - 1. EKF Predict Step (IMU Propagation)
    func predictIMU(rawGyro: simd_float3, rawAccelMetersPerSec2: simd_float3, dt: Float) {
        // 1. Subtract estimated IMU biases
        let omega = rawGyro - gyroBias
        let bodyAccel = rawAccelMetersPerSec2 - accelBias

        // 2. Propagate orientation on SO(3) using the Lie algebra exponential map
        orientation = SO3.normalize(orientation * SO3.exp(omega * dt))

        // 3. Rotate body linear acceleration into the world frame
        let worldAccel = orientation.act(bodyAccel)

        // Zero-velocity damping when the device is nearly stationary
        let isNearlyStationary = length(rawAccelMetersPerSec2) < 0.08 && length(omega) < 0.04
        if isNearlyStationary {
            velocity *= 0.90
        }

        // 4. Propagate nominal position and velocity
        position += velocity * dt + 0.5 * worldAccel * (dt * dt)
        velocity += worldAccel * dt

        // 5. Propagate EKF error covariance: P = F * P * F^T + Q
        let dt2 = dt * dt
        pPos = pPos + 2.0 * pPosVel * dt + pVel * dt2 + qPos * dt
        pPosVel = pPosVel + pVel * dt
        pVel = pVel + pRot * length_squared(bodyAccel) * dt2 + pBiasA * dt2 + qVel * dt
        pRot = pRot + pBiasG * dt2 + qRot * dt
        pBiasA = pBiasA + qBiasA * dt
        pBiasG = pBiasG + qBiasG * dt

        // 6. Sync SE(3) matrix
        pureSE3Transform = SE3.compose(rotation: orientation, translation: position)
    }

    // MARK: - 2. EKF Update Step (Camera Measurement Fusion)
    @discardableResult
    func updateWithCamera(
        cameraPosition: simd_float3,
        cameraVelocity: simd_float3,
        cameraOrientation: simd_quatf,
        isTrackingNormal: Bool
    ) -> Bool {
        // Measurement noise R adapts to ARKit tracking confidence
        let rPos = isTrackingNormal ? simd_float3(repeating: 0.002) : simd_float3(repeating: 0.05)
        let rVel = isTrackingNormal ? simd_float3(repeating: 0.02)  : simd_float3(repeating: 0.25)
        let rRot = isTrackingNormal ? simd_float3(repeating: 0.002) : simd_float3(repeating: 0.05)

        // 1. Compute 6-DOF Pose Innovations (Residuals)
        let posResidual = cameraPosition - position
        let rotResidual = SO3.betweenError(from: orientation, to: cameraOrientation)

        // 2. Form EKF Information Matrix H = P^{-1} + R^{-1} and Information Vector b = R^{-1} * y
        let safePPos = simd_max(pPos, simd_float3(repeating: 1e-6))
        let safePRot = simd_max(pRot, simd_float3(repeating: 1e-6))

        let infoPos = (1.0 / safePPos) + (1.0 / rPos)
        let infoRot = (1.0 / safePRot) + (1.0 / rRot)

        let hRows: [Int32] = [0, 1, 2, 3, 4, 5]
        let hCols: [Int32] = [0, 1, 2, 3, 4, 5]
        let hVals: [Double] = [
            Double(infoPos.x), Double(infoPos.y), Double(infoPos.z),
            Double(infoRot.x), Double(infoRot.y), Double(infoRot.z)
        ]

        let rhsPos = posResidual / rPos
        let rhsRot = rotResidual / rRot
        let bVector: [Double] = [
            Double(rhsPos.x), Double(rhsPos.y), Double(rhsPos.z),
            Double(rhsRot.x), Double(rhsRot.y), Double(rhsRot.z)
        ]

        // 3. Solve for Kalman error-state perturbation dx = [dp, dtheta] via Sparse Cholesky
        guard let dx = optimizer.solveGaussNewton(size: 6, hRows: hRows, hCols: hCols, hVals: hVals, b: bVector) else {
            return false
        }

        // 4. Inject error-state update onto the SE(3) manifold
        pureSE3Transform.updateManifold(dx: dx, startIndex: 0)
        let decomposed = SE3.decompose(pureSE3Transform)
        orientation = decomposed.rotation
        position = decomposed.translation

        // 5. Update velocity and IMU biases using Kalman gains
        let kPos = safePPos / (safePPos + rPos)
        let kRot = safePRot / (safePRot + rRot)

        let velResidual = cameraVelocity - velocity
        let kVelFromPos = pPosVel / (safePPos + rPos)
        let kVelDirect = pVel / (pVel + rVel)
        velocity += kVelFromPos * posResidual + kVelDirect * velResidual

        let bodyVelResidual = orientation.inverse.act(velResidual)
        accelBias = simd_clamp(
            accelBias - 0.02 * bodyVelResidual,
            simd_float3(repeating: -1.5),
            simd_float3(repeating: 1.5)
        )
        gyroBias = simd_clamp(
            gyroBias - 0.01 * rotResidual,
            simd_float3(repeating: -0.2),
            simd_float3(repeating: 0.2)
        )

        // 6. Update error covariance: P = (I - K * H) * P
        pPos = (simd_float3(repeating: 1.0) - kPos) * safePPos
        pRot = (simd_float3(repeating: 1.0) - kRot) * safePRot
        pVel = (simd_float3(repeating: 1.0) - kVelDirect) * pVel
        pPosVel = (simd_float3(repeating: 1.0) - kPos) * pPosVel

        return true
    }

    // MARK: - 3. Loop Closure Correction
    /// Injects a discrete loop closure translation update onto the nominal state and contracts positional uncertainty
    func applyLoopClosureCorrection(deltaTranslation: simd_float3) {
        position += deltaTranslation
        pureSE3Transform = SE3.compose(rotation: orientation, translation: position)
        pPos = simd_max(pPos * 0.4, simd_float3(repeating: 1e-4))
        pVel *= 0.8
    }

    // MARK: - 4. Reset State & Covariance
    func reset() {
        pureSE3Transform = matrix_identity_float4x4
        position = simd_float3(0, 0, 0)
        velocity = simd_float3(0, 0, 0)
        orientation = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)

        pPos = simd_float3(repeating: 0.01)
        pVel = simd_float3(repeating: 0.05)
        pPosVel = simd_float3(repeating: 0.0)
        pRot = simd_float3(repeating: 0.01)
    }
}
