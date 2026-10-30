//
//  HybridTracker.swift
//  IMU-test
//
//  Created by JohnQuavious JeanCarlo on 9/28/26.
//

import Foundation
import CoreMotion
import ARKit
import simd
import SwiftUI

/// Coordinates 100 Hz CoreMotion IMU updates, ARKit visual odometry, EKF fusion, and persistent point cloud mapping
@Observable
class HybridTracker: NSObject, ARSessionDelegate {
    private var motionManager = CMMotionManager()
    private var arSession = ARSession()
    private var ekf = EKFFusionEngine()
    private var pointCloudMap = PointCloudMap(voxelSize: 0.02, maxPoints: 40_000)
    private var loopManager = LoopClosureManager()

    // --- Published State for SwiftUI / SceneKit ---
    var pureSE3Transform: simd_float4x4 = matrix_identity_float4x4
    var position = simd_float3(0, 0, 0)
    var velocity = simd_float3(0, 0, 0)
    var path: [simd_float3] = []
    var featurePoints: [simd_float3] = []

    // --- Bag of Words Loop Closure Published State ---
    var isLoopClosureActive: Bool = false
    var latestLoopClosure: LoopClosureEvent? = nil
    var loopClosureCount: Int = 0
    var keyframeCount: Int = 0
    var loopClosureHistory: [LoopClosureEvent] = []
    var loopLinks: [(from: simd_float3, to: simd_float3)] = []
    var isPointCloudVisible: Bool = true
    private var loopIndicatorResetTimer: Timer?

    // --- Graph SLAM Hyperparameter Tuning ---
    var loopWeight: Float {
        get { loopManager.slamConfig.loopWeight }
        set { loopManager.slamConfig.loopWeight = newValue }
    }
    var maxDriftClamp: Float {
        get { loopManager.slamConfig.maxDriftClamp }
        set { loopManager.slamConfig.maxDriftClamp = newValue }
    }
    var huberThreshold: Float {
        get { loopManager.slamConfig.huberThreshold }
        set { loopManager.slamConfig.huberThreshold = newValue }
    }
    var livePoseDamping: Float {
        get { loopManager.slamConfig.livePoseDamping }
        set { loopManager.slamConfig.livePoseDamping = newValue }
    }

    // --- Sensor Timing & Latest Camera Observations ---
    private var lastIMUTime: TimeInterval = 0
    private var lastCameraTime: TimeInterval = 0
    private var latestCameraPosition = simd_float3(0, 0, 0)
    private var previousCameraPosition = simd_float3(0, 0, 0)
    private var latestCameraVelocity = simd_float3(0, 0, 0)
    private var latestCameraOrientation = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    private var hasReceivedCameraFrame = false
    private var cameraTrackingNormal = false
    private var optimizationTimer: Timer?

    // Fixed rotation from ARKit camera sensor axes to upright portrait device axes
    private let cameraToDeviceRotation = simd_quatf(angle: .pi / 2.0, axis: simd_float3(0, 0, 1))

    // MARK: - Lifecycle
    func start() {
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        arSession.delegate = self
        arSession.run(config)

        guard motionManager.isDeviceMotionAvailable else { return }

        motionManager.deviceMotionUpdateInterval = 1.0 / 100.0
        motionManager.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: .main) { [weak self] data, _ in
            guard let self = self, let data = data else { return }
            self.processIMU(data)
        }

        optimizationTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            self?.runEKFUpdate()
        }
    }

    func recenter() {
        path.removeAll()
        featurePoints.removeAll()
        pointCloudMap.reset()
        ekf.reset()
        loopManager.reset()
        loopLinks.removeAll()
        latestLoopClosure = nil
        loopClosureCount = 0
        keyframeCount = 0
        loopClosureHistory.removeAll()
        isLoopClosureActive = false
        loopIndicatorResetTimer?.invalidate()
        loopIndicatorResetTimer = nil

        pureSE3Transform = ekf.pureSE3Transform
        position = ekf.position
        velocity = ekf.velocity

        lastIMUTime = 0
        lastCameraTime = 0
        hasReceivedCameraFrame = false

        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        arSession.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    func applyCalibration(_ result: IMUCalibrationResult) {
        ekf.applyCalibration(result)
        syncPublishedState()
    }

    func stop() {
        optimizationTimer?.invalidate()
        optimizationTimer = nil
        loopIndicatorResetTimer?.invalidate()
        loopIndicatorResetTimer = nil
        motionManager.stopDeviceMotionUpdates()
        arSession.pause()
    }

    // MARK: - 100 Hz IMU Prediction
    private func processIMU(_ data: CMDeviceMotion) {
        let currentTime = data.timestamp
        if lastIMUTime == 0 {
            lastIMUTime = currentTime
            return
        }

        let dt = Float(currentTime - lastIMUTime)
        lastIMUTime = currentTime
        guard dt > 0, dt < 0.2 else { return }

        let rawGyro = simd_float3(
            Float(data.rotationRate.x),
            Float(data.rotationRate.y),
            Float(data.rotationRate.z)
        )
        let rawAccel = simd_float3(
            Float(data.userAcceleration.x),
            Float(data.userAcceleration.y),
            Float(data.userAcceleration.z)
        ) * 9.80665

        ekf.predictIMU(rawGyro: rawGyro, rawAccelMetersPerSec2: rawAccel, dt: dt)
        syncPublishedState()
    }

    // MARK: - ARKit Visual Observer & Persistent Point Cloud
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let camTransform = frame.camera.transform
        let col3 = camTransform.columns.3
        let newCamPos = simd_float3(col3.x, col3.y, col3.z)

        let timestamp = frame.timestamp
        if lastCameraTime > 0 {
            let camDt = Float(timestamp - lastCameraTime)
            if camDt > 1e-4 && camDt < 0.5 {
                let instVel = (newCamPos - previousCameraPosition) / camDt
                latestCameraVelocity = 0.6 * latestCameraVelocity + 0.4 * instVel
            }
        }
        lastCameraTime = timestamp
        previousCameraPosition = newCamPos
        latestCameraPosition = newCamPos
        latestCameraOrientation = SO3.normalize(simd_quatf(camTransform) * cameraToDeviceRotation)

        if !hasReceivedCameraFrame {
            ekf.initializePose(position: latestCameraPosition, orientation: latestCameraOrientation)
            syncPublishedState()
            hasReceivedCameraFrame = true
        }

        if case .normal = frame.camera.trackingState {
            cameraTrackingNormal = true
        } else {
            cameraTrackingNormal = false
        }

        if let rawFeatures = frame.rawFeaturePoints, pointCloudMap.accumulate(rawFeatures) {
            featurePoints = pointCloudMap.points
        }

        // Evaluate frame for Visual Bag-of-Words loop closure
        if hasReceivedCameraFrame {
            if let loop = loopManager.processFrame(
                frame: frame,
                currentPose: pureSE3Transform,
                currentPosition: position,
                currentOrientation: ekf.orientation,
                accumulatedFeatures: featurePoints
            ) {
                handleLoopClosure(loop)
            }
            keyframeCount = loopManager.keyframes.count
        }
    }

    // MARK: - Loop Closure Handling & Correction
    private func handleLoopClosure(_ loop: LoopClosureEvent) {
        // 1. Optimize the trajectory graph using Graph SLAM (Gauss-Newton with Accelerate Sparse Cholesky)
        let (optimizedPath, liveCorrection) = loopManager.optimizePoseGraph(path: path, loop: loop)
        path = optimizedPath

        // 2. Adjust EKF nominal state with clamped, damped Graph SLAM correction
        //    This prevents loop closures from dragging the current live pose far from physical reality!
        ekf.applyLoopClosureCorrection(deltaTranslation: liveCorrection)
        syncPublishedState()

        // 3. Record 3D visual link
        loopLinks.append((from: loop.queryPosition, to: loop.matchedPosition))

        // 4. Update UI indicator state
        latestLoopClosure = loop
        loopClosureCount = loopManager.loopHistory.count
        loopClosureHistory = loopManager.loopHistory
        isLoopClosureActive = true

        // 5. Schedule indicator pulse timeout
        loopIndicatorResetTimer?.invalidate()
        loopIndicatorResetTimer = Timer.scheduledTimer(withTimeInterval: 4.5, repeats: false) { [weak self] _ in
            self?.isLoopClosureActive = false
        }
    }

    /// Manually triggers a simulated loop closure for instant interactive UI demonstration
    func triggerSimulatedLoopClosure() {
        if let loop = loopManager.triggerSimulatedLoopClosure(
            currentPosition: position,
            currentPose: pureSE3Transform
        ) {
            handleLoopClosure(loop)
        }
    }

    // MARK: - 20 Hz EKF Fusion Update
    private func runEKFUpdate() {
        guard hasReceivedCameraFrame else { return }

        let updated = ekf.updateWithCamera(
            cameraPosition: latestCameraPosition,
            cameraVelocity: latestCameraVelocity,
            cameraOrientation: latestCameraOrientation,
            isTrackingNormal: cameraTrackingNormal
        )
        guard updated else { return }

        syncPublishedState()

        if let last = path.last {
            if distance(last, position) > 0.01 {
                path.append(position)
            }
        } else {
            path.append(position)
        }
    }

    private func syncPublishedState() {
        pureSE3Transform = ekf.pureSE3Transform
        position = ekf.position
        velocity = ekf.velocity
    }
}
