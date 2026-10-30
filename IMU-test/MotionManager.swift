//
//  MotionManager.swift
//  IMU-test
//
//  Created by JohnQuavious JeanCarlo on 9/28/26.
//  Pulls data from internal IMU (currently Accel, Magneto, and Gyro)
//

import Foundation
import CoreMotion
import SwiftUI
import SceneKit
import simd

/// Holds the measured resting calibration statistics (means, variances, and sample count)
struct IMUCalibrationResult {
    /// Accelerometer bias in G-force
    let accelBiasG: simd_float3
    /// Accelerometer bias in m/s^2
    let accelBiasMetersPerSec2: simd_float3
    /// Gyroscope bias in rad/s
    let gyroBiasRadPerSec: simd_float3
    /// Per-axis accelerometer noise variance in (m/s^2)^2
    let accelVarianceMetersPerSec2: simd_float3
    /// Per-axis gyroscope noise variance in (rad/s)^2
    let gyroVarianceRadPerSec: simd_float3
    /// Number of stationary samples used
    let sampleCount: Int
}

@Observable
class MotionManager {
    private var motionManager = CMMotionManager()
    
    // Quaternion
    var quaternion = SCNQuaternion(0, 0, 0, 1)
    
    // Calibrated output data
    var accel = (x: 0.0, y: 0.0, z: 0.0)
    var gyro = (x: 0.0, y: 0.0, z: 0.0)
    var mag = (x: 0.0, y: 0.0, z: 0.0)
    
    // Calibration States
    var isRestingCalibrating = false
    var isMagCalibrating = false
    
    private var accelSamples: [(Double, Double, Double)] = []
    private var gyroSamples: [(Double, Double, Double)] = []
    
    // Tracking for Magnetometer Extremes
    private var magMin = (x: Double.greatestFiniteMagnitude, y: Double.greatestFiniteMagnitude, z: Double.greatestFiniteMagnitude)
    private var magMax = (x: -Double.greatestFiniteMagnitude, y: -Double.greatestFiniteMagnitude, z: -Double.greatestFiniteMagnitude)
    
    // Computed Biases & Variances
    var accelBias = (x: 0.0, y: 0.0, z: 0.0)
    var gyroBias = (x: 0.0, y: 0.0, z: 0.0)
    var magBias = (x: 0.0, y: 0.0, z: 0.0)
    var accelVariance = (x: 0.0, y: 0.0, z: 0.0)
    var gyroVariance = (x: 0.0, y: 0.0, z: 0.0)
    
    /// Optional callback invoked whenever resting Accel/Gyro calibration completes
    var onRestingCalibrationComplete: ((IMUCalibrationResult) -> Void)?
    
    func start() {
        guard motionManager.isDeviceMotionAvailable else { return }
        
        // 60 Hz for smooth 3D graphics
        motionManager.deviceMotionUpdateInterval = 1.0 / 60.0
        
        motionManager.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: .main) { [weak self] data, _ in
            guard let data = data else { return }
            
            let q = data.attitude.quaternion
            // Map the CoreMotion quaternion to the SceneKit quaternion
            self?.quaternion = SCNQuaternion(q.x, q.y, q.z, q.w)
        }
        
        let updateInterval = 0.02 // 50 Hz for richer calibration sample statistics
        
        if motionManager.isAccelerometerAvailable {
            motionManager.accelerometerUpdateInterval = updateInterval
            motionManager.startAccelerometerUpdates(to: .main) { [weak self] data, _ in
                guard let self = self, let data = data else { return }
                self.processAccel(data.acceleration)
            }
        }
        
        if motionManager.isGyroAvailable {
            motionManager.gyroUpdateInterval = updateInterval
            motionManager.startGyroUpdates(to: .main) { [weak self] data, _ in
                guard let self = self, let data = data else { return }
                self.processGyro(data.rotationRate)
            }
        }
        
        if motionManager.isMagnetometerAvailable {
            motionManager.magnetometerUpdateInterval = updateInterval
            motionManager.startMagnetometerUpdates(to: .main) { [weak self] data, _ in
                guard let self = self, let data = data else { return }
                self.processMag(data.magneticField)
            }
        }
    }
    
    func startRestingCalibration() {
        accelSamples.removeAll()
        gyroSamples.removeAll()
        accelBias = (0, 0, 0)
        gyroBias = (0, 0, 0)
        isRestingCalibrating = true
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) { self.finishRestingCalibration() }
    }
    
    func startMagCalibration() {
        // Reset extremes and biases
        magMin = (.greatestFiniteMagnitude, .greatestFiniteMagnitude, .greatestFiniteMagnitude)
        magMax = (-.greatestFiniteMagnitude, -.greatestFiniteMagnitude, -.greatestFiniteMagnitude)
        magBias = (0, 0, 0)
        isMagCalibrating = true
        
        // Allow user to perform a figure-8 motion
        DispatchQueue.main.asyncAfter(deadline: .now() + 20.0) { self.finishMagCalibration() }
    }
    
    private func processAccel(_ raw: CMAcceleration) {
        if isRestingCalibrating { accelSamples.append((raw.x, raw.y, raw.z)) }
        accel = (raw.x - accelBias.x, raw.y - accelBias.y, raw.z - accelBias.z)
    }
    
    private func processGyro(_ raw: CMRotationRate) {
        if isRestingCalibrating { gyroSamples.append((raw.x, raw.y, raw.z)) }
        gyro = (raw.x - gyroBias.x, raw.y - gyroBias.y, raw.z - gyroBias.z)
    }
    
    private func processMag(_ raw: CMMagneticField) {
        if isMagCalibrating {
            // Continuously expand the bounding box of min/max values
            magMin.x = min(magMin.x, raw.x); magMax.x = max(magMax.x, raw.x)
            magMin.y = min(magMin.y, raw.y); magMax.y = max(magMax.y, raw.y)
            magMin.z = min(magMin.z, raw.z); magMax.z = max(magMax.z, raw.z)
        }
        mag = (raw.x - magBias.x, raw.y - magBias.y, raw.z - magBias.z)
    }
    
    private func finishRestingCalibration() {
        isRestingCalibrating = false
        guard accelSamples.count > 1, gyroSamples.count > 1 else { return }
        
        let nGyro = Double(gyroSamples.count)
        let nAccel = Double(accelSamples.count)
        
        // 1. Sample Means (Biases)
        let meanGx = gyroSamples.map { $0.0 }.reduce(0, +) / nGyro
        let meanGy = gyroSamples.map { $0.1 }.reduce(0, +) / nGyro
        let meanGz = gyroSamples.map { $0.2 }.reduce(0, +) / nGyro
        gyroBias = (meanGx, meanGy, meanGz)
        
        let meanAx = accelSamples.map { $0.0 }.reduce(0, +) / nAccel
        let meanAy = accelSamples.map { $0.1 }.reduce(0, +) / nAccel
        let meanAz = accelSamples.map { $0.2 }.reduce(0, +) / nAccel
        // When flat on a table, gravity is -1.0 G along Z
        accelBias = (meanAx, meanAy, meanAz - (-1.0))
        
        // 2. Unbiased Sample Variances: s^2 = sum((x_i - mean)^2) / (N - 1)
        let varGx = gyroSamples.map { ($0.0 - meanGx) * ($0.0 - meanGx) }.reduce(0, +) / (nGyro - 1.0)
        let varGy = gyroSamples.map { ($0.1 - meanGy) * ($0.1 - meanGy) }.reduce(0, +) / (nGyro - 1.0)
        let varGz = gyroSamples.map { ($0.2 - meanGz) * ($0.2 - meanGz) }.reduce(0, +) / (nGyro - 1.0)
        gyroVariance = (varGx, varGy, varGz)
        
        let varAxG = accelSamples.map { ($0.0 - meanAx) * ($0.0 - meanAx) }.reduce(0, +) / (nAccel - 1.0)
        let varAyG = accelSamples.map { ($0.1 - meanAy) * ($0.1 - meanAy) }.reduce(0, +) / (nAccel - 1.0)
        let varAzG = accelSamples.map { ($0.2 - meanAz) * ($0.2 - meanAz) }.reduce(0, +) / (nAccel - 1.0)
        accelVariance = (varAxG, varAyG, varAzG)
        
        let g = 9.80665
        let g2 = g * g
        
        let result = IMUCalibrationResult(
            accelBiasG: simd_float3(Float(accelBias.x), Float(accelBias.y), Float(accelBias.z)),
            accelBiasMetersPerSec2: simd_float3(Float(accelBias.x * g), Float(accelBias.y * g), Float(accelBias.z * g)),
            gyroBiasRadPerSec: simd_float3(Float(gyroBias.x), Float(gyroBias.y), Float(gyroBias.z)),
            accelVarianceMetersPerSec2: simd_float3(Float(varAxG * g2), Float(varAyG * g2), Float(varAzG * g2)),
            gyroVarianceRadPerSec: simd_float3(Float(varGx), Float(varGy), Float(varGz)),
            sampleCount: min(accelSamples.count, gyroSamples.count)
        )
        onRestingCalibrationComplete?(result)
    }
    
    private func finishMagCalibration() {
        isMagCalibrating = false
        // Hard-iron bias is the midpoint of the extremes
        magBias.x = (magMax.x + magMin.x) / 2.0
        magBias.y = (magMax.y + magMin.y) / 2.0
        magBias.z = (magMax.z + magMin.z) / 2.0
    }
    
    func stop() {
        motionManager.stopAccelerometerUpdates()
        motionManager.stopGyroUpdates()
        motionManager.stopMagnetometerUpdates()
        motionManager.stopDeviceMotionUpdates()
    }
}
