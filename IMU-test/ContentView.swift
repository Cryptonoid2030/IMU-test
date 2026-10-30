//
//  ContentView.swift
//  IMU-test
//
//  Created by JohnQuavious JeanCarlo on 9/28/26.
//

import SwiftUI
import SceneKit
import simd

struct ContentView: View {
    @State private var tracker = HybridTracker()
    @State private var motion = MotionManager()
    
    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                Text("SE(3) & Raw IMU Data")
                    .font(.title)
                    .fontWeight(.bold)
                
                // --- 1. Hybrid Tracker: 3D Path View ---
                VStack(spacing: 12) {
                    Text("Auto-Correcting 100Hz VIO")
                        .font(.headline)
                    
                    // PASS THE POINT CLOUD HERE
                    PathView(transform: tracker.pureSE3Transform,
                             path: tracker.path,
                             featurePoints: tracker.featurePoints)
                        .frame(maxWidth: .infinity)
                        .frame(height: 350)
                        .cornerRadius(16)
                        .shadow(radius: 5)
                    
                    Button(action: {
                        tracker.recenter()
                    }) {
                        Text("Recenter Tracking & Clear Path")
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.red)
                            .cornerRadius(12)
                    }
                }
                .padding(.horizontal)
                
                // --- 2. Pure IMU SE(3) Transform Matrix ---
                GroupBox("Pure IMU SE(3) Transform") {
                    Grid(horizontalSpacing: 16, verticalSpacing: 12) {
                        let cols = tracker.pureSE3Transform.columns
                        
                        GridRow {
                            Text("\(cols.0.x, specifier: "%+06.2f")")
                            Text("\(cols.1.x, specifier: "%+06.2f")")
                            Text("\(cols.2.x, specifier: "%+06.2f")")
                            Text("\(cols.3.x, specifier: "%+06.2f")").fontWeight(.bold).foregroundColor(.blue)
                        }
                        GridRow {
                            Text("\(cols.0.y, specifier: "%+06.2f")")
                            Text("\(cols.1.y, specifier: "%+06.2f")")
                            Text("\(cols.2.y, specifier: "%+06.2f")")
                            Text("\(cols.3.y, specifier: "%+06.2f")").fontWeight(.bold).foregroundColor(.blue)
                        }
                        GridRow {
                            Text("\(cols.0.z, specifier: "%+06.2f")")
                            Text("\(cols.1.z, specifier: "%+06.2f")")
                            Text("\(cols.2.z, specifier: "%+06.2f")")
                            Text("\(cols.3.z, specifier: "%+06.2f")").fontWeight(.bold).foregroundColor(.blue)
                        }
                        GridRow {
                            Text("\(cols.0.w, specifier: "%+06.2f")")
                            Text("\(cols.1.w, specifier: "%+06.2f")")
                            Text("\(cols.2.w, specifier: "%+06.2f")")
                            Text("\(cols.3.w, specifier: "%+06.2f")")
                        }.foregroundColor(.gray)
                    }
                    .monospacedDigit()
                    .font(.system(size: 14))
                    .frame(maxWidth: .infinity)
                }
                .padding(.horizontal)
                
                // --- 3. Raw IMU Data & Biases (From MotionManager) ---
                GroupBox("Accelerometer (G-Force)") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("X: \(motion.accel.x, specifier: "%+06.3f")")
                        Text("Y: \(motion.accel.y, specifier: "%+06.3f")")
                        Text("Z: \(motion.accel.z, specifier: "%+06.3f")")
                        Text("X - Bias: \(motion.accelBias.x, specifier: "%+06.3f")").foregroundColor(.secondary)
                        Text("Y - Bias: \(motion.accelBias.y, specifier: "%+06.3f")").foregroundColor(.secondary)
                        Text("Z - Bias: \(motion.accelBias.z, specifier: "%+06.3f")").foregroundColor(.secondary)
                    }
                    .monospacedDigit()
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding(.horizontal)
                
                GroupBox("Gyroscope (Rad/Sec)") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("X: \(motion.gyro.x, specifier: "%+06.3f")")
                        Text("Y: \(motion.gyro.y, specifier: "%+06.3f")")
                        Text("Z: \(motion.gyro.z, specifier: "%+06.3f")")
                        Text("X - Bias: \(motion.gyroBias.x, specifier: "%+06.3f")").foregroundColor(.secondary)
                        Text("Y - Bias: \(motion.gyroBias.y, specifier: "%+06.3f")").foregroundColor(.secondary)
                        Text("Z - Bias: \(motion.gyroBias.z, specifier: "%+06.3f")").foregroundColor(.secondary)
                    }
                    .monospacedDigit()
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding(.horizontal)
                
                GroupBox("Magnetometer (Microteslas)") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("X: \(motion.mag.x, specifier: "%+07.2f")")
                        Text("Y: \(motion.mag.y, specifier: "%+07.2f")")
                        Text("Z: \(motion.mag.z, specifier: "%+07.2f")")
                        Text("X - Bias: \(motion.magBias.x, specifier: "%+06.3f")").foregroundColor(.secondary)
                        Text("Y - Bias: \(motion.magBias.y, specifier: "%+06.3f")").foregroundColor(.secondary)
                        Text("Z - Bias: \(motion.magBias.z, specifier: "%+06.3f")").foregroundColor(.secondary)
                    }
                    .monospacedDigit()
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding(.horizontal)
                
                // --- 4. Calibration Controls (From MotionManager) ---
                VStack(spacing: 12) {
                    Button(action: { motion.startRestingCalibration() }) {
                        Text(motion.isRestingCalibrating ? "Calibrating..." : "1. Calibrate Accel/Gyro")
                            .font(.headline).foregroundColor(.white).frame(maxWidth: .infinity).padding()
                            .background(motion.isRestingCalibrating ? Color.gray : Color.blue)
                            .cornerRadius(12)
                    }.disabled(motion.isRestingCalibrating || motion.isMagCalibrating)
                    
                    Text("Place phone flat on a table.")
                        .font(.caption).foregroundColor(.secondary)
                    
                    Button(action: { motion.startMagCalibration() }) {
                        Text(motion.isMagCalibrating ? "Rotate in Figure-8..." : "2. Calibrate Magnetometer")
                            .font(.headline).foregroundColor(.white).frame(maxWidth: .infinity).padding()
                            .background(motion.isMagCalibrating ? Color.gray : Color.purple)
                            .cornerRadius(12)
                    }.disabled(motion.isRestingCalibrating || motion.isMagCalibrating)
                    
                    Text("Rotate phone in a Figure-8 for 20 seconds.")
                        .font(.caption).foregroundColor(.secondary)
                }
                .padding()
            }
            .padding(.vertical)
        }
        .onAppear {
            motion.onRestingCalibrationComplete = { result in
                tracker.applyCalibration(result)
            }
            tracker.start()
            motion.start()
        }
        .onDisappear {
            tracker.stop()
            motion.stop()
        }
    }
}
