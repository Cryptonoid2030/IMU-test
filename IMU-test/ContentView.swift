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
    @State private var isTuningExpanded: Bool = true
    
    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                Text("SE(3) & Raw IMU Data")
                    .font(.title)
                    .fontWeight(.bold)
                
                // --- 1. Hybrid Tracker: 3D Path View with Loop Closure Indicator ---
                VStack(spacing: 12) {
                    HStack {
                        Text("Auto-Correcting 100Hz VIO")
                            .font(.headline)
                        Spacer()

                        // Point Cloud Toggle Button
                        Button(action: {
                            tracker.isPointCloudVisible.toggle()
                        }) {
                            HStack(spacing: 5) {
                                Image(systemName: tracker.isPointCloudVisible ? "point.3.filled.connected.trianglepath.dotted" : "point.3.connected.trianglepath.dotted")
                                Text(tracker.isPointCloudVisible ? "Cloud ON" : "Cloud OFF")
                                    .font(.caption)
                                    .fontWeight(.bold)
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 5)
                            .background(tracker.isPointCloudVisible ? Color.yellow.opacity(0.2) : Color.gray.opacity(0.2))
                            .foregroundColor(tracker.isPointCloudVisible ? .yellow : .secondary)
                            .cornerRadius(8)
                        }

                        // Header Status Pill
                        if tracker.isLoopClosureActive {
                            HStack(spacing: 4) {
                                Circle().fill(Color.green).frame(width: 8, height: 8)
                                Text("LOOP DETECTED")
                                    .font(.caption2)
                                    .fontWeight(.bold)
                                    .foregroundColor(.green)
                            }
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.green.opacity(0.15))
                            .cornerRadius(8)
                        } else if tracker.loopClosureCount > 0 {
                            HStack(spacing: 4) {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.green)
                                    .font(.system(size: 10))
                                Text("\(tracker.loopClosureCount) Loops")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    
                    // 3D Scene View with floating Loop Closure Banner
                    ZStack(alignment: .top) {
                        PathView(
                            transform: tracker.pureSE3Transform,
                            path: tracker.path,
                            featurePoints: tracker.featurePoints,
                            loopLinks: tracker.loopLinks,
                            isLoopClosureActive: tracker.isLoopClosureActive,
                            showPointCloud: tracker.isPointCloudVisible
                        )
                        .frame(maxWidth: .infinity)
                        .frame(height: 350)
                        .cornerRadius(16)
                        .shadow(radius: 5)

                        // Real-time Loop Closure HUD Banner
                        if tracker.isLoopClosureActive, let loop = tracker.latestLoopClosure {
                            HStack(spacing: 10) {
                                Image(systemName: "arrow.triangle.2.circlepath.circle.fill")
                                    .font(.title2)
                                    .foregroundColor(.green)

                                VStack(alignment: .leading, spacing: 2) {
                                    Text("GRAPH SLAM LOOP CLOSED")
                                        .font(.system(size: 13, weight: .black))
                                        .foregroundColor(.white)
                                    Text("KF #\(loop.queryKeyframeId) ↔ #\(loop.matchedKeyframeId) • \(Int(loop.similarityScore * 100))% BoW • Δ \(String(format: "%.2fm", loop.driftDistance))")
                                        .font(.system(size: 11, weight: .medium))
                                        .foregroundColor(.white.opacity(0.9))
                                }

                                Spacer()

                                Text("OPTIMIZED")
                                    .font(.caption2)
                                    .fontWeight(.heavy)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 3)
                                    .background(Color.green)
                                    .foregroundColor(.black)
                                    .cornerRadius(6)
                            }
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .background(
                                RoundedRectangle(cornerRadius: 12)
                                    .fill(Color.black.opacity(0.85))
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 12)
                                            .stroke(Color.green, lineWidth: 2)
                                    )
                            )
                            .padding(10)
                            .transition(.move(edge: .top).combined(with: .opacity))
                        } else if let loop = tracker.latestLoopClosure {
                            // Subtle resting loop badge
                            HStack(spacing: 6) {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.green)
                                    .font(.caption)
                                Text("Last Loop: KF #\(loop.queryKeyframeId) ↔ #\(loop.matchedKeyframeId) (\(Int(loop.similarityScore * 100))% match)")
                                    .font(.caption2)
                                    .foregroundColor(.white)
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 5)
                            .background(Color.black.opacity(0.65))
                            .cornerRadius(8)
                            .padding(10)
                            .frame(maxWidth: .infinity, alignment: .topLeading)
                        }
                    }
                    .animation(.spring(response: 0.35, dampingFraction: 0.7), value: tracker.isLoopClosureActive)
                    
                    HStack(spacing: 12) {
                        Button(action: {
                            tracker.recenter()
                        }) {
                            Text("Recenter & Clear")
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .foregroundColor(.white)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.red)
                                .cornerRadius(12)
                        }

                        Button(action: {
                            tracker.triggerSimulatedLoopClosure()
                        }) {
                            HStack(spacing: 6) {
                                Image(systemName: "arrow.triangle.2.circlepath")
                                Text("Test Loop Closure")
                            }
                            .font(.subheadline)
                            .fontWeight(.semibold)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(12)
                        }
                    }
                }
                .padding(.horizontal)
                
                // --- 2. Bag-of-Words & Graph SLAM Panel ---
                GroupBox("Bag of Words & Graph SLAM Optimizer") {
                    VStack(alignment: .leading, spacing: 14) {
                        // Live Status Indicator Banner
                        HStack(spacing: 12) {
                            ZStack {
                                Circle()
                                    .fill(tracker.isLoopClosureActive ? Color.green.opacity(0.25) : Color.blue.opacity(0.15))
                                    .frame(width: 38, height: 38)
                                Image(systemName: tracker.isLoopClosureActive ? "arrow.triangle.2.circlepath" : "point.3.connected.trianglepath.dotted")
                                    .foregroundColor(tracker.isLoopClosureActive ? .green : .blue)
                                    .font(.system(size: 18, weight: .bold))
                            }

                            VStack(alignment: .leading, spacing: 2) {
                                Text(tracker.isLoopClosureActive ? "Graph SLAM Optimizing!" : "Graph SLAM Active")
                                    .font(.subheadline)
                                    .fontWeight(.bold)
                                    .foregroundColor(tracker.isLoopClosureActive ? .green : .primary)
                                Text("\(tracker.keyframeCount) Pose Nodes • Gauss-Newton Sparse Cholesky")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }

                            Spacer()

                            // Status Ring / Indicator Lamp
                            VStack(spacing: 2) {
                                Circle()
                                    .fill(tracker.isLoopClosureActive ? Color.green : (tracker.loopClosureCount > 0 ? Color.green.opacity(0.7) : Color.gray.opacity(0.4)))
                                    .frame(width: 14, height: 14)
                                    .shadow(color: tracker.isLoopClosureActive ? Color.green : Color.clear, radius: 4)
                                Text(tracker.isLoopClosureActive ? "CLOSED" : (tracker.loopClosureCount > 0 ? "READY" : "SEARCH"))
                                    .font(.system(size: 8, weight: .bold))
                                    .foregroundColor(.secondary)
                            }
                        }

                        Divider()

                        // Metrics Grid
                        Grid(horizontalSpacing: 12, verticalSpacing: 10) {
                            GridRow {
                                MetricCell(title: "Loops Closed", value: "\(tracker.loopClosureCount)", color: .green)
                                MetricCell(title: "Graph Nodes", value: "\(tracker.keyframeCount)", color: .blue)
                            }
                            GridRow {
                                let scoreStr = tracker.latestLoopClosure != nil
                                    ? String(format: "%.1f%%", tracker.latestLoopClosure!.similarityScore * 100)
                                    : "--"
                                let driftStr = tracker.latestLoopClosure != nil
                                    ? String(format: "%.2f m", tracker.latestLoopClosure!.driftDistance)
                                    : "--"
                                MetricCell(title: "Best BoW Match", value: scoreStr, color: .purple)
                                MetricCell(title: "Loop Drift", value: driftStr, color: .orange)
                            }
                        }

                        // Recent Loop Closures History
                        if !tracker.loopClosureHistory.isEmpty {
                            Divider()
                            VStack(alignment: .leading, spacing: 6) {
                                Text("Recent Loop Closures")
                                    .font(.caption)
                                    .fontWeight(.semibold)
                                    .foregroundColor(.secondary)

                                ForEach(tracker.loopClosureHistory.suffix(3).reversed()) { item in
                                    HStack {
                                        Image(systemName: "checkmark.circle.fill")
                                            .foregroundColor(.green)
                                            .font(.caption2)
                                        Text("KF #\(item.queryKeyframeId) ↔ #\(item.matchedKeyframeId)")
                                            .font(.caption)
                                            .monospacedDigit()
                                        Spacer()
                                        Text("\(Int(item.similarityScore * 100))% BoW")
                                            .font(.caption2)
                                            .foregroundColor(.secondary)
                                        Text("Δ \(String(format: "%.2fm", item.driftDistance))")
                                            .font(.caption2)
                                            .fontWeight(.bold)
                                            .foregroundColor(.orange)
                                    }
                                    .padding(.vertical, 2)
                                }
                            }
                        }

                        Divider()

                        // --- Graph SLAM Hyperparameter Tuning Section ---
                        DisclosureGroup(isExpanded: $isTuningExpanded) {
                            VStack(alignment: .leading, spacing: 14) {
                                Text("Tune these parameters to prevent loop closures from pulling the current pose away from the physical position:")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)

                                // 1. Loop Edge Weight Slider
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack {
                                        Text("Loop Weight (Stiffness)")
                                            .font(.caption)
                                            .fontWeight(.medium)
                                        Spacer()
                                        Text("\(Int(tracker.loopWeight * 100))%")
                                            .font(.caption)
                                            .fontWeight(.bold)
                                            .foregroundColor(.blue)
                                    }
                                    Slider(value: Binding(
                                        get: { tracker.loopWeight },
                                        set: { tracker.loopWeight = $0 }
                                    ), in: 0.05...1.0, step: 0.05)
                                    Text("Controls how strongly loop closures pull vs odometry. Lower = gentler.")
                                        .font(.system(size: 10))
                                        .foregroundColor(.secondary)
                                }

                                // 2. Maximum Drift Clamp
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack {
                                        Text("Max Live Jump Clamp")
                                            .font(.caption)
                                            .fontWeight(.medium)
                                        Spacer()
                                        Text(String(format: "%.2f m", tracker.maxDriftClamp))
                                            .font(.caption)
                                            .fontWeight(.bold)
                                            .foregroundColor(.orange)
                                    }
                                    Slider(value: Binding(
                                        get: { tracker.maxDriftClamp },
                                        set: { tracker.maxDriftClamp = $0 }
                                    ), in: 0.10...1.50, step: 0.05)
                                    Text("Limits maximum displacement on the current tracking pose per loop.")
                                        .font(.system(size: 10))
                                        .foregroundColor(.secondary)
                                }

                                // 3. Live Pose Damping
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack {
                                        Text("Live Pose Damping")
                                            .font(.caption)
                                            .fontWeight(.medium)
                                        Spacer()
                                        Text("\(Int(tracker.livePoseDamping * 100))%")
                                            .font(.caption)
                                            .fontWeight(.bold)
                                            .foregroundColor(.purple)
                                    }
                                    Slider(value: Binding(
                                        get: { tracker.livePoseDamping },
                                        set: { tracker.livePoseDamping = $0 }
                                    ), in: 0.05...1.0, step: 0.05)
                                    Text("Blends graph correction smoothly into live EKF without jolting ARKit.")
                                        .font(.system(size: 10))
                                        .foregroundColor(.secondary)
                                }

                                // 4. Huber Robust Kernel Threshold
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack {
                                        Text("Huber Outlier Rejection Threshold")
                                            .font(.caption)
                                            .fontWeight(.medium)
                                        Spacer()
                                        Text(String(format: "%.2f m", tracker.huberThreshold))
                                            .font(.caption)
                                            .fontWeight(.bold)
                                            .foregroundColor(.green)
                                    }
                                    Slider(value: Binding(
                                        get: { tracker.huberThreshold },
                                        set: { tracker.huberThreshold = $0 }
                                    ), in: 0.10...1.0, step: 0.05)
                                    Text("Residuals larger than this are attenuated to prevent bad loop matches.")
                                        .font(.system(size: 10))
                                        .foregroundColor(.secondary)
                                }

                                // Preset Tuning Buttons
                                HStack(spacing: 8) {
                                    Button("Gentle (Phone)") {
                                        tracker.loopWeight = 0.20
                                        tracker.maxDriftClamp = 0.25
                                        tracker.livePoseDamping = 0.25
                                        tracker.huberThreshold = 0.25
                                    }
                                    .font(.caption2)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(Color.blue.opacity(0.15))
                                    .cornerRadius(6)

                                    Button("Balanced") {
                                        tracker.loopWeight = 0.35
                                        tracker.maxDriftClamp = 0.50
                                        tracker.livePoseDamping = 0.40
                                        tracker.huberThreshold = 0.35
                                    }
                                    .font(.caption2)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(Color.green.opacity(0.15))
                                    .cornerRadius(6)

                                    Button("Aggressive") {
                                        tracker.loopWeight = 0.80
                                        tracker.maxDriftClamp = 1.00
                                        tracker.livePoseDamping = 0.75
                                        tracker.huberThreshold = 0.60
                                    }
                                    .font(.caption2)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(Color.orange.opacity(0.15))
                                    .cornerRadius(6)
                                }
                                .padding(.top, 4)
                            }
                            .padding(.top, 8)
                        } label: {
                            HStack {
                                Image(systemName: "slider.horizontal.3")
                                    .foregroundColor(.blue)
                                Text("Graph SLAM Tuning Controls")
                                    .font(.subheadline)
                                    .fontWeight(.semibold)
                            }
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding(.horizontal)

                // --- 3. Pure IMU SE(3) Transform Matrix ---
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
                
                // --- 4. Raw IMU Data & Biases (From MotionManager) ---
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
                
                // --- 5. Calibration Controls (From MotionManager) ---
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

// MARK: - Metric Cell Helper
private struct MetricCell: View {
    let title: String
    let value: String
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption2)
                .foregroundColor(.secondary)
            Text(value)
                .font(.subheadline)
                .fontWeight(.bold)
                .foregroundColor(color)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(8)
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(8)
    }
}
