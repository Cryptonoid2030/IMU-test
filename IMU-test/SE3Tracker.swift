//
//  SE3Tracker.swift
//  IMU-test
//
//  Created by JohnQuavious JeanCarlo on 9/28/26.
//  Manages the visualization
//

import Foundation
import ARKit
import SwiftUI
import simd

@Observable
class SE3Tracker: NSObject, ARSessionDelegate {
    private var session = ARSession()
    
    // SE(3) matrix containing both rotation and translation
    var transform: simd_float4x4 = matrix_identity_float4x4
    
    // Array to store the path of translations (X, Y, Z coordinates)
    var path: [simd_float3] = []
    
    func start() {
        session.delegate = self
        // ARWorldTrackingConfiguration provides 6-DOF tracking in real-world meters
        let configuration = ARWorldTrackingConfiguration()
        session.run(configuration)
    }
    
    func stop() {
        session.pause()
    }
    
    func recenter() {
        path.removeAll()
        let configuration = ARWorldTrackingConfiguration()
        // Restart tracking from the current physical location
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
    }
    
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        self.transform = frame.camera.transform
        
        // Extract the translation (position) from the 4th column of the SE(3) matrix
        let position = simd_float3(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
        
        // Only record a point if the device has moved at least 1 cm (0.01 meters)
        // This prevents memory bloat from drawing thousands of overlapping dots when standing still
        if let last = path.last {
            if distance(last, position) > 0.01 {
                path.append(position)
            }
        } else {
            path.append(position)
        }
    }
}
