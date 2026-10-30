//
//  PathView.swift
//  IMU-test
//
//  Created by JohnQuavious JeanCarlo on 9/28/26.
//  Visualization for point cloud, SE(3) pose, and Bag-of-Words loop closures
//

import SwiftUI
import SceneKit
import ARKit

struct PathView: UIViewRepresentable {
    var transform: simd_float4x4
    var path: [simd_float3]
    var featurePoints: [simd_float3]
    var loopLinks: [(from: simd_float3, to: simd_float3)] = []
    var isLoopClosureActive: Bool = false
    var showPointCloud: Bool = true

    class Coordinator {
        var lastRenderedPointCount: Int = 0
        var lastFirstPoint: simd_float3 = .zero
        var lastFinalPoint: simd_float3 = .zero
        var lastRenderedLoopCount: Int = 0
        var lastPathRevision: Int = 0
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.scene = SCNScene()
        view.allowsCameraControl = true
        view.autoenablesDefaultLighting = true
        view.backgroundColor = UIColor.systemBackground

        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.position = SCNVector3(x: 0, y: 0.5, z: 1.5)
        cameraNode.look(at: SCNVector3(0, 0, 0))
        view.scene?.rootNode.addChildNode(cameraNode)

        let box = SCNBox(width: 0.07, height: 0.14, length: 0.01, chamferRadius: 0.005)
        let screenMat = SCNMaterial(); screenMat.diffuse.contents = UIColor.black
        let backMat = SCNMaterial(); backMat.diffuse.contents = UIColor.systemBlue
        let edgeMat = SCNMaterial(); edgeMat.diffuse.contents = UIColor.systemGray
        box.materials = [screenMat, edgeMat, backMat, edgeMat, edgeMat, edgeMat]

        let phoneNode = SCNNode(geometry: box)
        phoneNode.name = "phone"
        view.scene?.rootNode.addChildNode(phoneNode)

        let pathContainerNode = SCNNode()
        pathContainerNode.name = "pathContainer"
        view.scene?.rootNode.addChildNode(pathContainerNode)

        // Container for the persistent point cloud
        let cloudContainerNode = SCNNode()
        cloudContainerNode.name = "cloudContainer"
        view.scene?.rootNode.addChildNode(cloudContainerNode)

        // Container for 3D loop closure constraint links
        let loopContainerNode = SCNNode()
        loopContainerNode.name = "loopContainer"
        view.scene?.rootNode.addChildNode(loopContainerNode)

        return view
    }

    func updateUIView(_ uiView: SCNView, context: Context) {
        if let phoneNode = uiView.scene?.rootNode.childNode(withName: "phone", recursively: false) {
            phoneNode.simdTransform = transform
        }

        // Render trajectory path dots, rebuilding when path is cleared or corrected via loop closure
        if let pathContainer = uiView.scene?.rootNode.childNode(withName: "pathContainer", recursively: false) {
            let currentDots = pathContainer.childNodes.count
            let needsFullRebuild = path.count < currentDots || (loopLinks.count != context.coordinator.lastRenderedLoopCount)

            if needsFullRebuild {
                pathContainer.childNodes.forEach { $0.removeFromParentNode() }
                for pt in path {
                    let sphere = SCNSphere(radius: 0.005)
                    sphere.firstMaterial?.diffuse.contents = UIColor.red
                    let dotNode = SCNNode(geometry: sphere)
                    dotNode.simdPosition = pt
                    pathContainer.addChildNode(dotNode)
                }
            } else if path.count > currentDots {
                for i in currentDots..<path.count {
                    let sphere = SCNSphere(radius: 0.005)
                    sphere.firstMaterial?.diffuse.contents = UIColor.red
                    let dotNode = SCNNode(geometry: sphere)
                    dotNode.simdPosition = path[i]
                    pathContainer.addChildNode(dotNode)
                }
            }
        }

        // Render 3D Loop Closure Constraint Beams and Markers
        if let loopContainer = uiView.scene?.rootNode.childNode(withName: "loopContainer", recursively: false) {
            if loopLinks.count != context.coordinator.lastRenderedLoopCount {
                context.coordinator.lastRenderedLoopCount = loopLinks.count
                loopContainer.childNodes.forEach { $0.removeFromParentNode() }

                for link in loopLinks {
                    let linkNode = Self.createLinkNode(from: link.from, to: link.to)
                    loopContainer.addChildNode(linkNode)
                }
            }
        }

        // Render the persistent point cloud efficiently, only rebuilding geometry when the cloud updates
        if let cloudContainer = uiView.scene?.rootNode.childNode(withName: "cloudContainer", recursively: false) {
            cloudContainer.isHidden = !showPointCloud

            if !showPointCloud || featurePoints.isEmpty {
                if featurePoints.isEmpty && !cloudContainer.childNodes.isEmpty {
                    cloudContainer.childNodes.forEach { $0.removeFromParentNode() }
                    context.coordinator.lastRenderedPointCount = 0
                }
                return
            }

            let firstPt = featurePoints.first ?? .zero
            let lastPt = featurePoints.last ?? .zero
            let cloudChanged = featurePoints.count != context.coordinator.lastRenderedPointCount
                || firstPt != context.coordinator.lastFirstPoint
                || lastPt != context.coordinator.lastFinalPoint

            guard cloudChanged else { return }

            context.coordinator.lastRenderedPointCount = featurePoints.count
            context.coordinator.lastFirstPoint = firstPt
            context.coordinator.lastFinalPoint = lastPt

            cloudContainer.childNodes.forEach { $0.removeFromParentNode() }

            let vertices = featurePoints.map { SCNVector3($0.x, $0.y, $0.z) }
            let source = SCNGeometrySource(vertices: vertices)
            let indices = Array(0..<Int32(vertices.count))
            let element = SCNGeometryElement(indices: indices, primitiveType: .point)

            element.pointSize = 3.0
            element.minimumPointScreenSpaceRadius = 2.0
            element.maximumPointScreenSpaceRadius = 5.0

            let geometry = SCNGeometry(sources: [source], elements: [element])
            geometry.firstMaterial?.diffuse.contents = UIColor.yellow
            geometry.firstMaterial?.lightingModel = .constant

            let cloudNode = SCNNode(geometry: geometry)
            cloudContainer.addChildNode(cloudNode)
        }
    }

    /// Generates a 3D visual link with glowing endpoint beacon spheres and a connecting beam
    private static func createLinkNode(from start: simd_float3, to end: simd_float3) -> SCNNode {
        let parent = SCNNode()

        // 1. Endpoint beacon spheres
        let beaconGeom = SCNSphere(radius: 0.012)
        beaconGeom.firstMaterial?.diffuse.contents = UIColor.systemGreen
        beaconGeom.firstMaterial?.emission.contents = UIColor(red: 0.2, green: 1.0, blue: 0.5, alpha: 1.0)

        let startBeacon = SCNNode(geometry: beaconGeom)
        startBeacon.simdPosition = start
        parent.addChildNode(startBeacon)

        let endBeacon = SCNNode(geometry: beaconGeom)
        endBeacon.simdPosition = end
        parent.addChildNode(endBeacon)

        // 2. Connecting cylindrical constraint beam
        let delta = end - start
        let dist = length(delta)
        guard dist > 1e-4 else { return parent }

        let cylinder = SCNCylinder(radius: 0.005, height: CGFloat(dist))
        cylinder.firstMaterial?.diffuse.contents = UIColor.systemGreen
        cylinder.firstMaterial?.emission.contents = UIColor(red: 0.1, green: 0.9, blue: 0.5, alpha: 0.9)

        let beamNode = SCNNode(geometry: cylinder)
        beamNode.simdPosition = (start + end) * 0.5

        // Orient along delta vector
        let yAxis = simd_float3(0, 1, 0)
        let dir = normalize(delta)
        let dotVal = dot(yAxis, dir)

        if abs(dotVal) > 0.9999 {
            if dotVal < 0 {
                beamNode.simdOrientation = simd_quatf(angle: .pi, axis: simd_float3(1, 0, 0))
            }
        } else {
            let axis = normalize(cross(yAxis, dir))
            let angle = acos(simd_clamp(dotVal, -1.0, 1.0))
            beamNode.simdOrientation = simd_quatf(angle: angle, axis: axis)
        }

        parent.addChildNode(beamNode)
        return parent
    }
}
