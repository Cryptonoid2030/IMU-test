//
//  PoseView.swift
//  IMU-test
//
//  Created by JohnQuavious JeanCarlo on 9/28/26.
//  Visualization for point cloud and pose data
//

import SwiftUI
import SceneKit
import ARKit

struct PathView: UIViewRepresentable {
    var transform: simd_float4x4
    var path: [simd_float3]
    var featurePoints: [simd_float3]

    class Coordinator {
        var lastRenderedPointCount: Int = 0
        var lastFirstPoint: simd_float3 = .zero
        var lastFinalPoint: simd_float3 = .zero
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

        return view
    }

    func updateUIView(_ uiView: SCNView, context: Context) {
        if let phoneNode = uiView.scene?.rootNode.childNode(withName: "phone", recursively: false) {
            phoneNode.simdTransform = transform
        }

        if let pathContainer = uiView.scene?.rootNode.childNode(withName: "pathContainer", recursively: false) {
            let currentDots = pathContainer.childNodes.count
            if path.count < currentDots {
                // Path was cleared via recenter()
                pathContainer.childNodes.forEach { $0.removeFromParentNode() }
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

        // Render the persistent point cloud efficiently, only rebuilding geometry when the cloud updates
        if let cloudContainer = uiView.scene?.rootNode.childNode(withName: "cloudContainer", recursively: false) {
            if featurePoints.isEmpty {
                if !cloudContainer.childNodes.isEmpty {
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
}
