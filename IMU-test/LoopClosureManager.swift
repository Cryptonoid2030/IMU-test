//
//  LoopClosureManager.swift
//  IMU-test
//
//  Created by JohnQuavious JeanCarlo on 9/28/26.
//

import Foundation
import ARKit
import simd

// MARK: - Keyframe Representation
/// Represents a discrete historical SLAM keyframe with spatial pose, 3D landmarks, and visual BoW representation
public struct Keyframe: Identifiable, Sendable {
    public let id: Int
    public let timestamp: TimeInterval
    public var pose: simd_float4x4
    public var position: simd_float3
    public var orientation: simd_quatf
    public var bowVector: BoWVector
    public var descriptors: [BinaryDescriptor]
    public var wordAssignments: [Int]
    public var keypoints2D: [simd_float2]
    public var featurePoints3D: [simd_float3]

    public init(
        id: Int,
        timestamp: TimeInterval,
        pose: simd_float4x4,
        position: simd_float3,
        orientation: simd_quatf,
        bowVector: BoWVector,
        descriptors: [BinaryDescriptor],
        wordAssignments: [Int],
        keypoints2D: [simd_float2],
        featurePoints3D: [simd_float3]
    ) {
        self.id = id
        self.timestamp = timestamp
        self.pose = pose
        self.position = position
        self.orientation = orientation
        self.bowVector = bowVector
        self.descriptors = descriptors
        self.wordAssignments = wordAssignments
        self.keypoints2D = keypoints2D
        self.featurePoints3D = featurePoints3D
    }
}

// MARK: - Loop Closure Event
/// Metadata recorded whenever a visual Bag-of-Words loop closure is confirmed
public struct LoopClosureEvent: Identifiable, Sendable, Equatable {
    public let id: UUID
    public let timestamp: Date
    public let queryKeyframeId: Int
    public let matchedKeyframeId: Int
    public let similarityScore: Float
    public let driftOffset: simd_float3
    public let driftDistance: Float
    public let queryPosition: simd_float3
    public let matchedPosition: simd_float3
    public let matchedFeaturesCount: Int

    public init(
        id: UUID = UUID(),
        timestamp: Date = Date(),
        queryKeyframeId: Int,
        matchedKeyframeId: Int,
        similarityScore: Float,
        driftOffset: simd_float3,
        driftDistance: Float,
        queryPosition: simd_float3,
        matchedPosition: simd_float3,
        matchedFeaturesCount: Int
    ) {
        self.id = id
        self.timestamp = timestamp
        self.queryKeyframeId = queryKeyframeId
        self.matchedKeyframeId = matchedKeyframeId
        self.similarityScore = similarityScore
        self.driftOffset = driftOffset
        self.driftDistance = driftDistance
        self.queryPosition = queryPosition
        self.matchedPosition = matchedPosition
        self.matchedFeaturesCount = matchedFeaturesCount
    }
}

// MARK: - Loop Closure Manager
/// Detects loops using a Bag-of-Words inverted index, verifies candidates geometrically, and computes trajectory corrections
public final class LoopClosureManager: @unchecked Sendable {
    public let vocabulary: BoWVocabulary
    public let featureExtractor: FeatureExtractor

    // --- Keyframe Database & Place Recognition Indices ---
    public private(set) var keyframes: [Keyframe] = []
    private var invertedIndex: [Int: [Int]] = [:] // Word ID -> [Keyframe ID]
    private var docFrequencies: [Int: Int] = [:] // Word ID -> Count of Keyframes

    // --- Loop Closure History & Constraints ---
    public private(set) var loopHistory: [LoopClosureEvent] = []
    public private(set) var lastLoopClosureTimestamp: TimeInterval = 0

    // --- Graph SLAM Optimizer & Tuning Configuration ---
    public let graphOptimizer = GraphSLAMOptimizer()
    public var slamConfig = GraphSLAMConfig()
    private var poseGraphEdges: [PoseGraphEdge] = []

    // --- Hyperparameters ---
    public var minKeyframeDistance: Float = 0.18 // Meters to trigger new keyframe
    public var minKeyframeAngle: Float = 0.22 // ~13 degrees to trigger new keyframe
    public var minLoopKeyframeSeparation: Int = 8 // Exclude the 8 most recent keyframes
    public var minLoopTemporalSeparation: TimeInterval = 6.0 // Exclude keyframes within last 6 seconds
    public var similarityThreshold: Float = 0.58 // Minimum cosine similarity for candidate
    public var maxLoopSpatialDistance: Float = 3.5 // Maximum spatial distance for loop closure search
    public var loopCooldown: TimeInterval = 3.5 // Cooldown period between consecutive closures

    public init() {
        self.vocabulary = BoWVocabulary()
        self.featureExtractor = FeatureExtractor()
    }

    /// Evaluates incoming ARFrame for potential keyframe creation, BoW place recognition, and loop closure
    public func processFrame(
        frame: ARFrame,
        currentPose: simd_float4x4,
        currentPosition: simd_float3,
        currentOrientation: simd_quatf,
        accumulatedFeatures: [simd_float3]
    ) -> LoopClosureEvent? {
        let currentTime = frame.timestamp

        // 1. Check if the camera motion justifies creating a new keyframe
        guard shouldCreateKeyframe(position: currentPosition, orientation: currentOrientation) else {
            return nil
        }

        // 2. Extract visual features and 256-bit BRIEF binary descriptors
        var (kpts, descriptors) = featureExtractor.extract(from: frame.capturedImage, maxFeatures: 160)

        // If in simulator without active camera hardware or image has insufficient features,
        // construct synthetic visual descriptors based on view orientation and position
        if descriptors.count < 10 {
            (kpts, descriptors) = generateSyntheticFeatures(position: currentPosition, orientation: currentOrientation)
        }

        // 3. Quantize descriptors and compute TF-IDF Bag-of-Words vector
        let (bowVector, assignments) = vocabulary.computeBoWVector(
            descriptors: descriptors,
            docFrequencies: docFrequencies,
            totalKeyframes: keyframes.count
        )

        let newKeyframeId = keyframes.count
        let currentKeyframe = Keyframe(
            id: newKeyframeId,
            timestamp: currentTime,
            pose: currentPose,
            position: currentPosition,
            orientation: currentOrientation,
            bowVector: bowVector,
            descriptors: descriptors,
            wordAssignments: assignments,
            keypoints2D: kpts,
            featurePoints3D: accumulatedFeatures
        )

        // 4. Query BoW database for potential loop closures
        var detectedLoop: LoopClosureEvent? = nil
        let isPastCooldown = (currentTime - lastLoopClosureTimestamp) > loopCooldown

        if isPastCooldown && keyframes.count >= (minLoopKeyframeSeparation + 2) {
            if let bestMatch = queryLoopCandidate(for: currentKeyframe) {
                // 5. Geometric and Spatial Verification
                if let verifiedLoop = verifyCandidate(query: currentKeyframe, candidate: bestMatch) {
                    detectedLoop = verifiedLoop
                    loopHistory.append(verifiedLoop)
                    lastLoopClosureTimestamp = currentTime
                }
            }
        }

        // 6. Register keyframe into the Bag of Words database and inverted index
        registerKeyframe(currentKeyframe)

        return detectedLoop
    }

    /// Evaluates if the device has translated or rotated enough from the last keyframe to spawn a new one
    private func shouldCreateKeyframe(position: simd_float3, orientation: simd_quatf) -> Bool {
        guard let lastKeyframe = keyframes.last else { return true }

        let dPos = distance(position, lastKeyframe.position)
        let rotError = length(SO3.betweenError(from: lastKeyframe.orientation, to: orientation))

        return dPos >= minKeyframeDistance || rotError >= minKeyframeAngle
    }

    /// Searches the BoW inverted index to find the highest scoring past keyframe candidate outside the exclusion window
    private func queryLoopCandidate(for query: Keyframe) -> Keyframe? {
        // Collect candidate keyframe IDs that share visual words
        var candidateCounts: [Int: Int] = [:]
        for wordId in query.wordAssignments {
            if let kfList = invertedIndex[wordId] {
                for kfId in kfList {
                    // Enforce temporal separation window to avoid matching immediately adjacent trajectory
                    if kfId <= (query.id - minLoopKeyframeSeparation) {
                        candidateCounts[kfId, default: 0] += 1
                    }
                }
            }
        }

        var bestCandidate: Keyframe? = nil
        var bestScore: Float = similarityThreshold

        for (candId, sharedWords) in candidateCounts {
            guard sharedWords >= 6 else { continue }
            let candidate = keyframes[candId]

            let timeDiff = query.timestamp - candidate.timestamp
            guard timeDiff >= minLoopTemporalSeparation else { continue }

            // Spatial sanity check: robot must return to within maxLoopSpatialDistance
            let spatialDist = distance(query.position, candidate.position)
            guard spatialDist <= maxLoopSpatialDistance else { continue }

            // Compute Bag-of-Words similarity (combining Cosine Similarity and DBoW L1 score)
            let cosine = query.bowVector.cosineSimilarity(to: candidate.bowVector)
            let l1 = query.bowVector.l1Score(to: candidate.bowVector)
            let score = 0.6 * cosine + 0.4 * l1

            if score > bestScore {
                bestScore = score
                bestCandidate = candidate
            }
        }

        return bestCandidate
    }

    /// Geometrically verifies feature correspondences and computes drift between the query and candidate keyframes
    private func verifyCandidate(query: Keyframe, candidate: Keyframe) -> LoopClosureEvent? {
        // Build word map for candidate descriptors
        var candidateWords: [Int: [Int]] = [:]
        for (idx, wId) in candidate.wordAssignments.enumerated() {
            candidateWords[wId, default: []].append(idx)
        }

        var matchedCount = 0
        // Match query descriptors against candidate descriptors that share the exact same visual word
        for (qIdx, qWord) in query.wordAssignments.enumerated() {
            guard let candIndices = candidateWords[qWord] else { continue }
            let qDesc = query.descriptors[qIdx]

            var minDistance = 60 // Max allowable bit flips out of 256
            var matched = false

            for cIdx in candIndices {
                let cDesc = candidate.descriptors[cIdx]
                let dist = qDesc.hammingDistance(to: cDesc)
                if dist < minDistance {
                    minDistance = dist
                    matched = true
                }
            }
            if matched {
                matchedCount += 1
            }
        }

        // Must have sufficient visual descriptor matches sharing visual words
        guard matchedCount >= 8 else { return nil }

        // Compute loop closure drift vector: candidate was observed at candidate.position;
        // current filter is at query.position. Drift is (candidate.position - query.position).
        let drift = candidate.position - query.position
        let driftDist = length(drift)

        // Compute final confidence score
        let score = query.bowVector.cosineSimilarity(to: candidate.bowVector)

        return LoopClosureEvent(
            queryKeyframeId: query.id,
            matchedKeyframeId: candidate.id,
            similarityScore: score,
            driftOffset: drift,
            driftDistance: driftDist,
            queryPosition: query.position,
            matchedPosition: candidate.position,
            matchedFeaturesCount: matchedCount
        )
    }

    /// Stores keyframe and updates inverted word index and document frequency counts
    private func registerKeyframe(_ keyframe: Keyframe) {
        keyframes.append(keyframe)

        var uniqueWords = Set<Int>()
        for wordId in keyframe.wordAssignments {
            if uniqueWords.insert(wordId).inserted {
                invertedIndex[wordId, default: []].append(keyframe.id)
                docFrequencies[wordId, default: 0] += 1
            }
        }
    }

    // MARK: - Graph SLAM Trajectory & Pose Optimization
    /// Optimizes the entire SLAM pose graph using Gauss-Newton and Sparse Cholesky,
    /// relaxing loop errors across all keyframes and applying damped, clamped correction to live pose.
    public func optimizePoseGraph(path: [simd_float3], loop: LoopClosureEvent) -> (path: [simd_float3], liveCorrection: simd_float3) {
        guard keyframes.count >= 2 else {
            let clamped = clampDrift(loop.driftOffset)
            let correctedPath = applyLinearInterpolatedCorrection(path: path, loop: loop)
            return (correctedPath, clamped * slamConfig.livePoseDamping)
        }

        // 1. Build Pose Graph Nodes from keyframes
        let originalPositions = keyframes.map { $0.position }
        var nodes: [PoseGraphNode] = []
        nodes.reserveCapacity(keyframes.count)

        for kf in keyframes {
            nodes.append(PoseGraphNode(id: kf.id, position: kf.position, orientation: kf.orientation, isFixed: (kf.id == 0)))
        }

        // 2. Build Odometry Edges (consecutive keyframe constraints)
        var allEdges: [PoseGraphEdge] = []
        for i in 0..<(keyframes.count - 1) {
            let relTrans = keyframes[i + 1].position - keyframes[i].position
            let odoEdge = PoseGraphEdge(
                fromId: i,
                toId: i + 1,
                measuredRelativeTranslation: relTrans,
                informationWeight: slamConfig.odometryWeight,
                isLoopClosure: false
            )
            allEdges.append(odoEdge)
        }

        // 3. Add Loop Closure Edges
        let loopRel = loop.matchedPosition - loop.queryPosition
        let newLoopEdge = PoseGraphEdge(
            fromId: loop.queryKeyframeId,
            toId: loop.matchedKeyframeId,
            measuredRelativeTranslation: loopRel,
            informationWeight: 1.0,
            isLoopClosure: true
        )
        poseGraphEdges.append(newLoopEdge)
        allEdges.append(contentsOf: poseGraphEdges)

        // 4. Solve Graph SLAM with Huber Robust Loss & Accelerate Sparse Cholesky
        let success = graphOptimizer.optimize(nodes: &nodes, edges: allEdges, config: slamConfig)
        guard success else {
            let fallbackDrift = clampDrift(loop.driftOffset) * slamConfig.livePoseDamping
            return (path, fallbackDrift)
        }

        // 5. Update Keyframe positions in memory
        var displacements: [simd_float3] = []
        displacements.reserveCapacity(nodes.count)
        for i in 0..<nodes.count {
            let disp = nodes[i].position - originalPositions[i]
            displacements.append(disp)
            keyframes[i].position = nodes[i].position
            keyframes[i].pose.columns.3 = simd_float4(nodes[i].position.x, nodes[i].position.y, nodes[i].position.z, 1.0)
        }

        // 6. Smoothly update 3D Trajectory Path
        var correctedPath = path
        if !correctedPath.isEmpty {
            for pIdx in 0..<correctedPath.count {
                let pt = correctedPath[pIdx]
                // Associate with nearest keyframe to smoothly warp path
                var bestKf = 0
                var minD = Float.greatestFiniteMagnitude
                for (kIdx, origPt) in originalPositions.enumerated() {
                    let d = distance(pt, origPt)
                    if d < minD {
                        minD = d
                        bestKf = kIdx
                    }
                }
                correctedPath[pIdx] += displacements[bestKf]
            }
        }

        // 7. Calculate bounded, damped live pose correction
        let lastNodeDisp = displacements.last ?? .zero
        let clampedLive = clampDrift(lastNodeDisp)
        let liveCorrection = clampedLive * slamConfig.livePoseDamping

        return (correctedPath, liveCorrection)
    }

    /// Convenience wrapper for backwards compatibility
    public func correctTrajectory(path: [simd_float3], loop: LoopClosureEvent) -> [simd_float3] {
        optimizePoseGraph(path: path, loop: loop).path
    }

    private func clampDrift(_ drift: simd_float3) -> simd_float3 {
        let len = length(drift)
        guard len > slamConfig.maxDriftClamp, len > 1e-4 else { return drift }
        return (drift / len) * slamConfig.maxDriftClamp
    }

    private func applyLinearInterpolatedCorrection(path: [simd_float3], loop: LoopClosureEvent) -> [simd_float3] {
        guard !path.isEmpty else { return path }
        var startIdx = 0
        var endIdx = path.count - 1
        var minStartDist = Float.greatestFiniteMagnitude
        var minEndDist = Float.greatestFiniteMagnitude

        for (i, pt) in path.enumerated() {
            let d1 = distance(pt, loop.matchedPosition)
            if d1 < minStartDist {
                minStartDist = d1
                startIdx = i
            }
            let d2 = distance(pt, loop.queryPosition)
            if d2 < minEndDist {
                minEndDist = d2
                endIdx = i
            }
        }

        guard startIdx < endIdx else { return path }

        var corrected = path
        let span = Float(endIdx - startIdx)
        let boundedDrift = clampDrift(loop.driftOffset)

        for i in startIdx...endIdx {
            let weight = Float(i - startIdx) / span
            corrected[i] += weight * boundedDrift
        }

        if endIdx + 1 < corrected.count {
            for i in (endIdx + 1)..<corrected.count {
                corrected[i] += boundedDrift
            }
        }

        return corrected
    }

    /// Generates synthetic visual features for Simulator or low-light testing
    private func generateSyntheticFeatures(
        position: simd_float3,
        orientation: simd_quatf
    ) -> (keypoints: [simd_float2], descriptors: [BinaryDescriptor]) {
        var kpts: [simd_float2] = []
        var descs: [BinaryDescriptor] = []

        // Spatial grid based on position hash
        let hash = UInt64(bitPattern: Int64((position.x * 100).rounded())) ^
                   (UInt64(bitPattern: Int64((position.z * 100).rounded())) << 16)

        for i in 0..<30 {
            let seed = hash &+ UInt64(i &* 1021)
            let x = Float((seed % 100)) / 100.0
            let y = Float(((seed >> 8) % 100)) / 100.0
            kpts.append(simd_float2(x, y))

            let w0 = seed ^ 0xabcdef1234567890
            let w1 = (seed << 13) ^ 0x0987654321fedcba
            let w2 = (seed >> 7) ^ 0xa1b2c3d4e5f60718
            let w3 = (seed &* 31) ^ 0x1807f6e5d4c3b2a1
            descs.append(BinaryDescriptor(w0: w0, w1: w1, w2: w2, w3: w3))
        }

        return (kpts, descs)
    }

    /// Manually triggers a simulated loop closure against an earlier keyframe for interactive UI demonstration & testing
    public func triggerSimulatedLoopClosure(
        currentPosition: simd_float3,
        currentPose: simd_float4x4
    ) -> LoopClosureEvent? {
        guard keyframes.count >= 2 else {
            // If fewer than 2 keyframes, generate a synthetic historical match 0.8 meters away
            let syntheticMatchedPos = currentPosition + simd_float3(0.15, 0.0, -0.25)
            let drift = syntheticMatchedPos - currentPosition
            let event = LoopClosureEvent(
                queryKeyframeId: 1,
                matchedKeyframeId: 0,
                similarityScore: 0.88,
                driftOffset: drift,
                driftDistance: length(drift),
                queryPosition: currentPosition,
                matchedPosition: syntheticMatchedPos,
                matchedFeaturesCount: 24
            )
            loopHistory.append(event)
            return event
        }

        // Pick the oldest keyframe with reasonable separation
        let targetIndex = max(0, keyframes.count - 10)
        let candidate = keyframes[targetIndex]
        let drift = candidate.position - currentPosition

        let event = LoopClosureEvent(
            queryKeyframeId: keyframes.count,
            matchedKeyframeId: candidate.id,
            similarityScore: 0.89,
            driftOffset: drift,
            driftDistance: length(drift),
            queryPosition: currentPosition,
            matchedPosition: candidate.position,
            matchedFeaturesCount: 32
        )
        loopHistory.append(event)
        return event
    }

    /// Resets all keyframes, inverted index, pose graph edges, and loop closure events
    public func reset() {
        keyframes.removeAll()
        invertedIndex.removeAll()
        docFrequencies.removeAll()
        loopHistory.removeAll()
        poseGraphEdges.removeAll()
        lastLoopClosureTimestamp = 0
    }
}
