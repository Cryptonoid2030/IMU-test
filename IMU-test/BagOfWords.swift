//
//  BagOfWords.swift
//  IMU-test
//
//  Created by JohnQuavious JeanCarlo on 9/28/26.
//

import Foundation
import CoreVideo
import simd

// MARK: - 256-Bit Binary Descriptor
/// A 256-bit binary descriptor (such as BRIEF or ORB) optimized for SIMD Hamming distance calculation
public struct BinaryDescriptor: Equatable, Hashable, Sendable {
    public var w0: UInt64
    public var w1: UInt64
    public var w2: UInt64
    public var w3: UInt64

    public init(w0: UInt64 = 0, w1: UInt64 = 0, w2: UInt64 = 0, w3: UInt64 = 0) {
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
    }

    /// Computes Hamming distance (number of bit differences) in O(1) via hardware popcount
    @inline(__always)
    public func hammingDistance(to other: BinaryDescriptor) -> Int {
        let d0 = (w0 ^ other.w0).nonzeroBitCount
        let d1 = (w1 ^ other.w1).nonzeroBitCount
        let d2 = (w2 ^ other.w2).nonzeroBitCount
        let d3 = (w3 ^ other.w3).nonzeroBitCount
        return d0 + d1 + d2 + d3
    }
}

// MARK: - Visual Word
/// Represents a visual word (cluster centroid) in the 256-bit descriptor space
public struct VisualWord: Identifiable, Sendable {
    public let id: Int
    public var centroid: BinaryDescriptor
    public var idfWeight: Float

    public init(id: Int, centroid: BinaryDescriptor, idfWeight: Float = 1.0) {
        self.id = id
        self.centroid = centroid
        self.idfWeight = idfWeight
    }
}

// MARK: - Bag of Words Vector
/// Sparse Bag-of-Words frequency and TF-IDF representation of an image / keyframe
public struct BoWVector: Sendable {
    /// Maps visual word ID -> TF-IDF normalized weight
    public var weights: [Int: Float] = [:]

    /// Euclidean (L2) norm of the weights
    public var l2Norm: Float = 0.0

    public init(weights: [Int: Float] = [:]) {
        self.weights = weights
        var sumSquares: Float = 0.0
        for (_, w) in weights {
            sumSquares += w * w
        }
        self.l2Norm = sqrt(max(sumSquares, 1e-8))
    }

    /// Computes Cosine Similarity between two BoW vectors: dot(v1, v2) / (|v1| * |v2|) in [0, 1]
    public func cosineSimilarity(to other: BoWVector) -> Float {
        guard !weights.isEmpty && !other.weights.isEmpty else { return 0.0 }
        var dot: Float = 0.0

        if weights.count <= other.weights.count {
            for (wordId, w1) in weights {
                if let w2 = other.weights[wordId] {
                    dot += w1 * w2
                }
            }
        } else {
            for (wordId, w2) in other.weights {
                if let w1 = weights[wordId] {
                    dot += w1 * w2
                }
            }
        }

        let denom = l2Norm * other.l2Norm
        return denom > 1e-6 ? min(max(dot / denom, 0.0), 1.0) : 0.0
    }

    /// Computes DBoW L1 score: 1 - 0.5 * ||v1 - v2||_1 = sum(min(v1, v2)) in [0, 1]
    public func l1Score(to other: BoWVector) -> Float {
        guard !weights.isEmpty && !other.weights.isEmpty else { return 0.0 }
        var score: Float = 0.0
        for (wordId, w1) in weights {
            if let w2 = other.weights[wordId] {
                score += min(w1, w2)
            }
        }
        return min(max(score, 0.0), 1.0)
    }
}

// MARK: - Visual Vocabulary (BoW Dictionary)
/// Manages quantization of binary descriptors into visual words and computes TF-IDF representations
public final class BoWVocabulary: @unchecked Sendable {
    public static let vocabularySize: Int = 256
    public private(set) var words: [VisualWord] = []

    public init() {
        self.words = Self.generatePredefinedVocabulary(size: Self.vocabularySize)
    }

    /// Deterministically initializes a diverse set of 256 visual word centroids spanning the 256-bit Hamming space
    private static func generatePredefinedVocabulary(size: Int) -> [VisualWord] {
        var result: [VisualWord] = []
        result.reserveCapacity(size)

        // Seeded 64-bit Linear Congruential Generator for deterministic, uniformly spaced centroids
        var state: UInt64 = 0x85435805a4f767dc

        func nextRandom64() -> UInt64 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return state
        }

        for i in 0..<size {
            // Mix sequential bits and pseudo-random variance so each centroid is distinct
            let prefix = UInt64(i)
            let w0 = nextRandom64() ^ (prefix << 56)
            let w1 = nextRandom64() ^ (prefix << 48)
            let w2 = nextRandom64() ^ (prefix << 40)
            let w3 = nextRandom64() ^ (prefix << 32)
            let descriptor = BinaryDescriptor(w0: w0, w1: w1, w2: w2, w3: w3)
            result.append(VisualWord(id: i, centroid: descriptor, idfWeight: 1.0))
        }

        return result
    }

    /// Quantizes a single descriptor to its nearest visual word ID via minimum Hamming distance
    @inline(__always)
    public func quantize(descriptor: BinaryDescriptor) -> Int {
        var bestId = 0
        var bestDist = Int.max

        for i in 0..<words.count {
            let dist = descriptor.hammingDistance(to: words[i].centroid)
            if dist < bestDist {
                bestDist = dist
                bestId = i
                if dist <= 12 { break } // Early exit for near-exact match
            }
        }
        return bestId
    }

    /// Computes the normalized BoW vector for a set of visual descriptors using TF-IDF weighting
    public func computeBoWVector(
        descriptors: [BinaryDescriptor],
        docFrequencies: [Int: Int],
        totalKeyframes: Int
    ) -> (vector: BoWVector, wordAssignments: [Int]) {
        guard !descriptors.isEmpty else {
            return (BoWVector(weights: [:]), [])
        }

        var wordCounts: [Int: Int] = [:]
        var assignments: [Int] = []
        assignments.reserveCapacity(descriptors.count)

        for desc in descriptors {
            let wordId = quantize(descriptor: desc)
            assignments.append(wordId)
            wordCounts[wordId, default: 0] += 1
        }

        let totalDescCount = Float(descriptors.count)
        var tfIdfWeights: [Int: Float] = [:]

        for (wordId, count) in wordCounts {
            let tf = Float(count) / totalDescCount
            let docCount = docFrequencies[wordId] ?? 1
            // Smooth IDF: ln((1 + N) / (1 + n_w)) + 1.0
            let idf = logf(Float(totalKeyframes + 1) / Float(docCount + 1)) + 1.0
            tfIdfWeights[wordId] = tf * idf
        }

        // L2 Normalization
        var sumSquares: Float = 0.0
        for (_, w) in tfIdfWeights {
            sumSquares += w * w
        }
        let norm = sqrt(max(sumSquares, 1e-8))
        for (wordId, w) in tfIdfWeights {
            tfIdfWeights[wordId] = w / norm
        }

        return (BoWVector(weights: tfIdfWeights), assignments)
    }
}

// MARK: - FAST & BRIEF Visual Feature Extractor
/// Extracts corner keypoints and 256-bit BRIEF binary descriptors directly from camera luminance frames
public final class FeatureExtractor: @unchecked Sendable {
    // 256 pre-defined coordinate test pairs (dx1, dy1) vs (dx2, dy2) in a 31x31 patch
    private static let briefPairs: [(x1: Int, y1: Int, x2: Int, y2: Int)] = {
        var pairs: [(x1: Int, y1: Int, x2: Int, y2: Int)] = []
        pairs.reserveCapacity(256)
        var rng: UInt64 = 0x9e3779b97f4a7c15

        func nextOffset() -> Int {
            rng = rng &* 2862933555777941757 &+ 3037000493
            let val = Int((rng >> 32) % 31) - 15
            return val
        }

        for _ in 0..<256 {
            let x1 = nextOffset()
            let y1 = nextOffset()
            let x2 = nextOffset()
            let y2 = nextOffset()
            pairs.append((x1, y1, x2, y2))
        }
        return pairs
    }()

    public init() {}

    /// Extracts keypoint locations and 256-bit binary descriptors from a CVPixelBuffer (Plane 0 luminance)
    public func extract(from pixelBuffer: CVPixelBuffer, maxFeatures: Int = 180) -> (keypoints: [simd_float2], descriptors: [BinaryDescriptor]) {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let planeCount = CVPixelBufferGetPlaneCount(pixelBuffer)
        let baseAddress: UnsafeMutableRawPointer?
        let width: Int
        let height: Int
        let bytesPerRow: Int

        if planeCount > 0 {
            baseAddress = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0)
            width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
            height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
            bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
        } else {
            baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)
            width = CVPixelBufferGetWidth(pixelBuffer)
            height = CVPixelBufferGetHeight(pixelBuffer)
            bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        }

        guard let buffer = baseAddress?.assumingMemoryBound(to: UInt8.self), width > 64, height > 64 else {
            return ([], [])
        }

        var keypoints: [simd_float2] = []
        var descriptors: [BinaryDescriptor] = []
        keypoints.reserveCapacity(maxFeatures)
        descriptors.reserveCapacity(maxFeatures)

        // Divide frame into a regular grid to ensure spatial coverage across the field of view
        let gridCols = 15
        let gridRows = 12
        let cellW = (width - 40) / gridCols
        let cellH = (height - 40) / gridRows

        for row in 0..<gridRows {
            for col in 0..<gridCols {
                if keypoints.count >= maxFeatures { break }

                let startX = 20 + col * cellW
                let startY = 20 + row * cellH
                let centerX = startX + cellW / 2
                let centerY = startY + cellH / 2

                // Sample candidate with highest local contrast in this cell
                var bestX = centerX
                var bestY = centerY
                var maxScore = 0

                let step = max(2, cellW / 4)
                for py in stride(from: startY + 4, to: startY + cellH - 4, by: step) {
                    for px in stride(from: startX + 4, to: startX + cellW - 4, by: step) {
                        guard px >= 16 && px < width - 16 && py >= 16 && py < height - 16 else { continue }
                        let centerVal = Int(buffer[py * bytesPerRow + px])
                        let top = Int(buffer[(py - 3) * bytesPerRow + px])
                        let bottom = Int(buffer[(py + 3) * bytesPerRow + px])
                        let left = Int(buffer[py * bytesPerRow + (px - 3)])
                        let right = Int(buffer[py * bytesPerRow + (px + 3)])

                        let score = abs(centerVal - top) + abs(centerVal - bottom) + abs(centerVal - left) + abs(centerVal - right)
                        if score > maxScore {
                            maxScore = score
                            bestX = px
                            bestY = py
                        }
                    }
                }

                guard maxScore > 25 else { continue }

                // Compute 256-bit BRIEF binary descriptor at (bestX, bestY)
                var w0: UInt64 = 0
                var w1: UInt64 = 0
                var w2: UInt64 = 0
                var w3: UInt64 = 0

                for i in 0..<256 {
                    let pair = Self.briefPairs[i]
                    let p1Val = buffer[(bestY + pair.y1) * bytesPerRow + (bestX + pair.x1)]
                    let p2Val = buffer[(bestY + pair.y2) * bytesPerRow + (bestX + pair.x2)]
                    let bit: UInt64 = (p1Val < p2Val) ? 1 : 0

                    if i < 64 {
                        w0 |= (bit << (i & 63))
                    } else if i < 128 {
                        w1 |= (bit << (i & 63))
                    } else if i < 192 {
                        w2 |= (bit << (i & 63))
                    } else {
                        w3 |= (bit << (i & 63))
                    }
                }

                let normX = Float(bestX) / Float(width)
                let normY = Float(bestY) / Float(height)
                keypoints.append(simd_float2(normX, normY))
                descriptors.append(BinaryDescriptor(w0: w0, w1: w1, w2: w2, w3: w3))
            }
        }

        return (keypoints, descriptors)
    }
}
