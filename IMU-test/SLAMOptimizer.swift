//
//  SLAMOptimizer.swift
//  IMU-test
//
//  Created by JohnQuavious JeanCarlo on 9/28/26.
//

import Foundation
import Accelerate

// MARK: - Accelerate Sparse Cholesky Solver
class SLAMOptimizer {
    /// Solves the normal equation H * dx = b using Sparse Cholesky Decomposition
    func solveGaussNewton(size: Int32, hRows: [Int32], hCols: [Int32], hVals: [Double], b: [Double]) -> [Double]? {
        var rowIndices = hRows
        var colIndices = hCols
        var values = hVals

        var attributes = SparseAttributes_t()
        attributes.triangle = SparseLowerTriangle
        attributes.kind = SparseSymmetric

        let H_COO = SparseConvertFromCoordinate(
            size, size,
            values.count,
            1,
            attributes,
            &rowIndices,
            &colIndices,
            &values
        )
        defer { SparseCleanup(H_COO) }

        let factorization = SparseFactor(SparseFactorizationCholesky, H_COO)
        defer { SparseCleanup(factorization) }

        guard factorization.status == SparseStatusOK else {
            print("Accelerate Matrix Solver Failed with status: \(factorization.status)")
            return nil
        }

        var bVector = b
        var dxVector = Array(repeating: 0.0, count: Int(size))

        bVector.withUnsafeMutableBufferPointer { bPtr in
            dxVector.withUnsafeMutableBufferPointer { dxPtr in
                guard let bBase = bPtr.baseAddress, let dxBase = dxPtr.baseAddress else { return }
                let bDense = DenseVector_Double(count: size, data: bBase)
                let dxDense = DenseVector_Double(count: size, data: dxBase)
                SparseSolve(factorization, bDense, dxDense)
            }
        }

        return dxVector
    }
}
