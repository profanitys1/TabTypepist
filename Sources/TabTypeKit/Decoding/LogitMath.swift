import Accelerate
import Foundation

/// Log-probability helpers over raw logits. Full-vocabulary passes use Accelerate
/// (vocabularies reach 262k entries); restricted passes touch only the given ids.
public enum LogitMath {
    /// log Σ exp(logit) over the whole vocabulary, skipping blocked tokens (nil:
    /// include everything, so end-of-text mass lowers every other probability).
    public static func logSumExp(_ logits: Logits, blocked: [Bool]?, scratch: inout [Float]) -> Float {
        let values = logits.values
        let n = values.count
        if scratch.count < n { scratch = [Float](repeating: 0, count: n) }
        var maxValue: Float = 0
        vDSP_maxv(values.baseAddress!, 1, &maxValue, vDSP_Length(n))
        // scratch = exp(values - max)
        var negMax = -maxValue
        scratch.withUnsafeMutableBufferPointer { out in
            vDSP_vsadd(values.baseAddress!, 1, &negMax, out.baseAddress!, 1, vDSP_Length(n))
            var count = Int32(n)
            vvexpf(out.baseAddress!, out.baseAddress!, &count)
        }
        var sum: Float = 0
        vDSP_sve(scratch, 1, &sum, vDSP_Length(n))
        // Blocked tokens (control/EOG markers) are rare; subtract their mass.
        if let blocked {
            for i in 0..<n where blocked[i] { sum -= scratch[i] }
        }
        return maxValue + log(max(sum, .leastNormalMagnitude))
    }

    /// log Σ exp(logit) over just `ids`.
    public static func logSumExp(_ logits: Logits, over ids: [TokenID]) -> Float {
        guard !ids.isEmpty else { return -.infinity }
        var maxValue = -Float.infinity
        for id in ids { maxValue = max(maxValue, logits.values[Int(id)]) }
        var sum: Float = 0
        for id in ids { sum += exp(logits.values[Int(id)] - maxValue) }
        return maxValue + log(sum)
    }

    /// The `k` highest-logit tokens among `candidates` (or the whole unblocked
    /// vocabulary when nil), best first. Ties keep the lower token id first.
    public static func topK(_ logits: Logits, k: Int, candidates: [TokenID]?, blocked: [Bool]) -> [TokenID] {
        guard k > 0 else { return [] }
        var best: [(TokenID, Float)] = []
        best.reserveCapacity(k + 1)
        let values = logits.values
        // The bar a value must beat once `k` are held; almost every token fails
        // it, so the common case is one comparison.
        var bar = -Float.infinity
        func insert(_ id: Int, _ v: Float) {
            let insertAt = best.firstIndex { v > $0.1 } ?? best.count
            best.insert((TokenID(id), v), at: insertAt)
            if best.count > k { best.removeLast() }
            if best.count == k { bar = best[k - 1].1 }
        }
        if let candidates {
            for id in candidates {
                let v = values[Int(id)]
                if best.count == k, v <= bar { continue }
                insert(Int(id), v)
            }
        } else {
            blocked.withUnsafeBufferPointer { blocked in
                for id in 0..<values.count {
                    let v = values[id]
                    if v <= bar, best.count == k { continue }
                    if blocked[id] { continue }
                    insert(id, v)
                }
            }
        }
        return best.map(\.0)
    }

    // MARK: Batched analysis

    /// What the phrase search reads from one position's logits.
    public struct Analysis {
        /// log Σ exp(logit) over the whole vocabulary.
        public var logNormalizer: Float
        public var maxValue: Float
        public var argmax: Int
        /// The `k` best unblocked tokens, best first.
        public var top: [TokenID]
    }

    /// Batches at least this large are analysed on several cores (the GPU is idle
    /// while logits are read, and a full-vocabulary pass per phrase adds up over a
    /// 9-wide beam). Int.max turns it off.
    nonisolated(unsafe) public static var parallelBatchThreshold = 3
    /// Vocabularies below this are not worth a thread hop.
    static let parallelVocabularyFloor = 16_384

    /// One scratch buffer kept between calls: a fresh several-megabyte allocation
    /// per decode step would page-fault its way through every step.
    nonisolated(unsafe) private static var spareWorkspace: UnsafeMutableBufferPointer<Float>?
    private static let workspaceLock = NSLock()

    private static func takeWorkspace(atLeast count: Int) -> UnsafeMutableBufferPointer<Float> {
        workspaceLock.lock()
        let spare = spareWorkspace
        spareWorkspace = nil
        workspaceLock.unlock()
        if let spare, spare.count >= count { return spare }
        spare?.deallocate()
        return UnsafeMutableBufferPointer<Float>.allocate(capacity: count)
    }

    private static func returnWorkspace(_ buffer: UnsafeMutableBufferPointer<Float>) {
        workspaceLock.lock()
        let previous = spareWorkspace
        spareWorkspace = buffer
        workspaceLock.unlock()
        previous?.deallocate()
    }

    /// `Analysis` for every entry of a decoded batch. Results are identical to
    /// calling `logSumExp`, `vDSP_maxvi` and `topK` per entry.
    public static func analyze(_ batch: [Logits], k: Int, blocked: [Bool]) -> [Analysis] {
        guard !batch.isEmpty else { return [] }
        let n = batch[0].values.count
        func one(_ logits: Logits, _ scratch: UnsafeMutablePointer<Float>) -> Analysis {
            let values = logits.values
            var maxValue: Float = 0
            var maxIndex: vDSP_Length = 0
            vDSP_maxvi(values.baseAddress!, 1, &maxValue, &maxIndex, vDSP_Length(n))
            var negMax = -maxValue
            vDSP_vsadd(values.baseAddress!, 1, &negMax, scratch, 1, vDSP_Length(n))
            var count = Int32(n)
            vvexpf(scratch, scratch, &count)
            var sum: Float = 0
            vDSP_sve(scratch, 1, &sum, vDSP_Length(n))
            return Analysis(logNormalizer: maxValue + log(max(sum, .leastNormalMagnitude)),
                            maxValue: maxValue, argmax: Int(maxIndex),
                            top: topK(logits, k: k, candidates: nil, blocked: blocked))
        }
        let parallel = batch.count >= parallelBatchThreshold && n >= parallelVocabularyFloor
        let needed = n * (parallel ? batch.count : 1)
        let workspace = takeWorkspace(atLeast: needed)
        defer { returnWorkspace(workspace) }
        let scratch = workspace.baseAddress!
        guard parallel else { return batch.map { one($0, scratch) } }
        let results = UnsafeMutablePointer<Analysis>.allocate(capacity: batch.count)
        defer { results.deallocate() }
        batch.withUnsafeBufferPointer { batch in
            DispatchQueue.concurrentPerform(iterations: batch.count) { i in
                (results + i).initialize(to: one(batch[i], scratch + i * n))
            }
        }
        let out = Array(UnsafeBufferPointer(start: results, count: batch.count))
        results.deinitialize(count: batch.count)
        return out
    }
}
