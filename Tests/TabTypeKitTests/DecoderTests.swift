import XCTest
@testable import TabTypeKit

final class HealSplitTests: XCTestCase {
    func testBoundaryHealsTheSpace() {
        XCTAssertEqual(HealSplit.split("I'll take a look "), HealSplit(prompt: "I'll take a look", heal: [0x20]))
    }

    func testMidwordHealsSpacePlusPartial() {
        XCTAssertEqual(HealSplit.split("please doub"), HealSplit(prompt: "please", heal: Array(" doub".utf8)))
    }

    func testWordAtStartOfTextHasNoSpace() {
        XCTAssertEqual(HealSplit.split("Doub"), HealSplit(prompt: "", heal: Array("Doub".utf8)))
    }

    func testWordAfterNewlineKeepsNewlineInPrompt() {
        XCTAssertEqual(HealSplit.split("Hi,\nThan"), HealSplit(prompt: "Hi,\n", heal: Array("Than".utf8)))
    }

    func testNewlineEndingHasNothingToHeal() {
        XCTAssertEqual(HealSplit.split("Hi,\n"), HealSplit(prompt: "Hi,\n", heal: []))
    }

    func testPunctuationIsPartOfThePartial() {
        XCTAssertEqual(HealSplit.split("see e.g"), HealSplit(prompt: "see", heal: Array(" e.g".utf8)))
    }

    func testOverlongRunsAreNotHealed() {
        let url = "go to https://example.com/some/very/long/path"
        XCTAssertEqual(HealSplit.split(url), HealSplit(prompt: url, heal: []))
    }
}

final class VocabIndexTests: XCTestCase {
    func testConsistentTokensCoverBothDirections() {
        let vocab = VocabIndex(
            pieces: [" t", " th", " the", " there", " a", "the", "<eos>"].map { Array($0.utf8) },
            blocked: [false, false, false, false, false, false, true],
            endOfGeneration: [false, false, false, false, false, false, true])
        let ids = Set(vocab.tokens(consistentWith: Array(" the".utf8)[...]))
        // Inside the requirement (" t", " th") or extending it (" the", " there");
        // never a token with a different start or a blocked one.
        XCTAssertEqual(ids, [0, 1, 2, 3])
    }
}

final class CompletionDecoderTests: XCTestCase {
    private let pieces = ["<eos>", " ", " t", " th", " the", " ther", " there", "re",
                          " cat", " dog", " sat", " on", ".", "\n", "Hi", " you"]

    private func model(_ rules: [String: [String: Float]]) -> FakeModel {
        FakeModel(pieces: pieces, rules: rules)
    }

    func testBoundaryCompletesNextWordWithoutLeadingSpace() throws {
        let m = model(["Hi": [" there": 4, " the": 1]])
        let r = try XCTUnwrap(CompletionDecoder.complete("Hi ", model: m))
        XCTAssertEqual(r.text, "there")
        XCTAssertGreaterThan(r.confidence, 0.9)
    }

    func testMidwordCompletesTheTypedWord() throws {
        // Typed "Hi the": the decoder must regenerate " the" and may extend it.
        let m = model(["Hi": [" there": 4, " ther": 2, " the": 1]])
        let r = try XCTUnwrap(CompletionDecoder.complete("Hi the", model: m))
        XCTAssertEqual(r.text, "re")
    }

    func testDifferentTokenizationsOfOneWordMerge() throws {
        // " there" directly, or " the" + "re": same visible word, probabilities add.
        let m = model(["Hi": [" there": 2, " the": 2], "the": ["re": 10]])
        let r = try XCTUnwrap(CompletionDecoder.complete("Hi ", model: m))
        XCTAssertEqual(r.text, "there")
        XCTAssertGreaterThan(r.confidence, 0.95)
        XCTAssertTrue(r.alternatives.isEmpty)
    }

    func testExtendsWhileNextWordsAreLikely() throws {
        let rules: [String: [String: Float]] = [
            "on": [" the": 5], "the": [" cat": 3, " dog": 2],
            "cat": [" sat": 4, ".": 1], "sat": [" on": 5],
        ]
        var options = DecoderOptions()
        options.extensionThreshold = 0.3
        options.maxWords = 4
        let r = try XCTUnwrap(CompletionDecoder.complete("on ", model: model(rules), options: options))
        XCTAssertEqual(r.text, "the cat sat on")
        XCTAssertEqual(r.words.map(\.text), ["the", " cat", " sat", " on"])

        options.extensionThreshold = 0.8   // " cat" is only ~0.73 likely
        let short = try XCTUnwrap(CompletionDecoder.complete("on ", model: model(rules), options: options))
        XCTAssertEqual(short.text, "the")
        XCTAssertTrue(short.predictedTrailingSpace)
    }

    func testNeverOpensWithALineBreak() throws {
        let m = model(["Hey\n": ["\n": 5, "Hi": 2]])
        let r = try XCTUnwrap(CompletionDecoder.complete("Hey\n", model: m))
        XCTAssertEqual(r.text, "Hi")
    }

    func testStopsAtLineBreakAndEndOfText() throws {
        let m = model(["Hi": [" there": 5], "there": ["\n": 5]])
        var options = DecoderOptions()
        options.extensionThreshold = 0
        let r = try XCTUnwrap(CompletionDecoder.complete("Hi ", model: m, options: options))
        XCTAssertEqual(r.text, "there")
        XCTAssertFalse(r.predictedTrailingSpace)
    }

    func testAlternativesAreTheRunnerUpWords() throws {
        let m = model(["the": [" cat": 3, " dog": 2.5]])
        var options = DecoderOptions()
        options.extensionThreshold = 2   // first word only
        let r = try XCTUnwrap(CompletionDecoder.complete("Hi the ", model: m, options: options))
        XCTAssertEqual(r.text, "cat")
        XCTAssertEqual(r.alternatives.first?.text, "dog")
    }

    func testCandidateSequencesAreReleased() throws {
        let m = model(["Hi": [" there": 4, " the": 3, " ther": 2]])
        _ = try CompletionDecoder.complete("Hi ", model: m)
        XCTAssertEqual(m.liveCandidateSequences, [])
    }

    func testCancellationAbortsBetweenSteps() {
        let m = model(["Hi": [" there": 4]])
        XCTAssertThrowsError(try CompletionDecoder.complete("Hi ", model: m, isCancelled: { true })) {
            XCTAssertTrue($0 is CancellationError)
        }
        XCTAssertEqual(m.liveCandidateSequences, [])
    }

    // MARK: Show-bar gate and early first word

    private let phraseRules: [String: [String: Float]] = [
        "on": [" the": 5], "the": [" cat": 3, " dog": 2],
        "cat": [" sat": 4, ".": 1], "sat": [" on": 5],
    ]

    func testFirstWordBelowTheShowBarSkipsThePhraseSearch() throws {
        var options = DecoderOptions()
        options.extensionThreshold = 0.3
        let searched = model(phraseRules)
        let full = try XCTUnwrap(CompletionDecoder.complete("on the ", model: searched, options: options))
        XCTAssertEqual(full.text, "cat sat on the")

        // "cat" is ~0.73 likely: a caller that needs 0.9 to show it never
        // displays the phrase, so no passes are spent finding it.
        options.minimumConfidenceToExtend = 0.9
        let gated = model(phraseRules)
        let short = try XCTUnwrap(CompletionDecoder.complete("on the ", model: gated, options: options))
        XCTAssertEqual(short.text, "cat")
        XCTAssertEqual(short.confidence, full.words[0].probability, accuracy: 1e-6)
        XCTAssertLessThan(gated.decodeCalls, searched.decodeCalls)
        XCTAssertEqual(gated.liveCandidateSequences, [])

        // The hint scale lowers the bar only for a first word that follows the hint.
        options.hint = Array("cat".utf8)
        options.hintFollowerConfidenceScale = 0.5
        let hinted = try XCTUnwrap(CompletionDecoder.complete("on the ", model: model(phraseRules), options: options))
        XCTAssertTrue(hinted.text.hasPrefix("cat sat"), hinted.text)
    }

    func testFirstWordIsReportedBeforeThePhrase() throws {
        var options = DecoderOptions()
        options.extensionThreshold = 0.3
        var early: [CompletionResult] = []
        let r = try XCTUnwrap(CompletionDecoder.complete("on the ", model: model(phraseRules), options: options,
                                                         onFirstWord: { early.append($0) }))
        XCTAssertEqual(early.count, 1)
        XCTAssertEqual(early.first?.text, "cat")
        XCTAssertTrue(r.text.hasPrefix(early.first?.text ?? "?"))
        XCTAssertEqual(early.first?.confidence ?? 0, r.words[0].probability, accuracy: 1e-6)
    }

    func testFirstWordIsNotReportedWithoutAPhraseSearch() throws {
        var early = 0
        var options = DecoderOptions()
        options.extensionThreshold = 2   // single words: nothing to wait for
        _ = try CompletionDecoder.complete("on the ", model: model(phraseRules), options: options,
                                           onFirstWord: { _ in early += 1 })
        XCTAssertEqual(early, 0)

        options.extensionThreshold = 0.3
        options.minimumConfidenceToExtend = 0.9   // below the show bar
        _ = try CompletionDecoder.complete("on the ", model: model(phraseRules), options: options,
                                           onFirstWord: { _ in early += 1 })
        XCTAssertEqual(early, 0)
    }

    func testNothingLikelyMeansNoSuggestion() throws {
        // Only end-of-text is plausible after the prompt.
        let m = model([:])
        XCTAssertNil(try CompletionDecoder.complete("Done.\n", model: m))
    }
}

final class RetrievalHintTests: XCTestCase {
    private let pieces = ["<eos>", " cat", " dog", " d", "og", " sat", " on", " the"]

    func testHintWinsCloseCallsWithHonestConfidence() throws {
        let m = FakeModel(pieces: pieces, rules: ["the": [" cat": 3, " dog": 2.5], "dog": [" sat": 5], "cat": [" sat": 5]])
        var options = DecoderOptions()
        options.extensionThreshold = 2
        let plain = try XCTUnwrap(CompletionDecoder.complete("Hi the ", model: m, options: options))
        XCTAssertEqual(plain.text, "cat")
        XCTAssertFalse(plain.followsHint)

        options.hint = Array("dog sat".utf8)
        let hinted = try XCTUnwrap(CompletionDecoder.complete("Hi the ", model: m, options: options))
        XCTAssertEqual(hinted.text, "dog")
        XCTAssertTrue(hinted.followsHint)
        XCTAssertLessThan(hinted.confidence, 0.5, "reported confidence is the model's own, not boosted")
    }

    func testImplausibleHintIsIgnored() throws {
        let m = FakeModel(pieces: pieces, rules: ["the": [" cat": 6]])
        var options = DecoderOptions()
        options.hint = Array("dog".utf8)
        let r = try XCTUnwrap(CompletionDecoder.complete("Hi the ", model: m, options: options))
        XCTAssertEqual(r.text, "cat")
        XCTAssertFalse(r.followsHint)
    }

    func testHintIsFollowedAcrossTokens() throws {
        // " dog" isn't one token here: the hint path goes " d" + "og".
        let m = FakeModel(pieces: ["<eos>", " cat", " d", "og", " sat"],
                          rules: ["the": [" cat": 2.6, " d": 2.5], " d": ["og": 4], "og": [" sat": 5], "cat": [" sat": 5]])
        var options = DecoderOptions()
        options.extensionThreshold = 2
        options.hint = Array("dog".utf8)
        let r = try XCTUnwrap(CompletionDecoder.complete("Hi the ", model: m, options: options))
        XCTAssertEqual(r.text, "dog")
        XCTAssertTrue(r.followsHint)
        XCTAssertEqual(m.liveCandidateSequences, [])
    }
}

final class WordsThatFitTests: XCTestCase {
    func testOffersOtherFittingWordsWithoutTheSelectedOne() throws {
        let m = FakeModel(pieces: ["<eos>", " cat", " dog", " sat", " on"],
                          rules: ["the": [" cat": 3, " dog": 2.5, " sat": 1], "cat": [" sat": 5], "dog": [" sat": 5]])
        let words = try CompletionDecoder.wordsThatFit(after: "Hi the ", excluding: "cat", model: m)
        XCTAssertEqual(words.first?.text, "dog")
        XCTAssertFalse(words.contains { $0.text.lowercased() == "cat" })
        XCTAssertEqual(m.liveCandidateSequences, [])
    }
}

final class GenerationGateTests: XCTestCase {
    func testNewerRequestInvalidatesOlder() {
        let gate = GenerationGate()
        let first = gate.next()
        XCTAssertTrue(gate.isCurrent(first))
        let second = gate.next()
        XCTAssertFalse(gate.isCurrent(first))
        XCTAssertTrue(gate.isCurrent(second))
    }

}

final class BeamSearchTests: XCTestCase {
    /// Searching from every first word, the beam ranks whole phrases: "look into"
    /// (≈0.45 × 1) beats "take a" (≈0.55 × 1/3) though "take" alone is likelier.
    func testBeamPrefersTheMoreProbablePhrase() throws {
        let m = FakeModel(pieces: ["<eos>", " take", " look", " a", " the", " it", " into"], rules: [
            "I will": [" take": 3.0, " look": 2.8],
            "will take": [" a": 1, " the": 1, " it": 1],
            "will look": [" into": 8],
            "look into": [" it": 8],
        ])
        var options = DecoderOptions()
        options.maxWords = 2
        options.extensionThreshold = 0
        options.beamFromBestFirstWord = false
        let r = try XCTUnwrap(CompletionDecoder.complete("I will", model: m, options: options))
        XCTAssertEqual(r.text, " look into")
        XCTAssertEqual(r.confidence, 0.45, accuracy: 0.03)
        XCTAssertEqual(r.alternatives.first?.text, " take")
        XCTAssertTrue(m.liveCandidateSequences.isEmpty, "beam sequences must be released")
    }
}

final class LogitMathTests: XCTestCase {
    private func randomLogits(_ n: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0..<n).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            // Coarse values so ties occur.
            return Float(Int(state >> 40) % 400) / 10 - 20
        }
    }

    private func referenceTopK(_ v: [Float], k: Int, blocked: [Bool]) -> [TokenID] {
        v.indices.filter { !blocked[$0] }
            .sorted { v[$0] != v[$1] ? v[$0] > v[$1] : $0 < $1 }
            .prefix(k).map { TokenID($0) }
    }

    func testTopKMatchesASortedReference() {
        let n = 5000
        let blocked = (0..<n).map { $0 % 7 == 0 }
        for seed in 1...4 {
            let v = randomLogits(n, seed: UInt64(seed))
            v.withUnsafeBufferPointer { buffer in
                let logits = Logits(buffer)
                for k in [1, 3, 9] {
                    XCTAssertEqual(LogitMath.topK(logits, k: k, candidates: nil, blocked: blocked),
                                   referenceTopK(v, k: k, blocked: blocked))
                }
                let some = stride(from: 1, to: n, by: 3).map { TokenID($0) }
                let expected = some.sorted { v[Int($0)] != v[Int($1)] ? v[Int($0)] > v[Int($1)] : $0 < $1 }.prefix(5)
                XCTAssertEqual(LogitMath.topK(logits, k: 5, candidates: some, blocked: blocked), Array(expected))
            }
        }
    }

    /// The batched (and, for large batches, multi-core) analysis must read the
    /// same numbers the per-entry helpers do.
    func testBatchedAnalysisMatchesPerEntryHelpers() {
        let n = 20_000   // above the parallel floor
        let blocked = (0..<n).map { $0 % 11 == 0 }
        let rows = (1...6).map { randomLogits(n, seed: UInt64($0 * 97)) }
        let buffers = rows.map { row -> UnsafeMutableBufferPointer<Float> in
            let b = UnsafeMutableBufferPointer<Float>.allocate(capacity: n)
            _ = b.initialize(from: row)
            return b
        }
        defer { buffers.forEach { $0.deallocate() } }
        let batch = buffers.map { Logits(UnsafeBufferPointer($0)) }

        let saved = LogitMath.parallelBatchThreshold
        defer { LogitMath.parallelBatchThreshold = saved }
        var scratch: [Float] = []
        for threshold in [1, Int.max] {   // multi-core, then serial
            LogitMath.parallelBatchThreshold = threshold
            let analyses = LogitMath.analyze(batch, k: 9, blocked: blocked)
            XCTAssertEqual(analyses.count, rows.count)
            for (i, a) in analyses.enumerated() {
                XCTAssertEqual(a.maxValue, rows[i].max())
                XCTAssertEqual(rows[i][a.argmax], a.maxValue)
                XCTAssertEqual(a.logNormalizer, LogitMath.logSumExp(batch[i], blocked: nil, scratch: &scratch),
                               accuracy: 1e-4)
                XCTAssertEqual(a.top, referenceTopK(rows[i], k: 9, blocked: blocked))
            }
        }
    }
}
