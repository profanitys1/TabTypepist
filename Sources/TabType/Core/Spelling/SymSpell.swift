import Foundation

/// Symmetric-delete spelling corrector (Wolf Garbe's SymSpell algorithm).
/// Builds a delete-index once (in the background) and answers nearest-word lookups.
final class SymSpell {
    private let maxEditDistance: Int
    private let prefixLength: Int

    private var words: [String: Int] = [:]                 // word -> frequency
    private var deletes: [String: [String]] = [:]          // delete-variant -> words
    /// Every word, ordered by UTF-8 bytes: the words starting with a prefix are
    /// one contiguous run, found by binary search instead of a scan of the whole
    /// dictionary (the typo gate asks before every mid-word suggestion).
    private var sortedWords: [String] = []
    private(set) var isReady = false

    init(maxEditDistance: Int = 2, prefixLength: Int = 7) {
        self.maxEditDistance = maxEditDistance
        self.prefixLength = prefixLength
    }

    /// Parse a "word<space>frequency" dictionary and build the index.
    func load(contents: String) {
        for line in contents.split(separator: "\n") {
            let parts = line.split(separator: " ")
            guard parts.count >= 2 else { continue }
            let word = parts[0].trimmingCharacters(in: CharacterSet(charactersIn: "\u{FEFF} \t\r"))
                .lowercased()
            guard !word.isEmpty, let freq = Int(parts[1]) else { continue }
            words[word] = freq
        }
        for word in words.keys {
            for d in editsPrefix(word) {
                deletes[d, default: []].append(word)
            }
        }
        sortedWords = words.keys.sorted { $0.utf8.lexicographicallyPrecedes($1.utf8) }
        isReady = true
    }

    /// The run of `sortedWords` whose UTF-8 bytes start with `prefix`'s.
    private func wordsWithBytePrefix(_ prefix: String) -> ArraySlice<String> {
        let p = prefix.utf8
        var low = 0, high = sortedWords.count
        while low < high {
            let mid = (low + high) / 2
            if sortedWords[mid].utf8.lexicographicallyPrecedes(p) { low = mid + 1 } else { high = mid }
        }
        var end = low
        while end < sortedWords.count, sortedWords[end].utf8.starts(with: p) { end += 1 }
        return sortedWords[low..<end]
    }

    /// Whether `candidate` (lowercased) is a known dictionary word or a prefix of one —
    /// used as a lightweight mid-word plausibility check (is `partial+fragment` on a
    /// path to a real word?).
    func isKnownOrPrefix(_ candidate: String) -> Bool {
        guard isReady, !candidate.isEmpty else { return true }   // fail open
        let c = candidate.lowercased()
        if words[c] != nil { return true }
        // `hasPrefix` compares whole characters; the byte run only narrows where
        // to look (a base letter is a byte prefix of the same letter with a
        // combining mark, but not a character prefix).
        return wordsWithBytePrefix(c).contains { $0.hasPrefix(c) }
    }

    /// Highest-frequency dictionary word that STARTS WITH `prefix` (lowercased) —
    /// instant word completion with no model involved. Returns nil when the best
    /// hit is the prefix itself (word already complete), the prefix is unknown, or
    /// the hit is too rare to be a confident guess.
    func topCompletion(prefix: String) -> String? {
        guard isReady, prefix.count >= 3 else { return nil }
        let p = prefix.lowercased()
        var best: (word: String, freq: Int)?
        for word in wordsWithBytePrefix(p) where word.count > p.count && word.hasPrefix(p) {
            let freq = words[word] ?? 0
            if best == nil || freq > best!.freq { best = (word, freq) }
        }
        guard let best, best.freq >= 1000 else { return nil }
        // If the typed prefix is itself a common word, completing it is usually
        // wrong more often than right ("the" → "them"?) — only complete past a
        // complete word when the continuation is far more frequent.
        if let ownFreq = words[p], ownFreq > best.freq { return nil }
        return best.word
    }

    /// Best correction for `input`, or nil if it's already a known word or no
    /// confident suggestion exists.
    func bestSuggestion(_ input: String) -> String? {
        guard isReady else { return nil }
        let word = input.lowercased()
        if words[word] != nil { return nil }               // already valid
        guard word.count >= 3 else { return nil }

        var best: (word: String, dist: Int, freq: Int)?
        var considered = Set<String>()
        for cand in candidateDeletes(word) {
            guard let list = deletes[cand] else { continue }
            for candidateWord in list where !considered.contains(candidateWord) {
                considered.insert(candidateWord)
                let dist = Self.damerauOSA(word, candidateWord, max: maxEditDistance)
                guard dist >= 0 else { continue }
                let freq = words[candidateWord] ?? 0
                if best == nil || dist < best!.dist || (dist == best!.dist && freq > best!.freq) {
                    best = (candidateWord, dist, freq)
                }
            }
        }
        return best?.word
    }

    // MARK: - Delete generation

    private func editsPrefix(_ word: String) -> Set<String> {
        let key = word.count > prefixLength ? String(word.prefix(prefixLength)) : word
        var result: Set<String> = [key]
        edits(key, distance: 0, into: &result)
        return result
    }

    private func edits(_ word: String, distance: Int, into set: inout Set<String>) {
        guard distance < maxEditDistance, word.count > 1 else { return }
        let chars = Array(word)
        for i in 0..<chars.count {
            var deleted = chars
            deleted.remove(at: i)
            let s = String(deleted)
            if set.insert(s).inserted {
                edits(s, distance: distance + 1, into: &set)
            }
        }
    }

    private func candidateDeletes(_ word: String) -> Set<String> {
        let key = word.count > prefixLength ? String(word.prefix(prefixLength)) : word
        var result: Set<String> = [key]
        edits(key, distance: 0, into: &result)
        return result
    }

    // MARK: - Distance

    /// Damerau optimal-string-alignment distance, bailing out past `max` (returns -1).
    static func damerauOSA(_ a: String, _ b: String, max: Int) -> Int {
        let s = Array(a), t = Array(b)
        if abs(s.count - t.count) > max { return -1 }
        let n = s.count, m = t.count
        if n == 0 { return m <= max ? m : -1 }
        if m == 0 { return n <= max ? n : -1 }

        var d = Array(repeating: Array(repeating: 0, count: m + 1), count: n + 1)
        for i in 0...n { d[i][0] = i }
        for j in 0...m { d[0][j] = j }

        for i in 1...n {
            var rowMin = Int.max
            for j in 1...m {
                let cost = s[i - 1] == t[j - 1] ? 0 : 1
                var val = Swift.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
                if i > 1, j > 1, s[i - 1] == t[j - 2], s[i - 2] == t[j - 1] {
                    val = Swift.min(val, d[i - 2][j - 2] + 1)  // transposition
                }
                d[i][j] = val
                rowMin = Swift.min(rowMin, val)
            }
            if rowMin > max { return -1 }
        }
        let result = d[n][m]
        return result <= max ? result : -1
    }
}
