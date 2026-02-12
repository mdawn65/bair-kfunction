def compute_wer(reference: str, hypothesis: str) -> int:
    """
    Computes WER = (S + D + I) / N
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        dp[i][0] = i
    for j in range(len(hyp_words) + 1):
        dp[0][j] = j

    # Fill matrix (Levenshtein distance)
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                substitute = dp[i - 1][j - 1] + 1
                insert = dp[i][j - 1] + 1
                delete = dp[i - 1][j] + 1
                dp[i][j] = min(substitute, insert, delete)

    wer = dp[len(ref_words)][len(hyp_words)] / len(ref_words)
    return wer


def compute_cer(reference, hypothesis):
    """
    Computes Character Error Rate (CER)
    CER = (S + D + I) / N
    """
    ref_chars = list(reference.lower())
    hyp_chars = list(hypothesis.lower())

    dp = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]

    # Initialize
    for i in range(len(ref_chars) + 1):
        dp[i][0] = i
    for j in range(len(hyp_chars) + 1):
        dp[0][j] = j

    # Fill DP matrix
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                substitute = dp[i - 1][j - 1] + 1
                insert = dp[i][j - 1] + 1
                delete = dp[i - 1][j] + 1
                dp[i][j] = min(substitute, insert, delete)

    cer = dp[len(ref_chars)][len(hyp_chars)] / len(ref_chars)
    return cer
