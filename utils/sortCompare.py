# To find the minimum number of elements in b that need to be moved to make it the same as a,
# we can use a technique that finds the length of the longest common subsequence.
# The elements not in this subsequence are the ones that need to be moved.

def longest_common_subsequence(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for i in range(m + 1)]

    # Building the L[m][n] in bottom up fashion
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    return L[m][n]

if __name__ == "__main__":
    a = [1, 2, 3, 4, 5, 7, 'nice']
    b = [1,3, 2,5, 4, 'nice', 7]

    # Length of the longest common subsequence
    lcs_length = longest_common_subsequence(a, b)

    # Minimum number of elements that need to be moved
    min_moves = len(a) - lcs_length
    print("Minimum number of elements that need to be moved is", min_moves)
