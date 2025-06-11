
def findMinimumDistance(corruptedMessage: list[int], codewords: list[list[int]], withStopLoss = False):
    min = float('inf')
    best_codeword = None
    for codeword in codewords:
        potentialHammingDistance = findHammingDistance(corruptedMessage, codeword)
        if potentialHammingDistance < min:
            best_codeword = codeword
            min = potentialHammingDistance
            if withStopLoss and potentialHammingDistance == 0:
                return best_codeword, min
            
    return best_codeword, min


def findHammingDistance(corruptedMessagge: list[int], potentialCodeWord: list[int]):
    hammingDistance = 0
    for index, bit in enumerate(corruptedMessagge):
        if bit != potentialCodeWord[index]:
            hammingDistance += 1
    return hammingDistance
