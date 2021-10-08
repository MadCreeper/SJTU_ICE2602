import bloomFilter
import random
import math
def readWordsFile(filename):
    # Open the file in read mode
    with open(filename, "r") as file:
        allText = file.read()
        words = list(map(str, allText.split()))
        # print random string
        #print(random.choice(words))
        return words



testSize = 2000
falsePositiveCount = 0

bitSize = 20000
optimal_k = bloomFilter.get_optimal_k(bitSize,testSize)
words = readWordsFile("words.txt")

print(f"Randomly choosing {(testSize)} words from {(len(words))} words")
print(f"BitarraySize: {(bitSize)}, k: {optimal_k}")

myBF = bloomFilter.BloomFilter(size = bitSize, k = optimal_k)
mySet = set()


for i in range(testSize):
    currentword = random.choice(words)
    if myBF.find(currentword):
        if currentword not in mySet:
            falsePositiveCount += 1
    else:
        myBF.add(currentword)
        mySet.add(currentword)


print(f"Has {(falsePositiveCount / testSize * 100)} % false positive rate in {(testSize)} strings ")



