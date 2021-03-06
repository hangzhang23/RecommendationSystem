from numpy import *

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset, C1)


# 其中D为全部数据集，
# # Ck为大小为k（包含k个元素）的候选项集，
# # minSupport为设定的最小支持度。
# # 返回值中retList为在Ck中找出的频繁项集（支持度大于minSupport的），
# # supportData记录各频繁项集的支持度
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                ssCnt[can] = ssCnt.get(can, 0) + 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems     # 计算频数
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

# 生成 k+1 项集的候选项集
# 注意其生成的过程中，首选对每个项集按元素排序，然后每次比较两个项集，只有在前k-1项相同时才将这两项合并。
# # 这样做是因为函数并非要两两合并各个集合，那样生成的集合并非都是k+1项的。在限制项数为k+1的前提下，只有在前k-1项相同、最后一项不相同的情况下合并才为所需要的新候选项集。
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            # 前k-2项相同时，将两个集合合并
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


dataSet = loadDataSet()
D = map(set, dataSet)
print dataSet
print D

C1 = createC1(dataSet)
print C1    # 其中C1即为元素个数为1的项集（非频繁项集，因为还没有同最小支持度比较）

L1, suppDat = scanD(D, C1, 0.5)
print "L1: ", L1
print "suppDat: ", suppDat


# 完整的频繁项集生成全过程
L, suppData = apriori(dataSet)
print "L: ",L
print "suppData:", suppData
