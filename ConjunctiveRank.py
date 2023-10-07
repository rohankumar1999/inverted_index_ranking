import sys
import os
import re
import math
c = {}
c_prev = {}
infty = 9999999999999999
if len(sys.argv)<4 :
    print("Please provide the arguments in the following format: python ConjunctiveRank.py filename num_results query")
    sys.exit(1)
text_file = open(sys.argv[1], 'r')
documents = text_file.read().lower().split("\n\n")
inverted_index = {}
num_results = int(sys.argv[2])
Q = sys.argv[3:]
Q = [x.lower() for x in Q]

################################################ Start of Inverted Index construction ################################################
formatted_docs = []
for doc in documents:
    formatted_docs.append([x for x in re.split(r'\s+|\d+|\W+', doc) if x])
for i in range(0, len(formatted_docs)):
    for j in range(len(formatted_docs[i])):
        inverted_index[formatted_docs[i][j]] = [[i+1, j]] if not inverted_index.get(formatted_docs[i][j]) else inverted_index[formatted_docs[i][j]] + [[i+1, j]]
inverted_index = dict(sorted(inverted_index.items()))
P = inverted_index
# print(P)
################################################ End of Inverted Index construction ################################################

############################################################ TF_IDF #####################################################
IndexKeys = inverted_index.keys()
TF_IDF = {}
for i in range(10) :
    TF_IDF['d' + str(i+1)] = []

for term in inverted_index :
    for i in range(10) :
        index = inverted_index[term]
        NumberOfDocOccurence = 0
        for each in index :
            if each[0] == i+1 :
                NumberOfDocOccurence+=1
        if NumberOfDocOccurence == 0 :
            TF_IDF['d' + str(i+1)].append(0)
            continue
        score = (math.log(NumberOfDocOccurence,2) + 1)*(math.log(len(formatted_docs)/len(set([x[0] for x in inverted_index[term]])),2))
        TF_IDF['d' + str(i+1)].append(score)
# print(TF_IDF)
############################################################ Query's TF_IDF ############################################################
query = {}
split_Q = []
for q in Q:
    split_Q += q.split('_')
for q in split_Q :
    if q in query :
        query[q] +=1
    else :
        query[q] = 1
TF_IDF_Q = []
# print(query)
for term in inverted_index :
    index = inverted_index[term]
    NumberOfDocOccurence = 0
    if term in query :
        NumberOfDocOccurence = query[term]
    if NumberOfDocOccurence == 0 :
        TF_IDF_Q.append(0.0)
        continue
    score = (math.log(NumberOfDocOccurence,2) + 1)*(math.log(len(formatted_docs)/len(set([x[0] for x in inverted_index[term]])),2))
    # print(term,len(set([x[0] for x in inverted_index[term]])),NumberOfDocOccurence,score)
    TF_IDF_Q.append(score)
# print(TF_IDF_Q)


################################################ Start Computing Cosine Similarities ################################################
cos_sim = {}
import numpy as np
def cossim(a,b):
    a=np.array(a,dtype=np.float32)
    b=np.array(b,dtype=np.float32)
    return np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b)
for doc,vector in TF_IDF.items():
    cos_sim[doc] = cossim(vector, TF_IDF_Q)
cos_sim = dict(sorted(cos_sim.items(), key=lambda item: -item[1]))
top_k_docs = list(cos_sim.keys())[:num_results]
print('DocId Score')
for doc in top_k_docs:
    print(doc[1:], cos_sim[doc])
################################################ End Computing Cosine Similarities ################################################

####### Function definitions for next(), prev(), first(), last(), nextPhrase(), prevPhase(), nextSolution(), allSolutions() ######

l = {}
for key in P.keys():
    l[key] = len(P[key])
    
def binarySearch(arr, low, high, x):
    while low < high:
        mid = low + (high - low) // 2
        if arr[mid][0] < x[0] or (arr[mid][0] == x[0] and arr[mid][1] <= x[1]):
            low = mid + 1
        else:
            high = mid
    return low

def binarySearch_2(arr, low, high, x):
    arr = arr[::-1]
    arr = [(-x, -y) for x,y in arr]
    x = [-x[0], -x[1]]
    low = len(arr)-1-low
    high = len(arr)-1-high
    return len(arr)-1-binarySearch(arr, high, low, x)


def next(t, current):
    c = {}
    if t not in c.keys():
        c[t] = 0
    if l[t] == 0 or P[t][l[t]-1] <= current:
        return [infty, infty]
    if P[t][0] > current:
        c[t] = 0
        return P[t][c[t]]
    if c[t] > 0 and P[t][c[t]-1] <= current :
        low = c[t] 
    else:
        low = 0

    jump = 1
    high = low + jump
    while high < l[t] and P[t][high] <= current: 
        low = high
        jump = 2*jump
        high = low + jump
    if high > l[t]:
        high = l[t]
    c[t] = binarySearch(P[t], low, high, current)
    return P[t][c[t]]

def prev(t, current):
    c_prev = {}
    if t not in c_prev.keys():
        c_prev[t] = l[t]
    if l[t] == 0 or P[t][0] >= current:
        return [-infty, -infty]
    if P[t][l[t]-1] < current:
        c_prev[t] = l[t]-1
        return P[t][c_prev[t]]
    if c_prev[t] < l[t] and P[t][c_prev[t]-1] >= current :
        high = c_prev[t] 
    else:
        high = l[t]-1

    jump = -1
    low = high + jump
    while low >=0 and P[t][low] >= current: 
        high=low
        jump = 2*jump
        low = high + jump
    if low < 0:
        low = 0
    c_prev[t] = binarySearch_2(P[t], low, high, current)
    return P[t][c_prev[t]]

def first(t):
    return next(t, [-infty, -infty])

def last(t):
    return prev(t, [infty, infty])

def nextPhrase(t, position):
    v = position
    n = len(t)
    for i in range(n):
        v = next(t[i], v)
    if v[0] == infty :
        return [[infty, infty], [infty, infty]]
    u = v
    for i in reversed(range(n-1)):
        u = prev(t[i],u)
    
    if v[0]==u[0] and v[1]-u[1] == n - 1:
        return [u, v]
    else:
        return nextPhrase(t, u)

def prevPhrase(t, position):
    v = position
    n = len(t)
    for i in reversed(range(n)):
        v = prev(t[i], v)
    if v[0] == -infty :
        return [[-infty, -infty], [-infty, -infty]]
    u = v
    for i in range(1, n):
        u = next(t[i],u)
    
    if v[0]==u[0] and u[1]-v[1] == n - 1:
        return [v, u]
    else:
        return prevPhrase(t, u)

def docRight(Q, u):
    return max([nextPhrase(x.split('_'), u)[0][0] for x in Q])

def docLeft(Q, u):
    return min([prevPhrase(x.split('_'), u)[0][0] for x in Q])

def nextSolution(Q, position):
    v = docRight(Q, position)
    if v == infty:
        return infty
    u = docLeft(Q, [v+1, -infty])
    if u == v:
        return u
    else:
        return nextSolution(Q, [v, -infty])

def allSolutions(Q):
    solutions = []
    u =  -infty
    while u < infty:
        u = nextSolution(Q, [u,-infty])
        if len(solutions) and solutions[-1] == u:
            u +=1
            continue
        if u < infty :
            solutions.append(u)
    return solutions

# print('prev document 3,1', prev('document', [3, 1]))
# print(prev('a', [1, 2]))
# print('next', next('a', [-infty, -infty]))
# print(prev('document', [2, 2]))
# print('next',next('document', [3, 0]))
# print(prev('document', [-infty, -infty]))
# print(nextPhrase('a'.split(), [-infty, -infty]))
# print(nextPhrase('document'.split(), [2, 2]))
# print(prevPhrase('document'.split(), [2, 2]))
# print(prevPhrase('awesome second'.split(), [2, 2]))
# Q = ['swimming_pool','diving', 'water']
# print(nextSolution(Q, [-infty, 0]))
# print(nextSolution(Q, [1, 0]))
# print(nextSolution(Q, [2, 0]))
# print(nextSolution(Q, [3, 0]))
# print(nextSolution(Q, [4, 0]))
# print(nextSolution(Q, [5, 0]))
# print(allSolutions(Q))
# print(binarySearch([[1, 4], [2, 2], [3, 2], [5, 4]], 0, 3, [3,0]))