import sys
import os
import re


def main():
    if len(sys.argv)!=2 :
        print("Please provide the arguments in the following format: python InvertedIndexPrinter.py filename")
        sys.exit(1)
    text_file = open(sys.argv[1], 'r')
    documents = text_file.read().lower().split("\n\n")
    formatted_docs = []
    inverted_index = {}
    for doc in documents:
        formatted_docs.append([x for x in re.split(r'\s+|\d+|\W+', doc) if x])
    
    for i in range(0, len(formatted_docs)):
        # for word in formatted_docs[i]:
        #     inverted_index[word] = {i+1} if not inverted_index.get(word) else inverted_index[word] | {i+1}
        for j in range(len(formatted_docs[i])):
            inverted_index[formatted_docs[i][j]] = [[i+1, j]] if not inverted_index.get(formatted_docs[i][j]) else inverted_index[formatted_docs[i][j]] + [[i+1, j]]
    inverted_index = dict(sorted(inverted_index.items()))
    print(inverted_index)
    line1 = ""
    num_words = "0000"+str(len(inverted_index))
    line1 += num_words[-4:]
    idx = 0
    for word in inverted_index.keys():
        line1 += ("0000"+str(idx))[-4:]
        idx += (len(word)+4)
    line2 = ""
    pos=0
    line3 = ""
    for word,occurences in inverted_index.items():
        line2 += word
        line2 += ("0000"+str(pos))[-4:]
        docs = {}
        for occurence in occurences:
            docs[occurence[0]] = [occurence[1]] if not docs.get(occurence[0]) else docs[occurence[0]] + [occurence[1]]
        pos += 2*len(docs.keys())
        for doc,positions in docs.items():
            print(positions)
            line3+=(str(doc)+':'+':'.join([str(x) for x in positions])+',')
            # line3
    line3 = line3[:-1]
    print(line1)
    print(line2)
    print(line3)

main()
