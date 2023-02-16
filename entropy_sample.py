import random

labels = ["A"] * 6 + ["B"] * 4
random.shuffle(labels)

from math import log2

def entropy(labels):
    n = len(labels)
    counts = {}
    for label in labels:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    ent = 0
    for count in counts.values():
        p = count / n
        ent -= p * log2(p)
    return ent

ent = entropy(labels)
print(ent)
