import math
from collections import Counter


class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, row):
        return row[self.column] == self.value


class Leaf:
    def __init__(self, rows):
        counts = Counter(row[-1] for row in rows)
        self.predictions = dict(counts)


class DecisionNode:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def class_counts(rows):
    return Counter(row[-1] for row in rows)


def entropy(rows):
    counts = class_counts(rows)
    total = len(rows)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def info_gain(left, right, current_uncertainty):
    total = len(left) + len(right)
    return current_uncertainty - (len(left) / total) * entropy(left) - (len(right) / total) * entropy(right)


def find_best_split(rows):
    best_gain = 0
    best_question = None
    current_uncertainty = entropy(rows)

    for col in range(len(rows[0]) - 1):
        values = set(row[col] for row in rows)
        for val in values:
            q = Question(col, val)
            true_rows, false_rows = partition(rows, q)
            if not true_rows or not false_rows:
                continue
            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain >= best_gain:
                best_gain = gain
                best_question = q

    return best_gain, best_question


def partition(rows, question):
    true_rows = [row for row in rows if question.match(row)]
    false_rows = [row for row in rows if not question.match(row)]
    return true_rows, false_rows


def build_tree(rows):
    gain, question = find_best_split(rows)

    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)

    return DecisionNode(question, true_branch, false_branch)
