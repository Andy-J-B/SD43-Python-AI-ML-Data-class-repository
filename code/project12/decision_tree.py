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
        
def classCount(rows):
    return Counter(row[-1] for row in rows)
def entropy(rows):
    counts=classCounts(rows)
    total=len(rows)
    return -sum((c/total)*math.log2(c/total)for c in counts.values())
def partition(rows, questions):
    trueRows = [row for row in rows if question.match(row)]
    falseRows = [row for row in rows if not questions.match(row)]
    #shrek yayyy
    return trueRows, falseRows#;