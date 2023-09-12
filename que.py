# Edge caching & recommender system
# Chuan Sun at Chongqing University
# 05/16 2021

class queue:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.queue = []

    def put(self, value):
        if len(self.queue) < self.maxsize:
            self.queue.append(value)
        else:
            self.queue.pop(0)
            self.queue.append(value)

    def pop(self, index=0):
        self.queue.pop(index)

    def get(self):
        return self.queue
