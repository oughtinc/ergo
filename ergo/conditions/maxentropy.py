from . import condition


class MaxEntropyCondition(condition.Condition):
    def loss(self, dist) -> float:
        return 1 - (self.weight * dist.entropy())

    def destructure(self):
        return ((MaxEntropyCondition,), (self.weight,))

    def __str__(self):
        return "Maximize the entropy of the distribution"
