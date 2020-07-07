from . import condition


class MaxEntropyCondition(condition.Condition):
    def loss(self, dist) -> float:
        return -self.weight * dist.entropy()

    def destructure(self):
        return ((MaxEntropyCondition,), (self.weight,))

    def __str__(self):
        return "Maximize the entropy of the distribution"

    def __repr__(self):
        return f"MaxEntropyCondition(weight:{self.weight})"
