"""
Base Distribution Class

Specifies interface for specific Distribution Classes
"""


class Distribution:
    def __mul__(self, x):
        raise NotImplementedError("This should be implemented by a subclass")

    def rv(self,):
        raise NotImplementedError("This should be implemented by a subclass")

    def sample(self):
        raise NotImplementedError("This should be implemented by a subclass")

    def cdf(self, x):
        raise NotImplementedError("This should be implemented by a subclass")

    def ppf(self, q):
        raise NotImplementedError("This should be implemented by a subclass")
