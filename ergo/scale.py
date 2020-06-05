from dataclasses import dataclass


@dataclass
class Scale:
    scale_min: float
    scale_max: float

    @property
    def range(self):
        return self.scale_max - self.scale_min

    def normalize_point(self, point, default=None):
        return (point - self.scale_min) / self.range if point is not None else default

    def denormalize_point(self, point, default=None):
        return (point * self.range) + self.scale_min if point is not None else default

    def normalize_variance(self, variance, default=None):
        return variance / (self.range ** 2) if variance is not None else default

    def denormalize_variance(self, variance, default=None):
        return variance * (self.range ** 2) if variance is not None else default
