from typing import List

from typing_extensions import TypedDict

HistogramEntry = TypedDict("HistogramEntry", {"x": float, "density": float})
Histogram = List[HistogramEntry]
