from matplotlib import pyplot as plt
import torch
import numpy as np
import requests
from ergo.ppl import uniform


class ForetoldDistribution:
    """"Aggregated distribution from a foretold question"""

    def __init__(self, id):
        """
            id: measurableId, the second id in the URL for a foretold question
        """
        self.id = id
        self.floatCdf = None
        # previousAggregate is the most recent aggregated distribution
        response = requests.post(
            "https://prediction-backend.herokuapp.com/graphql",
            json={
                "variables": {"measurableId": self.id},
                "query": """query ($measurableId: String!) {
      measurable(id:$measurableId) {
        id
        channelId
        previousAggregate {
          value {
            floatCdf {
              xs
              ys
            }
          }
        }
      }
    }
    """,
            },
        )
        j = response.json()
        try:
            self.channelId = j["data"]["measurable"]["channelId"]
            self.floatCdf = j["data"]["measurable"]["previousAggregate"]["value"][
                "floatCdf"
            ]
        except KeyError:
            raise (f"Error loading distribution {self.id} from Foretold")

    @property
    def url(self):
        return f"https://www.foretold.io/c/{self.channelId}/m/{self.id}"

    def sample(self):
        """Sample from CDF 

        First sample between 0 and 1, find the corresponding bin, then linearly interpolate within the bin"""
        xs = torch.tensor(self.floatCdf["xs"])
        ys = torch.tensor(self.floatCdf["ys"])
        y = uniform()
        i = np.argmax(ys > y)
        if i == len(ys) - 1:
            return xs[i]
        x0 = xs[i]
        x1 = xs[i + 1]
        y0 = ys[i]
        y1 = ys[i + 1]
        w = (y - y0) / (y1 - y0)
        return x1 * w + x0 * (1 - w)

    def plotCdf(self):
        plt.plot(self.floatCdf["xs"], self.floatCdf["ys"])
