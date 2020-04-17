import seaborn
import torch
import numpy as np
import requests
from ergo.ppl import uniform


class Foretold:
    """Interface to Foretold"""

    def __init__(self, token=None):
        """token (string): Specify an authorization token (supports Bot tokens from Foretold)"""
        self.token = token
        self.api_url = "https://prediction-backend.herokuapp.com/graphql"

    def get_question(self, id):
        question = ForetoldQuestion(id, self)
        question.refresh_question()
        return question

    def get_measurable(self, id):
        headers = {}
        if self.token is not None:
            headers["Authorization"] = f"Bearer {self.token}"
        response = requests.post(
            self.api_url,
            json={
                "variables": {"measurableId": id},
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
                            }""",
            },
            headers=headers,
        )
        return response.json()["data"]["measurable"]


class ForetoldQuestion:
    """"Information about foretold question, including aggregated distribution"""

    def __init__(self, id, foretold):
        """
            id: measurableId, the second id in the URL for a foretold question
        """
        self.id = id
        self.foretold = foretold
        self.floatCdf = None
        self.channelId = None

    def refresh_question(self):
        # previousAggregate is the most recent aggregated distribution
        try:
            measurable = self.foretold.get_measurable(self.id)
            self.channelId = measurable["channelId"]
            self.floatCdf = measurable["previousAggregate"]["value"]["floatCdf"]
        except KeyError:
            raise (ValueError(f"Error loading distribution {self.id} from Foretold"))

    @property
    def url(self):
        return f"https://www.foretold.io/c/{self.channelId}/m/{self.id}"

    def sample_community(self):
        """Sample from CDF

        Assumes that xs are the x coordinates of the left edge of bins,
        ys are the y coordinates of the left edge. First sample between 0 and 1,
        find the corresponding bin, then linearly interpolate within the bin.

        """
        xs = torch.tensor(self.floatCdf["xs"])
        ys = torch.tensor(self.floatCdf["ys"])
        y = uniform()
        # Finds the first index in ys where the value is greater than y.
        # y then falls between ys[i-1] and ys[i]
        i = np.argmax(ys > y)
        if i == 0:
            return xs[0]
        x0 = xs[i - 1]
        x1 = xs[i]
        y0 = ys[i - 1]
        y1 = ys[i]
        w = (y - y0) / (y1 - y0)
        return x1 * w + x0 * (1 - w)

    def plotCdf(self):
        seaborn.lineplot(self.floatCdf["xs"], self.floatCdf["ys"])
