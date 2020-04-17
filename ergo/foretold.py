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

    def quantile(self, q):
        """Quantile of distribution"""
        return np.interp(q, self.floatCdf["ys"], self.floatCdf["xs"])

    def sample_community(self):
        """Sample from CDF"""
        y = uniform()
        return torch.tensor(self.quantile(y))

    def plotCdf(self):
        seaborn.lineplot(self.floatCdf["xs"], self.floatCdf["ys"])
