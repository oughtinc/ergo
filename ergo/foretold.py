from dataclasses import dataclass
from typing import List, Union

import numpy as np
import pandas as pd
import requests
import seaborn
import torch

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

    def create_measurement(
        self, measureable_id: str, cdf: "ForetoldCdf"
    ) -> requests.Response:
        if self.token is None:
            raise Exception("A token is required to submit a prediction")
        if len(cdf) > 1000:
            raise Exception("Maximum CDF length exceeded")
        headers = {"Authorization": f"Bearer {self.token}"}
        query = measurement_query(measureable_id, cdf)
        response = requests.post(self.api_url, json={"query": query}, headers=headers)
        return response


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

    def submit_from_samples(
        self, samples: Union[np.ndarray, pd.Series], length: int
    ) -> requests.Response:
        """Submit a prediction to Foretold based on the given samples

        :param samples: Samples on which to base the submission
        :param length: The length of the CDF derived from the samples
        """
        cdf = ForetoldCdf.from_samples(samples, length)
        return self.foretold.create_measurement(self.id, cdf)


@dataclass
class ForetoldCdf:

    xs: List[float]
    ys: List[float]

    @staticmethod
    def from_samples(
        samples: Union[np.ndarray, pd.Series], length: int
    ) -> "ForetoldCdf":
        """Build a Foretold CDF representation from an array of samples

        See the following for details:
        https://docs.foretold.io/cumulative-distribution-functions-format

        :param samples: Samples from which to build the CDF
        :param length: The length of returned CDF
        """
        if length < 2:
            raise ValueError("`length` must be at least 2")
        hist, bin_edges = np.histogram(samples, bins=length - 1, density=True)  # type: ignore
        bin_width = bin_edges[1] - bin_edges[0]
        # Foretold expects `0 <= ys <= 1`, so we clip to that . This
        # is defensive -- at the time of implementation it isn't known
        # how the API handles violations of this.
        ys = np.clip(np.hstack([np.array([0.0]), np.cumsum(hist) * bin_width]), 0, 1)  # type: ignore
        return ForetoldCdf(bin_edges.tolist(), ys.tolist())  # type: ignore

    def __len__(self):
        return len(self.xs)


def measurement_query(measureable_id: str, cdf: ForetoldCdf) -> str:
    return f"""mutation {{
      measurementCreate(
        input: {{
          value: {{ floatCdf: {{ xs: {cdf.xs}, ys: {cdf.ys} }} }}
          competitorType: COMPETITIVE
          measurableId: "{measureable_id}"
        }}
      ) {{
        id
      }}
    }}
    """
