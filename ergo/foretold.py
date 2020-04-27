from dataclasses import dataclass
from typing import List, Union

import jax.numpy as np
import numpy as onp
import pandas as pd
import requests
import seaborn

from ergo.ppl import uniform


class Foretold:
    """Interface to Foretold"""

    def __init__(self, token=None):
        """token (string): Specify an authorization token
        (supports Bot tokens from Foretold)"""
        self.token = token
        self.api_url = "https://prediction-backend.herokuapp.com/graphql"

    def get_question(self, id):
        """Retrieve a single question by its id"""
        question = ForetoldQuestion(id, self)
        question.refresh_question()
        return question

    def get_questions(self, ids):
        """
        Retrieve many questions by their ids
            ids (List[string]): List of foretold question ids
                (should be less than 500 per request)
        Returns: List of questions corresponding to the ids,
            or None for questions that weren't found."""
        measurables = self._query_measurables(ids)
        return [
            ForetoldQuestion(measurable["id"], self, measurable) if measurable else None
            for measurable in measurables
        ]

    def _post(self, json_data):
        """Send a json post request to the foretold API, with proper authorization"""
        headers = {}
        if self.token is not None:
            headers["Authorization"] = f"Bearer {self.token}"
        response = requests.post(self.api_url, json=json_data, headers=headers)
        response.raise_for_status()
        return response.json()

    def _query_measurable(self, id):
        """Retrieve data from api about single question by its id"""
        response = self._post(
            {
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
            }
        )
        return response["data"]["measurable"]

    def _query_measurables(self, ids):
        """Retrieve data from api about many question by a list of ids"""
        if len(ids) > 500:
            # If we want to implement this later,
            # we can properly use the pageInfo in the request
            raise NotImplementedError(
                "We haven't implemented support for more than 500 ids per request"
            )
        response = self._post(
            {
                "variables": {"measurableIds": ids},
                "query": """query ($measurableIds: [String!]) {
                                measurables(measurableIds: $measurableIds, first: 500) {
                                    total
                                    pageInfo {
                                        hasPreviousPage
                                        hasNextPage
                                        startCursor
                                        endCursor
                                        __typename
                                    }
                                    edges {
                                    node {
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
                                }
                            }""",
            }
        )
        if "errors" in response:
            raise ValueError(
                "Error retrieving foretold measurables. You may not have authorization "
                "to load one or more measurables, or one of the measureable ids may be incorrect"
            )
        if response["data"]["measurables"]["pageInfo"]["hasNextPage"]:
            raise NotImplementedError(
                "We haven't implemented support for more than 500 ids per request"
            )
        measurables_dict = {}
        for edge in response["data"]["measurables"]["edges"]:
            measureable = edge["node"]
            measurables_dict[measureable["id"]] = measureable

        return [measurables_dict.get(id, None) for id in ids]

    def create_measurement(
        self, measureable_id: str, cdf: "ForetoldCdf"
    ) -> requests.Response:
        if self.token is None:
            raise Exception("A token is required to submit a prediction")
        if len(cdf) > 1000:
            raise Exception("Maximum CDF length of 1000 exceeded")
        headers = {"Authorization": f"Bearer {self.token}"}
        query = _measurement_query(measureable_id, cdf)
        response = requests.post(self.api_url, json={"query": query}, headers=headers)
        return response


class ForetoldQuestion:
    """"Information about foretold question, including aggregated distribution"""

    def __init__(self, id, foretold, data=None):
        """
            Should not be called directly, instead use Foretold.get_question

            id: measurableId, the second id in the URL for a foretold question
            foretold: Foretold api
            data: Data retrieved from the foretold api
        """
        self.id = id
        self.foretold = foretold
        self.floatCdf = None
        self.channelId = None
        if data is not None:
            self._update_from_data(data)

    def _update_from_data(self, data):
        """Update based on a dictionary of data from Foretold"""
        try:
            self.channelId = data["channelId"]
        except (KeyError, TypeError):
            raise ValueError(f"Foretold data missing or invalid")

        # If floatCdf is not available, we can just keep it as None
        try:
            self.floatCdf = data["previousAggregate"]["value"]["floatCdf"]
        except (KeyError, TypeError):
            self.floatCdf = None

    def refresh_question(self):
        # previousAggregate is the most recent aggregated distribution
        try:
            measurable = self.foretold._query_measurable(self.id)
            self._update_from_data(measurable)
        except ValueError:
            raise ValueError(f"Error loading distribution {self.id} from Foretold")

    @property
    def url(self):
        return f"https://www.foretold.io/c/{self.channelId}/m/{self.id}"

    @property
    def community_prediction_available(self):
        return self.floatCdf is not None

    def get_float_cdf_or_error(self):
        if not self.community_prediction_available:
            raise ValueError("No community prediction available")
        return self.floatCdf

    def quantile(self, q):
        """Quantile of distribution"""
        floatCdf = self.get_float_cdf_or_error()
        return onp.interp(q, floatCdf["ys"], floatCdf["xs"])

    def sample_community(self):
        """Sample from CDF"""
        y = uniform()
        return np.array(self.quantile(y))

    def plotCdf(self):
        """Plot the CDF"""
        floatCdf = self.get_float_cdf_or_error()
        seaborn.lineplot(floatCdf["xs"], floatCdf["ys"])

    def submit_from_samples(
        self, samples: Union[np.ndarray, pd.Series], length: int = 20
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
        hist, bin_edges = onp.histogram(samples, bins=length - 1, density=True)  # type: ignore
        bin_width = bin_edges[1] - bin_edges[0]
        # Foretold expects `0 <= ys <= 1`, so we clip to that . This
        # is defensive -- at the time of implementation it isn't known
        # how the API handles violations of this.
        ys = np.clip(np.hstack([onp.array([0.0]), onp.cumsum(hist) * bin_width]), 0, 1)  # type: ignore
        return ForetoldCdf(bin_edges.tolist(), ys.tolist())  # type: ignore

    def __len__(self):
        return len(self.xs)


def _measurement_query(measureable_id: str, cdf: ForetoldCdf) -> str:
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
