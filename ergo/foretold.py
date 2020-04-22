import numpy as np
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
        """Retrieve a single question by its id"""
        question = ForetoldQuestion(id, self)
        question.refresh_question()
        return question

    def get_questions(self, ids):
        """Retrieve many questions by their ids
            ids (List[string]): List of foretold question ids (should be less than 500 per request)
        Returns: List of questions corresponding to the ids, or None for questions that weren't found."""
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
            # If we want to implement this later, we can properly use the pageInfo in the request
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

    def getFloatCdfOrError(self):
        if not self.community_prediction_available:
            raise ValueError("No community prediction available")
        return self.floatCdf

    def quantile(self, q):
        """Quantile of distribution"""
        floatCdf = self.getFloatCdfOrError()
        return np.interp(q, floatCdf["ys"], floatCdf["xs"])

    def sample_community(self):
        """Sample from CDF"""
        y = uniform()
        return torch.tensor(self.quantile(y))

    def plotCdf(self):
        """Plot the CDF"""
        floatCdf = self.getFloatCdfOrError()
        seaborn.lineplot(self.floatCdf["xs"], self.floatCdf["ys"])
