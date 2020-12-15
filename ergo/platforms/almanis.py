from abc import abstractmethod
from typing import Dict, List, Sequence

import requests

from ergo.questions import BinaryQuestion, Question


class Almanis:
    """
    The main class for interacting with Almanis
    """

    question_id_request_json = {
        "customerID": "44d7f0b3-7e6f-4867-b00a-6816492bb510",
        "statuses": [{"Trading": {}}, {"ClosedForTrading": {}}],
    }

    BINARY_QUESTION_KEY = "BooleanQuestionOD"
    CONTINUOUS_QUESTION_KEY = "MultipleChoiceQuestionOD"

    def __init__(self):
        self.api_url = "https://proddysruptlabsr53tobackendapi.dysruptlabs.com/api/sharedCode/api/IntegrationLayerApi"
        self.s = requests.Session()
        self.question_ids = []

    def refresh_question_ids(self):
        response = self.s.post(
            f"{self.api_url}/getQuestionIDs", json=self.question_id_request_json
        )
        response_json = response.json()
        self.question_ids = [x["id"] for x in response_json]

    def get_questions(self, ids: List[str]) -> Sequence["AlmanisQuestion"]:
        request_json = {"questionIDs": ids}
        questions_response = self.s.post(
            f"{self.api_url}/getQuestionsWithIDs", json=request_json
        )
        questions_json = questions_response.json()

        prices_response = self.s.post(
            f"{self.api_url}/getCurrentPricesForQuestionIDs", json=request_json
        )
        prices_json = prices_response.json()
        prices_by_id = {p["questionID"]: p for p in prices_json}
        questions = []
        for question_json in questions_json:
            if self.BINARY_QUESTION_KEY in question_json:
                question_data = question_json[self.BINARY_QUESTION_KEY]
                binary_question = AlmanisBinaryQuestion(
                    self, question_data, prices_by_id[question_data["id"]]
                )
                questions.append(binary_question)
            else:
                pass  # For now, ignore the multiple choice questions
        return questions

    def get_question(self, question_id: str) -> "AlmanisQuestion":
        return self.get_questions([question_id])[0]

    def get_all_questions(self) -> Sequence["AlmanisQuestion"]:
        self.refresh_question_ids()
        return self.get_questions(self.question_ids)


class AlmanisQuestion(Question):
    """
    A question on the Almanis platform.

    :param market: Almanis market instance
    :param data: Contract JSON retrieved from PredictIt API

    :ivar str id: id of the question
    :ivar text: name of the question
    :ivar status: status of the question

    An Almanis question looks like:

    {
        "BooleanQuestionOD": {
            "id": "5d4350e5-7721-43f6-901d-9c4f362b992c",
            "customerID": "44d7f0b3-7e6f-4867-b00a-6816492bb510",
            "shortName": null,
            "text": "By 12 Dec 2020, will the S&P 500 fall below 2,800?",
            "description": "<p style=\"text-align:start;\"><span style=\"color: rgba(0,0,0,0.87)..."
            "settlementDescription": "",
            "settledAnswer": null,
            "pricesPriorToSettlement": null,
            "status": {
                "Trading": {}
            },
            "answers": [
                {
                    "TrueBooleanAnswerOD": {
                        "id": "b9fd4ad4-5575-4ce3-abb6-73f54573ed5e"
                    }
                },
                {
                    "FalseBooleanAnswerOD": {
                        "id": "fbc99287-3f2e-408b-ae24-cfcd903722c2"
                    }
                }
            ],
            "scheduledCloseTimeForForecasting": "1607778000000",
            "timeClosed": null,
            "timeSettled": null,
            "collections": [
                "3ceddbb9-af25-41b7-8ac8-dd1162aeb1ee"
            ],
            "liquidityFactor": 150,
            "socialData": {
                "iconPictureURL": null
            },
            "openForTradingTime": "1604275215128",
            "linkedQuestionIDs": []
        }
    }
    """

    def __init__(self, almanis: "Almanis", data: Dict, price_data: Dict):
        self.almanis = almanis
        self.id = data["id"]
        self._data = data
        self.price_data = price_data
        self.parse_price_data()

    def __getattr__(self, name: str):
        """
        If an attribute isn't directly on the class, check whether it's in the
        raw contract data.

        :param name:
        :return: attribute value
        """
        if name not in self._data:
            raise AttributeError(
                f"Attribute {name} is neither directly on this class nor in the raw question data"
            )
        return self._data[name]

    def get_text(self):
        return self._data["text"]

    @abstractmethod
    def parse_price_data(self):
        """
        Parses the price data from the price_data object
        """
        raise NotImplementedError("This should be implemented by a subclass")

    def refresh(self):
        """
        Refetch the prediction data from Almanis and reload the question.
        """
        question = self.almanis.get_question(self.id)
        self._data = question._data
        self.price_data = question.price_data
        self.parse_price_data()


class AlmanisBinaryQuestion(AlmanisQuestion, BinaryQuestion):
    """
    A single binary question on the Almanis platform.

    A price_data object for a Binary question from Almanis API looks like:

    {
        "questionID": "3a899310-c8b2-4dd8-a17d-f4cad1307d3f",
        "priceTime": "1604837474979",
        "scalarMean": 0,
        "probDist": {
            "values": [
                [
                    {
                        "TrueBooleanAnswerOD": {
                            "id": "db513ceb-b8e2-4fcd-952e-1c0bcc1f9246"
                        }
                    },
                    {
                        "prob": 0.021141193209703504
                    }
                ],
                [
                    {
                        "FalseBooleanAnswerOD": {
                            "id": "0c2e08e6-a868-46ae-a643-be45990239df"
                        }
                    },
                    {
                        "prob": 0.9788588067902965
                    }
                ]
            ],
            "id": "0c8eb5c1-5344-4713-b521-38a58e308590"
        }
    """

    def __init__(self, almanis: "Almanis", data: Dict, price_data: Dict):
        super().__init__(almanis, data, price_data)

    def __repr__(self):
        return f'<AlmanisBinaryQuestion text="{self.text}">'

    def parse_price_data(self):
        answers = self.price_data["probDist"]["values"]
        for answer in answers:
            if "TrueBooleanAnswerOD" in answer[0]:
                self.prob = answer[1]["prob"]  # See JSON structure above

    def get_community_prediction(self) -> float:
        """
        Get the latest community probability for the binary event
        """
        return self.prob

    def submit(self, p: float, confidence: float = 0) -> requests.Response:
        """
        Submit a prediction to the prediction platform

        :param p: how likely is the event to happen, from 0 to 1?
        :param confidence: Almanis points to invest in prediction
        """
        raise NotImplementedError("This is still TODO")
