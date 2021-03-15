"""
This module lets you get question and prediction information from Metaculus
and submit predictions, via the API (https://www.metaculus.com/api2/)
"""
from datetime import datetime
import json
from typing import Dict, List, Optional, Union

import pandas as pd
import requests
from typing_extensions import Literal

from .question import (
    BinaryQuestion,
    LinearDateQuestion,
    LinearQuestion,
    LogQuestion,
    MetaculusQuestion,
)


class Metaculus:
    """
    The main class for interacting with Metaculus

    :param api_domain: A Metaculus subdomain (e.g., www, pandemic, finance)
    :param username: A Metaculus username (deprecated)
    :param password: The password for the given Metaculus username (deprecated)
    """

    player_status_to_api_wording = {
        "predicted": "guessed_by",
        "not-predicted": "not_guessed_by",
        "author": "author",
        "interested": "upvoted_by",
    }

    def __init__(
        self,
        api_domain: Optional[str] = "www",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        if username or password:
            raise ValueError(
                "Username and password are no longer accepted on initializaion. Use login_via_username_and_password after initialization instead."
            )
        self.api_domain = api_domain
        self.api_url = f"https://{api_domain}.metaculus.com/api2"
        self.s = requests.Session()

    def login_via_username_and_password(self, username: str, password: str):
        """
        log in to Metaculus using your credentials and store cookies,
        etc. in the session object for future use
        """
        loginURL = f"{self.api_url}/accounts/login/"
        r = self.s.post(
            loginURL,
            headers={"Content-Type": "application/json"},
            data=json.dumps({"username": username, "password": password}),
        )

        r.raise_for_status()

        self.user_id = r.json()["user_id"]

    @property
    def is_logged_in_via_uname_pwd(self):
        return hasattr(self, "user_id")

    def login_via_api_keys(self, user_api_key: str, org_api_key: str):
        self.user_api_key = user_api_key
        self.org_api_key = org_api_key

    @property
    def has_api_keys(self):
        return hasattr(self, "user_api_key") and hasattr(self, "org_api_key")

    def predict(self, q_id: str, data: Dict) -> requests.Response:
        url = f"{self.api_url}/questions/{q_id}/predict/"
        if self.is_logged_in_via_uname_pwd:
            r = self.s.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "Referer": self.api_url,
                    "X-CSRFToken": self.s.cookies.get_dict()["csrftoken"],
                },
                data=json.dumps(data),
            )
        elif self.has_api_keys:
            r = self.s.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "Referer": self.api_url,
                    "X-USERKEY": self.user_api_key,
                    "X-APIKEY": self.org_api_key,
                },
                data=json.dumps(data),
            )
        else:
            raise ValueError("Must be authenticated to make a prediction")

        try:
            r.raise_for_status()

        except requests.exceptions.HTTPError as e:
            e.args = (
                str(e.args),
                f"request body: {e.request.body}",
                f"response json: {e.response.json()}",
            )
            raise

        return r

    def make_question_from_data(self, data: Dict, name=None) -> MetaculusQuestion:
        """
        Make a MetaculusQuestion given data about the question
        of the sort returned by the Metaculus API.

        :param data: the question data (usually from the Metaculus API)
        :param name: a custom name for the question
        :return: A MetaculusQuestion from the appropriate subclass
        """
        if not name:
            name = data.get("title")
        if data["possibilities"]["type"] == "binary":
            return BinaryQuestion(data["id"], self, data, name)
        if data["possibilities"]["type"] == "continuous":
            if data["possibilities"]["scale"]["deriv_ratio"] != 1:
                if data["possibilities"].get("format") == "date":
                    raise NotImplementedError(
                        "Logarithmic date-valued questions are not currently supported"
                    )
                else:
                    return LogQuestion(data["id"], self, data, name)
            if data["possibilities"].get("format") == "date":
                return LinearDateQuestion(data["id"], self, data, name)
            else:
                return LinearQuestion(data["id"], self, data, name)
        raise NotImplementedError(
            "We couldn't determine whether this question was binary, continuous, or something else"
        )

    def get_question(self, id: int, name=None) -> MetaculusQuestion:
        """
        Load a question from Metaculus

        :param id: Question id (can be read off from URL)
        :param name: Name to assign to this question (used in models)
        """
        r = self.s.get(f"{self.api_url}/questions/{id}/")
        data = r.json()
        if not data.get("possibilities"):
            print(id)
            print(data)
            raise ValueError(
                "Unable to find a question with that id. Are you using the right api_domain?"
            )
        return self.make_question_from_data(data, name)

    def get_questions(
        self,
        question_status: Literal[
            "all", "upcoming", "open", "closed", "resolved", "discussion"
        ] = "all",
        player_status: Literal[
            "any", "predicted", "not-predicted", "author", "interested", "private"
        ] = "any",  # 20 results per page
        cat: Union[str, None] = None,
        pages: int = 1,
        fail_silent: bool = False,
        load_detail: bool = True,
    ) -> List["MetaculusQuestion"]:
        """
        Retrieve multiple questions from Metaculus API.

        :param question_status: Question status
        :param player_status: Player's status on this question
        :param cat: Category slug
        :param pages: Number of pages of questions to retrieve
        """

        questions_json = self.get_questions_json(
            question_status=question_status,
            player_status=player_status,
            cat=cat,
            pages=pages,
            load_detail=load_detail,
        )

        def is_log_date(data: Dict) -> bool:
            return (
                data["possibilities"]["type"] == "continuous"
                and data["possibilities"]["scale"]["deriv_ratio"] != 1
                and data["possibilities"]["format"] == "date"
            )

        questions = []
        for q in questions_json:
            if not is_log_date(q):
                questions.append(self.make_question_from_data(q))

        return questions

    def get_questions_json(
        self,
        question_status: Literal[
            "all", "upcoming", "open", "closed", "resolved", "discussion"
        ] = "all",
        player_status: Literal[
            "any", "predicted", "not-predicted", "author", "interested", "private"
        ] = "any",  # 20 results per page
        cat: Union[str, None] = None,
        pages: int = 1,
        include_discussion_questions: bool = False,
        load_detail: bool = True,
    ) -> List[Dict]:
        """
        Retrieve JSON for multiple questions from Metaculus API.

        :param question_status: Question status
        :param player_status: Player's status on this question
        :param cat: Category slug
        :param pages: Number of pages of questions to retrieve
        :include_discussion_questions: If true, data for non-prediction
            questions will be included
        """
        query_params = [f"status={question_status}", "order_by=-publish_time"]
        if player_status != "any":
            if player_status == "private":
                query_params.append("access=private")
            else:
                if hasattr(self, "user_id"):
                    query_params.append(
                        f"{self.player_status_to_api_wording[player_status]}={self.user_id}"
                    )
                else:
                    raise ValueError(
                        f"username_and_password login must be used in order to filter by status {player_status}"
                    )

        if cat is not None:
            query_params.append(f"search=cat:{cat}")

        query_string = "&".join(query_params)

        def get_questions_for_pages(
            query_string: str, max_pages: int = 1, current_page: int = 1, results=[]
        ) -> List[Dict]:
            if current_page > max_pages:
                return results

            r = self.s.get(
                f"{self.api_url}/questions/?{query_string}&limit=20&offset={20*current_page}"
            )

            if len(r.json()["results"]) == 0:
                return results

            r.raise_for_status()

            return get_questions_for_pages(
                query_string, max_pages, current_page + 1, results + r.json()["results"]
            )

        questions = get_questions_for_pages(query_string, pages)

        # Add fields omitted by previous query
        if load_detail:
            for i, q in enumerate(questions):
                r = self.s.get(f"{self.api_url}/questions/{q['id']}")
                questions[i] = dict(r.json(), **q)

        if not include_discussion_questions:
            questions = [
                q for q in questions if q["possibilities"]["type"] != "discussion"
            ]

        return questions

    def make_questions_df(
        self, questions_json: List[Dict], columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Convert JSON returned by Metaculus API to dataframe.

        :param questions_json: List of questions (as dicts)
        :param columns: Optional list of column names to include
            (if omitted, every column is included)
        """
        if columns is not None:
            questions_df = pd.DataFrame(
                [{k: v for (k, v) in q.items() if k in columns} for q in questions_json]
            )
        else:
            questions_df = pd.DataFrame(questions_json)

        for col in ["created_time", "publish_time", "close_time", "resolve_time"]:
            if col in questions_df.columns:
                questions_df[col] = questions_df[col].apply(
                    lambda x: datetime.strptime(x[:19], "%Y-%m-%dT%H:%M:%S")
                )

        if "author" in questions_df.columns:
            questions_df["i_created"] = questions_df["author"] == self.user_id

        if "my_predictions" in questions_df.columns:
            questions_df["i_predicted"] = questions_df["my_predictions"].apply(
                lambda x: x is not None
            )

        return questions_df
