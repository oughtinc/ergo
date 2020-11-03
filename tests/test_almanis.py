from math import isclose
import pprint

pp = pprint.PrettyPrinter(indent=4)


def test_get_question_binary(almanis):
    binary_question = almanis.get_question("662d7cfd-623e-4f92-b05b-124c9a061420")
    assert binary_question.id == "662d7cfd-623e-4f92-b05b-124c9a061420"
    assert isclose(
        binary_question.get_community_prediction(), 1.0
    )  # Question already resolved as "yes"
    assert binary_question.sample_community()
    assert (
        binary_question.get_text()
        == "Will the Argentine government default on its debt obligations by 12 Sep 2019?"
    )
    assert binary_question.status == {"Settled":{}}


def test_refresh_question(almanis):
    question = almanis.get_question("662d7cfd-623e-4f92-b05b-124c9a061420")
    question.refresh()
    assert question.id == "662d7cfd-623e-4f92-b05b-124c9a061420"


def test_get_all_questions(almanis):
    questions = almanis.get_all_questions()
    assert len(questions) > 10
