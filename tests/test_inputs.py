import json

import pytest

from pipeline.inputs import MAX_OBSERVATIONS, InputError, parse_knowledge, parse_observations

QUESTIONS = {
    "mode": {
        "type": "choice",
        "instructions": "Which signal type is this most likely?",
        "criteria": {"morse": "Morse code; OOK", "other": "any other signal"},
    }
}
ROW = {"center_frequency_hz": 7030000, "bandwidth_hz": 100, "modulation": "OOK"}


def knowledge_error(text):
    with pytest.raises(InputError) as raised:
        parse_knowledge(text)
    assert raised.value.field == "knowledge"
    return str(raised.value)


def data_error(text):
    with pytest.raises(InputError) as raised:
        parse_observations(text)
    assert raised.value.field == "data"
    return str(raised.value)


def test_knowledge_is_returned_as_parsed():
    assert parse_knowledge(json.dumps(QUESTIONS)) == QUESTIONS


def test_knowledge_syntax_error_names_line_and_column():
    message = knowledge_error('{\n  "mode": {,\n}')
    assert "line 2" in message
    assert "column" in message


def test_empty_knowledge_file():
    assert "empty" in knowledge_error("  \n")


def test_knowledge_given_a_list_points_at_the_data_file():
    assert "data file" in knowledge_error(json.dumps([ROW]))


def test_knowledge_given_jsonl_points_at_the_data_file():
    text = json.dumps(ROW) + "\n" + json.dumps(ROW) + "\n"
    assert "data file" in knowledge_error(text)


def test_knowledge_without_questions():
    assert "no questions" in knowledge_error("{}")


@pytest.mark.parametrize(
    "change, expected",
    [
        ({"type": "number"}, "type"),
        ({"instructions": ""}, "instructions"),
        ({"criteria": {}}, "criteria"),
        ({"criteria": {"morse": "only one"}}, "criteria"),
        ({"criteria": {"morse": 1, "other": "x"}}, "morse"),
    ],
)
def test_knowledge_question_problems_name_the_question(change, expected):
    message = knowledge_error(json.dumps({"mode": {**QUESTIONS["mode"], **change}}))
    assert "'mode'" in message
    assert expected in message


def test_observations_from_a_json_array_get_an_id():
    observations, total = parse_observations(json.dumps([ROW, {**ROW, "id": "b7"}]))
    assert total == 2
    assert [observation["id"] for observation in observations] == [0, "b7"]
    assert observations[0]["modulation"] == "OOK"


def test_observations_from_jsonl():
    text = "\n".join(json.dumps({**ROW, "id": index}) for index in range(3)) + "\n"
    observations, total = parse_observations(text)
    assert total == 3
    assert [observation["id"] for observation in observations] == [0, 1, 2]


def test_a_single_observation_object_is_one_row():
    observations, total = parse_observations(json.dumps(ROW))
    assert total == 1
    assert observations[0]["id"] == 0


def test_observations_are_capped():
    rows = [{**ROW, "id": index} for index in range(MAX_OBSERVATIONS + 5)]
    observations, total = parse_observations(json.dumps(rows))
    assert len(observations) == MAX_OBSERVATIONS
    assert total == MAX_OBSERVATIONS + 5


def test_observations_syntax_error_names_line_and_column():
    message = data_error('[\n  {"center_frequency_hz": 1,}\n]')
    assert "line 2" in message


def test_broken_jsonl_names_the_line():
    message = data_error(json.dumps(ROW) + "\n" + '{"center_frequency_hz": \n')
    assert "line 2" in message


def test_empty_observations():
    assert "no observations" in data_error("[]")


def test_observations_given_a_knowledge_file_says_so():
    assert "knowledge file" in data_error(json.dumps(QUESTIONS))


@pytest.mark.parametrize(
    "row, expected",
    [
        ({"bandwidth_hz": 100, "modulation": "OOK"}, "center_frequency_hz"),
        ({**ROW, "bandwidth_hz": "wide"}, "bandwidth_hz"),
        ({**ROW, "center_frequency_hz": True}, "center_frequency_hz"),
        ({**ROW, "center_frequency_hz": -5}, "center_frequency_hz"),
        ({"center_frequency_hz": 1, "bandwidth_hz": 1}, "modulation"),
        ({**ROW, "modulation": 3}, "modulation"),
        ("not an object", "object"),
    ],
)
def test_observation_problems_name_the_row_and_field(row, expected):
    message = data_error(json.dumps([ROW, row]))
    assert "Row 2" in message
    assert expected in message
