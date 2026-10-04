from pipeline.context import LOW_CONFIDENCE, MAX_LOW_CONFIDENCE, build_context, build_summary

NAMES = {"knowledge": "questions.json", "data": "observations.json"}
QUESTIONS = {
    "mode": {
        "type": "choice",
        "instructions": "Which signal type is this most likely?",
        "criteria": {
            "morse": "Morse code; OOK",
            "rtty": "radioteletype; FSK",
            "other": "any other signal",
        },
    }
}


def row(identifier, choice, confidence):
    return {
        "id": identifier,
        "state": f"state of {identifier}.",
        "answers": {"mode": (choice, confidence)},
    }


ROWS = [
    row(0, "morse", 0.91),
    row(1, "other", 0.42),
    row(2, "morse", 0.80),
    row(3, "other", 0.18),
]


def test_context_lists_every_answer_with_count_and_ids():
    context = build_context(NAMES, QUESTIONS, ROWS, total=4)
    assert "- morse (2): 0, 2" in context
    assert "- other (2): 1, 3" in context
    assert "- rtty (0): none" in context


def test_context_explains_the_question_and_its_answers():
    context = build_context(NAMES, QUESTIONS, ROWS, total=4)
    assert "Which signal type is this most likely?" in context
    assert "- rtty: radioteletype; FSK" in context
    assert "4 observations" in context
    assert "observations.json" in context
    assert "questions.json" in context


def test_context_reports_confidence():
    context = build_context(NAMES, QUESTIONS, ROWS, total=4)
    assert "mean 0.58" in context
    assert "lowest 0.18" in context


def test_context_lists_low_confidence_least_certain_first():
    context = build_context(NAMES, QUESTIONS, ROWS, total=4)
    worst = context.index("- 3: other at 0.18; state of 3.")
    next_worst = context.index("- 1: other at 0.42; state of 1.")
    assert worst < next_worst
    assert "state of 0." not in context


def test_context_says_when_nothing_is_low_confidence():
    context = build_context(NAMES, QUESTIONS, [row(0, "morse", 0.9)], total=1)
    assert f"No observation is below {LOW_CONFIDENCE:.2f}" in context


def test_context_caps_the_low_confidence_list():
    rows = [row(index, "other", 0.1) for index in range(MAX_LOW_CONFIDENCE + 5)]
    context = build_context(NAMES, QUESTIONS, rows, total=len(rows))
    assert context.count("other at 0.10") == MAX_LOW_CONFIDENCE
    assert f"{MAX_LOW_CONFIDENCE + 5} observations are below" in context
    assert f"Only the {MAX_LOW_CONFIDENCE} least certain are listed" in context


def test_context_does_not_claim_a_cap_it_did_not_apply():
    context = build_context(NAMES, QUESTIONS, ROWS, total=4)
    assert "Only the" not in context
    assert "Least certain first" in context


def test_context_notes_truncation_only_when_rows_were_dropped():
    assert "first 4 of 774" in build_context(NAMES, QUESTIONS, ROWS, total=774)
    assert "first 4 of" not in build_context(NAMES, QUESTIONS, ROWS, total=4)


def test_summary():
    summary = build_summary(NAMES, QUESTIONS, ROWS, total=774, seconds=9.44)
    assert summary == {
        "knowledge_name": "questions.json",
        "data_name": "observations.json",
        "observations": 4,
        "total_rows": 774,
        "questions": [
            {
                "name": "mode",
                "instructions": "Which signal type is this most likely?",
                "counts": {"morse": 2, "rtty": 0, "other": 2},
            }
        ],
        "low_confidence": 2,
        "seconds": 9.4,
    }
