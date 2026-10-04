"""Parse and check the uploaded files before Laya sees them."""

import json

MAX_OBSERVATIONS = 100


class InputError(ValueError):
    """An uploaded file that cannot be used, and which of the two it was."""

    def __init__(self, field, message):
        super().__init__(message)
        self.field = field


def _kind(value):
    """Name a JSON value the way an operator would."""
    if isinstance(value, dict):
        return "an object"
    if isinstance(value, list):
        return "a list"
    if isinstance(value, str):
        return "text"
    if value is None:
        return "null"
    return "a number" if not isinstance(value, bool) else "true or false"


def _syntax(exc):
    return f"Not valid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}."


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _looks_like_questions(value):
    return bool(value) and all(
        isinstance(question, dict) and "criteria" in question for question in value.values()
    )


def _load_questions(text, partial):
    """Read one knowledge file. A partial file may omit type and instructions."""
    if not text.strip():
        raise InputError("knowledge", "The file is empty.")
    try:
        questions = json.loads(text)
    except json.JSONDecodeError as exc:
        if exc.msg == "Extra data":
            raise InputError(
                "knowledge",
                "This has one object per line, which looks like a data file. "
                "A knowledge file is a single object of questions.",
            ) from exc
        raise InputError("knowledge", _syntax(exc)) from exc
    if isinstance(questions, list):
        raise InputError(
            "knowledge",
            "This is a list, which looks like a data file. "
            "A knowledge file is an object of questions.",
        )
    if not isinstance(questions, dict):
        raise InputError("knowledge", f"Expected an object of questions, got {_kind(questions)}.")
    if not questions:
        raise InputError("knowledge", "The file has no questions.")
    for name, question in questions.items():
        _check_question(name, question, partial)
    return questions


def _check_question(name, question, partial):
    if not isinstance(question, dict):
        raise InputError("knowledge", f"Question {name!r} must be an object, got {_kind(question)}.")
    criteria = question.get("criteria")
    if not isinstance(criteria, dict) or (not partial and len(criteria) < 2):
        raise InputError("knowledge", f"Question {name!r} needs criteria with at least two answers.")
    for option, description in criteria.items():
        if not isinstance(description, str):
            raise InputError(
                "knowledge",
                f"Question {name!r}: the description of {option!r} must be text, "
                f"got {_kind(description)}.",
            )
    provides = "type" in question or "instructions" in question
    if partial and not provides:
        return
    if question.get("type") != "choice":
        raise InputError("knowledge", f'Question {name!r} must have type "choice".')
    instructions = question.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise InputError("knowledge", f"Question {name!r} needs instructions, as text.")


def parse_knowledge(text):
    """Return the Laya questions in one complete knowledge file."""
    return _load_questions(text, partial=False)


def merge_knowledge(parts):
    """Merge knowledge files in selection order.

    One file supplies type and instructions for each question. The others add
    criteria under the same question name. parts is a list of (name, text).
    """
    if not parts:
        raise InputError("knowledge", "The file has no questions.")
    multiple = len(parts) > 1
    merged = {}
    instruction_file = {}
    answer_file = {}
    question_files = {}
    for filename, text in parts:
        try:
            questions = _load_questions(text, partial=True)
        except InputError as exc:
            if multiple:
                raise InputError("knowledge", f"{filename}: {exc}") from exc
            raise
        for name, question in questions.items():
            question_files.setdefault(name, []).append(filename)
            slot = merged.setdefault(name, {"criteria": {}})
            if "type" in question or "instructions" in question:
                if name in instruction_file:
                    raise InputError(
                        "knowledge",
                        f"Question {name!r} has type and instructions in "
                        f"{instruction_file[name]} and {filename}. "
                        "Only one knowledge file can include them.",
                    )
                instruction_file[name] = filename
                slot["type"] = question.get("type")
                slot["instructions"] = question.get("instructions")
            for option, description in question["criteria"].items():
                if option in slot["criteria"]:
                    raise InputError(
                        "knowledge",
                        f"{filename}: Question {name!r}: answer {option!r} is already defined in "
                        f"{answer_file[name, option]}.",
                    )
                slot["criteria"][option] = description
                answer_file[name, option] = filename
    for name, question in merged.items():
        if name not in instruction_file:
            files = ", ".join(question_files[name])
            raise InputError(
                "knowledge",
                f"Question {name!r} in {files} needs instructions, as text.",
            )
        if len(question["criteria"]) < 2:
            raise InputError("knowledge", f"Question {name!r} needs criteria with at least two answers.")
    return merged


def _rows(text):
    """Read observations from a JSON array, or from one JSON object per line."""
    if not text.strip():
        raise InputError("data", "The file is empty.")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        lines = [(number, line) for number, line in enumerate(text.splitlines(), 1) if line.strip()]
        try:
            first = json.loads(lines[0][1])
        except json.JSONDecodeError:
            first = None
        # Only a file whose first line is a whole object is read as JSONL.
        if not isinstance(first, dict):
            raise InputError("data", _syntax(exc)) from exc
        rows = []
        for number, line in lines:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as line_exc:
                raise InputError(
                    "data",
                    f"Not valid JSON at line {number}, column {line_exc.colno}: {line_exc.msg}.",
                ) from line_exc
        return rows
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and "center_frequency_hz" in value:
        return [value]
    if isinstance(value, dict) and _looks_like_questions(value):
        raise InputError(
            "data",
            "This looks like a knowledge file. A data file is a list of observations.",
        )
    raise InputError("data", f"Expected a list of observations, got {_kind(value)}.")


def _observation(row, label, default_id):
    if not isinstance(row, dict):
        raise InputError("data", f"{label} must be an object, got {_kind(row)}.")
    for field in ("center_frequency_hz", "bandwidth_hz"):
        if field not in row:
            raise InputError("data", f"{label} is missing {field}.")
        if not _is_number(row[field]) or row[field] < 0:
            raise InputError("data", f"{label}: {field} must be a number of Hz, zero or more.")
    if "modulation" not in row:
        raise InputError("data", f"{label} is missing modulation.")
    if not isinstance(row["modulation"], str):
        raise InputError("data", f"{label}: modulation must be text, got {_kind(row['modulation'])}.")
    return {**row, "id": row.get("id", default_id)}


def combine_observations(parts, limit=MAX_OBSERVATIONS, offset=0):
    """Concatenate data files, then keep one slice of rows.

    parts is a list of (name, text). Each file is a JSON list or one object per
    line, in the same shape as a single data file. A row keeps its id when the
    file sets one. The default slice is the first MAX_OBSERVATIONS rows. Rows
    outside the slice are not checked.
    """
    if not parts:
        raise InputError("data", "The file has no observations.")
    multiple = len(parts) > 1
    combined = []
    for filename, text in parts:
        try:
            rows = _rows(text)
        except InputError as exc:
            message = f"{filename}: {exc}" if multiple else str(exc)
            raise InputError("data", message) from exc
        combined.extend((filename, index, row) for index, row in enumerate(rows))
    if not combined:
        raise InputError("data", "The file has no observations.")
    observations = []
    for position, (filename, index, row) in enumerate(combined[offset:offset + limit]):
        label = f"{filename}, row {index + 1}" if multiple else f"Row {index + 1}"
        observations.append(_observation(row, label, offset + position))
    return observations, len(combined)


def parse_observations(text):
    """Return the observations Laya will classify, and how many rows the file had."""
    return combine_observations([("", text)])
