from deepeval.metrics import GEval

try:
    from deepeval.test_case import SingleTurnParams as P
except ImportError:
    from deepeval.test_case import LLMTestCaseParams as P

from configs import get_settings

STEPS = {
    "meaning": [
        "The input gives the translation direction and the source text.",
        "Compare the actual output with the source and the expected output.",
        "Penalize added, missing or changed information (names, numbers, negation).",
        "Penalize extra explanations, repeated source text, or output in the wrong language.",
        "Accept correct paraphrases.",
    ],
    "fluency": [
        "Judge grammar and naturalness in the target language of the stated direction.",
        "For Egyptian Arabic, colloquial forms (مفيش, مش, بيقول, إيه, دا/دي) are correct.",
        "For Egyptian Arabic, do NOT penalize missing إعراب, diacritics or spelling variation.",
        "For English, use standard natural English grammar.",
    ],
    "dialect": [
        "If the target is Egyptian Arabic: score whether it sounds like everyday spoken Egyptian, "
        "not formal Arabic (الفصحى) and not another dialect.",
        "If the target is English: score whether Egyptian slang and idioms in the source "
        "were understood, not translated literally.",
    ],
    "overall": [
        "Judge overall quality: meaning, grammar and naturalness together.",
        "For Egyptian Arabic, natural means colloquial Egyptian, not formal Arabic.",
        "Give a high score only if the output is both faithful and natural.",
    ],
}

# Which parts of the test case each criterion can see
WITH_REFERENCE = [P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT]
NO_REFERENCE = [P.INPUT, P.ACTUAL_OUTPUT]
PARAMS = {
    "meaning": WITH_REFERENCE,
    "fluency": NO_REFERENCE,
    "dialect": NO_REFERENCE,
    "overall": WITH_REFERENCE,
}


def build_metrics(judge):
    criteria = get_settings().CRITERIA
    unknown = [c for c in criteria if c not in STEPS]
    if unknown:
        raise ValueError(f"Unknown criteria {unknown}; choose from {list(STEPS)}")
    return [
        GEval(
            name=c,
            evaluation_steps=STEPS[c],
            evaluation_params=PARAMS[c],
            model=judge,
            async_mode=False,
        )
        for c in criteria
    ]
