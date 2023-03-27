import pytest

from spgraph import SPGraph
from spgraph import layout  # pylint: disable=unused-import
from spgraph.testutil import make_random


TESTCASES = [
    (
        "_",
        [
            "-",
            "|",
            "-",
        ],
    ),
    (
        "(s _ _)",
        [
            "-",
            "|",
            "-",
            "|",
            "-",
        ],
    ),
    (
        "(s _ _ _)",
        [
            "-",
            "|",
            "-",
            "|",
            "-",
            "|",
            "-",
        ],
    ),
    (
        "(p _ _)",
        [
            "-----",
            " | | ",
            "-----",
        ],
    ),
    (
        "(p _ _ _)",
        [
            "-------",
            " | | | ",
            "-------",
        ],
    ),
    (
        "(s (p _ _) (p _ _))",
        [
            "-----",
            " | | ",
            "-----",
            " | | ",
            "-----",
        ],
    ),
    (
        "(p (s _ _) (s _ _))",
        [
            "-----",
            " | | ",
            " - - ",
            " | | ",
            "-----",
        ],
    ),
    (
        "(p (s _ _) _)",
        [
            "-----",
            " | | ",
            " - | ",
            " | | ",
            "-----",
        ],
    ),
    (
        "(s (p _ _) _)",
        [
            "-----",
            " | | ",
            "-----",
            "  |  ",
            "-----",
        ],
    ),
    (
        "(s _ (p _ _))",
        [
            "-----",
            "  |  ",
            "-----",
            " | | ",
            "-----",
        ],
    ),
]
@pytest.mark.parametrize("sexp, expect", TESTCASES)
def test_layout(sexp, expect):
    spg = SPGraph.from_sexp(sexp)
    answer = spg.draw()
    assert len(expect) == len(answer)
    for exp, ans in zip(expect, answer):
        assert exp == ans


def test_layout_random():
    spg = make_random(3, 5)
    answer = spg.draw()
    for l in answer:
        print(l)
