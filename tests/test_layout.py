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
    (
        """
        (p
            (s
                (p
                    (s _ _ _ _)
                    (s _ _ _)
                    (s _ _ _)
                )
                _
            )
            (s
                (p (s _ _ _) (s _ _ _) (s _ _ _ _) (s _ _))
                _
                (p (s _ _ _) (s _ _ _ _))
            )
        )""",
        [
            "-------------------",
            "  | | |   | | | |  ",
            "  | | |   | | - |  ",
            "  | | |   - - | |  ",
            "  | | |   | | - -  ",
            "  | | |   - - | |  ",
            "  - | |   | | - |  ",
            "  | - -   | | | |  ",
            "  - | |  --------- ",
            "  | - -      |     ",
            "  - | |    -----   ",
            "  | | |     | |    ",
            " -------    | -    ",
            "    |       - |    ",
            "    |       | -    ",
            "    |       - |    ",
            "    |       | -    ",
            "    |       | |    ",
            "-------------------",
        ]
    )
]
@pytest.mark.parametrize("sexp, expect", TESTCASES)
def test_layout_simple(sexp, expect):
    spg = SPGraph.from_sexpr(sexp)
    answer = spg.draw().split("\n")
    assert len(expect) == len(answer)
    for exp, ans in zip(expect, answer):
        assert exp == ans


TESTCASES = [
    (
        "_",
"""\
┯
│
┷""".split("\n"),
    ),
    (
        """(p
            (s (p (s _ _ _ _) (s _ _ _) (s _ _ _)) _)
            (s _ (p (s _ _) (s _ _) (s _ _ _ _) (s _ _ _)))
            (s (p (s _ _ _) (s _ _ _) (s _ _ _ _) (s _ _)) _ (p (s _ _ _) (s _ _ _ _)))
        )""",
"""\
╺━┯━┯━┯━━━━━━┯━━━━━━┯━┯━┯━┯━╸
  │ │ │      │      │ │ │ │  
  │ │ │      │      │ │ ┿ │  
  │ │ │      │      ┿ ┿ │ │  
  │ │ │      │      │ │ ┿ ┿  
  │ │ │      │      ┿ ┿ │ │  
  ┿ │ │  ╺┯━┯┷┯━┯╸  │ │ ┿ │  
  │ ┿ ┿   │ │ │ │   │ │ │ │  
  ┿ │ │   │ │ ┿ │  ╺┷━┷┯┷━┷╸ 
  │ ┿ ┿   │ │ │ ┿      │     
  ┿ │ │   ┿ ┿ ┿ │    ╺┯┷┯╸   
  │ │ │   │ │ │ ┿     │ │    
 ╺┷━┿━┷╸  │ │ ┿ │     │ ┿    
    │     │ │ │ │     ┿ │    
    │     │ │ │ │     │ ┿    
    │     │ │ │ │     ┿ │    
    │     │ │ │ │     │ ┿    
    │     │ │ │ │     │ │    
╺━━━┷━━━━━┷━┷━┷━┷━━━━━┷━┷━━━╸""".split("\n")
    )
]
@pytest.mark.parametrize("sexp, expect", TESTCASES)
def test_layout_pretty(sexp, expect):
    spg = SPGraph.from_sexpr(sexp)
    answer = spg.draw(pretty=True)
    answer = answer.split("\n")
    assert len(expect) == len(answer)
    for exp, ans in zip(expect, answer):
        assert exp == ans


def test_layout_random():
    spg = make_random(5, 4, 10)
    answer = spg.draw(pretty=True)
    print()
    print(answer)
