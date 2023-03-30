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
        "(a _ _)",
        [
            "-----",
            " | | ",
            "-----",
        ],
    ),
    (
        "(a _ _ _)",
        [
            "-------",
            " | | | ",
            "-------",
        ],
    ),
    (
        "(s (a _ _) (a _ _))",
        [
            "-----",
            " | | ",
            "-----",
            " | | ",
            "-----",
        ],
    ),
    (
        "(a (s _ _) (s _ _))",
        [
            "-----",
            " | | ",
            " - - ",
            " | | ",
            "-----",
        ],
    ),
    (
        "(a (s _ _) _)",
        [
            "-----",
            " | | ",
            " - | ",
            " | | ",
            "-----",
        ],
    ),
    (
        "(s (a _ _) _)",
        [
            "-----",
            " | | ",
            "-----",
            "  |  ",
            "-----",
        ],
    ),
    (
        "(s _ (a _ _))",
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
        (a
            (s
                (a
                    (s _ _ _ _)
                    (s _ _ _)
                    (s _ _ _)
                )
                _
            )
            (s
                (a (s _ _ _) (s _ _ _) (s _ _ _ _) (s _ _))
                _
                (a (s _ _ _) (s _ _ _ _))
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
        """(a
            (s (a (s _ _ _ _) (s _ _ _) (s _ _ _)) _)
            (s _ (a (s _ _) (s _ _) (s _ _ _ _) (s _ _ _)))
            (s (a (s _ _ _) (s _ _ _) (s _ _ _ _) (s _ _)) _ (a (s _ _ _) (s _ _ _ _)))
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
