from __future__ import annotations

from pathlib import Path

from idea_graph.benchmarks.liveideabench import get_liveideabench_record


def _write_live_csv(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "liveideabench_hf.csv"
    csv_path.write_text(
        "\n".join(
            [
                "keywords,idea_model,critic_model,idea,full_response,raw_critique,parsed_score,orig,feas,flue,avg",
                "meteorology,m1,c1,idea0,response0,crit0,score0,1,2,3,2",
                "earthquakes,m2,c2,idea1,response1,crit1,score1,4,5,6,5",
                "endocrinology,m3,c3,idea2,response2,crit2,score2,7,8,9,8",
                "astronomy,m4,c4,idea3,response3,crit3,score3,1,1,1,1",
                "astronomy,m5,c5,idea4,response4,crit4,score4,2,2,2,2",
            ]
        ),
        encoding="utf-8",
    )


def test_get_liveideabench_record_accepts_absolute_row_index_with_keyword(tmp_path: Path) -> None:
    _write_live_csv(tmp_path)

    record = get_liveideabench_record(
        tmp_path,
        row_index=2,
        keyword="endocrinology",
    )

    assert record.row_index == 2
    assert record.keyword == "endocrinology"
    assert record.idea == "idea2"


def test_get_liveideabench_record_keeps_keyword_offset_fallback(tmp_path: Path) -> None:
    _write_live_csv(tmp_path)

    record = get_liveideabench_record(
        tmp_path,
        row_index=1,
        keyword="astronomy",
    )

    assert record.row_index == 4
    assert record.keyword == "astronomy"
    assert record.idea == "idea4"
