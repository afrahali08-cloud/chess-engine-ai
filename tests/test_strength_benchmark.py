from scripts.benchmark_evaluators import benchmark


def test_benchmark_swaps_colors_and_writes_results(tmp_path):
    output_path = tmp_path / "strength.json"

    result = benchmark(
        "handcrafted",
        "ridge",
        output_path,
        depth=1,
        time_limit=0.02,
        max_plies=2,
        opening_count=1,
    )

    assert output_path.is_file()
    assert result["settings"]["game_count"] == 2
    assert result["games"][0]["white_evaluator"] == "handcrafted"
    assert result["games"][1]["white_evaluator"] == "ridge"
    assert set(result["summary"]) == {"handcrafted", "ridge"}
    assert result["format_version"] == 2
    assert result["dataset"]["total_positions"] == 4_000_000
    assert result["model_metadata"]["handcrafted"]["artifact_path"] is None
    assert result["model_metadata"]["ridge"]["artifact_sha256"]
