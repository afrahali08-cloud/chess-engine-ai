from scripts.run_pipeline import build_commands, build_parser


def test_pipeline_builds_a_4m_benchmark_command(tmp_path):
    input_path = tmp_path / "positions.csv"
    input_path.write_text("game_id,fen,cp,ply\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--models-dir",
            str(tmp_path / "models"),
            "--stages",
            "benchmark",
            "--overwrite",
        ]
    )

    commands = build_commands(args)

    assert len(commands) == 1
    stage, command = commands[0]
    assert stage == "benchmark"
    expected_index = command.index("--expected-positions") + 1
    assert command[expected_index] == "4000000"
    assert "--overwrite" in command


def test_pipeline_dry_run_does_not_require_a_local_dataset(tmp_path):
    missing_input = tmp_path / "missing.csv"
    args = build_parser().parse_args(
        ["--input", str(missing_input), "--stages", "benchmark", "--dry-run"]
    )

    commands = build_commands(args)

    assert commands[0][0] == "benchmark"
