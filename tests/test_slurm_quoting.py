"""Regression guards for shell_arg quoting of the generated SLURM job scripts.

The audit found that operation main.py files interpolated user-derived values
(paths, extras.user) into bash scripts with a naive space-guard pattern, so
command substitution ($(), backticks) and newline injection executed arbitrary
commands on compute nodes as the pipeline user. These tests pin the fix.
"""

import shlex
import subprocess

import pytest

from utils.slurm import shell_arg


def test_shell_arg_round_trip_hostile_payloads():
    payloads = [
        "x; rm -rf ~; y",
        "/h20/Public/iana/$(cat /etc/shadow)",
        "/h20/Public/iana/`reboot`",
        "data $(reboot) path",
        "it's a path",
        "a && b || c | d > e",
        "/h20/Public/iana/*",
        "/h20/Public/iana/x\ny",
        '/h20/Public/iana/"quoted"',
    ]
    for payload in payloads:
        expected = payload.replace("\n", " ").replace("\r", " ")
        argv = shlex.split(f"python do_x.py {shell_arg(payload)}")
        assert argv[2:] == [expected], (
            f"payload must arrive as ONE literal argv token, byte-identical: {payload!r}"
        )


def test_shell_arg_strips_newlines():
    assert "\n" not in shell_arg("a\nb\rc")
    assert shell_arg("a\nb") == shell_arg("a b")


def test_no_execution_via_command_substitution(tmp_path):
    """The definitive behavioral check: bash must not execute anything inside a
    shell_arg-quoted value (the naive double-quote pattern would)."""
    marker = tmp_path / "pwned"
    payload = f"/h20/Public/iana/x$(touch {marker})"
    line = f"echo {shell_arg(payload)}"
    result = subprocess.run(["bash", "-c", line], capture_output=True, text=True)
    assert not marker.exists(), "command substitution must not execute inside quoted args"
    assert result.stdout.strip() == payload, "the value must arrive literally"


def test_script_generation_simulated():
    """Simulate the operation script pattern with hostile values: a newline in
    extras.user must not inject a bash line, and no metacharacter may survive."""
    from utils.slurm import shell_arg as sa

    user = "iana\nrm -rf ~"
    input_path = "/h20/Public/iana/data; rm -rf /h20/Public/other/important"
    log_path = "/h20/Public/iana/out/slurm_%j.out"
    script = "\n".join([
        "#!/bin/bash",
        f"#SBATCH -J {sa(user + '-stretch-contrast')}",
        f"#SBATCH -o {sa(log_path)}",
        f"python do_x.py {sa(input_path)} {sa('/tmp/out')} $SLURM_ARRAY_TASK_ID",
    ])
    lines = script.split("\n")
    executable = [l for l in lines if l.strip() and not l.startswith("#")]
    assert len(executable) == 1, "a hostile value must not inject extra script lines"
    tokens = shlex.split(executable[0])
    assert tokens[:3] == ["python", "do_x.py", input_path]
    assert tokens[3] == "/tmp/out"
    assert "$SLURM_ARRAY_TASK_ID" in tokens, "runtime variables must stay unquoted"


@pytest.mark.parametrize("value", ["plain", "with space", "50%", "tmp$old", "a-b_c.d"])
def test_shell_arg_transparent_for_benign_values(value):
    assert shlex.split(f"python do_x.py {shell_arg(value)}")[2:] == [value]
