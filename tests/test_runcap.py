"""Tests for lobora/runcap.py -- one declaration of the run's cumulative cap.

A watcher whose target is a step the run has already passed concludes that a live
run has finished. That is not a loud failure, so the number lives in exactly one
file and everything else reads it.
"""

from __future__ import annotations

from lobora import runcap


def test_reads_the_declaration_from_h3_env_sh():
    declared = runcap.from_env_file()
    assert declared > 0, "scripts/vast/h3_env.sh no longer declares STOP_TARGET_STEP"
    assert declared == runcap.target_step("LOBORA_TEST_UNSET_CAP")


def test_a_missing_or_silent_env_file_falls_back_rather_than_crashing(tmp_path):
    assert runcap.from_env_file(tmp_path / "nope.sh") == -1
    quiet = tmp_path / "quiet.sh"
    quiet.write_text("HEIGHT=480\n")
    assert runcap.from_env_file(quiet) == -1
    assert runcap.target_step("LOBORA_TEST_UNSET_CAP", env_file=quiet,
                              fallback=1234) == 1234


def test_precedence_is_argument_then_env_then_file(tmp_path, monkeypatch):
    envfile = tmp_path / "h3_env.sh"
    envfile.write_text("STOP_TARGET_STEP=${STOP_TARGET_STEP:-4321}\n")
    monkeypatch.delenv("LOBORA_TEST_CAP", raising=False)
    assert runcap.target_step("LOBORA_TEST_CAP", env_file=envfile) == 4321
    monkeypatch.setenv("LOBORA_TEST_CAP", "777")
    assert runcap.target_step("LOBORA_TEST_CAP", env_file=envfile) == 777
    assert runcap.target_step("LOBORA_TEST_CAP", override=99, env_file=envfile) == 99


def test_a_non_numeric_override_is_ignored_rather_than_crashing(tmp_path, monkeypatch):
    envfile = tmp_path / "h3_env.sh"
    envfile.write_text("STOP_TARGET_STEP=${STOP_TARGET_STEP:-4321}\n")
    monkeypatch.setenv("LOBORA_TEST_CAP", "soon")
    assert runcap.target_step("LOBORA_TEST_CAP", env_file=envfile) == 4321


def test_source_of_names_where_the_number_came_from(tmp_path, monkeypatch):
    envfile = tmp_path / "h3_env.sh"
    envfile.write_text("STOP_TARGET_STEP=${STOP_TARGET_STEP:-4321}\n")
    monkeypatch.delenv("LOBORA_TEST_CAP", raising=False)
    assert runcap.source_of("LOBORA_TEST_CAP", envfile) == "h3_env.sh:STOP_TARGET_STEP"
    monkeypatch.setenv("LOBORA_TEST_CAP", "777")
    assert runcap.source_of("LOBORA_TEST_CAP", envfile) == "$LOBORA_TEST_CAP"
