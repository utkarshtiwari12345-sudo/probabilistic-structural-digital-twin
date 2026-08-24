"""Tests for strict OpenFAST output ingestion and observation mapping."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from probdt.openfast import (
    ChannelMapping,
    OpenFASTError,
    map_openfast_channels,
    read_openfast_output,
    read_openfast_scalar_settings,
    uniform_time_grid_error,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "openfast"
    / "v5.0.0"
    / "5MW_OC3Mnpl_DLL_WTurb_WavesIrr"
    / "5MW_OC3Mnpl_DLL_WTurb_WavesIrr.outb"
)


def _fixed_width(values: tuple[str, ...], width: int) -> bytes:
    return b"".join(value.encode("ascii").ljust(width) for value in values)


def _write_binary_fixture(path: Path) -> None:
    width = 16
    names = ("Time", "Disp", "Force")
    units = ("(s)", "(m)", "(kN)")
    packed = np.asarray([[100, 50], [200, 70], [300, 90]], dtype="<i2")
    content = b"".join(
        (
            struct.pack("<h", 4),
            struct.pack("<h", width),
            struct.pack("<i", 2),
            struct.pack("<i", 3),
            struct.pack("<d", 0.0),
            struct.pack("<d", 0.1),
            np.asarray([100.0, 10.0], dtype="<f4").tobytes(),
            np.asarray([0.0, 0.0], dtype="<f4").tobytes(),
            struct.pack("<i", 7),
            b"fixture",
            _fixed_width(names, width),
            _fixed_width(units, width),
            packed.tobytes(),
        )
    )
    path.write_bytes(content)


def test_binary_output_reader_decompresses_fixture(tmp_path: Path) -> None:
    path = tmp_path / "fixture.outb"
    _write_binary_fixture(path)

    output = read_openfast_output(path)

    assert output.format_id == 4
    assert output.channel_names == ("Time", "Disp", "Force")
    assert output.channel_units == ("s", "m", "kN")
    assert output.data == pytest.approx(
        np.asarray([[0.0, 1.0, 5.0], [0.1, 2.0, 7.0], [0.2, 3.0, 9.0]])
    )


def test_ascii_reader_and_si_mapping(tmp_path: Path) -> None:
    path = tmp_path / "fixture.out"
    path.write_text(
        "OpenFAST fixture\nTime Angle\n(s) (deg)\n0 0\n0.5 90\n1 180\n",
        encoding="utf-8",
    )
    output = read_openfast_output(path)
    batch = map_openfast_channels(
        output,
        [
            ChannelMapping(
                "Angle",
                "deg",
                "angle_rad",
                "rad",
                np.pi / 180.0,
            )
        ],
        start_s=0.5,
        end_s=1.0,
    )

    assert output.file_format == "ascii"
    assert batch.time_s == pytest.approx([0.5, 1.0])
    assert batch.values[:, 0] == pytest.approx([np.pi / 2.0, np.pi])
    assert batch.channel_units == ("rad",)


def test_mapping_rejects_unit_mismatch_and_duplicate_targets(tmp_path: Path) -> None:
    path = tmp_path / "fixture.outb"
    _write_binary_fixture(path)
    output = read_openfast_output(path)

    with pytest.raises(OpenFASTError, match="Unit mismatch"):
        map_openfast_channels(
            output,
            [ChannelMapping("Disp", "mm", "displacement_m", "m", 0.001)],
        )
    duplicate = ChannelMapping("Disp", "m", "response", "m")
    with pytest.raises(OpenFASTError, match="target names"):
        map_openfast_channels(output, [duplicate, duplicate])


def test_truncated_binary_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "fixture.outb"
    _write_binary_fixture(path)
    path.write_bytes(path.read_bytes()[:-3])

    with pytest.raises(OpenFASTError, match="Truncated"):
        read_openfast_output(path)


def test_scalar_input_settings_are_read_without_general_deck_editing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fixture.fst"
    path.write_text(
        "60 TMax - duration\n0.01 DT - integration step\n4 OutFileFmt - output\n",
        encoding="utf-8",
    )

    settings = read_openfast_scalar_settings(path, ["TMax", "DT", "OutFileFmt"])

    assert settings == {"TMax": "60", "DT": "0.01", "OutFileFmt": "4"}


def test_uniform_time_grid_error_detects_jitter() -> None:
    assert uniform_time_grid_error(np.asarray([0.0, 0.1, 0.2])) == pytest.approx(0.0)
    assert uniform_time_grid_error(np.asarray([0.0, 0.1, 0.201])) > 0.0


def test_official_openfast_v5_reference_output_is_supported() -> None:
    output = read_openfast_output(REFERENCE_OUTPUT)

    assert output.format_id == 3
    assert output.data.shape == (1201, 62)
    assert output.time_step_s == pytest.approx(0.05)
    assert {"TwHt1TPxi", "Wave1Elev", "HydroFxi"} <= set(
        output.channel_names
    )
