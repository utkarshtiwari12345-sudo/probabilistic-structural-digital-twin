"""Strict readers and unit-aware observation mapping for OpenFAST outputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class OpenFASTError(ValueError):
    """Raised when an OpenFAST output or channel mapping is invalid."""


def _clean_unit(unit: str) -> str:
    cleaned = unit.strip().strip("()[]").replace("sec", "s")
    return cleaned.replace(" ", "")


@dataclass(frozen=True)
class OpenFASTOutput:
    """Validated time-series data read from one OpenFAST output file."""

    source_path: Path
    description: str
    file_format: str
    format_id: int | None
    channel_names: tuple[str, ...]
    channel_units: tuple[str, ...]
    data: FloatArray

    def __post_init__(self) -> None:
        values = np.asarray(self.data, dtype=np.float64)
        if self.file_format not in {"ascii", "binary"}:
            raise OpenFASTError("file_format must be 'ascii' or 'binary'.")
        if not self.channel_names or self.channel_names[0].lower() != "time":
            raise OpenFASTError("The first OpenFAST channel must be Time.")
        if len(set(self.channel_names)) != len(self.channel_names):
            raise OpenFASTError("OpenFAST channel names must be unique.")
        if len(self.channel_names) != len(self.channel_units):
            raise OpenFASTError("Every OpenFAST channel must have one unit.")
        if values.ndim != 2 or values.shape[1] != len(self.channel_names):
            raise OpenFASTError("OpenFAST data shape does not match its channels.")
        if values.shape[0] < 2 or not np.all(np.isfinite(values)):
            raise OpenFASTError("OpenFAST output must contain finite time-series data.")
        time = values[:, 0]
        if _clean_unit(self.channel_units[0]) != "s":
            raise OpenFASTError("The OpenFAST Time channel must use seconds.")
        if np.any(np.diff(time) <= 0.0):
            raise OpenFASTError("OpenFAST time must be strictly increasing.")

    @property
    def time_s(self) -> FloatArray:
        """Return the output time grid in seconds."""

        return np.asarray(self.data[:, 0], dtype=np.float64)

    @property
    def time_step_s(self) -> float:
        """Return the median output time step."""

        return float(np.median(np.diff(self.time_s)))

    @property
    def duration_s(self) -> float:
        """Return the elapsed duration represented in the file."""

        return float(self.time_s[-1] - self.time_s[0])

    def channel(self, name: str) -> FloatArray:
        """Return a channel by its exact OpenFAST name."""

        try:
            index = self.channel_names.index(name)
        except ValueError as error:
            raise OpenFASTError(f"OpenFAST channel is unavailable: {name}") from error
        return np.asarray(self.data[:, index], dtype=np.float64)

    def unit(self, name: str) -> str:
        """Return the normalized unit for one exact channel name."""

        try:
            index = self.channel_names.index(name)
        except ValueError as error:
            raise OpenFASTError(f"OpenFAST channel is unavailable: {name}") from error
        return _clean_unit(self.channel_units[index])


@dataclass(frozen=True)
class ChannelMapping:
    """Map one named OpenFAST channel into a canonical SI observation."""

    source_channel: str
    source_unit: str
    target_name: str
    target_unit: str
    scale_to_target: float = 1.0
    offset_to_target: float = 0.0

    def __post_init__(self) -> None:
        if not self.source_channel.strip() or not self.target_name.strip():
            raise OpenFASTError("Channel mapping names must be nonempty.")
        if not _clean_unit(self.source_unit) or not _clean_unit(self.target_unit):
            raise OpenFASTError("Channel mapping units must be nonempty.")
        coefficients = np.asarray(
            [self.scale_to_target, self.offset_to_target],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(coefficients)) or self.scale_to_target == 0.0:
            raise OpenFASTError(
                "Channel mapping coefficients must be finite and nonzero."
            )


@dataclass(frozen=True)
class OpenFASTObservationBatch:
    """Canonical SI observations ready for a digital-twin adapter."""

    time_s: FloatArray
    values: FloatArray
    channel_names: tuple[str, ...]
    channel_units: tuple[str, ...]
    source_channels: tuple[str, ...]

    def __post_init__(self) -> None:
        time = np.asarray(self.time_s, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)
        n_channels = len(self.channel_names)
        if time.ndim != 1 or time.size < 2 or np.any(np.diff(time) <= 0.0):
            raise OpenFASTError("Observation time must be strictly increasing.")
        if values.shape != (time.size, n_channels):
            raise OpenFASTError("Observation values do not match time and channels.")
        if not np.all(np.isfinite(values)):
            raise OpenFASTError("Mapped OpenFAST observations must be finite.")
        if (
            n_channels < 1
            or len(set(self.channel_names)) != n_channels
            or len(self.channel_units) != n_channels
            or len(self.source_channels) != n_channels
        ):
            raise OpenFASTError("Mapped channel metadata is inconsistent.")


def _read_exact_array(
    stream: BinaryIO,
    count: int,
    dtype: str,
    label: str,
) -> np.ndarray:
    if count < 0:
        raise OpenFASTError(f"Invalid negative {label} count.")
    item_size = np.dtype(dtype).itemsize
    raw = stream.read(count * item_size)
    if len(raw) != count * item_size:
        raise OpenFASTError(f"Truncated OpenFAST binary while reading {label}.")
    return np.frombuffer(raw, dtype=dtype, count=count)


def _read_scalar(stream: BinaryIO, dtype: str, label: str) -> int | float:
    return _read_exact_array(stream, 1, dtype, label)[0].item()


def _decode_fixed_strings(
    raw: np.ndarray,
    count: int,
    width: int,
) -> tuple[str, ...]:
    encoded = raw.reshape(count, width)
    return tuple(
        bytes(row).decode("ascii", errors="strict").strip("\x00 ")
        for row in encoded
    )


def _read_binary_output(path: Path) -> OpenFASTOutput:
    with path.open("rb") as stream:
        format_id = int(_read_scalar(stream, "<i2", "format identifier"))
        if format_id not in {1, 2, 3, 4}:
            raise OpenFASTError(f"Unsupported OpenFAST binary format id: {format_id}")
        name_width = (
            int(_read_scalar(stream, "<i2", "channel-name width"))
            if format_id == 4
            else 10
        )
        n_output_channels = int(
            _read_scalar(stream, "<i4", "output-channel count")
        )
        n_steps = int(_read_scalar(stream, "<i4", "time-step count"))
        if not 1 <= name_width <= 1024:
            raise OpenFASTError("OpenFAST channel-name width is implausible.")
        if not 1 <= n_output_channels <= 100_000 or not 2 <= n_steps <= 100_000_000:
            raise OpenFASTError("OpenFAST binary dimensions are implausible.")

        first_time_value = float(_read_scalar(stream, "<f8", "time coefficient"))
        second_time_value = float(_read_scalar(stream, "<f8", "time coefficient"))
        if format_id == 3:
            column_scales = np.ones(n_output_channels, dtype=np.float64)
            column_offsets = np.zeros(n_output_channels, dtype=np.float64)
        else:
            column_scales = _read_exact_array(
                stream,
                n_output_channels,
                "<f4",
                "channel scales",
            ).astype(np.float64)
            column_offsets = _read_exact_array(
                stream,
                n_output_channels,
                "<f4",
                "channel offsets",
            ).astype(np.float64)

        description_length = int(
            _read_scalar(stream, "<i4", "description length")
        )
        if not 0 <= description_length <= 10_000_000:
            raise OpenFASTError("OpenFAST description length is implausible.")
        description_bytes = _read_exact_array(
            stream,
            description_length,
            "u1",
            "description",
        )
        description = bytes(description_bytes).decode("ascii", errors="replace").strip()
        metadata_count = n_output_channels + 1
        names = _decode_fixed_strings(
            _read_exact_array(
                stream,
                metadata_count * name_width,
                "u1",
                "channel names",
            ),
            metadata_count,
            name_width,
        )
        raw_units = _decode_fixed_strings(
            _read_exact_array(
                stream,
                metadata_count * name_width,
                "u1",
                "channel units",
            ),
            metadata_count,
            name_width,
        )
        units = tuple(_clean_unit(unit) for unit in raw_units)

        if format_id == 1:
            packed_time = _read_exact_array(stream, n_steps, "<i4", "packed time")
            if not np.isfinite(first_time_value) or first_time_value == 0.0:
                raise OpenFASTError("OpenFAST packed-time scale is invalid.")
            time = (
                packed_time.astype(np.float64) - second_time_value
            ) / first_time_value
        else:
            if not np.all(np.isfinite([first_time_value, second_time_value])):
                raise OpenFASTError("OpenFAST time origin or increment is invalid.")
            time = first_time_value + second_time_value * np.arange(n_steps)

        packed_dtype = "<f8" if format_id == 3 else "<i2"
        packed_values = _read_exact_array(
            stream,
            n_steps * n_output_channels,
            packed_dtype,
            "channel data",
        ).reshape(n_steps, n_output_channels)
        if stream.read(1):
            raise OpenFASTError("OpenFAST binary contains unexpected trailing bytes.")

    if format_id == 3:
        values = packed_values.astype(np.float64)
    else:
        invalid = (~np.isfinite(column_scales)) | (column_scales == 0.0)
        if np.any(invalid):
            raise OpenFASTError("OpenFAST channel scales must be finite and nonzero.")
        values = (
            packed_values.astype(np.float64) - column_offsets[np.newaxis, :]
        ) / column_scales[np.newaxis, :]
    data = np.column_stack((time, values))
    return OpenFASTOutput(
        source_path=path,
        description=description,
        file_format="binary",
        format_id=format_id,
        channel_names=names,
        channel_units=units,
        data=data,
    )


def _read_ascii_output(path: Path) -> OpenFASTOutput:
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        lines = stream.readlines()
    header_index: int | None = None
    for index, line in enumerate(lines[:50]):
        words = line.split()
        if words and words[0].lower() == "time":
            header_index = index
            break
    if header_index is None or header_index + 2 >= len(lines):
        raise OpenFASTError("Could not locate the OpenFAST ASCII Time header.")
    names = tuple(lines[header_index].split())
    units = tuple(_clean_unit(unit) for unit in lines[header_index + 1].split())
    try:
        data = np.loadtxt(lines[header_index + 2 :], dtype=np.float64)
    except ValueError as error:
        raise OpenFASTError(
            "OpenFAST ASCII data are not a rectangular table."
        ) from error
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return OpenFASTOutput(
        source_path=path,
        description="".join(lines[:header_index]).strip(),
        file_format="ascii",
        format_id=None,
        channel_names=names,
        channel_units=units,
        data=data,
    )


def read_openfast_output(path: str | Path) -> OpenFASTOutput:
    """Read a standard OpenFAST ASCII ``.out`` or binary ``.outb`` file."""

    source = Path(path)
    if not source.is_file():
        raise OpenFASTError(f"OpenFAST output does not exist: {source}")
    if source.stat().st_size == 0:
        raise OpenFASTError("OpenFAST output is empty.")
    if source.suffix.lower() == ".outb":
        return _read_binary_output(source)
    if source.suffix.lower() == ".out":
        return _read_ascii_output(source)
    raise OpenFASTError("OpenFAST output extension must be .out or .outb.")


def read_openfast_scalar_settings(
    path: str | Path,
    setting_names: Sequence[str],
) -> dict[str, str]:
    """Read declared scalar settings from an OpenFAST-style input file.

    This intentionally handles only the standard ``value name - description``
    records needed for provenance checks. It is not a general input-deck editor.
    """

    source = Path(path)
    if not source.is_file():
        raise OpenFASTError(f"OpenFAST input does not exist: {source}")
    requested = tuple(setting_names)
    if not requested or any(not name.strip() for name in requested):
        raise OpenFASTError("At least one nonempty setting name is required.")
    if len(set(requested)) != len(requested):
        raise OpenFASTError("Requested OpenFAST setting names must be unique.")
    found: dict[str, str] = {}
    with source.open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            fields = line.split()
            for name in requested:
                if name in fields:
                    index = fields.index(name)
                    if index == 0:
                        raise OpenFASTError(f"OpenFAST setting has no value: {name}")
                    if name in found:
                        raise OpenFASTError(f"Duplicate OpenFAST setting: {name}")
                    found[name] = fields[index - 1].strip('"')
    missing = [name for name in requested if name not in found]
    if missing:
        raise OpenFASTError(
            "OpenFAST settings are unavailable: " + ", ".join(missing)
        )
    return found


def map_openfast_channels(
    output: OpenFASTOutput,
    mappings: Sequence[ChannelMapping],
    *,
    start_s: float | None = None,
    end_s: float | None = None,
) -> OpenFASTObservationBatch:
    """Select channels, enforce declared units, and convert observations to SI."""

    mapping_sequence = tuple(mappings)
    if not mapping_sequence:
        raise OpenFASTError("At least one OpenFAST channel mapping is required.")
    target_names = tuple(mapping.target_name for mapping in mapping_sequence)
    if len(set(target_names)) != len(target_names):
        raise OpenFASTError("Mapped target names must be unique.")
    lower = output.time_s[0] if start_s is None else float(start_s)
    upper = output.time_s[-1] if end_s is None else float(end_s)
    if not np.all(np.isfinite([lower, upper])) or lower >= upper:
        raise OpenFASTError("Mapping interval must have finite start_s < end_s.")
    if lower < output.time_s[0] or upper > output.time_s[-1]:
        raise OpenFASTError("Mapping interval lies outside the OpenFAST output.")
    selected = (output.time_s >= lower) & (output.time_s <= upper)
    if np.count_nonzero(selected) < 2:
        raise OpenFASTError("Mapping interval contains fewer than two samples.")

    columns: list[FloatArray] = []
    for mapping in mapping_sequence:
        observed_unit = output.unit(mapping.source_channel)
        expected_unit = _clean_unit(mapping.source_unit)
        if observed_unit != expected_unit:
            raise OpenFASTError(
                f"Unit mismatch for {mapping.source_channel}: "
                f"expected {expected_unit}, found {observed_unit}."
            )
        source_values = output.channel(mapping.source_channel)[selected]
        columns.append(
            source_values * mapping.scale_to_target + mapping.offset_to_target
        )
    return OpenFASTObservationBatch(
        time_s=output.time_s[selected].copy(),
        values=np.column_stack(columns),
        channel_names=target_names,
        channel_units=tuple(
            _clean_unit(mapping.target_unit) for mapping in mapping_sequence
        ),
        source_channels=tuple(
            mapping.source_channel for mapping in mapping_sequence
        ),
    )


def uniform_time_grid_error(time_s: FloatArray) -> float:
    """Return maximum relative deviation from the median time increment."""

    time = np.asarray(time_s, dtype=np.float64)
    if time.ndim != 1 or time.size < 2 or np.any(np.diff(time) <= 0.0):
        raise OpenFASTError("time_s must be a strictly increasing vector.")
    increments = np.diff(time)
    median = float(np.median(increments))
    return float(np.max(np.abs(increments - median)) / median)
