"""Time synchronization utilities for Resonate clients."""

from __future__ import annotations

import math
from dataclasses import dataclass

ADAPTIVE_FORGETTING_CUTOFF = 0.75


@dataclass(slots=True)
class TimeElement:
    """Time transformation parameters."""

    last_update: int = 0
    offset: float = 0.0
    drift: float = 0.0


class ResonateTimeFilter:
    """Simple 2-state Kalman filter used to track clock offset and drift."""

    def __init__(self, process_std_dev: float = 2_000.0, forget_factor: float = 1.5) -> None:
        """Initialise the Kalman filter with noise and forgetting parameters."""
        if process_std_dev <= 0:
            raise ValueError("process_std_dev must be positive")
        if forget_factor < 1.0:
            raise ValueError("forget_factor must be >= 1")
        self._process_variance = process_std_dev * process_std_dev
        self._forget_variance_factor = forget_factor * forget_factor
        self._time_element = TimeElement()
        self._count = 0
        self._offset = 0.0
        self._drift = 0.0
        self._offset_covariance = math.inf
        self._offset_drift_covariance = 0.0
        self._drift_covariance = 0.0
        self._last_update = 0

    def reset(self) -> None:
        """Reset the filter state."""
        self._time_element = TimeElement()
        self._count = 0
        self._offset = 0.0
        self._drift = 0.0
        self._offset_covariance = math.inf
        self._offset_drift_covariance = 0.0
        self._drift_covariance = 0.0
        self._last_update = 0

    def update(self, measurement: float, max_error: float, time_added: int) -> None:
        """Update the filter with a new offset measurement."""
        if time_added == self._last_update:
            return

        dt = float(time_added - self._last_update)
        if self._last_update == 0:
            dt = 0.0
        self._last_update = time_added

        measurement_variance = max(max_error, 1.0) ** 2

        if self._count <= 0:
            self._count += 1
            self._offset = float(measurement)
            self._offset_covariance = measurement_variance
            self._drift = 0.0
            self._time_element = TimeElement(
                last_update=time_added,
                offset=self._offset,
                drift=self._drift,
            )
            return

        if self._count == 1 and dt > 0:
            self._count += 1
            self._drift = (measurement - self._offset) / dt
            self._offset = float(measurement)
            self._drift_covariance = (self._offset_covariance + measurement_variance) / max(dt, 1.0)
            self._offset_covariance = measurement_variance
            self._time_element = TimeElement(
                last_update=time_added,
                offset=self._offset,
                drift=self._drift,
            )
            return

        if dt <= 0:
            # We need positive dt from now on; ignore bogus updates
            return

        offset_pred = self._offset + self._drift * dt
        dt_squared = dt * dt

        new_drift_covariance = self._drift_covariance
        new_offset_drift_covariance = self._offset_drift_covariance + self._drift_covariance * dt
        offset_process_variance = dt * self._process_variance
        new_offset_covariance = (
            self._offset_covariance
            + 2 * self._offset_drift_covariance * dt
            + self._drift_covariance * dt_squared
            + offset_process_variance
        )

        residual = measurement - offset_pred
        max_residual_cutoff = max_error * ADAPTIVE_FORGETTING_CUTOFF

        if self._count < 100:
            self._count += 1
        elif abs(residual) > max_residual_cutoff:
            factor = self._forget_variance_factor
            new_drift_covariance *= factor
            new_offset_drift_covariance *= factor
            new_offset_covariance *= factor

        innovation_covariance = new_offset_covariance + measurement_variance
        if innovation_covariance <= 0:
            return
        uncertainty = 1.0 / innovation_covariance
        offset_gain = new_offset_covariance * uncertainty
        drift_gain = new_offset_drift_covariance * uncertainty

        self._offset = offset_pred + offset_gain * residual
        self._drift += drift_gain * residual

        self._drift_covariance = new_drift_covariance - drift_gain * new_offset_drift_covariance
        self._offset_drift_covariance = (
            new_offset_drift_covariance - drift_gain * new_offset_covariance
        )
        self._offset_covariance = new_offset_covariance - offset_gain * new_offset_covariance

        self._time_element = TimeElement(
            last_update=time_added,
            offset=self._offset,
            drift=self._drift,
        )

    @property
    def covariance(self) -> float:
        """Return the covariance (variance) estimate for the offset."""
        return self._offset_covariance

    @property
    def error(self) -> float:
        """Return the standard deviation estimate in microseconds."""
        if math.isinf(self._offset_covariance):
            return float("inf")
        return math.sqrt(max(self._offset_covariance, 0.0))

    @property
    def ready(self) -> bool:
        """Return True when the filter has enough measurements for conversions."""
        return self._count >= 2 and not math.isinf(self._offset_covariance)

    def compute_server_time(self, client_time: int) -> int:
        """Map a client timestamp to the server clock."""
        dt = float(client_time - self._time_element.last_update)
        offset = self._time_element.offset + self._time_element.drift * dt
        return round(client_time + offset)

    def compute_client_time(self, server_time: int) -> int:
        """Map a server timestamp to the client clock."""
        drift = self._time_element.drift
        denom = 1.0 + drift
        if denom == 0:
            return server_time
        numerator = server_time - self._time_element.offset + drift * self._time_element.last_update
        return round(numerator / denom)
