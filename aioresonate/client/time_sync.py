"""
Time synchronization utilities for Resonate clients.

1 to 1 port of the ESPHome implementation.
"""

# ruff: noqa: ERA001

from __future__ import annotations

import math
from dataclasses import dataclass

# Residual threshold as fraction of max_error for triggering adaptive forgetting.
# When residual > CUTOFF * max_error, the filter applies forgetting to recover from outliers.
ADAPTIVE_FORGETTING_CUTOFF = 0.75


@dataclass(slots=True)
class TimeElement:
    """Time transformation parameters."""

    last_update: int = 0
    offset: float = 0.0
    drift: float = 0.0


class ResonateTimeFilter:
    """
    Two-dimensional Kalman filter for NTP-style time synchronization.

    This class implements a time synchronization filter that tracks both the timestamp
    offset and clock drift rate between a client and server. It processes measurements
    obtained with NTP-style time messages that contain round-trip timing information to
    optimally estimate the time relationship while accounting for network latency
    uncertainty.

    The filter maintains a 2D state vector [offset, drift] with associated covariance
    matrix to track estimation uncertainty. An adaptive forgetting factor helps the
    filter recover quickly from network disruptions or server clock adjustments.
    """

    _last_update: int = 0
    _count: int = 0

    _offset: float = 0.0
    _drift: float = 0.0

    _offset_covariance: float = math.inf
    _offset_drift_covariance: float = 0.0
    _drift_covariance: float = 0.0

    _process_variance: float
    _forget_variance_factor: float

    _current_time_element: TimeElement

    def __init__(self, process_std_dev: float = 0.01, forget_factor: float = 1.001) -> None:
        """Initialise the Kalman filter with noise and forgetting parameters."""
        self._process_variance = process_std_dev * process_std_dev
        self._forget_variance_factor = forget_factor * forget_factor
        self._current_time_element = TimeElement()

    def update(self, measurement: int, max_error: int, time_added: int) -> None:
        """
        Process a new time synchronization measurement through the Kalman filter.

        Updates the filter's offset and drift estimates using a two-stage Kalman filter
        algorithm: predict based on the drift model then correct using the new
        measurement. The measurement uncertainty is derived from the network round-trip
        delay.

        Note:
            Thread-safe when called concurrently with compute_server_time() or
            compute_client_time().

        Args:
            measurement: Computed offset from NTP-style exchange: ((T2-T1)+(T3-T4))/2
                in microseconds.
            max_error: Half the round-trip delay: ((T4-T1)-(T3-T2))/2, representing
                maximum measurement uncertainty in microseconds.
            time_added: Client timestamp when this measurement was taken in
                microseconds.
        """
        if time_added == self._last_update:
            # Skip duplicate timestamps to avoid division by zero in drift calculation
            return

        dt: float = float(time_added - self._last_update)
        self._last_update = time_added

        update_std_dev: float = float(max_error)
        measurement_variance: float = update_std_dev * update_std_dev

        # Filter initialization: First measurement establishes offset baseline
        if self._count <= 0:
            self._count += 1

            self._offset = float(measurement)
            self._offset_covariance = measurement_variance
            self._drift = 0.0  # No drift information available yet

            self._current_time_element = TimeElement(
                last_update=self._last_update,
                offset=self._offset,
                drift=self._drift,
            )

            return

        # Second measurement: Initial drift estimation from finite differences
        if self._count == 1:
            self._count += 1

            self._drift = (measurement - self._offset) / dt
            self._offset = float(measurement)

            # Drift variance estimated from propagation of offset uncertainties
            self._drift_covariance = (self._offset_covariance + measurement_variance) / dt
            self._offset_covariance = measurement_variance

            self._current_time_element = TimeElement(
                last_update=self._last_update,
                offset=self._offset,
                drift=self._drift,
            )

            return

        ### Kalman Prediction Step ###
        ## State prediction: x_k|k-1 = F * x_k-1|k-1

        offset: float = self._offset + self._drift * dt

        # Covariance prediction: P_k|k-1 = F * P_k-1|k-1 * F^T + Q
        # State transition matrix F = [1, dt; 0, 1]
        dt_squared: float = dt * dt

        # Process noise only applied to offset (modeling clock jitter/wander)
        drift_process_variance: float = 0.0  # Drift assumed stable
        new_drift_covariance: float = self._drift_covariance + drift_process_variance

        offset_drift_process_variance: float = 0.0
        new_offset_drift_covariance: float = (
            self._offset_drift_covariance
            + self._drift_covariance * dt
            + offset_drift_process_variance
        )

        offset_process_variance: float = dt * self._process_variance
        new_offset_covariance = (
            self._offset_covariance
            + 2 * self._offset_drift_covariance * dt
            + self._drift_covariance * dt_squared
            + offset_process_variance
        )

        ### Innovation and Adaptive Forgetting ###
        residual: float = measurement - offset  # Innovation: y_k = z_k - H * x_k|k-1
        max_residual_cutoff: float = max_error * ADAPTIVE_FORGETTING_CUTOFF

        if self._count < 100:
            # Build sufficient history before enabling adaptive forgetting
            self._count += 1
        elif residual > max_residual_cutoff:
            # Large prediction error detected - likely network disruption or clock adjustment
            # Apply forgetting factor to increase Kalman gain and accelerate convergence
            new_drift_covariance *= self._forget_variance_factor
            new_offset_drift_covariance *= self._forget_variance_factor
            new_offset_covariance *= self._forget_variance_factor

        ### Kalman Update Step ###
        # Innovation covariance: S = H * P * H^T + R, where H = [1, 0]
        uncertainty: float = 1.0 / (new_offset_covariance + measurement_variance)

        # Kalman gain: K = P * H^T * S^(-1)
        offset_gain: float = new_offset_covariance * uncertainty
        drift_gain: float = new_offset_drift_covariance * uncertainty

        # State update: x_k|k = x_k|k-1 + K * y_k
        self._offset = offset + offset_gain * residual
        self._drift += drift_gain * residual

        # Covariance update: P_k|k = (I - K*H) * P_k|k-1
        # Using simplified form to ensure numerical stability
        self._drift_covariance = new_drift_covariance - drift_gain * new_offset_drift_covariance
        self._offset_drift_covariance = (
            new_offset_drift_covariance - drift_gain * new_offset_covariance
        )
        self._offset_covariance = new_offset_covariance - offset_gain * new_offset_covariance

        self._current_time_element = TimeElement(
            last_update=self._last_update,
            offset=self._offset,
            drift=self._drift,
        )

    def compute_server_time(self, client_time: int) -> int:
        """
        Convert a client timestamp to the equivalent server timestamp.

        Applies the current offset and drift compensation to transform from client time
        domain to server time domain. The transformation accounts for both static offset
        and dynamic drift accumulated since the last filter update.

        Note:
            Not thread-safe when called concurrently with compute_client_time().

        Args:
            client_time: Client timestamp in microseconds.

        Returns:
            Equivalent server timestamp in microseconds.
        """
        # Transform: T_server = T_client + offset + drift * (T_client - T_last_update)
        # Compute instantaneous offset accounting for linear drift:
        # offset(t) = offset_base + drift * (t - t_last_update)

        # Retrieve latest time transformation parameters

        dt = float(client_time - self._current_time_element.last_update)
        offset = round(self._current_time_element.offset + self._current_time_element.drift * dt)
        return client_time + offset

    def compute_client_time(self, server_time: int) -> int:
        """
        Convert a server timestamp to the equivalent client timestamp.

        Inverts the time transformation to convert from server time domain to client
        time domain. Accounts for both offset and drift effects in the inverse
        transformation.

        Note:
            Not thread-safe when called concurrently with compute_server_time().

        Args:
            server_time: Server timestamp in microseconds.

        Returns:
            Equivalent client timestamp in microseconds.
        """
        # Inverse transform solving for T_client:
        # T_server = T_client + offset + drift * (T_client - T_last_update)
        # T_server = (1 + drift) * T_client + offset - drift * T_last_update
        # T_client = (T_server - offset + drift * T_last_update) / (1 + drift)

        return round(
            (
                float(server_time)
                - self._current_time_element.offset
                + self._current_time_element.drift * self._current_time_element.last_update
            )
            / (1.0 + self._current_time_element.drift)
        )

    def reset(self) -> None:
        """Reset the filter state."""
        self._count = 0
        self._offset = 0.0
        self._drift = 0.0
        self._offset_covariance = math.inf
        self._offset_drift_covariance = 0.0
        self._drift_covariance = 0.0

        self._current_time_element = TimeElement()

    @property
    def error(self) -> int:
        """Return the standard deviation estimate in microseconds."""
        return round(math.sqrt(self._offset_covariance))

    @property
    def covariance(self) -> int:
        """Return the covariance (variance) estimate for the offset."""
        return round(self._offset_covariance)
