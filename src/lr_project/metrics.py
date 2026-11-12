# Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np

from lr_project.rollout import Rollout

GRAVITY = 9.81  # m/s^2
FOOT_RADIUS = 0.022  # m


#############################
# HELPERS (provided)
#############################


def touchdowns(contact: np.ndarray) -> np.ndarray:
    """Step indices where a 1-D contact signal switches from False to True."""
    return np.flatnonzero(~contact[:-1] & contact[1:]) + 1


def lift_offs(contact: np.ndarray) -> np.ndarray:
    """Step indices where a 1-D contact signal switches from True to False."""
    return np.flatnonzero(contact[:-1] & ~contact[1:]) + 1


def phases(contact: np.ndarray, stance: bool) -> list[tuple[int, int]]:
    """Complete stance (or swing) phases of a 1-D contact signal, as
    (start, end) step indices: phase = contact[start:end]. Phases cut off by
    the start or end of the rollout are ignored."""
    starts = touchdowns(contact) if stance else lift_offs(contact)
    ends = lift_offs(contact) if stance else touchdowns(contact)
    out = []
    for start in starts:
        later = ends[ends > start]
        if len(later):
            out.append((int(start), int(later[0])))
    return out


def body_weight(r: Rollout) -> np.ndarray:
    """Weight of each robot, m g, in N. Shape (N,)."""
    return r.mass * GRAVITY


#############################
# EXAMPLE (provided): signal + metric pair
#############################


def lin_vel_error_per_step(r: Rollout) -> np.ndarray:
    """Norm of the (vx, vy) tracking error at each step, in m/s. Shape (N, T)."""
    actual = r.base_lin_vel[..., :2]  # (N, T, 2)
    command = r.cmd[:, None, :2]  # (N, 1, 2), broadcast over time
    return np.linalg.norm(actual - command, axis=-1)


def lin_vel_error(r: Rollout) -> np.ndarray:
    """RMS (vx, vy) tracking error of each robot over its rollout, in m/s. Shape (N,)."""
    error = lin_vel_error_per_step(r)
    return np.sqrt(np.mean(error**2, axis=1))


#############################
# PER-STEP SIGNALS, shape (N, T)
#############################


def yaw_rate_error_per_step(r: Rollout) -> np.ndarray:
    """Absolute yaw rate tracking error at each step, in rad/s. Shape (N, T)."""
    actual = r.base_ang_vel[..., 2]  # (N, T)
    command = r.cmd[:, None, 2]  # (N, 1), broadcast over time
    return np.abs(actual - command)


def mechanical_power(r: Rollout) -> np.ndarray:
    """Sum over joints of |torque * joint velocity| at each step, in W. Shape (N, T)."""
    return np.sum(np.abs(r.joint_torque * r.joint_vel), axis=-1)


def total_vertical_grf(r: Rollout) -> np.ndarray:
    """Sum of the vertical GRF of the four feet at each step, in body weights. Shape (N, T)."""
    # Absolute value: robust to the sign convention of the contact force.
    vertical = np.abs(r.foot_grf[..., 2])  # (N, T, 4)
    return np.sum(vertical, axis=-1) / body_weight(r)[:, None]


def base_ang_vel_xy_per_step(r: Rollout) -> np.ndarray:
    """Norm of the roll and pitch angular velocity of the base, in rad/s. Shape (N, T).
    Zero for a perfectly level, non-rocking base."""
    return np.linalg.norm(r.base_ang_vel[..., :2], axis=-1)


def base_tilt_per_step(r: Rollout) -> np.ndarray:
    """Angle between the base vertical axis and the world vertical, in rad. Shape (N, T).
    Quaternions are (w, x, y, z), the MuJoCo convention."""
    x, y = r.base_quat[..., 1], r.base_quat[..., 2]
    cos_tilt = 1.0 - 2.0 * (x**2 + y**2)  # z component of the body z axis in the world
    return np.arccos(np.clip(cos_tilt, -1.0, 1.0))


#############################
# PER-ROBOT METRICS, shape (N,)
#############################


def _horizontal_speed(r: Rollout) -> np.ndarray:
    """Mean horizontal base speed of each robot, in m/s. Shape (N,)."""
    return np.mean(np.linalg.norm(r.base_lin_vel[..., :2], axis=-1), axis=1)


def yaw_rate_error(r: Rollout) -> np.ndarray:
    """RMS yaw rate tracking error of each robot, in rad/s. Shape (N,)."""
    error = yaw_rate_error_per_step(r)
    return np.sqrt(np.mean(error**2, axis=1))


def cost_of_transport(r: Rollout) -> np.ndarray:
    """Mechanical cost of transport of each robot:
    mean power / (m g mean horizontal speed). Dimensionless. Shape (N,).
    NaN for robots slower than 0.05 m/s (undefined when standing or turning in place)."""
    power = np.mean(mechanical_power(r), axis=1)
    speed = _horizontal_speed(r)
    cot = np.full_like(power, np.nan, dtype=float)
    moving = speed > 0.05
    cot[moving] = power[moving] / (body_weight(r)[moving] * speed[moving])
    return cot


def duty_factor(r: Rollout) -> np.ndarray:
    """Fraction of time the feet are on the ground, averaged over the four
    feet (> 0.5 walk, < 0.5 trot/run). Shape (N,)."""
    return np.mean(r.foot_contact, axis=(1, 2))


def stride_frequency(r: Rollout) -> np.ndarray:
    """Strides per second of each robot: 1 / mean time between two
    touchdowns of the same foot, averaged over the feet. Shape (N,).
    NaN if no foot touched down twice (e.g. standing still)."""
    out = np.full(r.num_robots, np.nan)
    for i in range(r.num_robots):
        freqs = []
        for foot in range(r.foot_contact.shape[2]):
            td = touchdowns(r.foot_contact[i, :, foot])
            if len(td) >= 2:
                period = np.mean(np.diff(td)) * r.dt
                freqs.append(1.0 / period)
        if freqs:
            out[i] = np.mean(freqs)
    return out


def stride_length(r: Rollout) -> np.ndarray:
    """Distance travelled per stride: mean horizontal speed / stride frequency, in m. Shape (N,)."""
    return _horizontal_speed(r) / stride_frequency(r)


def foot_clearance(r: Rollout) -> np.ndarray:
    """Mean over swing phases of the maximum foot height above the ground,
    in m (flat ground only). Shape (N,). NaN if no complete swing phase."""
    out = np.full(r.num_robots, np.nan)
    for i in range(r.num_robots):
        peaks = []
        for foot in range(r.foot_contact.shape[2]):
            height = r.foot_height[i, :, foot]
            for start, end in phases(r.foot_contact[i, :, foot], stance=False):
                peaks.append(np.max(height[start:end]) - FOOT_RADIUS)
        if peaks:
            out[i] = np.mean(peaks)
    return out


def peak_vertical_grf(r: Rollout) -> np.ndarray:
    """Mean over stance phases of the peak vertical GRF, in body weights. Shape (N,).
    NaN if no complete stance phase."""
    weight = body_weight(r)
    out = np.full(r.num_robots, np.nan)
    for i in range(r.num_robots):
        peaks = []
        for foot in range(r.foot_contact.shape[2]):
            force = np.abs(r.foot_grf[i, :, foot, 2])
            for start, end in phases(r.foot_contact[i, :, foot], stance=True):
                peaks.append(np.max(force[start:end]) / weight[i])
        if peaks:
            out[i] = np.mean(peaks)
    return out


def weight_support(r: Rollout) -> np.ndarray:
    """Mean total vertical GRF in body weights. About 1 for steady walking
    (sanity check of the force measurement). Shape (N,)."""
    return np.mean(total_vertical_grf(r), axis=1)


def action_rate(r: Rollout) -> np.ndarray:
    """Mean norm of the change in action between two control steps. Shape (N,)."""
    change = np.diff(r.action, axis=1)  # (N, T-1, 12)
    return np.mean(np.linalg.norm(change, axis=-1), axis=1)


#############################
# OWN METRICS (base stability and smoothness), shape (N,)
#############################


def base_ang_vel_xy(r: Rollout) -> np.ndarray:
    """RMS roll and pitch angular velocity of the base, in rad/s. Shape (N,).
    Base stability: lower = calmer body (lecture 4 compares gaits with the mean
    base angular velocity)."""
    return np.sqrt(np.mean(base_ang_vel_xy_per_step(r) ** 2, axis=1))


def base_tilt(r: Rollout) -> np.ndarray:
    """Mean tilt of the base with respect to the vertical, in deg. Shape (N,).
    Base stability: lower = more level body."""
    return np.degrees(np.mean(base_tilt_per_step(r), axis=1))


def base_height_std(r: Rollout) -> np.ndarray:
    """Standard deviation of the base height over the rollout, in m. Shape (N,).
    Base stability: vertical bouncing of the body."""
    return np.std(r.base_pos[..., 2], axis=1)


def joint_acceleration(r: Rollout) -> np.ndarray:
    """RMS joint acceleration (finite difference of joint velocities), in rad/s^2. Shape (N,).
    Smoothness of the motion (lecture 4 compares gaits with the mean joint acceleration)."""
    acc = np.diff(r.joint_vel, axis=1) / r.dt  # (N, T-1, 12)
    return np.sqrt(np.mean(acc**2, axis=(1, 2)))
