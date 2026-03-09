# Fish Robot – RL PID Gain Policy Deployment Guide

This document explains how to deploy a trained reinforcement learning (RL) policy that outputs **PID gains** for:

- Heading control (servo angle `delta`)
- Forward speed control (throttle)

The policy does NOT output direct motor commands.
It outputs normalized PID gains in `[-1, 1]`, which are scaled and applied inside classical PID controllers.

---

# 1. Environment Configuration

```yaml
dt: 0.01
max_steps: 1000

delta_max: 1.3
delta_rate_max: 5.0

max_heading_angle: 3.14159
min_heading_angle: -3.14159

max_ux: 0.5
min_ux: 0.1

# Heading PID limits
max_kp: 2.0
max_kd: 0.5
max_ki: 0.5

min_kp: 0.0
min_kd: 0.0
min_ki: 0.0

# Velocity PID limits
max_v_kp: 3.0
max_v_kd: 0.5
max_v_ki: 0.5

min_v_kp: 0.0
min_v_kd: 0.0
min_v_ki: 0.0

heading_max_int_error: 0.2
speed_max_int_error: 0.1
```

---

# 2. Observation Construction

```python
hd_err = qh - heading_des
hd_err = jnp.arctan2(jnp.sin(hd_err), jnp.cos(hd_err))
hd_err = hd_err / jnp.pi

speed_err = ux - state.desired_ux
speed_err = speed_err / cfg.max_ux

obs = jnp.concatenate([
    ux[:, None],
    speed_err[:, None],
    hd_err[:, None],
    state.delta_prev[:, None],
], axis=1)

if key is not None:
    obs = add_obs_noise(key, obs)
```

## Observation Vector

```
obs = [
    ux,
    speed_error,
    heading_error,
    delta_prev
]
```

---

# 3. Policy Output

The RL policy outputs 6 values in `[-1, 1]`:

```
action = {
    "kp", "kd", "ki",
    "v_kp", "v_kd", "v_ki"
}
```

These are NOT physical gains yet. They must be scaled.

---

# 4. Scaling Policy Output to Physical Gains

## Scaling Function

```python
def scale_from_unit(raw, min_val, max_val):
    """
    Maps raw ∈ [-1, 1] → [min_val, max_val]
    """
    return min_val + 0.5 * (raw + 1.0) * (max_val - min_val)
```

## Gain Mapping

```python
def get_pid_gains(action, cfg):

    kp  = scale_from_unit(action["kp"],  cfg.min_kp,  cfg.max_kp)
    kd  = scale_from_unit(action["kd"],  cfg.min_kd,  cfg.max_kd)
    ki  = scale_from_unit(action["ki"],  cfg.min_ki,  cfg.max_ki)

    v_kp = scale_from_unit(action["v_kp"], cfg.min_v_kp, cfg.max_v_kp)
    v_kd = scale_from_unit(action["v_kd"], cfg.min_v_kd, cfg.max_v_kd)
    v_ki = scale_from_unit(action["v_ki"], cfg.min_v_ki, cfg.max_v_ki)

    return kp, kd, ki, v_kp, v_kd, v_ki
```

---

# 5. Heading (Servo) Control

## 5.1 Heading Error

```python
hd_error = qh - state.heading_desired
hd_error = jnp.arctan2(jnp.sin(hd_error), jnp.cos(hd_error))
hd_error = hd_error / jnp.pi
```

## 5.2 Integral Term

```python
heading_error_int = state.heading_error_int + hd_error * cfg.dt
heading_error_int = jnp.clip(
    heading_error_int,
    -cfg.heading_max_int_error,
    cfg.heading_max_int_error
)
```

## 5.3 PID Law

```python
delta = (
    kp * hd_error
    + kd * (hd_error - state.heading_error_prev) / cfg.dt
    + ki * heading_error_int
)
```

---

# 6. Servo Safety Limits

## Rate Limit

```python
delta_change = delta - state.delta_prev
delta_change = jnp.clip(
    delta_change,
    -cfg.delta_rate_max * cfg.dt,
    cfg.delta_rate_max * cfg.dt
)

delta = state.delta_prev + delta_change
```

## Absolute Limit

```python
delta = jnp.clip(delta, -cfg.delta_max, cfg.delta_max)
```

These limits are critical for stability.

---

# 7. Velocity (Throttle) Control

## 7.1 Speed Error

```python
speed_error = ux - state.desired_ux
speed_error = speed_error / cfg.max_ux
```

## 7.2 Integral Term

```python
speed_error_int = state.speed_error_int + speed_error * cfg.dt
speed_error_int = jnp.clip(
    speed_error_int,
    -cfg.speed_max_int_error,
    cfg.speed_max_int_error
)
```

## 7.3 PID Law

```python
throttle = (
    v_kp * speed_error
    + v_kd * (speed_error - state.velocity_error_prev) / cfg.dt
    + v_ki * speed_error_int
)
```

---

# 8. Full Deployment Loop

At each control cycle:

1. Read sensors (heading `qh`, velocity `ux`)
2. Build observation
3. Normalize observation (use training mean/std)
4. Run policy → get raw gains
5. Scale gains
6. Compute:
   - `delta`
   - `throttle`
7. Apply:
   - Servo angle = `delta`
   - Convert throttle → propulsion command

---

# 9. Critical Deployment Requirements

- Use exact same `dt = 0.01`
- Use same observation normalization stats
- Keep servo rate limits
- Keep integral clipping
- Keep gain scaling identical to training

If any of these change, behavior may become unstable.

---

# 10. System Architecture

```
Observation → Policy → PID Gains → PID Controller → Safety Limits → Robot
```

This gives:

- Stability from PID
- Adaptivity from RL
- Safety from limits
- Smooth behavior from rate limiting

---

# End of File