
Observation vector shape: **(6,)**

```
obs = [
    u_x,          # body forward velocity (m/s)
    u_y,          # body lateral velocity (m/s)
    qh,           # heading (rad)
    ct_err,       # cross-track error (m)
    hd_err,       # heading error wrt path (rad)
    delta_prev    # previous steering angle (rad)
]
```


# Action from Policy

Policy output:

```
delta_raw ∈ [-1, 1]
```

Single continuous action.

---

# servo angle


```
delta_change = delta_rate_max * dt * delta_raw
delta = delta_prev + delta_change
delta = clip(delta, -delta_max, delta_max)
```

Parameters:

```
dt = 0.01 s
delta_max = 1.0 rad
delta_rate_max = 5.0 rad/s
```

So:

* max servo angle = ±1 rad
* max servo rate = 5 rad/s

---

# PID Branch Variation (`feature/pid-servo-training`)

This branch changes the control setup from a single steering action to a
policy that outputs PID gains for heading and speed loops.

## Key difference vs current/main notes

- Main/current notes above: policy outputs one action (`delta_raw`).
- PID branch: policy outputs 6 actions (normalized in `[-1, 1]`):

```
[
  kp, kd, ki,      # heading PID gains
  v_kp, v_kd, v_ki # forward-speed PID gains
]
```

## Gain scaling (PID branch)

Raw policy outputs are scaled to physical ranges from config:

```
gain = min_gain + 0.5 * (raw + 1.0) * (max_gain - min_gain)
```

See: `fish/env/action_parser.py` (`get_pid_gains`).

## Controller behavior (PID branch)

- Heading loop computes steering command `delta` using `kp/kd/ki`
  with angle wrapping and integral anti-windup.
- Speed loop computes throttle command using `v_kp/v_kd/v_ki`
  with integral anti-windup.
- Steering and throttle are then clipped/rate-limited by env constraints.

## Files to review in PID branch

- `fish/env/action_parser.py`
- `fish/env/env_fish.py`
- `fish/env/observation.py`
- `fish/env/reward.py`
- `fish/configs/ppo_fish.yaml`

Use `feature/pid-servo-training` when reproducing PID-gain training results.
