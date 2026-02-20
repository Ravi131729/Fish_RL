
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

