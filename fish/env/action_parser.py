def parse_action(action_vec, env_state, cfg):
    """
    Convert NN action vector → physical controls.
    """
    assert action_vec.shape[-1] == 6, \
    f"Expected 6 actions, got shape {action_vec.shape}"

    kp = action_vec[:, 0]
    kd = action_vec[:, 1]
    ki = action_vec[:, 2]

    v_kp = action_vec[:, 3]
    v_kd = action_vec[:, 4]
    v_ki = action_vec[:, 5]

    # L = action_vec[:, 6]

    return {
        "kp": kp,
        "kd": kd,
        "ki": ki,
        "v_kp": v_kp,
        "v_kd": v_kd,
        "v_ki": v_ki,

    }


def scale_from_unit(raw, min_val, max_val):
    """
    Maps raw ∈ [-1, 1] → [min_val, max_val]
    """
    return min_val + 0.5 * (raw + 1.0) * (max_val - min_val)

def get_pid_gains(action, cfg):
    """
    action: dict or struct with fields:
        kp_raw, kd_raw, ki_raw
        vkp_raw, vkd_raw, vki_raw
    """

    kp  = scale_from_unit(action["kp"],  cfg.min_kp,  cfg.max_kp)
    kd  = scale_from_unit(action["kd"],  cfg.min_kd,  cfg.max_kd)
    ki  = scale_from_unit(action["ki"],  cfg.min_ki,  cfg.max_ki)

    v_kp = scale_from_unit(action["v_kp"], cfg.min_v_kp, cfg.max_v_kp)
    v_kd = scale_from_unit(action["v_kd"], cfg.min_v_kd, cfg.max_v_kd)
    v_ki = scale_from_unit(action["v_ki"], cfg.min_v_ki, cfg.max_v_ki)

    return kp, kd, ki, v_kp, v_kd, v_ki

# def get_lookahead(action, cfg):
#     L = scale_from_unit(action["L"], cfg.min_L, cfg.max_L)
#     return L