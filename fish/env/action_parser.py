def parse_action(action_vec, env_state, cfg):
    """
    Convert NN action vector → physical controls.
    ONLY place you change when action space changes.
    """
    assert action_vec.shape[-1] == 2, \
    f"Expected 2 actions, got shape {action_vec.shape}"
    # ===== CURRENT: only delta =====
    kp = action_vec[:, 0]
    kd = action_vec[:, 1]
    L = action_vec[:, 2]

    return {
        "kp": kp,
        "kd": kd,
        "L": L,
       
    }

    # ===== FUTURE EXAMPLES =====
    # delta = action_vec[:,0]
    # alpha = action_vec[:,1]
    # return {"delta": delta, "alpha": alpha}

    # A = cfg.A_max * action_vec[:,0]
    # w = cfg.w_max * action_vec[:,1]
    # return {"A":A, "w":w}