def parse_action(action_vec, env_state, cfg):
    """
    Convert NN action vector → physical controls.
    ONLY place you change when action space changes.
    """

    # ===== CURRENT: only delta =====
    delta = action_vec[:, 0]

    return {
        "delta": delta,
    }

    # ===== FUTURE EXAMPLES =====
    # delta = action_vec[:,0]
    # alpha = action_vec[:,1]
    # return {"delta": delta, "alpha": alpha}

    # A = cfg.A_max * action_vec[:,0]
    # w = cfg.w_max * action_vec[:,1]
    # return {"A":A, "w":w}