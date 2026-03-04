"""
Entry point for the federated learning experiment.

Configure hyperparameters here and run with:
    python main.py
"""


from experiment import run_experiment

if __name__ == "__main__":
    run_experiment(
        # ---- Federated setup ----
        rounds=350,
        n_clients=50,
        k_select=15,                # K clients selected per round
        dir_alpha=0.3,              # Dirichlet heterogeneity (lower = more heterogeneous)
        # ---- Tracks ----

        run_random=True,            # FedAvg baseline with random selection
        run_vdn=False,              # VDN-based MARL selection

        # ---- Initial attack ----
        initial_flip_fraction=0.4,  # fraction of clients that are attackers from round 1
        flip_rate_initial=1.0,      # fraction of samples flipped per attacker

        # ---- Cumulative attack  ----
        flip_add_fraction=0.0,      # fraction of honest clients converted per attack round
        attack_rounds=[600],
        flip_rate_new_attack=0.0,

        # ---- Attack type ----
        targeted_only_map_classes=True,
        target_map=None,            # uses default visually similar class swaps

        # ---- Local training ----
        max_per_client=2500,
        local_lr=0.005,
        #local_steps=10,
        local_epochs = 5,            # epochs for selected clients
        probe_batches=10,            # batches used to compute gener metric

        # ---- Server gradient EMA ----
        mom_beta=0.90,               # beta for server gradient momentum

        # ---- Reward ----
        reward_window_W=5,           # window size for windowed reward

        # ---- MARL / VDN ----
        marl_eps=0.15,
        marl_swap_m=2,
        marl_lr=1e-4,
        marl_gamma=0.90,
        marl_hidden=128,
        marl_target_sync_every=20,
        warmup_transitions=50,
        start_train_round=50,
        updates_per_round=50,
        train_every=1,

        # ---- Replay buffer ----
        buf_size=20000,
        batch_base=32,
        batch_max=256,
        batch_buffer_ratio=4,

        # ---- PER ----
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_end=1.0,
        per_beta_steps=4000,
        per_eps=1e-3,

        # ---- Evaluation ----
        val_shuffle=False,
        val_per_class=200,
        eval_max_batches=20,
        print_every=1,
        print_advfo_every=20,       # prints client ranking every N rounds

        out_dir=".",
    )
