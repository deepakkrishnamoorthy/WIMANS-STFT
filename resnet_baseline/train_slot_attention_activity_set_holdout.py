from train_activity_set_common import run_experiment


if __name__ == "__main__":
    run_experiment(
        "slot_attention_activity_set",
        "ResNet18 plus slot-attention head for 9-label room-level activity-set prediction.",
    )
