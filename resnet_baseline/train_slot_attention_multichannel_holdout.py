from train_multichannel_common import run_experiment


if __name__ == "__main__":
    run_experiment(
        "slot_attention_multichannel",
        "ResNet18 slot-attention head on PCA or no-PCA multi-channel STFT Top-5 features.",
    )
