from train_multichannel_common import run_experiment


if __name__ == "__main__":
    run_experiment(
        "resnet18_multichannel",
        "ResNet18 on PCA or no-PCA multi-channel STFT Top-5 features.",
    )
