from train_multichannel_common import run_experiment


if __name__ == "__main__":
    run_experiment(
        "clstm_multichannel",
        "ResNet18 feature extractor with CLSTM head on PCA or no-PCA multi-channel STFT Top-5 features.",
    )
