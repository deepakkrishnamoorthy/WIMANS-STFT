# End-to-End Experimentation Checklist

To maintain a rigorous "Big Lab" research environment, every single experiment MUST strictly follow this end-to-end process. This ensures full reproducibility, traceability, and immediate translation of EDA insights into Phase 2 machine learning architectures.

## Pre-Experiment
- [ ] **Hypothesis Formulation**: Define a clear objective and hypothesis for the experiment.
- [ ] **Script Creation**: Write a modular, reusable Python script in the `scripts/` directory that leverages `experiment_utils.setup_experiment()`.
- [ ] **Data Loading**: Ensure the script pulls the correct data subset (e.g., filtering `annotation.csv` for specific users or WiFi bands).

## Execution
- [ ] **Run Script**: Execute the experiment using the designated conda environment: `conda run -n wimans python scripts/<script_name>.py`.
- [ ] **Output Verification**: Verify that the script successfully generated plots and saved them in a timestamped folder inside `outputs/`.
- [ ] **Finalize**: Ensure `experiment_utils.finalize_experiment()` was called to log the JSON metrics and generate the local `report.md`.

## Post-Experiment Documentation (Mandatory Updates)
- [ ] **Task Tracker** (`task.md`): Mark the experiment as `[x]` Completed.
- [ ] **Decision Log** (`decision_log.md`): Log the Hypothesis, the Tradeoff Addressed, and the final Outcome/Insight.
- [ ] **Model Architecture Ideas** (`possible_model_ideas.md`): Explicitly write down how the results of this experiment will influence the structural design of the Phase 2 Neural Networks (e.g., "Use 2D CNNs because correlation was found").
- [ ] **IEEE Research Paper** (`research_paper_EDA_WiMANS.md`): Add the formal Methodology, Results, and Implication sections. **CRITICAL**: Embed the generated visualization images and include a "How to Interpret the Figures (Layman's Terms)" block.
- [ ] **Walkthrough** (`walkthrough.md`): Append the visual results, the layman's interpretation, and the image embeds so the user has an easy-to-read summary of the EDA.
