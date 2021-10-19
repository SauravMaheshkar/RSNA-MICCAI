My codebase and model files for the "[RSNA-MICCAI Brain Tumor Radiogenomic Classification](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/overview)" Kaggle Competition.

[Link to the Kernel to Reproduce Experiments](https://www.kaggle.com/sauravmaheshkar/rsna-miccai-the-random-seed-fluke/) | [Weights and Biases Report ⭐️](https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/The-Fluke--VmlldzoxMDA2MDQy) | [Weights and Biases Project](https://wandb.ai/sauravmaheshkar/RSNA-MICCAI) |
# Key Takeaways

## [Changing Random Seeds can get you from 0.540 to 0.710 in no time](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/271214)


For an in-depth comparison of the various models such as b0 vs b1 vs b2 or b0 with different seeds, head over to the [**accompanying wandb report**](https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/The-Fluke--VmlldzoxMDA2MDQy).

The motivation for these experiments come from [Chai Time Kaggle Talks with Anjum Sayed (Datasaurus)](https://youtu.be/udw-uSV66EQ) Video on the [Weights and Biases Channel](https://www.youtube.com/WeightsBiases). Anjum mentioned that a good way to check if the models are learning anything is to just change the random seeds and see if it affects the performance.

## Models Don't Learn 🤷🏻

![](https://raw.githubusercontent.com/SauravMaheshkar/RSNA-MICCAI/main/assets/Fluke-Training-Loss.svg)

![](https://raw.githubusercontent.com/SauravMaheshkar/RSNA-MICCAI/main/assets/Fluke-Validation-Loss.svg)

Naming Convention - `arch-seed`

|**Name**                           |**Training Loss**|**Validation Loss**|**EPOCHS**|**BATCH_SIZE**|
|-------------------------------|-------------|---------------|:------:|:----------:|
|baseline-efficientnet3d-b0-42  |0.6814|0.5914|10    |4         |64        |
|baseline-efficientnet3d-b0-12  |0.6913|0.6420|10    |4         |64        |
|augment-efficientnet3d-b2-42   |0.7125|5.6498|10    |4         |64        |
|baseline-efficientnet3d-b0-2021|0.7293|0.6019|10    |4         |64        |
|baseline-efficientnet3d-b2-12  |0.7431|0.8652|10    |4         |64        |
|baseline-efficientnet3d-b1-12  |0.7461|0.5110|10    |4         |64        |
|baseline-efficientnet3d-b0-21  |0.7643|0.7170|10    |4         |64        |
