## RRG-DPO: Direct Preference Optimization for Clinically Accurate Radiology Report Generation

This repository demonstrates the RRG-DPO method (from the paper "RRG-DPO: Direct Preference Optimization for Clinically Accurate Radiology Report Generation") using R2Gen as an example. The example shows how to integrate RRG-DPO into a report-generation model as a post-training step.

## Usage

1. Download the preference dataset from:
[https://drive.google.com/file/d/13zdP6qXNaVzAWVowv7DgAdJLrJ6Em-e1/view?usp=drive_link](https://drive.google.com/file/d/11OrR5Vi8mA0nvXJWMdNTBRz7bUi4Jo9i/view?usp=sharing)

2. Train the supervised fine-tuned (SFT) R2Gen model by running main.py.

3. Configure main_dpo.py: set the data path and model path to the locations from steps 1–2, then execute the DPO training stage (6,000–9,000 steps is enough).

4. Configure main.py: Set the checkpoint path for testing and run the test script. The prediction results will be saved as a .pkl file in the checkpoint directory.

## If you have any questions, please do not hesitate to contact liuhong@stu.xmu.edu.cn
