## RRG-DPO: Direct Preference Optimization for Clinically Accurate Radiology Report Generation

This repository demonstrates the RRG-DPO method (from the paper "RRG-DPO: Direct Preference Optimization for Clinically Accurate Radiology Report Generation") using R2Gen as an example. The example shows how to integrate RRG-DPO into a report-generation model as a post-training step.

## Usage

Download the preference dataset from:
https://drive.google.com/file/d/13zdP6qXNaVzAWVowv7DgAdJLrJ6Em-e1/view?usp=drive_link

Train the supervised fine-tuned (SFT) R2Gen model by running main.py.

Configure main_dpo.py: set the data path and model path to the locations from steps 1–2, then execute the DPO training stage.

## If you have any questions, please do not hesitate to contact liuhong@stu.xmu.edu.cn
