# Feature Extraction from Images

## Overview

This repository contains the code and models for the **Feature Extraction from Images** machine learning challenge. The objective is to extract entity values such as weight, volume, voltage, and dimensions from product images using machine learning and Optical Character Recognition (OCR) techniques.

## Problem Statement

In this hackathon, the goal is to create a model that extracts important product information (like weight, volume, and dimensions) directly from images. These details are critical for digital stores and marketplaces, where many products lack textual descriptions.

## Repository Structure

```bash
├── dataset/                # Contains train.csv, test.csv, and sample files
├── src/                    # Contains utility scripts and sanity checkers
│   ├── utils.py            # Helper functions (including image downloading)
│   ├── constants.py        # Allowed units for the predictions
│   ├── sanity.py           # Sanity checker for output format
│   └── test.ipynb          # Sample code notebook for testing
├── models/                 # Directory for saving trained models
├── notebooks/              # Jupyter notebooks for experimentation
├── sample_code.py          # Sample script for prediction format
├── test_out.csv            # Output file with final predictions
├── README.md               # This readme file
└── requirements.txt        # Python dependencies
```
Setup Instructions
1. Environment Setup
Ensure you have Python 3.7+ installed. Install the required dependencies by running the following command:

bash
Copy code
pip install -r requirements.txt
The requirements.txt file includes libraries such as:

torch for deep learning models
torchvision for image processing
pytesseract for Optical Character Recognition (OCR)
pandas, scikit-learn, and tqdm for data handling and processing
2. Data
Place the following files in the dataset/ directory:

train.csv: The training data with image links and entity values.
test.csv: The test data without entity values (to be predicted).
sample_test_out.csv: Sample output file for reference.
3. Image Download
To download images from the provided URLs in train.csv and test.csv, use the download_images function from utils.py. Run the following command in a Python environment:
```
python
Copy code
from src.utils import download_images
import pandas as pd

train_data = pd.read_csv('dataset/train.csv')
download_images(train_data['image_link'], 'train_images/')
This will download all the images to the specified folder (train_images/).
```
4. Model Training
The provided solution extracts features using a pre-trained ResNet model and applies OCR for text extraction. To train the model, open the Jupyter notebook (notebooks/training.ipynb) and run the code cells step by step.

Alternatively, you can run the sample_code.py script, which generates a dummy output file for testing purposes:
```
bash
Copy code
python sample_code.py
```
5. Prediction
Once the model is trained, use it to generate predictions for the test dataset. The predictions will be saved in the required format.

Run the following command to make predictions:

```bash
Copy code
python predict.py
```
This will generate a test_out.csv file, which will be formatted according to the submission guidelines.

6. Sanity Check
Before submitting, ensure the output passes the sanity checker provided in src/sanity.py. Run the checker as follows:
```
bash
Copy code
python src/sanity.py --file test_out.csv
```
You should see a message like:
```bash
Copy code
Parsing successful for file: test_out.csv
```
7. Submission
Once the output passes the sanity checker, submit test_out.csv as your final prediction.

Evaluation Criteria
The solution will be evaluated based on the F1 score. The F1 score calculation involves comparing the predicted entity values against the ground truth in the test dataset.
```
F1 Score Calculation
True Positives (TP): OUT != "" and GT != "" and OUT == GT
False Positives (FP): OUT != "" and GT != "" and OUT != GT
False Positives (FP): OUT != "" and GT == ""
False Negatives (FN): OUT == "" and GT != ""
True Negatives (TN): OUT == "" and GT == ""
```
The F1 score is computed as:
```
bash
Copy code
F1 = 2 * (Precision * Recall) / (Precision + Recall)
Where:

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
```
Notes
Ensure that the predictions use the correct format: "x unit", where x is a float, and unit is one of the allowed units.
Handle images where no entity value is detected by returning an empty string ("").
Ensure that the output file has the same number of rows as the input test.csv.
Acknowledgements
Special thanks to the organizers of this challenge and the open-source community for providing valuable tools and resources used in this project.
