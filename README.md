# Payment Behaviour & Digital Payment Intelligence

**Student:** Anubhav Dash  
**UID:** A501132524011  
**Course:** Fundamentals of Machine Learning  
**Problem Statement:** 7 — Payment Behaviour & Digital Payment Intelligence

## Project overview

This project explores customer payment behaviour using customer-level unsupervised clustering. It compares segmentation based on payment-mode preferences alone with segmentation that combines payment preferences, spending, transaction frequency, cashback and EMI usage.

## Repository contents

- `Payment_Behaviour_Digital_Payment_Intelligence.ipynb` — executable notebook with analysis and model outputs
- `Payment_Behaviour_Digital_Payment_Intelligence.py` — standalone Python script
- `payment_transactions_synthetic.csv` — transaction-level input data
- `customer_payment_features.csv` — engineered customer-level features
- `cluster_profiles.csv` and `model_comparison.csv` — cluster summaries and model evaluations
- `outputs/` — cleaned data, selected model metrics, cluster-selection results and recommendations
- `plots/` — EDA, clustering and evaluation figures in PNG/JPG formats
- `Payment_Behaviour_Digital_Payment_Intelligence_Report.pdf` — detailed project report
- `Payment_Behaviour_Digital_Payment_Intelligence_Report.docx` — editable report
- `Payment_Behaviour_Digital_Payment_Intelligence_Presentation.pptx` — presentation
- `Payment_Behaviour_Digital_Payment_Intelligence_Code.pdf` — PDF export of code
- `requirements.txt` — Python dependencies

## How to run

1. Clone or download this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Open the notebook in Jupyter and run cells from top to bottom. Keep the input CSV in the notebook's expected relative path.

## Methods covered

- Exploratory data analysis and data-quality checks
- Duplicate and abnormal transaction handling
- Customer-level feature engineering
- Encoding categorical payment information and standardizing features
- K-Means clustering with Elbow-based cluster selection
- Hierarchical clustering comparison
- Silhouette Score and Davies–Bouldin Index
- PCA visualizations and cluster-profile interpretation
- Industry-oriented payment incentive and checkout recommendations

## Dataset and limitations

**The dataset is synthetic.** It is generated for educational demonstration and is not real banking, UPI, card-network or customer data. Findings are illustrative and should not be treated as evidence about real customers or financial institutions. Clustering metrics measure internal structure, not business impact or causal effects.

## Author

Anubhav Dash — A501132524011
