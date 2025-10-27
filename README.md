# DA5401 Assignment 7 - Model Selection for Landsat Classification

## File Structure
```
├── assn_7.ipynb # Main analysis notebook
├── README.md    # This file
└── data/
    ├── sat.trn  # Landsat training data
    └── sat.tst  # Landsat test data
```

## Notebook Structure
### Imports & Configuration
- Import required libraries (scikit-learn, pandas, numpy, matplotlib, seaborn, xgboost)
- Set up plotting configurations and random seeds for reproducibility

### Data Loading & Preprocessing
- Load UCI Landsat Satellite dataset from local files
- Explore dataset structure and class distributions
- Filter out class 6 (mixture class) as specified
- Perform stratified train-test split and feature standardization

### Model Implementation & Training
Train six different classifiers:
- **K-Nearest Neighbors** (n_neighbors=5)
- **Decision Tree** (max_depth=10)
- **Dummy Classifier** (strategy='prior')
- **Logistic Regression** (multi_class='ovr')
- **Gaussian Naive Bayes**
- **Support Vector Machine** (probability=True, kernel='rbf')

### Baseline Performance Evaluation
- Calculate accuracy, weighted F1-score, and training times
- Generate comparative visualizations for initial model assessment

### Multi-Class ROC Analysis
- Implement One-vs-Rest (OvR) approach for multi-class ROC curves
- Compute macro and weighted AUC scores
- Create comprehensive ROC visualizations and per-class analysis

### Precision-Recall Curve Analysis
- Compute PRC metrics using OvR methodology
- Generate Average Precision (AP) scores for each model
- Analyze performance trade-offs, especially for imbalanced classes

### Comprehensive Model Comparison
- Ranking analysis across multiple metrics (Accuracy, F1, ROC-AUC, PRC-AP)
- Correlation analysis between different evaluation approaches
- Performance vs efficiency trade-off analysis

### Synthesis & Recommendations
- Detailed ranking comparison and correlation analysis
- Model-specific trade-off explanations
- Final recommendation with quantitative justification

### Brownie Points - Additional Experiments
- **RandomForest Classifier** (ensemble method evaluation)
- **XGBoost Classifier** (gradient boosting performance)
- **Inverted Classifier** (custom model designed to achieve AUC < 0.5)
- Comprehensive comparison including additional models


## Input Data
The analysis uses the **UCI Landsat Satellite dataset**, a multi-class classification benchmark for remote sensing and land cover analysis. This dataset contains multi-spectral satellite image data with 36 features representing different spectral bands. The dataset is split into training (`sat.trn`) and test (`sat.tst`) files, which are combined for our own stratified splitting approach.

### Citations
Blake, C. and Merz, C.J. (1998). UCI Repository of machine learning
databases. Irvine, CA: University of California, Department of Information and
Computer Science.

## How to Use
1. Open `assn_7.ipynb` for complete model selection analysis with results
2. Ensure dataset files (`sat.trn`, `sat.tst`) are placed in the `data/` directory as shown above
3. Run cells sequentially for comprehensive model comparison

## Requirements
- Python 3.x  
- pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, scipy

Install dependencies with:  
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost scipy
```

## Author

*Jerry Jose (BE22B022)*  
*IITM Data Analytics Lab, Semester 7*