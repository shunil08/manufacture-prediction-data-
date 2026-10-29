# 🏭 Manufacturing Defect Detection Project

## 📋 Project Overview

**Kaggle Competition:** [Product Defect Detection](https://www.kaggle.com/competitions/product-defect-detection)  
**Problem:** Predict manufacturing defects using sensor data and production parameters  
**Approach:** Machine Learning classification with multiple algorithms  
**Dataset:** Tabular manufacturing data with sensor readings and production metrics from Kaggle competition
 
**Problem:** Predict manufacturing defects using sensor data and production parameters  
**Approach:** Machine Learning classification with multiple algorithms  
**Dataset:** Tabular manufacturing data with sensor readings and production metrics  

## 🎯 What This Project Does

This project addresses the **Kaggle Product Defect Detection Competition**, which challenges participants to build machine learning models that can predict manufacturing defects based on sensor data and production parameters. The goal is to improve quality control in manufacturing processes by identifying potential defects before they reach customers.

This project analyzes manufacturing data to predict whether a product will be defective based on various production parameters including:
- **Production Line** - Manufacturing line identifier
- **Temperature** - Operating temperature during production
- **Pressure** - System pressure readings
- **Assembly Time** - Time taken for assembly
- **Sensor Readings** - Readings from 3 different sensors
- **Operator Experience** - Experience level of the operator

## 📊 Dataset Information

**Source:** [Kaggle Product Defect Detection Competition](https://www.kaggle.com/competitions/product-defect-detection)

- **Training Data:** `train_data.csv` (6,700 samples, 10 features)
- **Test Data:** `test_data.csv` (3,300 samples, 9 features)
- **Sample Submission:** `sample_submission.csv` (3,300 predictions to fill)
- **Target Variable:** `Defective` (0 = Good, 1 = Defect)

### Dataset Features
The competition dataset includes manufacturing sensor data and production parameters that are commonly used in quality control systems.

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Project
1. **Open** `manufacturing_defect_detection.ipynb` in Jupyter Notebook
2. **Run** all cells sequentially from top to bottom
3. **Wait** for model training and evaluation
4. **Check** the generated `my_submission.csv` file

## 🤖 Machine Learning Models

### 1. Support Vector Machine (SVM)
- **Kernel:** RBF (Radial Basis Function)
- **Features:** Scaled using StandardScaler
- **Purpose:** Baseline classification model

### 2. Random Forest
- **Estimators:** 100 decision trees
- **Features:** Raw feature values
- **Purpose:** Primary model with feature importance analysis

## 📈 What You'll Get

### Visualizations
- **Defect Distribution** - Target variable balance
- **Model Comparison** - Accuracy and ROC AUC charts
- **Feature Importance** - Which factors matter most
- **Confusion Matrices** - Model performance details

### Results
- **Model Performance Metrics** (Accuracy, ROC AUC)
- **Feature Importance Rankings**
- **Predictions for Test Data**
- **Submission File** (`my_submission.csv`)

## 🔧 Technical Details

### Data Preprocessing
- **Missing Values:** Handled with median imputation
- **Categorical Variables:** Encoded using LabelEncoder
- **Feature Scaling:** Applied for SVM model

### Model Training
- **Data Split:** 80% training, 20% validation
- **Cross-validation:** Stratified sampling for balanced classes
- **Hyperparameters:** Optimized for manufacturing data

## 📁 Project Structure

```
22BSM023 LAB-3/
├── manufacturing_defect_detection.ipynb  # Main notebook
├── train_data.csv                        # Training dataset
├── test_data.csv                         # Test dataset
├── sample_submission.csv                 # Submission template
├── README.md                             # This file
└── my_submission.csv                     # Generated predictions
```

## ⏱️ Expected Runtime

- **Total Time:** 15-25 minutes
- **Data Loading:** 2-3 minutes
- **Preprocessing:** 1-2 minutes
- **SVM Training:** 3-5 minutes
- **Random Forest:** 5-10 minutes
- **Evaluation & Visualization:** 2-3 minutes
- **Final Predictions:** 1-2 minutes

## 🎉 Key Features

✅ **Complete ML Pipeline** - From data loading to predictions  
✅ **Multiple Algorithms** - SVM and Random Forest comparison  
✅ **Beautiful Visualizations** - Charts, heatmaps, and comparisons  
✅ **Feature Analysis** - Understanding what drives defects  
✅ **Production Ready** - Generates submission file automatically  
✅ **Beginner Friendly** - Well-commented and modular code  

## 🔍 Understanding the Results

### Model Performance
- **Accuracy:** Percentage of correct predictions
- **ROC AUC:** Model's ability to distinguish between classes
- **Higher values = Better performance**

### Competition Context
This project was developed for the Kaggle Product Defect Detection competition, where the goal is to achieve the highest possible prediction accuracy on the test dataset. The models are evaluated on their ability to correctly classify products as defective or non-defective based on manufacturing sensor data.

### Feature Importance
- **Shows which factors most influence defect detection**
- **Helps identify critical production parameters**
- **Useful for process optimization**

## 🚨 Troubleshooting

### Common Issues
1. **Missing Libraries:** Install required packages using pip
2. **File Paths:** Ensure CSV files are in the same directory
3. **Memory Issues:** Close other applications if running slowly

### If Something Goes Wrong
- Check that all CSV files are present
- Verify Python environment has required packages
- Restart Jupyter kernel if needed

## 📚 Learning Outcomes

This project demonstrates:
- **Data Preprocessing** for manufacturing datasets
- **Feature Engineering** and selection
- **Model Training** and evaluation
- **Performance Comparison** between algorithms
- **Result Visualization** and interpretation
- **Production Deployment** with submission generation

## 🎯 Perfect for Assignment Submission

- **Professional Quality** code and documentation
- **Complete Analysis** with visualizations
- **Working Models** that generate predictions
- **Clear Results** and performance metrics
- **Ready to Present** to instructors

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Verify all files are in the correct location
3. Ensure Python environment is properly set up

---

**🎉 Happy Manufacturing Defect Detection! 🏭✨**

*This project is designed for educational purposes and demonstrates practical machine learning applications in manufacturing quality control.*
