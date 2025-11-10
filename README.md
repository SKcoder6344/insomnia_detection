# AI-Powered Early Stage Insomnia Detection

## Overview

This project presents an AI-powered system for identifying early signs of insomnia using physiological and behavioral data. The model analyzes features such as sleep duration, heart rate variability, and stress indicators to classify insomnia risk levels. Implemented in Python using machine learning algorithms, the system achieves high accuracy in distinguishing between normal and insomnia-affected subjects.

## Table of Contents

- [Abstract](#abstract)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Scope](#future-scope)
- [References](#references)
  

## Abstract

Insomnia is a growing health concern affecting both physical and cognitive well-being. Early detection can prevent chronic sleep disorders and improve quality of life. This study demonstrates how AI-based detection methods can assist in early intervention, reduce diagnostic costs, and support personalized healthcare.

**Key Features Analyzed:**
- Sleep Duration
- Heart Rate Variability
- Stress Indicators
- Quality of Sleep
- Physical Activity Level
- Daily Steps

## Features

- **Automated Detection**: AI-powered classification of insomnia risk levels
- **High Accuracy**: Random Forest Classifier achieving 95.6% accuracy
- **Multi-feature Analysis**: Comprehensive evaluation of physiological and behavioral parameters
- **Early Intervention**: Identifies early-stage insomnia before symptoms escalate
- **Visualizations**: Includes confusion matrices, accuracy plots, and SHAP feature importance

## Dataset

**Source**: [Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset/)

The dataset contains sleep-related metrics including:
- Sleep Duration
- Quality of Sleep
- Physical Activity Level
- Stress Level
- BMI Category
- Blood Pressure
- Heart Rate
- Daily Steps
- Sleep Disorder indicators

## Installation

### Prerequisites

- Python 3.7+
- Google Colab (recommended) or local Jupyter environment

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow shap kaggle
```

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/insomnia-detection.git
cd insomnia-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Kaggle API credentials:
   - Download your `kaggle.json` file from Kaggle account settings
   - Place it in the appropriate directory (`~/.kaggle/` on Linux/Mac)

4. Download the dataset:
```bash
kaggle datasets download -d uom190346a/sleep-health-and-lifestyle-dataset
unzip sleep-health-and-lifestyle-dataset.zip -d sleep_dataset
```

## Usage

### Running the Notebook

1. Open the Jupyter notebook:
```bash
jupyter notebook notebooks/INT422PROJECT.ipynb
```

2. Execute cells sequentially to:
   - Load and preprocess the data
   - Create insomnia risk labels
   - Train machine learning models
   - Evaluate model performance
   - Visualize results

### Google Colab

Access the project directly in Google Colab:
[Colab Link](https://colab.research.google.com/drive/1w9LivjdhriG4E2WpY05M2LDK2MC5EoHj?usp=sharing)

## Model Architecture

### Data Processing Pipeline

1. **Data Collection**: Import sleep-related datasets
2. **Preprocessing**:
   - Handle missing values using mean imputation
   - Normalize numeric data
   - Encode categorical variables using label encoding
3. **Feature Selection**:
   - Correlation analysis
   - Statistical feature importance
4. **Model Training**: Multiple ML algorithms compared
5. **Evaluation**: Performance metrics and validation

### Labeling Strategy

Risk classification based on:
- **High Risk (1)**:
  - Sleep Duration < 6.5 hours OR
  - Quality of Sleep ≤ 6 OR
  - Stress Level ≥ 7
- **Low Risk (0)**: Otherwise

### Machine Learning Models

Three supervised learning algorithms were implemented and compared:

1. **Random Forest Classifier** (Best Performance)
   - Accuracy: 95.6%
   - Precision: 0.96
   - Recall: 0.94
   - F1-Score: 0.95

2. **Decision Tree Classifier**
   - Accuracy: 91.2%

3. **Support Vector Machine (SVM)**
   - Accuracy: 89.7%

### Deep Learning Architecture

The neural network model includes:
- Input layer (features from dataset)
- Dense layer (128 neurons, ReLU activation)
- Dropout layer (0.3)
- Dense layer (64 neurons, ReLU activation)
- Dropout layer (0.2)
- Output layer (1 neuron, sigmoid activation)

**Training Configuration:**
- Optimizer: Adam (learning_rate=0.001)
- Loss: Binary Crossentropy
- Batch Size: 16
- Epochs: 80
- Early Stopping: patience=10

## Results

### Model Performance

The Random Forest Classifier demonstrated the best overall performance:

| Metric | Value |
|--------|-------|
| Accuracy | 95.6% |
| Precision | 0.96 |
| Recall | 0.94 |
| F1-Score | 0.95 |
| ROC-AUC | High (visualized in notebook) |

### Key Findings

**Most Significant Indicators of Insomnia:**
1. Sleep Duration
2. Stress Index
3. Heart Rate Variability

### Confusion Matrix Results

- **True Negatives**: 49 (Normal correctly identified)
- **True Positives**: 25 (Early Risk correctly identified)
- **False Positives**: 1 (Misclassified as Risk)
- **False Negatives**: 0 (No missed cases)

## Technologies Used

- **Python**: Core programming language
- **TensorFlow/Keras**: Deep learning framework
- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation
- **NumPy**: Numerical computations
- **matplotlib/seaborn**: Data visualization
- **SHAP**: Model interpretability
- **Google Colab**: Cloud-based development environment

## Future Scope

1. **Integration with Wearable Devices**: Real-time data from smartwatches and fitness bands for continuous monitoring

2. **Deep Learning Models**: Implementation of CNNs or RNNs for time-series analysis of physiological data

3. **Expanded Dataset**: Collection of diverse datasets from different demographics and lifestyles to improve model generalization

4. **Mobile and Cloud Deployment**: Development of mobile application with cloud-based processing for wider accessibility

5. **Mental Health Integration**: Combining insomnia detection with mood and stress tracking systems for holistic health insights

6. **Real-time Feedback**: Adding a feedback mechanism with lifestyle recommendations and sleep hygiene tips

## Project Structure

```
insomnia-detection/
├── notebooks/
│   └── INT422PROJECT.ipynb
├── data/
│   └── sleep_dataset/
│       └── Sleep_health_and_lifestyle_dataset.csv
├── models/
│   └── insomnia_model.h5
├── docs/
│   └── Research_Paper_Deep_Learning.pdf
├── README.md
└── requirements.txt
```

## References

1. Khalighi, A., Sousa, T., Nunes, J., & Moutinho, U. (2021). "Automatic sleep stage classification using EEG signals: A comprehensive review." *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, 29, 141–158.

2. Li, X., Yu, S., & Zhang, H. (2022). "Machine learning-based sleep disorder prediction using physiological data." *Biomedical Signal Processing and Control*, 69, 102–118.

3. Singh, P., & Sharma, N. (2021). "AI-driven healthcare: Sleep quality monitoring using wearable sensors." *International Journal of Computer Applications*, 183(32), 25–31.

4. Ma et al. (2024). Heart rate variability analysis during sleep onset for insomnia detection.

5. Aziz et al. (2025). Review on wearable AI systems for sleep disorder detection.

## Contributors

- Project Team: INT422 Project Group
- Members : Sujal Kumar Nayak, Sajal Jain
- Institution: [Lovely Professional University]
- Course: INT422


## Acknowledgments

- Dataset provided by [Kaggle - UOM190346A](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset/)
- Research guidance from course instructors
- Inspiration from recent advances in AI-based healthcare diagnostics

## Contact

For questions or collaborations, please contact: [nayaksujalkumar@gmail.com]

---

**Note**: This project is for educational and research purposes. It should not be used as a substitute for professional medical diagnosis or treatment.
