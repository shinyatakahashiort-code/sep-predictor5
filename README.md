# 👁️ Spherical Equivalent (SE) Prediction App

A multilingual web application for predicting spherical equivalent (SE) from ophthalmological examination data using multiple machine learning models with SE classification and batch processing capabilities.

眼科検査データから球面度数を予測する多言語対応Webアプリケーション（SE分類・一括処理対応）

## 🌟 Features

### Core Features
- **13 Machine Learning Models**: Compare predictions from Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost, SVR, KNN, and MLP
- **Multilingual Support**: Switch between English and Japanese (日本語)
- **Two Input Modes**: 
  - Single patient input for individual predictions
  - Batch processing via Excel file upload for multiple patients

### 🎯 SE Classification Meter
Automatic classification of predicted SE values:
- **< -10D**: High Myopia (Extreme) / 最強度近視 🔴
- **-10D to -6D**: High Myopia / 強度近視 🟠
- **-6D to -3D**: Moderate Myopia / 中等度近視 🟡
- **-3D to -0.5D**: Mild Myopia / 弱度近視 🟢
- **-0.5D to 0.5D**: Emmetropia / 正視 ✅
- **> 0.5D**: Hyperopia / 遠視 🔵

### 📊 Batch Processing
- Upload Excel files with multiple patient records
- Process hundreds of patients at once
- Download results with predictions and classifications
- Visual analytics: distribution charts, pie charts, histograms
- Classification distribution analysis

### 📈 Visualization
- Interactive gauge meter for SE classification
- Bar charts comparing model predictions
- Distribution histograms for batch results
- Pie charts for classification breakdown
- Real-time performance metrics

## 🚀 Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/se-prediction-app.git
cd se-prediction-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Deploy to Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository
5. Set main file path to `app.py`
6. Click "Deploy"!

## 📊 Model Performance

| Model | Test R² | Test RMSE | CV RMSE |
|-------|---------|-----------|---------|
| **Gradient Boosting** | 0.9640 | 0.7369 | 0.9030 |
| Extra Trees | 0.9609 | 0.7674 | 0.9017 |
| MLP | 0.9601 | 0.7753 | 1.4036 |
| XGBoost | 0.9562 | 0.8123 | 0.9912 |
| Ridge | 0.9547 | 0.8261 | 0.9182 |

## 🔧 Training Your Own Models

If you want to train models with your own data:

```python
# 1. Prepare your data (CSV with required columns)
import pandas as pd
sep_df_cleaned = pd.read_csv('your_data.csv')

# Required columns: 年齢, 性別, K（AVG）, AL, LT, ACD, SE_p

# 2. Run the training script
python train_and_save_models.py

# 3. Models will be saved in the models/ directory
```

## 📁 Project Structure

```
se-prediction-app/
├── app.py                      # Main Streamlit application
├── train_and_save_models.py    # Script to train and save models
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── models/                     # Directory for saved models (optional)
│   ├── scaler.pkl
│   ├── Gradient_Boosting.pkl
│   ├── Extra_Trees.pkl
│   └── ...
└── .gitignore                  # Git ignore file
```

## 🎯 Input Parameters

### Required Input Features

- **Age (年齢)**: Patient age (0-100 years)
- **Sex (性別)**: Patient sex (Male/Female)
- **K (AVG)**: Average Keratometry (40.0-50.0)
- **AL**: Axial Length (眼軸長, 20.0-30.0 mm)
- **LT**: Lens Thickness (水晶体厚, 3.0-6.0 mm)
- **ACD**: Anterior Chamber Depth (前房深度, 2.0-4.0 mm)

## 🌐 Language Support

The app supports:
- **English** - Full interface in English
- **日本語** - 完全日本語対応

Switch languages using the sidebar dropdown.

## 📈 Usage Examples

### Example 1: Single Patient Prediction
1. Select language (English/日本語)
2. Choose "Single Patient" mode
3. Enter patient data in the sidebar:
   - Age: 30 years
   - Sex: Male
   - K (AVG): 43.5
   - AL: 24.0 mm
   - LT: 4.5 mm
   - ACD: 3.0 mm
4. Click "Predict" button
5. View results:
   - **SE Classification Meter**: Visual gauge showing classification
   - **Top 3 Models**: Quick comparison of best models
   - **All Model Predictions**: Detailed table with all results
   - **Comparison Chart**: Bar chart comparing predictions

### Example 2: Batch Processing
1. Select "Batch (Excel)" mode
2. Download the template Excel file
3. Fill in patient data (required columns: 年齢, 性別, K（AVG）, AL, LT, ACD)
4. Upload the completed Excel file
5. Click "Process Batch"
6. View results:
   - Summary statistics (mean, median, std dev, range)
   - Classification distribution pie chart
   - Detailed results table
   - SE distribution histogram
7. Download results Excel file with all predictions

### Example 3: Model Comparison
1. Uncheck "Show All Models"
2. Select specific models to compare (e.g., Gradient Boosting, XGBoost, Extra Trees)
3. Run prediction (single or batch)
4. Analyze differences between selected models

## 📋 Excel File Format

### Required Columns
Your Excel file must contain these columns (exact names):

| Column | Description | Example Values |
|--------|-------------|----------------|
| 年齢 | Age in years | 30, 45, 25 |
| 性別 | Sex (0=Male, 1=Female) | 0, 1 |
| K（AVG） | Average Keratometry | 43.5, 44.0 |
| AL | Axial Length (mm) | 24.0, 25.5 |
| LT | Lens Thickness (mm) | 4.5, 4.8 |
| ACD | Anterior Chamber Depth (mm) | 3.0, 3.2 |

### Optional Columns
- Patient ID / 患者ID: Will be auto-generated if not provided

### Sample Excel Data
```
年齢  性別  K（AVG）   AL    LT   ACD
30    0     43.5    24.0  4.5  3.0
45    1     44.0    25.5  4.8  3.2
25    0     43.0    23.5  4.3  2.9
```

Download the template from the app for a properly formatted example.

## 🛠️ Technical Details

- **Framework**: Streamlit 1.31.0
- **ML Libraries**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Data Processing**: pandas, numpy
- **Visualization**: plotly, matplotlib, seaborn
- **Excel Processing**: openpyxl
- **Cross-Validation**: 5-Fold CV
- **Random Seed**: 2025 (for reproducibility)

## 📝 Model Descriptions

### Top Performing Models

1. **Gradient Boosting** (Best Overall) ⭐
   - Test R²: 0.9640
   - Test RMSE: 0.7369
   - CV RMSE: 0.9030
   - Balanced performance and stability
   - Recommended for production use

2. **Extra Trees**
   - Test R²: 0.9609
   - Test RMSE: 0.7674
   - CV RMSE: 0.9017
   - Fast predictions
   - Good generalization

3. **MLP (Neural Network)**
   - Test R²: 0.9601
   - Test RMSE: 0.7753
   - CV RMSE: 1.4036
   - Complex patterns recognition
   - Higher variance in CV

## 🎨 SE Classification System

The app uses a color-coded classification system based on clinical standards:

| SE Range (D) | Classification (EN) | Classification (JA) | Color | Emoji |
|-------------|---------------------|---------------------|-------|-------|
| < -10 | High Myopia (Extreme) | 最強度近視 | Dark Red | 🔴 |
| -10 to -6 | High Myopia | 強度近視 | Crimson | 🟠 |
| -6 to -3 | Moderate Myopia | 中等度近視 | Orange | 🟡 |
| -3 to -0.5 | Mild Myopia | 弱度近視 | Gold | 🟢 |
| -0.5 to 0.5 | Emmetropia | 正視 | Green | ✅ |
| > 0.5 | Hyperopia | 遠視 | Blue | 🔵 |

## 📊 Batch Processing Outputs

When processing batch data, the app provides:

1. **Excel Results File** containing:
   - Original input data
   - SE predictions from all selected models
   - Classification for each prediction
   - Patient IDs (auto-generated if not provided)

2. **Statistical Summary**:
   - Mean, median, standard deviation
   - Range (min to max)
   - Count by classification category

3. **Visual Analytics**:
   - Distribution histogram with classification boundaries
   - Pie chart of classification breakdown
   - Interactive plots for exploration

## 💾 Data Privacy

- All data processing occurs in-memory
- No data is stored on servers
- Uploaded files are not saved permanently
- Results are only available during the session

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Built with Streamlit
- Machine learning models from scikit-learn, XGBoost, LightGBM, and CatBoost
- Ophthalmological data analysis

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This app is for educational and research purposes only. Always consult with qualified healthcare professionals for medical decisions.

**注意**: このアプリは教育・研究目的のみです。医療上の判断については必ず専門医にご相談ください。
