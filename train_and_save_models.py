"""
このスクリプトは全てのモデルを訓練して保存します
Train and save all models for the Streamlit app
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

SEED = 2025
np.random.seed(SEED)

# ==========================================
# データ準備（sep_df_cleanedを読み込む必要があります）
# ==========================================
# 注: 実際のデータを読み込んでください
# 例: sep_df_cleaned = pd.read_csv('your_data.csv')

print("=" * 80)
print("モデル訓練スクリプト - Model Training Script")
print("=" * 80)

# データが存在する場合のみ実行
try:
    feature_cols = ['年齢', '性別', 'K（AVG）', 'AL', 'LT', 'ACD']
    X = sep_df_cleaned[feature_cols]
    y = sep_df_cleaned['SE_p']
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    
    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # modelsディレクトリを作成
    os.makedirs('models', exist_ok=True)
    
    # スケーラーを保存
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ Scaler saved")
    
    # ==========================================
    # 全モデルを定義
    # ==========================================
    models_to_train = {
        # 基本モデル（スケーリング不要）
        'Linear_Regression': (LinearRegression(), False),
        'Ridge': (Ridge(alpha=1.0), True),
        'Lasso': (Lasso(alpha=0.1), True),
        'ElasticNet': (ElasticNet(alpha=0.1, l1_ratio=0.5), True),
        
        # ツリーベース（スケーリング不要）
        'Random_Forest': (RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1), False),
        'Extra_Trees': (ExtraTreesRegressor(n_estimators=100, random_state=SEED, n_jobs=-1), False),
        'Gradient_Boosting': (GradientBoostingRegressor(n_estimators=100, random_state=SEED), False),
        
        # 勾配ブースティング（スケーリング不要）
        'XGBoost': (xgb.XGBRegressor(n_estimators=100, random_state=SEED, n_jobs=-1), False),
        'LightGBM': (lgb.LGBMRegressor(n_estimators=100, random_state=SEED, n_jobs=-1, verbose=-1), False),
        'CatBoost': (CatBoostRegressor(iterations=100, random_state=SEED, verbose=False), False),
        
        # その他（スケーリング必要）
        'SVR': (SVR(kernel='rbf'), True),
        'KNN': (KNeighborsRegressor(n_neighbors=5), True),
        'MLP': (MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=SEED), True),
    }
    
    # ==========================================
    # 全モデルを訓練して保存
    # ==========================================
    print("\n訓練開始 - Starting training...")
    print("-" * 80)
    
    for name, (model, needs_scaling) in models_to_train.items():
        try:
            print(f"Training {name}...", end=" ")
            
            # 訓練
            if needs_scaling:
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train, y_train)
            
            # 保存
            model_path = f'models/{name}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            print("✓ Saved")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("\n" + "=" * 80)
    print("全モデルの訓練と保存が完了しました")
    print("All models trained and saved successfully!")
    print("=" * 80)
    print("\nSaved files:")
    print("- models/scaler.pkl")
    for name in models_to_train.keys():
        print(f"- models/{name}.pkl")
    
except NameError:
    print("\n" + "=" * 80)
    print("⚠️ データが見つかりません - Data not found")
    print("=" * 80)
    print("""
使用方法 - Usage:

1. データを読み込む - Load your data:
   sep_df_cleaned = pd.read_csv('your_data.csv')

2. このスクリプトを実行 - Run this script:
   python train_and_save_models.py

3. モデルが models/ ディレクトリに保存されます
   Models will be saved in the models/ directory

注意 - Note:
データには以下のカラムが必要です:
Required columns: 年齢, 性別, K（AVG）, AL, LT, ACD, SE_p
    """)
