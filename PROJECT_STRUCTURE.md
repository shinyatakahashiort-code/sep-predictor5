# 📁 プロジェクト構造 / Project Structure

## ファイル一覧 / File List

```
se-prediction-app/
├── 📄 app.py                           # メインアプリケーション / Main application
├── 📄 requirements.txt                 # Python依存パッケージ / Python dependencies
├── 📄 README.md                        # プロジェクト説明（英語） / Project README (English)
├── 📄 USER_GUIDE_JA.md                 # 使用ガイド（日本語） / User guide (Japanese)
├── 📄 DEPLOYMENT.md                    # デプロイ手順 / Deployment guide
├── 📄 PROJECT_STRUCTURE.md             # このファイル / This file
├── 📄 train_and_save_models.py         # モデル訓練スクリプト / Model training script
├── 📄 se_prediction_template.xlsx      # Excelテンプレート / Excel template
├── 📄 .gitignore                       # Git除外設定 / Git ignore file
├── 📁 .streamlit/                      # Streamlit設定 / Streamlit config
│   └── config.toml                     # アプリ設定 / App configuration
└── 📁 models/ (optional)               # 訓練済みモデル / Trained models
    ├── scaler.pkl
    ├── Gradient_Boosting.pkl
    ├── Extra_Trees.pkl
    ├── XGBoost.pkl
    └── ...
```

## ファイル説明 / File Descriptions

### 📄 app.py
**用途 / Purpose**: メインのStreamlitアプリケーション

**機能 / Features**:
- 多言語対応（日本語・英語）
- 単一患者予測
- Excel一括処理
- SE分類メーター
- 13種類の機械学習モデル
- インタラクティブな可視化

**重要な関数 / Key Functions**:
- `classify_se()`: SE値を分類
- `create_se_gauge()`: ゲージチャート作成
- `load_models()`: モデル読み込み

### 📄 requirements.txt
**用途 / Purpose**: Python依存パッケージのリスト

**主要パッケージ / Main Packages**:
- streamlit: Webアプリフレームワーク
- scikit-learn: 基本的な機械学習
- xgboost, lightgbm, catboost: 勾配ブースティング
- plotly: インタラクティブグラフ
- openpyxl: Excel処理

### 📄 train_and_save_models.py
**用途 / Purpose**: モデルを訓練して保存するスクリプト

**使用方法 / Usage**:
```bash
python train_and_save_models.py
```

**出力 / Output**:
- models/ディレクトリに.pklファイルを保存
- 全13モデルの訓練済みバージョン

### 📄 se_prediction_template.xlsx
**用途 / Purpose**: 一括処理用のExcelテンプレート

**含まれる内容 / Contents**:
- サンプルデータ（6人分）
- 必須カラムの例
- 正しいフォーマット

**必須カラム / Required Columns**:
- 年齢, 性別, K（AVG）, AL, LT, ACD

### 📄 README.md
**用途 / Purpose**: プロジェクトの総合説明書（英語）

**内容 / Contents**:
- 機能紹介
- インストール方法
- デプロイ手順
- モデル性能データ
- 使用例

### 📄 USER_GUIDE_JA.md
**用途 / Purpose**: 日本語の詳細な使用ガイド

**内容 / Contents**:
- 機能説明
- ステップバイステップの使用方法
- 結果の見方
- トラブルシューティング

### 📄 DEPLOYMENT.md
**用途 / Purpose**: デプロイメント手順書

**内容 / Contents**:
- GitHub設定
- Streamlit Cloudデプロイ
- カスタムドメイン設定
- トラブルシューティング

### 📄 .gitignore
**用途 / Purpose**: Gitで管理しないファイルの設定

**除外ファイル / Excluded Files**:
- __pycache__/
- *.pyc
- venv/
- .DS_Store
- データファイル（オプション）

### 📁 .streamlit/config.toml
**用途 / Purpose**: Streamlitアプリの設定

**設定項目 / Settings**:
- テーマカラー
- サーバー設定
- ブラウザ設定

### 📁 models/ (オプション)
**用途 / Purpose**: 訓練済みモデルの保存

**ファイル / Files**:
- scaler.pkl: データ標準化スケーラー
- [Model_Name].pkl: 各モデルの訓練済みバージョン

**注意 / Note**: 
- モデルファイルは大きいのでGit LFSの使用を推奨
- または、アプリ起動時に自動初期化

## 🔧 開発フロー / Development Flow

### 初期セットアップ / Initial Setup
```bash
# 1. リポジトリをクローン
git clone https://github.com/yourusername/se-prediction-app.git
cd se-prediction-app

# 2. 仮想環境を作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 依存パッケージをインストール
pip install -r requirements.txt

# 4. アプリを起動
streamlit run app.py
```

### モデル訓練 / Model Training
```bash
# データを用意
# sep_df_cleaned = pd.read_csv('your_data.csv')

# モデルを訓練
python train_and_save_models.py

# modelsディレクトリにモデルが保存される
```

### デプロイ / Deployment
```bash
# GitHubにプッシュ
git add .
git commit -m "Update app"
git push origin main

# Streamlit Cloudで自動デプロイ
```

## 📊 データフロー / Data Flow

### 単一患者モード
```
入力 (Sidebar) 
  ↓
pandas DataFrame作成
  ↓
全モデルで予測
  ↓
分類 (classify_se)
  ↓
結果表示 (ゲージ、テーブル、チャート)
```

### 一括処理モード
```
Excelファイルアップロード
  ↓
pandas DataFrameに変換
  ↓
バリデーション (必須カラムチェック)
  ↓
全行・全モデルで予測
  ↓
分類を各行に追加
  ↓
統計分析・可視化
  ↓
結果Excelファイル作成
  ↓
ダウンロード
```

## 🎯 カスタマイズポイント / Customization Points

### モデルの追加
`app.py`の`load_models()`関数に新しいモデルを追加:
```python
models = {
    ...
    'Your Model': YourModelClass(params),
}
```

### 分類基準の変更
`classify_se()`関数の条件を変更:
```python
def classify_se(se_value):
    if se_value < YOUR_THRESHOLD:
        return {...}
```

### 言語の追加
`TRANSLATIONS`辞書に新しい言語を追加:
```python
TRANSLATIONS = {
    'en': {...},
    'ja': {...},
    'fr': {...},  # 新しい言語
}
```

### UIのカスタマイズ
`.streamlit/config.toml`でテーマを変更:
```toml
[theme]
primaryColor = "#YOUR_COLOR"
backgroundColor = "#YOUR_COLOR"
```

## 🔐 セキュリティ / Security

### データプライバシー
- すべての処理はインメモリ
- ファイルは永続保存されない
- セッション終了後に自動削除

### 推奨事項 / Recommendations
- 個人情報は含めない
- 患者IDは匿名化
- HTTPSを使用（Streamlit Cloudは自動対応）

## 📈 パフォーマンス最適化 / Performance Optimization

### キャッシング
`@st.cache_resource`を使用してモデル読み込みを高速化

### バッチ処理
- 大量データ（1000+患者）の場合は分割処理を推奨
- プログレスバーで進捗表示

### メモリ管理
- 不要な変数は削除
- 大きなDataFrameはチャンク処理

## 🐛 デバッグ / Debugging

### ログ出力
```python
st.write("Debug info:", variable)
```

### エラー表示
```python
st.error("Error message")
st.warning("Warning message")
st.info("Info message")
```

### ローカルテスト
```bash
streamlit run app.py --server.port 8501
```

---

**作成日 / Created**: 2025-10-23
**バージョン / Version**: 2.0.0
**ライセンス / License**: MIT
