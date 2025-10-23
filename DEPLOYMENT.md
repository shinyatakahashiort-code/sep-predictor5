# 🚀 デプロイメントガイド / Deployment Guide

## 📋 準備 / Preparation

### 必要なもの / Requirements
- GitHubアカウント / GitHub account
- Streamlit Community Cloudアカウント（無料）/ Streamlit Community Cloud account (free)

## 🔧 セットアップ手順 / Setup Steps

### 1. GitHubリポジトリの作成 / Create GitHub Repository

```bash
# 新しいリポジトリを作成
# Create a new repository
git init
git add .
git commit -m "Initial commit: SE Prediction App"

# GitHubで新しいリポジトリを作成後
# After creating a new repository on GitHub
git remote add origin https://github.com/yourusername/se-prediction-app.git
git branch -M main
git push -u origin main
```

### 2. Streamlit Community Cloudでデプロイ / Deploy on Streamlit Community Cloud

#### 手順 / Steps:

1. **https://share.streamlit.io** にアクセス
   Visit https://share.streamlit.io

2. **Sign in with GitHub**
   GitHubアカウントでサインイン

3. **New app** をクリック
   Click "New app"

4. リポジトリ情報を入力 / Enter repository information:
   - **Repository**: `yourusername/se-prediction-app`
   - **Branch**: `main`
   - **Main file path**: `app.py`

5. **Deploy!** をクリック
   Click "Deploy!"

6. 数分待つと、アプリが公開されます！
   Wait a few minutes, and your app will be live!

## 🎨 カスタマイズ / Customization

### アプリのURLを変更 / Change App URL

1. Streamlit Cloudのダッシュボードにアクセス
   Access Streamlit Cloud dashboard

2. アプリの設定から "App settings" を開く
   Open "App settings" from your app

3. "General" タブで URL を変更
   Change URL in the "General" tab

## 🔒 モデルファイルの扱い / Handling Model Files

### オプション1: モデルをGitHubにコミット / Option 1: Commit Models to GitHub

```bash
# モデルを訓練して保存
# Train and save models
python train_and_save_models.py

# .gitignore からmodels/を削除
# Remove models/ from .gitignore

# コミット
# Commit
git add models/
git commit -m "Add trained models"
git push
```

**注意**: 大きなファイルの場合、Git LFSの使用を推奨
**Note**: For large files, consider using Git LFS

### オプション2: アプリ起動時に訓練 / Option 2: Train on Startup

app.py内で、モデルが存在しない場合は自動的に初期化されます。
Models are automatically initialized in app.py if they don't exist.

**注意**: 初回起動時に時間がかかります
**Note**: First startup will be slower

## 📊 データの準備 / Data Preparation

### 訓練データの配置 / Place Training Data

1. データファイル（CSV）を準備
   Prepare your data file (CSV)

2. リポジトリのルートに配置
   Place it in the repository root

3. `train_and_save_models.py` を編集:
   Edit `train_and_save_models.py`:

```python
# データを読み込む
sep_df_cleaned = pd.read_csv('your_data.csv')
```

4. ローカルで訓練
   Train locally:

```bash
python train_and_save_models.py
```

5. 訓練済みモデルをコミット
   Commit trained models:

```bash
git add models/
git commit -m "Add trained models"
git push
```

## 🔍 トラブルシューティング / Troubleshooting

### エラー: Package not found

**解決方法 / Solution**:
```bash
# requirements.txt を確認
# Check requirements.txt
pip install -r requirements.txt

# バージョンの互換性を確認
# Check version compatibility
```

### エラー: Model file not found

**解決方法 / Solution**:
- モデルファイルがリポジトリに含まれているか確認
  Check if model files are included in the repository
- `train_and_save_models.py` を実行
  Run `train_and_save_models.py`

### アプリが遅い / App is slow

**改善方法 / Improvements**:
- `@st.cache_resource` デコレータを活用
  Use `@st.cache_resource` decorator
- モデル数を減らす
  Reduce number of models
- より軽量なモデルを使用
  Use lighter models

## 🌐 独自ドメインの設定 / Custom Domain Setup

Streamlit Community Cloudでは独自ドメインも設定可能:
You can set up a custom domain with Streamlit Community Cloud:

1. ドメインを取得（例: GoDaddy, Namecheap）
   Purchase a domain (e.g., GoDaddy, Namecheap)

2. Streamlit Cloudの設定で "Custom domain" を選択
   Select "Custom domain" in Streamlit Cloud settings

3. DNS設定を更新
   Update DNS settings

詳細: https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app
Details: https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app

## 📈 使用状況の監視 / Monitor Usage

Streamlit Cloudダッシュボードで以下を確認可能:
Check the following in Streamlit Cloud dashboard:

- アクセス数 / Number of visitors
- リソース使用量 / Resource usage
- エラーログ / Error logs
- 稼働時間 / Uptime

## 🎯 次のステップ / Next Steps

1. ✅ アプリをテスト
   Test the app

2. ✅ フィードバックを収集
   Collect feedback

3. ✅ 機能を追加
   Add features

4. ✅ パフォーマンスを最適化
   Optimize performance

5. ✅ ドキュメントを更新
   Update documentation

## 💡 ヒント / Tips

### パフォーマンス向上 / Performance Improvement

```python
# キャッシュを効果的に使用
@st.cache_resource
def load_models():
    # Heavy computation here
    return models

# セッション状態を活用
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
```

### ユーザー体験の向上 / Improve UX

```python
# プログレスバーを表示
with st.spinner('Calculating predictions...'):
    predictions = calculate_predictions()

# 成功メッセージ
st.success('Predictions completed!')
```

## 📞 サポート / Support

問題が発生した場合:
If you encounter issues:

- GitHub Issues: リポジトリで問題を報告
  Report issues in the repository
- Streamlit Forum: https://discuss.streamlit.io
- Streamlit Docs: https://docs.streamlit.io

---

**成功を祈ります！/ Good luck!** 🚀
