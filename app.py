import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
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

# ==========================================
# SE分類関数
# ==========================================
def classify_se(se_value):
    """球面度数を分類"""
    if se_value < -10:
        return {
            'en': 'High Myopia (Extreme)',
            'ja': '最強度近視',
            'color': '#8B0000',  # Dark red
            'emoji': '🔴'
        }
    elif se_value < -6:
        return {
            'en': 'High Myopia',
            'ja': '強度近視',
            'color': '#DC143C',  # Crimson
            'emoji': '🟠'
        }
    elif se_value < -3:
        return {
            'en': 'Moderate Myopia',
            'ja': '中等度近視',
            'color': '#FFA500',  # Orange
            'emoji': '🟡'
        }
    elif se_value < -0.5:
        return {
            'en': 'Mild Myopia',
            'ja': '弱度近視',
            'color': '#FFD700',  # Gold
            'emoji': '🟢'
        }
    elif se_value < 0.5:
        return {
            'en': 'Emmetropia',
            'ja': '正視',
            'color': '#32CD32',  # Lime green
            'emoji': '✅'
        }
    else:
        return {
            'en': 'Hyperopia',
            'ja': '遠視',
            'color': '#1E90FF',  # Dodger blue
            'emoji': '🔵'
        }

def create_se_gauge(se_value, lang='en'):
    """SE値のゲージチャートを作成"""
    classification = classify_se(se_value)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = se_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"{classification['emoji']} {classification[lang]}", 
                'font': {'size': 20}},
        delta = {'reference': 0},
        gauge = {
            'axis': {'range': [-15, 5], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': classification['color'], 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-15, -10], 'color': '#FFE6E6'},  # 最強度近視
                {'range': [-10, -6], 'color': '#FFE6F0'},   # 強度近視
                {'range': [-6, -3], 'color': '#FFF4E6'},    # 中等度近視
                {'range': [-3, -0.5], 'color': '#FFFACD'},  # 弱度近視
                {'range': [-0.5, 0.5], 'color': '#E6FFE6'}, # 正視
                {'range': [0.5, 5], 'color': '#E6F3FF'}     # 遠視
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        font={'size': 14}
    )
    
    return fig

# ==========================================
# 多言語対応
# ==========================================
TRANSLATIONS = {
    'en': {
        'title': '👁️ Spherical Equivalent (SE) Prediction App',
        'subtitle': 'Predict spherical equivalent from ophthalmological examination data',
        'input_header': 'Input Examination Data',
        'age': 'Age',
        'sex': 'Sex',
        'male': 'Male',
        'female': 'Female',
        'k_avg': 'K (AVG)',
        'al': 'AL (Axial Length)',
        'lt': 'LT (Lens Thickness)',
        'acd': 'ACD (Anterior Chamber Depth)',
        'predict_button': 'Predict',
        'prediction_results': 'Prediction Results',
        'model': 'Model',
        'predicted_value': 'Predicted Value',
        'classification': 'Classification',
        'model_info': '📊 Model Information',
        'all_predictions': '🎯 All Model Predictions',
        'top_models': '🏆 Top 3 Models',
        'comparison_chart': '📈 Model Comparison',
        'test_r2': 'Test R²',
        'test_rmse': 'Test RMSE',
        'cv_rmse': 'CV RMSE',
        'input_data': 'Input Data',
        'select_models': 'Select Models to Display',
        'show_all': 'Show All Models',
        'language': 'Language',
        'input_mode': 'Input Mode',
        'single_input': 'Single Patient',
        'batch_input': 'Batch (Excel)',
        'upload_file': 'Upload Excel File',
        'download_template': '📥 Download Template',
        'file_format': 'Required columns: 年齢, 性別, K（AVG）, AL, LT, ACD',
        'process_batch': 'Process Batch',
        'batch_results': '📊 Batch Prediction Results',
        'download_results': '📥 Download Results',
        'se_gauge': 'SE Classification Meter',
        'patient_id': 'Patient ID',
        'se_categories': {
            'extreme_myopia': 'High Myopia (Extreme)',
            'high_myopia': 'High Myopia',
            'moderate_myopia': 'Moderate Myopia',
            'mild_myopia': 'Mild Myopia',
            'emmetropia': 'Emmetropia',
            'hyperopia': 'Hyperopia'
        }
    },
    'ja': {
        'title': '👁️ 球面度数（SE）予測アプリ',
        'subtitle': '眼科検査データから球面度数を予測します',
        'input_header': '検査データ入力',
        'age': '年齢',
        'sex': '性別',
        'male': '男性',
        'female': '女性',
        'k_avg': 'K（AVG）',
        'al': 'AL（眼軸長）',
        'lt': 'LT（水晶体厚）',
        'acd': 'ACD（前房深度）',
        'predict_button': '予測実行',
        'prediction_results': '予測結果',
        'model': 'モデル',
        'predicted_value': '予測値',
        'classification': '分類',
        'model_info': '📊 モデル情報',
        'all_predictions': '🎯 全モデルの予測結果',
        'top_models': '🏆 Top 3 モデル',
        'comparison_chart': '📈 モデル比較',
        'test_r2': 'テストR²',
        'test_rmse': 'テストRMSE',
        'cv_rmse': 'CV RMSE',
        'input_data': '入力データ',
        'select_models': '表示するモデルを選択',
        'show_all': '全モデルを表示',
        'language': '言語',
        'input_mode': '入力モード',
        'single_input': '単一患者',
        'batch_input': '一括処理（Excel）',
        'upload_file': 'Excelファイルをアップロード',
        'download_template': '📥 テンプレートダウンロード',
        'file_format': '必須カラム: 年齢, 性別, K（AVG）, AL, LT, ACD',
        'process_batch': '一括処理実行',
        'batch_results': '📊 一括予測結果',
        'download_results': '📥 結果をダウンロード',
        'se_gauge': 'SE分類メーター',
        'patient_id': '患者ID',
        'se_categories': {
            'extreme_myopia': '最強度近視',
            'high_myopia': '強度近視',
            'moderate_myopia': '中等度近視',
            'mild_myopia': '弱度近視',
            'emmetropia': '正視',
            'hyperopia': '遠視'
        }
    }
}

# ==========================================
# ページ設定
# ==========================================
st.set_page_config(
    page_title="SE Prediction App",
    page_icon="👁️",
    layout="wide"
)

# 言語選択
lang = st.sidebar.selectbox(
    "🌐 Language / 言語",
    options=['en', 'ja'],
    format_func=lambda x: 'English' if x == 'en' else '日本語'
)

t = TRANSLATIONS[lang]

# ==========================================
# タイトル
# ==========================================
st.title(t['title'])
st.write(t['subtitle'])

# ==========================================
# モデル定義
# ==========================================
SEED = 2025

@st.cache_resource
def load_models():
    """全てのモデルを読み込み（または初期化）"""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=SEED),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=SEED, n_jobs=-1, verbose=-1),
        'CatBoost': CatBoostRegressor(iterations=100, random_state=SEED, verbose=False),
        'SVR': SVR(kernel='rbf'),
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'MLP': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=SEED)
    }
    
    # 保存されたモデルを読み込む（存在する場合）
    for name in models.keys():
        model_path = f'models/{name.replace(" ", "_")}.pkl'
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    models[name] = pickle.load(f)
            except:
                pass
    
    return models

# モデル性能データ（実際の評価結果）
MODEL_PERFORMANCE = {
    'Gradient Boosting': {'Test R²': 0.9640, 'Test RMSE': 0.7369, 'CV RMSE': 0.9030},
    'Extra Trees': {'Test R²': 0.9609, 'Test RMSE': 0.7674, 'CV RMSE': 0.9017},
    'MLP': {'Test R²': 0.9601, 'Test RMSE': 0.7753, 'CV RMSE': 1.4036},
    'XGBoost': {'Test R²': 0.9562, 'Test RMSE': 0.8123, 'CV RMSE': 0.9912},
    'Ridge': {'Test R²': 0.9547, 'Test RMSE': 0.8261, 'CV RMSE': 0.9182},
    'CatBoost': {'Test R²': 0.9546, 'Test RMSE': 0.8271, 'CV RMSE': 0.9294},
    'Linear Regression': {'Test R²': 0.9546, 'Test RMSE': 0.8271, 'CV RMSE': 0.9176},
    'Lasso': {'Test R²': 0.9468, 'Test RMSE': 0.8958, 'CV RMSE': 0.9962},
    'Random Forest': {'Test R²': 0.9455, 'Test RMSE': 0.9065, 'CV RMSE': 0.9939},
    'ElasticNet': {'Test R²': 0.9396, 'Test RMSE': 0.9538, 'CV RMSE': 1.0263},
    'LightGBM': {'Test R²': 0.9349, 'Test RMSE': 0.9902, 'CV RMSE': 1.0008},
    'SVR': {'Test R²': 0.8685, 'Test RMSE': 1.4076, 'CV RMSE': 1.3254},
    'KNN': {'Test R²': 0.8433, 'Test RMSE': 1.5366, 'CV RMSE': 1.5308}
}

models = load_models()

# ==========================================
# サイドバー入力
# ==========================================
st.sidebar.header(t['input_header'])

# 入力モード選択
input_mode = st.sidebar.radio(
    t['input_mode'],
    options=['single', 'batch'],
    format_func=lambda x: t['single_input'] if x == 'single' else t['batch_input']
)

if input_mode == 'single':
    # 単一患者入力
    age = st.sidebar.number_input(
        t['age'],
        min_value=0,
        max_value=100,
        value=30,
        step=1
    )

    sex = st.sidebar.selectbox(
        t['sex'],
        options=[0, 1],
        format_func=lambda x: t['male'] if x == 0 else t['female']
    )

    k_avg = st.sidebar.number_input(
        t['k_avg'],
        min_value=40.0,
        max_value=50.0,
        value=43.5,
        step=0.1,
        format="%.2f"
    )

    al = st.sidebar.number_input(
        t['al'],
        min_value=20.0,
        max_value=30.0,
        value=24.0,
        step=0.1,
        format="%.2f"
    )

    lt = st.sidebar.number_input(
        t['lt'],
        min_value=3.0,
        max_value=6.0,
        value=4.5,
        step=0.1,
        format="%.2f"
    )

    acd = st.sidebar.number_input(
        t['acd'],
        min_value=2.0,
        max_value=4.0,
        value=3.0,
        step=0.1,
        format="%.2f"
    )
else:
    # バッチ入力
    st.sidebar.markdown(f"### {t['upload_file']}")
    st.sidebar.info(t['file_format'])
    
    # テンプレートダウンロード
    template_df = pd.DataFrame({
        '年齢': [30, 45, 25],
        '性別': [0, 1, 0],
        'K（AVG）': [43.5, 44.0, 43.0],
        'AL': [24.0, 25.5, 23.5],
        'LT': [4.5, 4.8, 4.3],
        'ACD': [3.0, 3.2, 2.9]
    })
    
    # Excelファイルとして出力
    from io import BytesIO
    template_buffer = BytesIO()
    with pd.ExcelWriter(template_buffer, engine='openpyxl') as writer:
        template_df.to_excel(writer, index=False, sheet_name='Template')
    template_buffer.seek(0)
    
    st.sidebar.download_button(
        label=t['download_template'],
        data=template_buffer,
        file_name="se_prediction_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose Excel file",
        type=['xlsx', 'xls'],
        help=t['file_format']
    )

# モデル選択
st.sidebar.markdown("---")
show_all = st.sidebar.checkbox(t['show_all'], value=True)

if not show_all:
    selected_models = st.sidebar.multiselect(
        t['select_models'],
        options=list(models.keys()),
        default=['Gradient Boosting', 'Extra Trees', 'XGBoost']
    )
else:
    selected_models = list(models.keys())

# ==========================================
# 予測実行
# ==========================================
if input_mode == 'single':
    # 単一患者の予測
    if st.sidebar.button(t['predict_button'], type="primary", use_container_width=True):
        
        # 入力データ作成
        input_data = pd.DataFrame({
            '年齢': [age],
            '性別': [sex],
            'K（AVG）': [k_avg],
            'AL': [al],
            'LT': [lt],
            'ACD': [acd]
        })
        
        # 全モデルで予測
        predictions = {}
        for model_name in selected_models:
            try:
                model = models[model_name]
                pred = model.predict(input_data)[0]
                predictions[model_name] = pred
            except Exception as e:
                st.error(f"Error in {model_name}: {str(e)}")
                predictions[model_name] = None
        
        # ==========================================
        # 結果表示
        # ==========================================
        st.markdown(f"## {t['prediction_results']}")
        
        # 入力データ表示
        with st.expander(t['input_data'], expanded=False):
            st.dataframe(input_data, use_container_width=True)
        
        # ベストモデルの予測値でゲージ表示
        best_model = 'Gradient Boosting'
        if best_model in predictions and predictions[best_model] is not None:
            st.markdown(f"### {t['se_gauge']}")
            best_pred = predictions[best_model]
            classification = classify_se(best_pred)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = create_se_gauge(best_pred, lang)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### ")
                st.markdown("### ")
                st.metric(
                    label=f"{classification['emoji']} {classification[lang]}",
                    value=f"{best_pred:.2f} D"
                )
                st.markdown(f"**Model:** {best_model}")
                st.markdown(f"**R²:** {MODEL_PERFORMANCE[best_model]['Test R²']:.4f}")
        
        # Top 3 モデル
        st.markdown(f"### {t['top_models']}")
        
        top_models = ['Gradient Boosting', 'Extra Trees', 'MLP']
        cols = st.columns(3)
        
        for i, model_name in enumerate(top_models):
            if model_name in predictions and predictions[model_name] is not None:
                with cols[i]:
                    classification = classify_se(predictions[model_name])
                    st.metric(
                        label=f"{classification['emoji']} {model_name}",
                        value=f"{predictions[model_name]:.2f} D",
                        delta=classification[lang]
                    )
        
        # 全モデルの予測結果
        st.markdown(f"### {t['all_predictions']}")
        
        # データフレーム作成
        results_data = []
        for model_name, pred_value in predictions.items():
            if pred_value is not None and model_name in MODEL_PERFORMANCE:
                classification = classify_se(pred_value)
                results_data.append({
                    t['model']: model_name,
                    t['predicted_value']: f"{pred_value:.2f} D",
                    t['classification']: f"{classification['emoji']} {classification[lang]}",
                    t['test_r2']: f"{MODEL_PERFORMANCE[model_name]['Test R²']:.4f}",
                    t['test_rmse']: f"{MODEL_PERFORMANCE[model_name]['Test RMSE']:.4f}",
                    t['cv_rmse']: f"{MODEL_PERFORMANCE[model_name]['CV RMSE']:.4f}"
                })
        
        results_df = pd.DataFrame(results_data)
        
        # 予測値で並び替え
        results_df_sorted = results_df.copy()
        results_df_sorted['_pred_numeric'] = [predictions[row[t['model']]] for _, row in results_df.iterrows()]
        results_df_sorted = results_df_sorted.sort_values('_pred_numeric', ascending=False)
        results_df_sorted = results_df_sorted.drop('_pred_numeric', axis=1)
        
        st.dataframe(
            results_df_sorted,
            use_container_width=True,
            hide_index=True
        )
        
        # ==========================================
        # 可視化
        # ==========================================
        st.markdown(f"### {t['comparison_chart']}")
        
        # 予測値の比較
        chart_data = pd.DataFrame({
            'Model': list(predictions.keys()),
            'Predicted SE': [predictions[m] for m in predictions.keys()]
        })
        chart_data = chart_data.sort_values('Predicted SE', ascending=False)
        
        st.bar_chart(
            chart_data.set_index('Model'),
            use_container_width=True,
            height=400
        )
        
        # 統計情報
        col1, col2, col3, col4 = st.columns(4)
        
        pred_values = [v for v in predictions.values() if v is not None]
        
        with col1:
            st.metric("Mean", f"{np.mean(pred_values):.2f} D")
        with col2:
            st.metric("Median", f"{np.median(pred_values):.2f} D")
        with col3:
            st.metric("Std Dev", f"{np.std(pred_values):.2f}")
        with col4:
            st.metric("Range", f"{np.max(pred_values) - np.min(pred_values):.2f}")

else:
    # バッチ処理
    if uploaded_file is not None:
        if st.sidebar.button(t['process_batch'], type="primary", use_container_width=True):
            try:
                # Excelファイルを読み込む
                batch_df = pd.read_excel(uploaded_file)
                
                # 必須カラムのチェック
                required_cols = ['年齢', '性別', 'K（AVG）', 'AL', 'LT', 'ACD']
                missing_cols = [col for col in required_cols if col not in batch_df.columns]
                
                if missing_cols:
                    st.error(f"Missing columns: {', '.join(missing_cols)}")
                else:
                    # 患者IDがない場合は追加
                    if t['patient_id'] not in batch_df.columns and 'Patient ID' not in batch_df.columns:
                        batch_df.insert(0, t['patient_id'], [f"P{i+1:03d}" for i in range(len(batch_df))])
                    
                    # 全モデルで予測
                    st.markdown(f"## {t['batch_results']}")
                    st.markdown(f"Processing {len(batch_df)} patients...")
                    
                    progress_bar = st.progress(0)
                    
                    # 各モデルの予測を格納
                    for idx, model_name in enumerate(selected_models):
                        try:
                            model = models[model_name]
                            predictions = model.predict(batch_df[required_cols])
                            batch_df[f'{model_name}_SE'] = predictions
                            
                            # 分類を追加
                            classifications = [classify_se(p)[lang] for p in predictions]
                            batch_df[f'{model_name}_Classification'] = classifications
                            
                            progress_bar.progress((idx + 1) / len(selected_models))
                        except Exception as e:
                            st.error(f"Error in {model_name}: {str(e)}")
                    
                    progress_bar.empty()
                    
                    # 結果表示
                    st.success(f"✅ Completed! Processed {len(batch_df)} patients with {len(selected_models)} models.")
                    
                    # 基本統計
                    best_model = 'Gradient Boosting'
                    if f'{best_model}_SE' in batch_df.columns:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        se_values = batch_df[f'{best_model}_SE']
                        
                        with col1:
                            st.metric("Mean SE", f"{se_values.mean():.2f} D")
                        with col2:
                            st.metric("Median SE", f"{se_values.median():.2f} D")
                        with col3:
                            st.metric("Std Dev", f"{se_values.std():.2f}")
                        with col4:
                            st.metric("Range", f"{se_values.max() - se_values.min():.2f}")
                    
                    # 分類の分布
                    if f'{best_model}_Classification' in batch_df.columns:
                        st.markdown("### Classification Distribution")
                        
                        classification_counts = batch_df[f'{best_model}_Classification'].value_counts()
                        
                        fig = px.pie(
                            values=classification_counts.values,
                            names=classification_counts.index,
                            title=f'SE Classification Distribution ({best_model})'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 詳細結果テーブル
                    st.markdown("### Detailed Results")
                    st.dataframe(batch_df, use_container_width=True, height=400)
                    
                    # 結果をダウンロード
                    output_buffer = BytesIO()
                    with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                        batch_df.to_excel(writer, index=False, sheet_name='Results')
                    output_buffer.seek(0)
                    
                    st.download_button(
                        label=t['download_results'],
                        data=output_buffer,
                        file_name="se_prediction_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    # ヒストグラム
                    if f'{best_model}_SE' in batch_df.columns:
                        st.markdown("### SE Distribution")
                        
                        fig = px.histogram(
                            batch_df,
                            x=f'{best_model}_SE',
                            nbins=30,
                            title=f'SE Distribution ({best_model})',
                            labels={f'{best_model}_SE': 'Spherical Equivalent (D)'}
                        )
                        fig.add_vline(x=-10, line_dash="dash", line_color="red", annotation_text="High Myopia")
                        fig.add_vline(x=-6, line_dash="dash", line_color="orange", annotation_text="Moderate")
                        fig.add_vline(x=-3, line_dash="dash", line_color="yellow", annotation_text="Mild")
                        fig.add_vline(x=-0.5, line_dash="dash", line_color="green", annotation_text="Emmetropia")
                        fig.add_vline(x=0.5, line_dash="dash", line_color="blue", annotation_text="Hyperopia")
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.error("Please make sure your Excel file has the correct format.")
    else:
        st.info(f"👆 {t['upload_file']}")
        st.markdown(f"**{t['file_format']}**")

# ==========================================
# モデル情報
# ==========================================
with st.expander(t['model_info']):
    st.markdown("""
    ### Model Performance Summary
    
    | Model | Test R² | Test RMSE | CV RMSE |
    |-------|---------|-----------|---------|
    | Gradient Boosting | 0.9640 | 0.7369 | 0.9030 |
    | Extra Trees | 0.9609 | 0.7674 | 0.9017 |
    | MLP | 0.9601 | 0.7753 | 1.4036 |
    | XGBoost | 0.9562 | 0.8123 | 0.9912 |
    | Ridge | 0.9547 | 0.8261 | 0.9182 |
    
    **Features Used:**
    - Age (年齢)
    - Sex (性別)
    - K (AVG) - Average Keratometry
    - AL - Axial Length (眼軸長)
    - LT - Lens Thickness (水晶体厚)
    - ACD - Anterior Chamber Depth (前房深度)
    
    **Training Details:**
    - 5-Fold Cross Validation
    - Random Seed: 2025
    - Test Size: 20%
    """)

# フッター
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: gray;'>Built with Streamlit | "
    f"{'眼科検査データから球面度数を予測' if lang == 'ja' else 'Ophthalmological SE Prediction'}</div>",
    unsafe_allow_html=True
)
