from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import metrics

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title='Подборка вин', layout='wide')

@st.cache_resource
def load_model():
    return joblib.load(BASE / 'model.joblib')

@st.cache_data
def load_test():
    return pd.read_csv(BASE / 'test.csv')

@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]

needed = ['model.joblib', 'test.csv', 'config.json']
if any(not (BASE / name).exists() for name in needed):
    st.error('Сначала сохраните model.joblib, test.csv и config.json из ноутбука рядом с app.py.')
    st.stop()
model = load_model()
test = load_test()
config = json.loads((BASE / 'config.json').read_text(encoding='utf-8'))
features = list(model.feature_names_in_)
X_test = test[features]
y_test = test['target'].to_numpy()
p_test = test_proba(model, X_test)
threshold = st.sidebar.slider('Порог хорошего вина', 0.0, 1.0, float(config['threshold']), step=0.001, key='threshold')
st.title('Хорошее вино для подписки')
st.caption('Модель оценивает вероятность экспертной оценки не ниже 7.')
one, quality, batch, weights = st.tabs(
    ['Одно вино', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])
with one:
    values = {}
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            if feature == 'is_red':
                values[feature] = int(st.checkbox('Красное вино', value=bool(
                    X_test[feature].median()), key='input_is_red'))
            else:
                lo, hi = float(X_test[feature].min()), float(X_test[feature].max())
                default = float(X_test[feature].median())
                if lo == hi:
                    values[feature] = lo
                    st.caption(f'{feature}: {lo}')
                else:
                    values[feature] = st.number_input(feature, min_value=lo, max_value=hi,
                        value=default, format='%.5f', key='input_' + feature)
    row = pd.DataFrame([values], columns=features)
    probability = float(model.predict_proba(row)[0, 1])
    st.metric('Вероятность хорошего вина', f'{probability:.1%}')
    st.write('Включить в подборку' if probability >= threshold else 'Не включать в подборку')
with quality:
    pred = (p_test >= threshold).astype(int)
    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred, zero_division=0)
    f1 = metrics.f1_score(y_test, pred, zero_division=0)
    for col, label, value in zip(st.columns(3), ['Precision', 'Recall', 'F1'], [precision, recall, f1]):
        col.metric(label, f'{value:.3f}')
    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    fig, ax = plt.subplots()
    metrics.ConfusionMatrixDisplay(cm, display_labels=['остальное', 'хорошее']).plot(ax=ax)
    st.pyplot(fig); plt.close(fig)
    tn, fp, fn, tp = cm.ravel()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fpr, tpr, _ = metrics.roc_curve(y_test, p_test)
    pr, re, _ = metrics.precision_recall_curve(y_test, p_test)
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].scatter(fp / (fp + tn), recall, color='red', label=f't={threshold:.3f}')
    axes[0].set(xlabel='FPR', ylabel='TPR', title='ROC')
    axes[1].plot(re, pr)
    axes[1].axhline(y_test.mean(), ls='--', color='gray')
    axes[1].scatter(recall, precision, color='red', label=f't={threshold:.3f}')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='PR')
    for ax in axes: ax.legend()
    st.pyplot(fig); plt.close(fig)
    st.caption(f'ROC-AUC: {metrics.roc_auc_score(y_test, p_test):.3f}; ' f'PR-AUC (AP): {metrics.average_precision_score(y_test, p_test):.3f}')
    st.write('При увеличении порога подборка сокращается: хороших вин можно пропустить больше. ' 'При уменьшении порога подборка расширяется: охват растёт, но могут добавиться посредственные вина. ' 'Precision на конечной выборке не обязан меняться монотонно.')
with batch:
    st.write('Загрузите CSV без target. Нужные колонки:', ', '.join(features))
    st.download_button('Скачать пример CSV', X_test.head(5).to_csv(index=False).encode('utf-8'), file_name='wine_example.csv', mime='text/csv')
    uploaded = st.file_uploader('CSV с признаками вин', type=['csv'])
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
            missing = [f for f in features if f not in data.columns]
            if missing:
                st.error('Не хватает колонок: ' + ', '.join(missing))
            elif data.empty:
                st.error('CSV не содержит строк.')
            else:
                numeric = data[features].apply(pd.to_numeric, errors='raise')
                if not np.isfinite(numeric.to_numpy()).all():
                    st.error('В признаках есть пропуски или бесконечные значения.')
                elif not numeric['is_red'].isin([0, 1]).all():
                    st.error('is_red должен содержать только 0 и 1.')
                else:
                    result = data.copy()
                    result['good_probability'] = model.predict_proba(numeric)[:, 1]
                    result['good_prediction'] = (result.good_probability >= threshold).astype(int)
                    st.dataframe(result, hide_index=True)
                    st.download_button('Скачать прогнозы', result.to_csv(index=False).encode('utf-8'), file_name='wine_predictions.csv', mime='text/csv')
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
            st.error(f'Не удалось обработать CSV. Проверьте разделитель и числовые признаки: {exc}')
with weights:
    lr = model.steps[-1][1]
    coefficients = pd.Series(lr.coef_[0], index=features).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    coefficients.plot.barh(ax=ax, color=['seagreen' if v > 0 else 'salmon' for v in coefficients])
    ax.axvline(0, color='black', lw=1)
    ax.set_xlabel('Коэффициент после StandardScaler')
    st.pyplot(fig); plt.close(fig)
    positive = coefficients[coefficients > 0].sort_values(ascending=False).head(3)
    negative = coefficients[coefficients < 0].head(3)
    st.write('Сильнее за хорошее вино: ' + ', '.join(positive.index))
    st.write('Сильнее против: ' + ', '.join(negative.index))
    st.caption('Знак коэффициента описывает изменение log-odds при прочих равных. ' 'Это связь в модели, а не доказательство причинности; коррелирующие признаки влияют на интерпретацию.')