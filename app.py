import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


app_dir = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(app_dir / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(app_dir / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


st.title('Подбор хороших вин')
model = load_model()
test = load_test()
X_test = test.drop(columns='target')
y_test = test['target']
features = list(X_test.columns)
config = json.loads((app_dir / 'config.json').read_text())

threshold = st.sidebar.slider(
    'Порог', 0.0, 1.0, float(config['threshold']), step=0.001,
    format='%.3f'
)

single_tab, metrics_tab, batch_tab, weights_tab = st.tabs([
    'Одно вино', 'Модельные метрики', 'Загрузка CSV', 'Коэффициенты'
])

with single_tab:
    values = {}
    for feature in features:
        if feature == 'is_red':
            values[feature] = int(st.toggle('Красное вино'))
        else:
            values[feature] = st.number_input(
                feature,
                min_value=float(X_test[feature].min()),
                max_value=float(X_test[feature].max()),
                value=float(X_test[feature].median()),
                format='%.5f'
            )

    sample = pd.DataFrame([values], columns=features)
    probability = model.predict_proba(sample)[0, 1]
    st.write(f'Вероятность хорошего вина: {probability:.1%}')
    st.write('Хорошее вино' if probability >= threshold else 'Не хорошее вино')

with metrics_tab:
    proba = test_proba(model, X_test)
    pred = (proba >= threshold).astype(int)
    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred)
    f1 = metrics.f1_score(y_test, pred)

    columns = st.columns(3)
    columns[0].metric('Precision', f'{precision:.3f}')
    columns[1].metric('Recall', f'{recall:.3f}')
    columns[2].metric('F1', f'{f1:.3f}')

    fig, ax = plt.subplots()
    metrics.ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=['Не хорошее', 'Хорошее'], ax=ax
    )
    st.pyplot(fig)
    plt.close(fig)

    fpr, tpr, _ = metrics.roc_curve(y_test, proba)
    pr_precision, pr_recall, _ = metrics.precision_recall_curve(y_test, proba)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].scatter(fp / (fp + tn), recall, color='red')
    axes[0].set(xlabel='FPR', ylabel='TPR', title='ROC-кривая')

    axes[1].plot(pr_recall, pr_precision)
    axes[1].axhline(y_test.mean(), linestyle='--', color='gray')
    axes[1].scatter(recall, precision, color='red')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='PR-кривая')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write(
        'Если повысить порог, в подборку попадет меньше вин. '
        'Отбор обычно станет точнее, но больше хороших вин будет пропущено. '
        'Если снизить порог, подборка станет шире, '
        'но в нее может попасть больше посредственных вин.'
    )

with batch_tab:
    uploaded = st.file_uploader('CSV с признаками вин без target', type='csv')
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
            missing = [feature for feature in features if feature not in data.columns]
            if missing:
                st.error('Не хватает столбцов: ' + ', '.join(missing))
            elif data.empty:
                st.error('В файле нет строк с винами.')
            else:
                batch = data[features].apply(pd.to_numeric, errors='raise')
                if not np.isfinite(batch.to_numpy()).all():
                    st.error('В признаках есть пропуски или бесконечные значения.')
                else:
                    data['probability'] = model.predict_proba(batch)[:, 1]
                    data['prediction'] = (data['probability'] >= threshold).astype(int)
                    st.dataframe(data)
                    st.download_button(
                        'Скачать результат',
                        data.to_csv(index=False).encode('utf-8'),
                        file_name='predictions.csv', mime='text/csv'
                    )
        except (ValueError, pd.errors.ParserError, UnicodeError):
            st.error('Не удалось прочитать данные. Проверьте CSV и числовые признаки.')

with weights_tab:
    coefficients = pd.Series(
        model.steps[-1][1].coef_[0], index=features
    ).sort_values()
    fig, ax = plt.subplots()
    coefficients.plot.barh(ax=ax)
    ax.set_xlabel('Коэффициент после масштабирования')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write('Сильнее всего за хорошее вино:', ', '.join(coefficients.tail(3).index[::-1]))
    st.write('Сильнее всего против:', ', '.join(coefficients.head(3).index))
