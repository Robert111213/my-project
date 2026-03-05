# app.py
# Веб-интерфейс для калькулятора цен на квартиры

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Калькулятор цен на квартиры",
    page_icon="",
    layout="wide"
)

st.title(" Калькулятор рыночной стоимости квартиры")
st.markdown("---")

@st.cache_resource
def load_model():
    try:
        model = joblib.load('flat_price_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Файл модели не найден! Сначала запусти train_model.py")
        return None

@st.cache_data
def load_model_info():
    try:
        return joblib.load('model_info.pkl')
    except:
        return None

model = load_model()
model_info = load_model_info()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(" Параметры квартиры")
    
    with st.form("apartment_form"):
        area = st.number_input(
            "Площадь (м²)", 
            min_value=10.0, 
            max_value=200.0, 
            value=50.0,
            step=1.0
        )
        
        rooms = st.selectbox(
            "Количество комнат",
            options=[1, 2, 3, 4, 5],
            index=1
        )
        
        floor_type = st.selectbox(
            "Этаж",
            options=['первый', 'средний', 'последний'],
            index=1
        )
        
        district = st.selectbox(
            "Район",
            options=['Центр', 'Северный', 'Южный'],
            index=0
        )
        
        condition = st.selectbox(
            "Состояние квартиры",
            options=['требует ремонта', 'хорошее', 'евроремонт'],
            index=1
        )
        
        submitted = st.form_submit_button("Рассчитать стоимость", type="primary", use_container_width=True)

with col2:
    st.subheader(" Результат расчета")
    
    if model is not None:
        if submitted:
            # ВАЖНО: названия столбцов должны совпадать с train_model.py
            input_data = pd.DataFrame({
                'Площадь': [area],
                'Комнат': [rooms],           # именно 'Комнат'
                'Этаж': [floor_type],
                'Район': [district],
                'Состояние': [condition]
            })
            
            try:
                predicted_price = model.predict(input_data)[0]
                
                price_min = predicted_price * 0.95
                price_max = predicted_price * 1.05
                
                st.metric(
                    label="Ориентировочная стоимость",
                    value=f"{predicted_price:,.0f} ₽"
                )
                
                st.info(f" Диапазон вероятной стоимости:\n{price_min:,.0f} ₽ — {price_max:,.0f} ₽")
                
                st.subheader(" Введенные параметры")
                params_df = pd.DataFrame({
                    'Параметр': ['Площадь', 'Комнат', 'Этаж', 'Район', 'Состояние'],
                    'Значение': [f"{area} м²", rooms, floor_type, district, condition]
                })
                st.table(params_df)
                
                if 'history' not in st.session_state:
                    st.session_state.history = []
                
                st.session_state.history.append({
                    'datetime': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'area': area,
                    'rooms': rooms,
                    'floor': floor_type,
                    'district': district,
                    'condition': condition,
                    'price': predicted_price
                })
                
            except Exception as e:
                st.error(f"Ошибка при расчете: {e}")
        else:
            st.info(" Заполните параметры квартиры слева и нажмите 'Рассчитать стоимость'")
    else:
        st.error(" Модель не загружена. Сначала запустите train_model.py")

st.markdown("---")

tab1, tab2, tab3 = st.tabs([" О модели", " Статистика", " История расчетов"])

with tab1:
    st.subheader("О модели машинного обучения")
    
    col1, col2, col3 = st.columns(3)
    
    if model_info:
        with col1:
            st.metric("Точность модели (R²)", f"{model_info.get('r2_test', 0):.3f}")
        with col2:
            st.metric("Средняя ошибка", f"{model_info.get('mae_test', 0):,.0f} ₽")
        with col3:
            avg_price = 5000000
            st.metric("Относительная ошибка", f"{model_info.get('mae_test', 0) / avg_price * 100:.1f}%")
    else:
        st.info("Информация о модели пока недоступна")

with tab2:
    st.subheader("Анализ рынка недвижимости")
    st.write("Здесь можно добавить графики анализа")

with tab3:
    st.subheader("История расчетов")
    
    if 'history' in st.session_state and st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("Очистить историю"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("История расчетов пуста")

with st.sidebar:
    st.title("О проекте")
    st.markdown("""
    **Калькулятор цен на квартиры**
    
    **Технологии:**
    - Python
    - Scikit-learn
    - Pandas
    - Streamlit
    """)

st.markdown("---")
st.markdown("© 2026 Калькулятор цен на квартиры")