import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pathlib
from pathlib import Path
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
import sklearn
import xgboost
from xgboost import XGBClassifier

# 设置页面标题
st.set_page_config(
    page_title="TDXd progression prediction system", 
    page_icon="🩺",
    layout="wide"
)
st.title("TDXd progression prediction system")

# 加载模型
model = joblib.load('xgb_model.pkl')

# 侧边栏 - 关于
st.sidebar.header("About")
st.sidebar.info("""
Input the relevant parameters of the patient, and the system will calculate the probability of TDXd progression occurrence.
""")

# 风险等级解释
st.sidebar.subheader("Risk level description")
st.sidebar.markdown("""
- **Low risk**: < 20% 
- **Concentration risk**: 20% - 50% 
- **High Risk**: > 50% 
""")

# 主界面
st.header("Please enter the patient parameters")

# 创建输入表单
with st.form("prediction_form"):
    # 使用多列布局
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Laboratory index")
        # 实验室指标输入
        ast_alt = st.number_input(
            "AST/ALT Ratio",
            min_value=0.0,
            max_value=10.0,
            value=1.5,
            step=0.1,
            help="Aspartate aminotransferase / Alanine aminotransferase ratio"
        )
        basophil_pct = st.number_input(
            "Basophil percentage (%)",
            min_value=0.0,
            max_value=10.0,
            value=0.5,
            step=0.1,
            help="Percentage of basophils in white blood cells"
        )
        
    with col2:
        st.subheader("Tumor markers")
        ca125 = st.number_input(
            "CA125 (U/mL)",
            min_value=0.0,
            max_value=1000.0,
            value=20.0,
            step=1.0,
            help="Cancer antigen 125"
        )
        ca153 = st.number_input(
            "CA153 (U/mL)",
            min_value=0.0,
            max_value=500.0,
            value=22.0,
            step=1.0,
            help="Cancer antigen 15-3"
        )
    
    with col3:
        st.subheader("Clinical indicators")
        # 中性粒细胞减少症选择
        neutropenia4 = st.selectbox(
            "Neutropenia Grade 4",
            options=[("NO", 0), ("YES", 1)],
            format_func=lambda x: x[0]
        )[1]
        
        tba = st.number_input(
            "Total bile acid (TBA) (μmol/L)",
            min_value=0.0,
            max_value=200.0,
            value=4.0,
            step=0.5,
            help="Total bile acid level"
        )
    
    # 提交按钮
    submitted = st.form_submit_button("Predict the risk of TDXd progression")
    
    if submitted:
        # 创建输入数据框 - 按照模型训练时的特征顺序
        input_data = pd.DataFrame([[
            ast_alt, basophil_pct, ca125, ca153, neutropenia4, tba
        ]], columns=[
            'AST/ALT ratio', 'Basophil percentage', 'CA125', 'CA153', 'Neutropenia4', 'Total bile acid (TBA)'
        ])
        
        # 预测概率
        try:
            proba = model.predict_proba(input_data)[0][1]  # 获取进展的概率
            
            # 显示结果
            st.subheader("Predict the outcome")
            
            # 使用进度条和指标显示概率
            col_res1, col_res2 = st.columns([1, 3])
            with col_res1:
                st.metric(label="Probability of TDXd progression", value=f"{proba:.1%}")
            with col_res2:
                st.progress(float(proba))
            
            # 风险等级评估
            if proba > 0.5:
                st.error("High risk")
            elif proba > 0.2:
                st.warning("Concentration risk")
            else:
                st.success("Low risk")
            
            # 详细解释
            st.markdown("### Clinical practice recommendations")
            if proba > 0.5:
                st.markdown("""
                - **Do it now**:
                  - Consider adjusting treatment regimen
                  - Conduct imaging assessment for disease progression
                  - Evaluate alternative treatment options
                
                - **Supervise**:
                  - Monitor clinical symptoms weekly
                  - Regular laboratory tests including tumor markers
                  - Assess quality of life and treatment tolerance
                """)
            elif proba > 0.2:
                st.markdown("""
                - **Further inspection**:
                  - Schedule follow-up imaging in 4-6 weeks
                  - Monitor tumor marker trends
                  - Assess clinical symptom changes
                
                - **Preventive measures**:
                  - Maintain regular follow-up schedule
                  - Monitor for new symptoms
                  - Consider supportive care optimization
                """)
            else:
                st.markdown("""
                - **Regular management**:
                  - Continue current monitoring protocol
                  - Routine follow-up as scheduled
                  - Regular laboratory testing
                
                - **Prevention suggestions**:
                  - Maintain regular health check-ups
                  - Monitor for any new symptoms
                  - Adhere to scheduled follow-up visits
                """)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

# 添加使用说明
st.markdown("---")
st.subheader("instructions")
st.markdown("""
1. Enter all the parameters of the patient in the form
2. The system will calculate and display the probability and risk level of TDXd progression

**Parameter specification**:
- **Laboratory index**: AST/ALT ratio and basophil percentage
- **Tumor markers**: CA125 and CA153 levels
- **Clinical indicators**: Neutropenia grade and total bile acid levels

**Risk Level Definition**:
- **Low risk**: < 20% progression probability
- **Medium risk**: 20% - 50% progression probability  
- **High risk**: > 50% progression probability
""")

# 添加页脚
st.markdown("---")
st.caption("© TDXd progression prediction model")
