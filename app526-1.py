import os
import sys
import pandas as pd
import streamlit as st
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
model = joblib.load('训练好的模型/xgb_model.pkl')

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
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Laboratory Indicators")
        
        ast_alt = st.number_input(
            "AST/ALT Ratio",
            min_value=0.0,
            max_value=10.0,
            value=1.5,
            step=0.1,
            help="Aspartate aminotransferase / Alanine aminotransferase ratio"
        )
        
        basophil_pct = st.number_input(
            "Basophil Percentage (%)",
            min_value=0.0,
            max_value=10.0,
            value=0.5,
            step=0.1,
            help="Percentage of basophils in white blood cells"
        )
        
        ca125 = st.number_input(
            "CA125 (U/mL)",
            min_value=0.0,
            max_value=1000.0,
            value=20.0,
            step=1.0,
            help="Cancer antigen 125"
        )
        
    with col2:
        st.subheader("Clinical Indicators")
        
        ca153 = st.number_input(
            "CA153 (U/mL)",
            min_value=0.0,
            max_value=500.0,
            value=22.0,
            step=1.0,
            help="Cancer antigen 15-3"
        )
        
        neutropenia4 = st.selectbox(
            "Neutropenia Grade 4",
            options=[("No", 0), ("Yes", 1)],
            format_func=lambda x: x[0]
        )[1]
        
        tba = st.number_input(
            "Total Bile Acid (TBA) (μmol/L)",
            min_value=0.0,
            max_value=200.0,
            value=4.0,
            step=0.5,
            help="Total bile acid level"
        )
    
    # 提交按钮
    submitted = st.form_submit_button("Predict Progression Risk")
    
    if submitted:
        # 创建输入数据框 - 按照模型训练时的特征顺序
        input_data = pd.DataFrame([[
            ast_alt, basophil_pct, ca125, ca153, neutropenia4, tba
        ]], columns=[
            'AST/ALT ratio', 'Basophil percentage', 'CA125', 'CA153', 'Neutropenia4', 'Total bile acid (TBA)'
        ])
        
        # 预测概率
        try:
            proba = model.predict_proba(input_data)[0][1]
            
            # 显示结果
            st.subheader("Prediction Result")
            
            # 使用进度条和指标显示概率
            col_res1, col_res2 = st.columns([1, 3])
            with col_res1:
                st.metric(label="Progression Probability", value=f"{proba:.1%}")
            with col_res2:
                st.progress(float(proba))
            
            # 风险等级评估
            if proba > 0.5:
                st.error("⚠️ High Risk")
            elif proba > 0.2:
                st.warning("⚠️ Medium Risk")
            else:
                st.success("✅ Low Risk")
            
            # 详细解释
            st.markdown("### Clinical Recommendations")
            if proba > 0.5:
                st.markdown("""
                **High Risk Patients**:
                - **Immediate Actions**:
                  - Close monitoring of disease progression
                  - Consider more frequent follow-up imaging
                  - Evaluate treatment response regularly
                
                - **Monitoring**:
                  - Assess need for treatment adjustment
                  - Monitor clinical symptoms weekly
                  - Regular laboratory tests
                """)
            elif proba > 0.2:
                st.markdown("""
                **Medium Risk Patients**:
                - **Further Evaluation**:
                  - Regular follow-up every 1-2 months
                  - Monitor clinical symptoms
                  - Routine laboratory tests
                
                - **Preventive Measures**:
                  - Consider further evaluation if symptoms progress
                  - Maintain regular follow-up schedule
                """)
            else:
                st.markdown("""
                **Low Risk Patients**:
                - **Regular Management**:
                  - Continue standard monitoring protocol
                  - Routine follow-up as scheduled
                  - Maintain regular laboratory testing
                
                - **Prevention Suggestions**:
                  - Regular health check-ups
                  - Monitor for any new symptoms
                """)
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# 添加使用说明
st.markdown("---")
st.subheader("Instructions")
st.markdown("""
1. Enter all patient parameters in the form
2. Click the 'Predict' button
3. The system will display the progression probability and risk level

**Parameter Specification**:
- **Laboratory Indicators**: Current laboratory test values
- **Clinical Indicators**: Patient clinical status and tumor markers

**Risk Level Definition**:
- **Low Risk**: < 20% progression probability
- **Medium Risk**: 20% - 50% progression probability
- **High Risk**: > 50% progression probability
""")

# 添加页脚
st.markdown("---")
st.caption("© Disease Progression Prediction Model | Based on XGBoost Algorithm")
