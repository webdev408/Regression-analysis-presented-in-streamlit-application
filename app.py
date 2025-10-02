import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import plotly.express as px
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Set page configuration
st.set_page_config(page_title="Sales Revenue Analysis", layout="wide")

# Title
st.title("📊 Sales Revenue Regression Analysis")
st.markdown("---")

# Generate Data
st.header("1. Data Generation")
with st.expander("View Data Generation Code"):
    st.code("""
np.random.seed(124)
sales = np.random.uniform(25, 100, size=50)
borrowing_cost = np.random.uniform(5.0, 25.0, size=50)
population = np.random.uniform(200, 800, size=50)
tv_ad = np.random.choice([0, 1], size=50, p=[0.6, 0.4])
    """, language="python")

np.random.seed(124)
sales = np.random.uniform(25, 100, size=50)
borrowing_cost = np.random.uniform(5.0, 25.0, size=50)
population = np.random.uniform(200, 800, size=50)
tv_ad = np.random.choice([0, 1], size=50, p=[0.6, 0.4])

df = pd.DataFrame({
    'sales(millions)': sales,
    'interest_rate(%)': borrowing_cost,
    'population(000)': population,
    'tv_ad': tv_ad
})

st.subheader("Dataset Preview")
st.dataframe(df.tail().round(2), use_container_width=True)
st.write(f"**Total observations:** {len(df)}")

# Exploratory Plot
st.markdown("---")
st.header("2. Exploratory Analysis")
fig1 = px.scatter(df, x='interest_rate(%)', y='sales(millions)',
                trendline='ols', size='interest_rate(%)',
                title='Sales vs Interest Rate')
st.plotly_chart(fig1, use_container_width=True)

# Regression Model
st.markdown("---")
st.header("3. Multiple Linear Regression")
st.write("**Model:** `sales(millions) ~ population(000) + interest_rate(%) + tv_ad`")

reg = smf.ols(formula='Q("sales(millions)") ~ Q("population(000)")+Q("interest_rate(%)")+tv_ad', 
            data=df).fit()

# Display regression results
st.subheader("Regression Summary")
with st.expander("View Full Regression Output", expanded=True):
    st.text(reg.summary())

# Key metrics in columns
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("R-squared", f"{reg.rsquared:.4f}")
with col2:
    st.metric("Adj. R-squared", f"{reg.rsquared_adj:.4f}")
with col3:
    st.metric("F-statistic", f"{reg.fvalue:.2f}")
with col4:
    st.metric("Prob (F-statistic)", f"{reg.f_pvalue:.4f}")

# Residual Analysis
st.markdown("---")
st.header("4. Diagnostic Plots")

# Create columns for side-by-side plots
col1, col2 = st.columns(2)

# Residuals vs Fitted Values
with col1:
    st.subheader("Residuals vs Fitted Values")
    sales_hat = reg.fittedvalues
    u_hat = reg.resid
    
    resid_df = pd.DataFrame({
        'sales_hat': sales_hat,
        'u_hat': u_hat
    })
    
    fig2 = px.scatter(resid_df, x='sales_hat', y='u_hat',
                      labels={
                          'sales_hat': 'Fitted Values',
                          'u_hat': 'Residuals'
                      },
                      title='Residuals vs Fitted Values')
    fig2.add_hline(y=0, line_dash='dash', line_color='crimson')
    st.plotly_chart(fig2, use_container_width=True)
    
    st.write("**Interpretation:** Points should be randomly scattered around the horizontal line at 0.")

# Q-Q Plot
with col2:
    st.subheader("Q-Q Plot (Normality Check)")
    fig3, ax = plt.subplots(figsize=(8, 6))
    stats.probplot(u_hat, dist='norm', plot=ax)
    ax.set_title('Q-Q Plot of Residuals')
    plt.tight_layout()
    st.pyplot(fig3)
    
    st.write("**Interpretation:** Points should follow the diagonal line if residuals are normally distributed.")

# VIF Analysis
st.markdown("---")
st.header("5. Multicollinearity Check (VIF)")

X = df[['interest_rate(%)', 'population(000)', 'tv_ad']]
vif_data = pd.DataFrame()
vif_data['Variables'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

st.dataframe(vif_data, use_container_width=True)
st.write("**Rule of thumb:** VIF < 5 indicates low multicollinearity, VIF > 10 suggests high multicollinearity.")

# Conclusions
st.markdown("---")
st.header("6. Conclusions")

st.write("""
### Key Findings:
- **Model Fit:** The R-squared value indicates how much variance in sales is explained by the model.
- **Residual Analysis:** Check if residuals are randomly distributed and normally distributed.
- **Multicollinearity:** VIF values suggest whether independent variables are highly correlated.

### Potential Improvements:
- Add interaction terms (e.g., `tv_ad * interest_rate`)
- Try non-linear transformations (log, polynomial)
- Include additional variables that might affect sales
- Consider functional form misspecification if R² is low despite good diagnostics
""")

st.markdown("---")
st.caption("Built with Streamlit 🎈 | Data generated using NumPy with seed=124")
