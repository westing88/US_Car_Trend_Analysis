
import streamlit as st
import pandas as pd
import plotly.express as px

import gdown

# Use your Google Drive file ID
file_id = "1jVv5775qiV95L2DVodSZ1mjz6oFwrIXo"
url = f"https://drive.google.com/uc?id={file_id}"

# Download it
gdown.download(url, "large_data.csv", quiet=False)

# Now read it as a local file
df = pd.read_csv("large_data.csv")


# Combine year columns into long format
year_data = df[['STATE', 'YEAR1', 'YEAR2', 'YEAR3', 'YEAR4']].melt(
    id_vars='STATE', value_name='YEAR').dropna()

year_data['YEAR'] = year_data['YEAR'].astype(int)

# Split data into two time periods
pre_2000 = year_data[(year_data['YEAR'] >= 1981) & (year_data['YEAR'] <= 2000)]
post_2000 = year_data[year_data['YEAR'] > 2000]

# Count by state
pre_2000_counts = pre_2000.groupby('STATE').size().reset_index(name='count')
post_2000_counts = post_2000.groupby('STATE').size().reset_index(name='count')

# Create heatmaps
fig_pre2000 = px.choropleth(
    pre_2000_counts,
    locations="STATE",
    locationmode="USA-states",
    color="count",
    scope="usa",
    color_continuous_scale="Greens",
    title="Car Purchases by State (1981–2000)"
)

fig_post2000 = px.choropleth(
    post_2000_counts,
    locations="STATE",
    locationmode="USA-states",
    color="count",
    scope="usa",
    color_continuous_scale="Blues",
    title="Car Purchases by State (2001–Present)"
)

# Streamlit layout
st.title("U.S. Car Purchases Heatmap: Then and Now")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_pre2000)

with col2:
    st.plotly_chart(fig_post2000)


# Load forecasted results

# Use your Google Drive file ID
file_id = "1rpvvtqjFkU48JQaXif24biixEJ1gOtNr"
url = f"https://drive.google.com/uc?id={file_id}"

# Download it
gdown.download(url, "forecasting_data.csv", quiet=False)

# Now read it as a local file
forecast_df = pd.read_csv("large_data.csv")

# Replace negative predictions with zero
forecast_df['PREDICTED'] = forecast_df['PREDICTED'].clip(lower=0)

# Dropdown to select state
selected_state = st.selectbox("Select a state to explore brand trends and forecasts:", forecast_df['STATE'].unique())

# Filter data by state
state_data = forecast_df[forecast_df['STATE'] == selected_state]

# Identify top 3 brands in this state by purchase count
top_brands = (
    state_data[state_data['PURCHASES'].notna()]
    .groupby("BRAND")['PURCHASES'].sum()
    .sort_values(ascending=False)
    .head(3)
    .index.tolist()
)

# Dropdown for a specific brand (optional)
selected_brand = st.selectbox("Select a brand for forecast comparison:", top_brands)

# Filter data for selected brand
brand_data = state_data[state_data['BRAND'] == selected_brand]

# Plot actual and forecast
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=brand_data['YEAR'], y=brand_data['PURCHASES'],
                         mode='lines+markers', name='Actual Purchases'))
fig.add_trace(go.Scatter(x=brand_data['YEAR'], y=brand_data['PREDICTED'],
                         mode='lines+markers', name='Forecasted'))

fig.update_layout(title=f"{selected_brand} Purchase Trend & Forecast in {selected_state}",
                  xaxis_title="Year", yaxis_title="Number of Cars")

st.plotly_chart(fig)
