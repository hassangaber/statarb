import plotly.graph_objs as go
import pandas as pd
import numpy as np
import plotly.express as px

def filter_df(df, selected_id, start_date, end_date):
    return df[(df["ID"] == selected_id) & (df["DATE"].between(start_date, end_date))]

def create_trace(x, y, name, type='scatter', yaxis='y'):
    if type == 'scatter':
        return go.Scatter(x=x, y=y, mode='lines', name=name, yaxis=yaxis)
    elif type == 'bar':
        return go.Bar(x=x, y=y, name=name, yaxis=yaxis, opacity=0.3)
    
def calculate_mean_variance(df, window=252):
    """Calculate rolling mean and variance of returns"""
    returns = df['RETURNS']
    rolling_mean = returns.rolling(window=window).mean() * 252  # Annualized
    rolling_var = returns.rolling(window=window).var() * np.sqrt(252)  # Annualized std dev
    return pd.DataFrame({'mean': rolling_mean, 'std_dev': rolling_var}).dropna()

def create_layout(title, xaxis_title, yaxis_title, yaxis2_title=None, height=300):
    layout = go.Layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis=dict(title=yaxis_title, side='left'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    if yaxis2_title:
        layout.update(yaxis2=dict(title=yaxis2_title, side='right', overlaying='y'))
    return layout

def highlight_periods(df, column, threshold_percentile=95):
    threshold = df[column].quantile(threshold_percentile/100)
    highlighted_periods = df[df[column] > threshold]
    return [
        dict(
            type="rect",
            xref="x", yref="paper",
            x0=period.DATE, x1=period.DATE,
            y0=0, y1=1,
            fillcolor="rgba(255, 0, 0, 0.2)",
            layer="below",
            line_width=0,
        ) for _, period in highlighted_periods.iterrows()
    ]

def update_graph(selected_ids, start_date, end_date, df):
    graphs = {}
    
    for selected_id in selected_ids:
        filtered_df = filter_df(df, selected_id, start_date, end_date)
        
        # Price and Volume
        price_volume_traces = [
            create_trace(filtered_df["DATE"], filtered_df["CLOSE"], f"{selected_id} Close"),
            create_trace(filtered_df["DATE"], filtered_df["VOLUME"], f"{selected_id} Volume", 'bar', 'y2')
        ]
        graphs['price_volume'] = {
            'data': price_volume_traces,
            'layout': create_layout("Close & Volume", "Date", "Price", "Volume")
        }
        
        # Volatility
        volatility_trace = create_trace(filtered_df["DATE"], filtered_df["VOLATILITY_90D"], f"{selected_id} Volatility 90D")
        volatility_layout = create_layout("Volatility 90D", "Date", "Volatility")
        volatility_layout['shapes'] = highlight_periods(filtered_df, "VOLATILITY_90D")
        graphs['volatility'] = {
            'data': [volatility_trace],
            'layout': volatility_layout
        }
        
        # Rate of Change
        roc_trace = create_trace(filtered_df["DATE"], filtered_df["CLOSE_ROC_21D"], f"{selected_id} ROC")
        roc_layout = create_layout("Close Rate of Change", "Date", "ROC")
        roc_layout['shapes'] = highlight_periods(filtered_df, "CLOSE_ROC_21D")
        graphs['roc'] = {
            'data': [roc_trace],
            'layout': roc_layout
        }
        
        returns = filtered_df["RETURNS"].dropna()
        mean_return = returns.mean()
        variance_return = returns.var()
        
        returns_trace = go.Histogram(
            x=returns,
            name=f"{selected_id} Returns",
            opacity=0.7,
            xbins=dict(size=0.001)  # Adjust bin size as needed
        )
        
        returns_layout = go.Layout(
            title="Returns Distribution",
            xaxis=dict(title="Returns"),
            yaxis=dict(title="Frequency"),
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            annotations=[
                dict(
                    x=0.05,
                    y=0.95,
                    xref="paper",
                    yref="paper",
                    text=f"Mean Return: {mean_return:.4f}<br>Variance: {variance_return:.4f}",
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                    align="left"
                )
            ]
        )
        
        graphs['returns_dist'] = {
            'data': [returns_trace],
            'layout': returns_layout
        }
        
        # Mean-Variance Analysis
        mv_df = calculate_mean_variance(filtered_df)
        
        # Create a color scale based on dates
        date_range = pd.date_range(start=mv_df.index.min(), end=mv_df.index.max(), periods=8)
        color_scale = px.colors.sample_colorscale("Viridis", len(date_range))
        
        mv_trace = go.Scatter(
            x=mv_df['std_dev'],
            y=mv_df['mean'],
            mode='markers',
            name=f"{selected_id}",
            marker=dict(
                size=8,
                color=mv_df.index,
                colorscale=list(zip([d.timestamp() for d in date_range], color_scale)),
                showscale=True,
                colorbar=dict(
                    title="Date",
                    thickness=20,
                    len=0.5,
                    tickmode='array',
                    tickvals=[d.timestamp() for d in date_range],
                    ticktext=[d.strftime('%Y-%m-%d') for d in date_range],
                    tickangle=45
                )
            ),
            hovertemplate='Risk (Std Dev): %{x:.4f}<br>Return: %{y:.2f}%<extra></extra>'
        )
        
        # Add a diagonal line for the risk-return trade-off
        min_x, max_x = mv_df['std_dev'].min(), mv_df['std_dev'].max()
        min_y, max_y = mv_df['mean'].min(), mv_df['mean'].max()
        diagonal_trace = go.Scatter(
            x=[min_x, max_x],
            y=[min_y, max_y],
            mode='lines',
            name='Risk-Return Trade-off',
            line=dict(color='red', dash='dash'),
            hoverinfo='skip'
        )
        
        mv_layout = go.Layout(
            title="Mean-Variance Analysis",
            xaxis=dict(title="Risk (Annualized Standard Deviation)"),
            yaxis=dict(title="Annualized Return (%)", tickformat='.1f'),
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='closest',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        graphs['mean_variance'] = {
            'data': [mv_trace, diagonal_trace],
            'layout': mv_layout
        }
    
    return graphs
