import pandas as pd
import plotly.graph_objs as go
from dash import html, dcc
from dash.dependencies import Input, Output

def filter_df(df, selected_id, start_date, end_date):
    return df[(df["ID"] == selected_id) & (df["DATE"].between(start_date, end_date))]

def create_trace(x, y, name, type='scatter', yaxis='y'):
    if type == 'scatter':
        return go.Scatter(x=x, y=y, mode='lines', name=name, yaxis=yaxis)
    elif type == 'bar':
        return go.Bar(x=x, y=y, name=name, yaxis=yaxis, opacity=0.3)

def create_layout(title, xaxis_title, yaxis_title, yaxis2_title=None):
    layout = go.Layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis=dict(title=yaxis_title, side='left'),
        legend=dict(x=1.05, y=1, xanchor='left', yanchor='top')
    )
    if yaxis2_title:
        layout.update(yaxis2=dict(title=yaxis2_title, side='right', overlaying='y'))
    return layout

def update_graph(selected_ids, selected_data, start_date, end_date, df):
    traces = {
        'stock': [], 'returns': [], 'volatility': [], 'roc': []
    }
    
    for selected_id in selected_ids:
        filtered_df = filter_df(df, selected_id, start_date, end_date)
        
        if 'CLOSE' in selected_data:
            traces['stock'].append(create_trace(filtered_df["DATE"], filtered_df["CLOSE"], f"{selected_id} Close"))
            traces['stock'].append(create_trace(filtered_df["DATE"], filtered_df["VOLUME"], f"{selected_id} Volume", 'bar', 'y2'))
        
        traces['returns'].append(go.Histogram(x=filtered_df["RETURNS"], name=f"{selected_id} Returns"))
        traces['volatility'].append(create_trace(filtered_df["DATE"], filtered_df["VOLATILITY_90D"], f"{selected_id} Volatility 90D"))
        traces['roc'].append(create_trace(filtered_df["DATE"], filtered_df["CLOSE_ROC_21D"], f"{selected_id} ROC"))

    return (
        {'data': traces['stock'], 'layout': create_layout("Stock Data", "Date", "Price", "Volume")},
        {'data': traces['returns'], 'layout': create_layout("Returns Distribution", "Returns", "Frequency")},
        {'data': traces['volatility'], 'layout': create_layout("Volatility 90D", "Date", "Volatility")},
        {'data': traces['roc'], 'layout': create_layout("Rate of Change", "Date", "ROC")}
    )
