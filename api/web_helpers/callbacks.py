
import dash
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, State
from plotly.subplots import make_subplots
from scipy import stats

from typing import Final

MACRO: Final[list[str]] = ['MACRO_INFLATION_EXPECTATION', 'MACRO_US_ECONOMY', 'MACRO_TREASURY_10Y', 'MACRO_TREASURY_5Y','MACRO_TREASURY_2Y','MACRO_VIX', 'MACRO_US_DOLLAR','MACRO_GOLD','MACRO_OIL']
FUNDAMENTALS: Final[list[str]] = ['HIGH', 'VOLUME', 'VOLATILITY_90D']
BETA: Final[list[str]] = ['BETA_TS']

ALL_FEATURES = MACRO + FUNDAMENTALS + BETA





def register_callbacks(app: dash.Dash, df: pd.DataFrame) -> None:



    @app.callback(
        Output('unified-graph', 'figure'),
        [Input('asset-dropdown', 'value'),
        Input('feature-dropdown', 'value')]
    )
    def update_macro_dataset_dashboard(selected_asset: str, selected_features: list[str]) -> go.Figure:
        if selected_asset not in df['ID'].unique():
            return go.Figure().add_annotation(text="Invalid asset selected", showarrow=False)
        
        asset_data = df[df['ID'] == selected_asset]
        
        if asset_data.empty:
            return go.Figure().add_annotation(text="No data available for selected asset", showarrow=False)

        required_columns = ['DATE', 'CLOSE', 'VOLUME']
        if not all(col in asset_data.columns for col in required_columns):
            return go.Figure().add_annotation(text="Missing required data columns", showarrow=False)

        asset_data['DATE'] = pd.to_datetime(asset_data['DATE'])
        asset_data = asset_data.sort_values('DATE')

        log_returns = np.diff(np.log(asset_data['CLOSE']))
        asset_data['Log_Return'] = pd.Series([np.nan] + log_returns.tolist(), index=asset_data.index)
        
        fig = make_subplots(rows=4, cols=1, 
                            vertical_spacing=0.1,  # Increased spacing between subplots
                            row_heights=[0.4, 0.2, 0.2, 0.3],  # Adjusted row heights
                            specs=[[{"secondary_y": True}],
                                [{"secondary_y": False}],
                                [{"secondary_y": False}],
                                [{"secondary_y": False}]],
                            subplot_titles=("Asset Price and Features", "Volume", "Log Returns Distribution", "Correlation Heatmap"))

        # Add asset price (close)
        fig.add_trace(go.Scatter(x=asset_data['DATE'], y=asset_data['CLOSE'],
                                name='Close Price', line=dict(color='#1f77b4', width=2)),
                    row=1, col=1, secondary_y=False)

        # Add volume bars
        fig.add_trace(go.Bar(x=asset_data['DATE'], y=asset_data['VOLUME'],
                            name='Volume', marker_color='rgba(31, 119, 180, 0.3)'),
                    row=2, col=1)

        # Add selected features
        for feature in selected_features:
            if feature not in asset_data.columns:
                print(f"Warning: {feature} not found in data")
                continue
            
            if feature in MACRO:
                color = '#ff7f0e'  # Orange for macro
                name = f'Macro: {feature}'
            elif feature in FUNDAMENTALS:
                color = '#2ca02c'  # Green for fundamental
                name = f'Fundamental: {feature}'
            else:
                color = '#9467bd'  # Purple for beta
                name = f'Beta: {feature}'
            
            fig.add_trace(go.Scatter(x=asset_data['DATE'], y=asset_data[feature],
                                    name=name, line=dict(color=color, width=1.5)),
                        row=1, col=1, secondary_y=True)

        # Add log returns distribution
        returns = asset_data['Log_Return'].dropna()
        mean_return = returns.mean()
        std_return = returns.std()
        
        iqr = stats.iqr(returns)
        bin_width = 2 * iqr / (len(returns) ** (1/3))
        num_bins = int((returns.max() - returns.min()) / bin_width)
        
        fig.add_trace(
            go.Histogram(x=returns, nbinsx=num_bins, name='Log Returns',
                        marker_color='rgba(31, 119, 180, 0.7)'),
            row=3, col=1
        )
        
        fig.add_vline(x=mean_return, line_dash="dash", line_color="red",
                    annotation_text=f"Mean: {mean_return:.4f}", 
                    annotation_position="top right",
                    row=3, col=1)

        # Add correlation heatmap
        corr_data = asset_data[['CLOSE'] + selected_features].corr()
        fig.add_trace(go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns,
            y=corr_data.columns,
            colorscale='RdBu_r',  # Red-White-Blue diverging colorscale
            zmin=-1, zmax=1,
            name='Correlation Heatmap',
            showscale=False  # This removes the color gradient key
        ), row=4, col=1)

        # Update layout
        fig.update_layout(
            title=f'{selected_asset} - Comprehensive Analysis',
            height=1600,  # Increased height to accommodate the larger spacing
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='rgba(240,240,240,0.5)',
            paper_bgcolor='white',
            font=dict(family="Arial", size=10),
        )

        # Update axes
        for i in range(1, 5):
            fig.update_xaxes(showgrid=True, gridcolor='white', linecolor='lightgrey', row=i, col=1)
            fig.update_yaxes(showgrid=True, gridcolor='white', linecolor='lightgrey', row=i, col=1)

        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Log Return", row=3, col=1)
        fig.update_yaxes(title_text="Close Price", secondary_y=False, row=1, col=1)
        fig.update_yaxes(title_text="Feature Value", secondary_y=True, row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=3, col=1)

        # Add range selector
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            row=1, col=1
        )

        # Clip the returns graph
        returns_range = 6 * std_return
        fig.update_xaxes(range=[mean_return - returns_range, mean_return + returns_range], row=3, col=1)

        return fig
    


    