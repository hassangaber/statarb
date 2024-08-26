
import pandas as pd


import dash
from dash import callback_context, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np



macro_columns = ['MACRO_INFLATION_EXPECTATION', 'MACRO_US_ECONOMY', 'MACRO_TREASURY_10Y', 'MACRO_TREASURY_5Y','MACRO_TREASURY_2Y','MACRO_VIX', 'MACRO_US_DOLLAR','MACRO_GOLD','MACRO_OIL']
fundamental_columns = ['HIGH', 'VOLUME', 'VOLATILITY_90D']
beta_columns = ['BETA_TS']

all_feature_columns = macro_columns + fundamental_columns + beta_columns





def register_callbacks(app: dash.Dash, df: pd.DataFrame) -> None:

    @app.callback(
        Output('unified-graph', 'figure'),
        [Input('asset-dropdown', 'value'),
        Input('feature-dropdown', 'value')]
    )
    def update_graph(selected_asset, selected_features):
        # Error checking
        if selected_asset not in df['ID'].unique():
            return go.Figure().add_annotation(text="Invalid asset selected", showarrow=False)
        
        asset_data = df[df['ID'] == selected_asset]
        
        if asset_data.empty:
            return go.Figure().add_annotation(text="No data available for selected asset", showarrow=False)

        # Ensure all required columns are present
        required_columns = ['DATE', 'CLOSE', 'VOLUME']
        if not all(col in asset_data.columns for col in required_columns):
            return go.Figure().add_annotation(text="Missing required data columns", showarrow=False)

        # Convert DATE to datetime if it's not already
        asset_data['DATE'] = pd.to_datetime(asset_data['DATE'])

        # Sort data by date
        asset_data = asset_data.sort_values('DATE')

        # Create subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # Add asset price (close)
        fig.add_trace(go.Scatter(x=asset_data['DATE'], y=asset_data['CLOSE'],
                                name='Close Price', line=dict(color='blue')),
                    row=1, col=1)

        # # Add asset price (high)
        # fig.add_trace(go.Scatter(x=asset_data['DATE'], y=asset_data['HIGH'],
        #                         name='High Price', line=dict(color='lightblue', dash='dot')),
        #             row=1, col=1)

        # Add volume bars
        fig.add_trace(go.Bar(x=asset_data['DATE'], y=asset_data['VOLUME'],
                            name='Volume', marker_color='rgba(0, 0, 255, 0.3)'),
                    row=2, col=1)

        # Add selected features
        for feature in selected_features:
            if feature not in asset_data.columns:
                print(f"Warning: {feature} not found in data")
                continue
            
            if feature in macro_columns:
                name = f'Macro: {feature}'
                color = 'rgba(255, 0, 0, 0.5)'  # Red for macro
            elif feature in fundamental_columns:
                name = f'Fundamental: {feature}'
                color = 'rgba(0, 255, 0, 0.5)'  # Green for fundamental
            else:
                name = f'Beta: {feature}'
                color = 'rgba(128, 0, 128, 0.5)'  # Purple for beta
            
            fig.add_trace(go.Scatter(x=asset_data['DATE'], y=asset_data[feature],
                                    name=name, yaxis='y3', line=dict(color=color)),
                        row=1, col=1)

        # Add significant events
        events = [
                ('2020-03-23', 'COVID-19 Market Bottom'),
                #('2021-01-06', 'US Capitol Riot'),
                #('2022-02-24', 'Russia-Ukraine War Begins'),
                ('2021-02-08', 'Bitcoin Exceeds $44,000'),
                #('2021-05-12', 'Colonial Pipeline Cyberattack'),
                ('2022-07-27', 'Fed Hike by 75 bps'),
                ('2022-09-15', 'Ethereum Merge'),
                ('2020-08-31', 'Apple,Tesla Splits'),
                #('2021-10-28', 'Facebook Rebrands as Meta'),
                #('2022-04-25', 'Elon Musk Acquires Twitter'),
                #('2022-11-11', 'FTX Cryptocurrency Exchange Collapse'),
                ('2022-11-30', 'OpenAI ChatGPT'),
                #('2023-03-14', 'SVB Collapse'),
                ('2022-05-04', 'Fed Largest Rate Hike Since 2000'),
                #('2022-08-16', 'Inflation Reduction Act Signed'),
                #('2023-03-10', 'Signature Bank Collapse'),
                #('2023-05-01', 'First Republic Bank Fails'),
                #('2023-06-29', 'Supreme Court Strikes Down Student Loan Forgiveness'),
                ('2023-07-13', 'Inflation Falls in US'),
                #('2023-11-08', 'Bitcoin Surpasses $37,000'),
                ('2024-01-10', 'SEC Approves Spot Bitcoin ETFs'),
                ('2023-11-17', 'NVIDIA Hits $1 Trillion Market Cap'),
                #('2023-11-06', 'OpenAI Launches GPT-4 Turbo')
            ]

        for date, event in events:
            event_date = pd.to_datetime(date)
            if event_date in asset_data['DATE'].values:
                y_value = asset_data.loc[asset_data['DATE'] == event_date, 'HIGH'].values[0]
                fig.add_annotation(x=event_date, y=y_value,
                                text=event, showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                                arrowcolor='rgba(255, 0, 0, 0.5)', ax=0, ay=-40)

        # Update layout for a more professional look
        fig.update_layout(
            title=f'{selected_asset}',
            height=950,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12),
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=True,
            legend_title_text='Indicators',
            legend=dict(x=1.05, y=1, bordercolor='Black', borderwidth=1)
        )

        # Update y-axes
        fig.update_yaxes(title_text="Price", secondary_y=False, row=1, col=1)
        fig.update_yaxes(title_text="Feature Value", secondary_y=True, row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        # Update x-axis to show date range only once
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(showticklabels=False, row=1, col=1)  # Hide x-axis labels for the top subplot

        # Add range selector to bottom subplot
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
            row=2, col=1
        )

        # Add gridlines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', zerolinecolor='LightGrey')

        return fig