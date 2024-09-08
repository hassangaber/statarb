import os

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dash_table, dcc, html

from api.web_helpers.utils import (create_experience_card,
                                   create_professional_timeline)

macro_columns = ['MACRO_INFLATION_EXPECTATION', 'MACRO_US_ECONOMY', 'MACRO_TREASURY_10Y', 'MACRO_TREASURY_5Y','MACRO_TREASURY_2Y','MACRO_VIX', 'MACRO_US_DOLLAR','MACRO_GOLD','MACRO_OIL']
fundamental_columns = ['HIGH', 'VOLUME', 'VOLATILITY_90D']
beta_columns = ['BETA_TS']

all_feature_columns = macro_columns + fundamental_columns + beta_columns


def render_intro():
    return html.Div([
        dcc.Store(id="theme-store", data={"primary-color": "#007bff"}),
        dbc.Container([
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H1("Hassan Gaber", className="text-primary text-center mb-3", style={"fontSize": "36px"}),
                        html.P(
                            "Showcasing small projects and professional journey. My interests are data mining, quantitative trading, and programming.",
                            className="lead text-center mb-3 small",
                        ),
                        html.P(
                            "Currently, the website explores topics in quantitative strategy formulation, starting with risk management and onto assets",
                            className="lead text-center mb-3 small",
                        ),
                        html.Div([
                            dbc.Button("Email", href="mailto:hassansameh90@gmail.com", color="primary", size="sm", className="me-1 mb-1"),
                            dbc.Button("Resume", href="https://drive.google.com/file/d/1wIPUDhL86DAmzJoxc_aPlWqWzqlwahU-/view?usp=sharing", color="primary", size="sm", className="me-1 mb-1"),
                            dbc.Button("LinkedIn", href="https://www.linkedin.com/in/hassansgaber/", color="primary", size="sm", className="me-1 mb-1"),
                            dbc.Button("GitHub", href="https://github.com/hassangaber", color="primary", size="sm", className="mb-1"),
                        ], className="d-flex justify-content-center flex-wrap mb-4"),
                    ], className="py-4"),
                    width=12,
                ),
            ]),
            dbc.Row([
                dbc.Col(
                    create_professional_timeline(),
                    width=12,
                ),
            ]),
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H2("Experience", className="text-primary mb-3"),
                        create_experience_card(
                            "Data Scientist Intern",
                            "PSP Investments",
                            "January 2023 – April 2024",
                            [
                                "Supported quantitative investing team managing $8.5B with predictive analytics.",
                                "Built a feature selector that significantly improved model performance, directly impacting trading strategies.",
                                "Implemented a CNN in PyTorch to predict earnings per share surprises across 2,000 stocks, deployed on AzureML.",
                            ]
                        ),
                        create_experience_card(
                            "Data Scientist Intern",
                            "National Bank of Canada",
                            "May 2022 – August 2022",
                            [
                                "Patented and created automatic data drift detection tool saving over $765k annually.",
                                "Developed drift detection framework with PyTorch, AWS Sagemaker, OpenCV.",
                                "Designed a feedback loop to incorporate user input into data labeling for continuously improving drift detection."
                            ]
                        ),
                        create_experience_card(
                            "Research Assistant - Data Scientist",
                            "McGill University Health Center",
                            "September 2021 – August 2022",
                            [
                                "Innovated data computing scheme to speed up website load times by 5-fold.",
                                "Developed visualization methods to find new gene relationships in chronic lung diseases.",
                                "Designed and implemented a testing framework for models and data pipelines, allowing researchers to incorporate custom datasets into the drug discovery platform."
                            ]
                        ),
                    ], className="py-4"),
                    width=12,
                ),
            ]),
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H2("Education", className="text-primary mb-3"),
                        create_experience_card(
                            "Bachelor of Engineering, Electrical Engineering",
                            "McGill University",
                            "September 2019 – April 2024",
                            [
                                "Took grad-level courses in statistical & deep learning, control theory, and microprocessors."
                            ] 
                        ),
                    ], className="py-4"),
                    width=12,
                ),
            ]),
        ], fluid=True, className="px-3"),
    ])

# section 1
def render_strat1(df:pd.DataFrame) -> html.Div:
    large_text ="""
                The dataset includes 9 macroeconomic features, 1 sector-beta feature, and 4 fundamentals (close price, high price, volume, and realized volatility). These features are the basis by which I make the hypotheses below. These features are related to the observation either by sector or by country (so AAPL observations would have the United States Inflation, technology sector beta time-series). The data displayed in the exploration tab is raw and not representitative of the stationary series given to the models.

                ---

                An important note regarding trading signals: due to the nature of the fundamental and beta features, they are observed at the end of the trading day, trading decisions and signals can only be generated at the end of the trading day. Otherwise, look-a-ahead bias will be introduced. Most likely, a signal will be generated using observations from time tick N and the signal will be applied on time tick N + 1.

                ---

                Trading hypotheses

                | H:          | Description | Signal Generation |
                |------------|-------------|-------------------|
                | 1:    | There's a non-linear relationship between inflation expectations and stock returns, particularly for certain sectors. | - When inflation expectations (measured by the TIP ETF) rise above a certain threshold, the model might generate a signal to underweight consumer discretionary stocks (like AMZN) and overweight companies with pricing power or inflation-protected revenues (like utilities, SOset).- If inflation expectations are rising but still below the threshold, the model might favor growth stocks (like NVDA) that can potentially outpace inflation. |
                | 2:    | The technology sector's beta increases during economic expansion and decreases during contraction. | - Use the US_Economy proxy (SPY) to determine the economic condition.- Compare the current beta of the technology sector (calculated from stocks like AAPL, NVDA) to its historical average.- Generate buy/sell signals based on the divergence of current beta from historical average, considering the economic condition. |
                | 3:    | Changes in interest rates (Treasury yields) lead changes in financial sector stock performance with a lag of 1-3 months. | - Monitor changes in the 10-Year Treasury Yield (^TNX). - Generate signals for financial sector stocks (like JPM) based on significant yield changes from 1-3 months ago. |
                | 4:    | There's a positive correlation between market volatility (VIX) and gold prices, but the relationship strengthens during periods of high uncertainty. | - Monitor the VIX index and gold prices (GLD).- Generate signals for gold-related assets when VIX spikes occur, especially if other uncertainty indicators (like economic policy uncertainty) are also elevated. |
                | 5:    | Different sectors outperform at different stages of the economic cycle. | - Use a combination of indicators (GDP growth, unemployment, inflation) to determine the current stage of the economic cycle.- Generate sector rotation signals based on the identified stage. |
                | 6:    | A strong US dollar negatively impacts the earnings of U.S. multinational companies with significant overseas revenues. | - Monitor the US Dollar Index (DX-Y.NYB).- Generate signals for U.S. multinationals (like AAPL, MSFT) when the dollar strength exceeds certain thresholds. |

                ---

                ## Relevant Literature
                1. Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. The Review of Financial Studies.
                - Utilized 94 firm characteristics and macro variables
                - Found neural networks outperform other ML methods in stock return prediction

                2. Kelly, B., Pruitt, S., & Su, Y. (2019). Characteristics Are Covariances: A Unified Model of Risk and Return. Journal of Financial Economics.
                - Developed IPCA (Instrumented Principal Components Analysis)
                - combined stock characteristics and macro factors

                3. Rapach, D. E., Ringgenberg, M. C., & Zhou, G. (2016). Short interest and aggregate stock returns. Journal of Financial Economics.
                - Demonstrated the predictive power of aggregate short interest for market returns

            """
    

    return html.Div([
        html.H1("Generating Trading Signals for Equities with Macro & Risk Data"),

        dcc.Tabs([
            dcc.Tab(label='Making the Dataset & Hypothesis', children=[
                html.Div(),
                html.H3("Below is the link to ingesting and manipulating the dataset into the master dataset:"),
                html.A('Jupyter Notebook On GitHub', href='https://github.com/hassangaber/statarb/blob/master/api/src/eda.ipynb', target='_blank', style={'font-size':'20px'}),
                html.Div(),
                dcc.Markdown(large_text, style={'padding': '10px','font-size':'20px'}), # explaining the hypothesis
            ]),

            dcc.Tab(label='Explore dataset', children=[
        html.Div([
            html.Div([
                html.Label('Select Asset:', style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id='asset-dropdown',
                    options=[{'label': i, 'value': i} for i in df['ID'].unique()],
                    value=df['ID'].unique()[0],
                    style={'width': '100%'}
                ),
            ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
            
            html.Div([
                html.Label('Select Features:', style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[
                        {'label': f'Macro: {i}', 'value': i} for i in macro_columns
                    ] + [
                        {'label': f'Fundamental: {i}', 'value': i} for i in fundamental_columns
                    ] + [
                        {'label': f'Beta: {i}', 'value': i} for i in beta_columns
                    ],
                    value=[macro_columns[0]],
                    multi=True,
                    style={'width': '100%'}
                ),
            ], style={'width': '65%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '5%'})
        ], style={'margin-bottom': '20px'}),
        
        dcc.Graph(id='unified-graph', style={'height': '120vh'}),
    ]),

            dcc.Tab(label='Processing & Triple-barrier Target', children=[
                
            ]),

            dcc.Tab(label='Some OOS Testing', children=[
                
            ]),
            
        ])
    ])

