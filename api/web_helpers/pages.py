import os
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html
from api.web_helpers.utils import create_experience_card, create_professional_timeline, get_encoded_image
from api.web_helpers.callback.rintro import notebook_to_html


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
def render_strat1() -> html.Div:
    large_text ="""
                # Macro-Driven Stock Performance Prediction

                ## Hypothesis
                - Macroeconomic factors and sector-specific risk (beta) influence individual stock returns
                - Granular hypotheses to test:
                1. Correlation (consider non-linear) strength of specific macro indicators with stock returns
                2. Sector beta impact during varying economic conditions
                3. Lag effects between macro changes and stock price directions/momentum
                4. Volatility-macro indicator relationships

                ## Dataset Overview
                - 16 stocks: price, volume, volatility (2020-01-01 to 2024-05-03)
                - 13 macro indicators (e.g., inflation, unemployment, Treasury yields)
                - 11 sector ETFs for beta calculation

                ## Data Structure
                - `equity_data_with_macro`:
                - Stock features: ID, DATE, CLOSE, HIGH, VOLUME, VOLATILITY_90D
                - Macro indicators: MACRO_*
                - BETA_TS: 6-month rolling window

                ## Key Assumptions
                1. End-of-day trading only (due to data frequency)
                2. Proper macro-stock data alignment
                3. Some market inefficiency exists
                4. Potential non-stationarity in time series
                5. Possible non-linear relationships
                6. VOLATILITY_90D assumes gradual volatility changes

                ## Statistical Considerations
                - Autocorrelation in time series/Heteroscedasticity
                - Multicollinearity among macro indicators
                - Non-normal return distributions

                ## Relevant Literature
                1. Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. The Review of Financial Studies.
                - Utilized 94 firm characteristics and macro variables
                - Found neural networks outperform other ML methods in stock return prediction

                2. Kelly, B., Pruitt, S., & Su, Y. (2019). Characteristics Are Covariances: A Unified Model of Risk and Return. Journal of Financial Economics.
                - Developed IPCA (Instrumented Principal Components Analysis)
                - combined stock characteristics and macro factors

                3. Rapach, D. E., Ringgenberg, M. C., & Zhou, G. (2016). Short interest and aggregate stock returns. Journal of Financial Economics.
                - Demonstrated the predictive power of aggregate short interest for market returns
                

                ## Execution Plan

                ### Current Status
                - dataset combining stock-specific features, macro indicators, and sector betas
                - Data aligned and preprocessed for analysis (2020-01-01 to 2024-05-03)
                - Hypotheses formulated linking macro factors to stock performance

                ### Objective
                - Develop a model to generate excess returns using the triple barrier method

                ### Triple Barrier Method Implementation
                1. Label Generation:
                - Use CLOSE prices to calculate (log) returns
                - Define upper and lower return thresholds (e.g., +/-5%)
                - Set a maximum holding period (e.g., 10 trading days)
                - Label each instance as 1 (upper barrier hit), -1 (lower barrier hit), or 0 (time barrier hit)

                2. Feature Engineering:
                - Utilize MACRO_* indicators as primary features
                - Incorporate BETA_TS for sector-specific risk
                - Consider lagged features to capture delayed market reactions
                - Create interaction terms between macro indicators and BETA_TS

                3. Model Development:
                - Train classification models (e.g., Random Forest, XGBoost)
                - Use VOLATILITY_90D for sample weighting to address heteroscedasticity
                - Implement cross-validation with purged K-fold to prevent look-ahead bias

                4. Performance Evaluation:
                - Sharpe ratio, maximum drawdown, and hit rate
                - Compare against buy-and-hold and sector ETF benchmarks

                ### Rationale for Dataset and Method
                1. Macro-Micro Integration:
                - Dataset uniquely combines firm-specific, sector, and macro-level data
                - Allows for capturing complex interactions across different economic scales

                2. Time Series Nature:
                - Daily frequency captures short-term market reactions to macro changes
                - BETA_TS provides dynamic risk assessment

                3. Triple Barrier Relevance:
                - Symmetric approach to defining trading opportunities
                - Aligns with real-world trading constraints (stop-loss, take-profit, time limits)

                4. Volatility Incorporation:
                - VOLATILITY_90D enables risk-adjusted predictions and position sizing

            """
    
    image_paths = [
    '/Users/hassan/Desktop/website/assets/Screen Shot 2024-08-22 at 12.24.46 AM.png',
    '/Users/hassan/Desktop/website/assets/Screen Shot 2024-08-22 at 12.24.58 AM.png',
    ]
    
    image_components = [
        html.Div([
            html.Img(src=get_encoded_image(path), style={'width': '100%', 'marginBottom': '20px'})
        ]) for path in image_paths if os.path.exists(path)
    ]

    return html.Div([
        html.H1("Generating Trading Signals for Equities with Macro & Risk Data"),

        dcc.Tabs([
            dcc.Tab(label='Making the Dataset & Hypothesis', children=[
                dcc.Markdown(large_text, style={'padding': '20px'}), # explaining the hypothesis
                html.Iframe(
                srcDoc=notebook_to_html('/Users/hassan/Desktop/website/api/src/eda.ipynb'),
                style={'width': '85%', 'height': '1000px', 'border': 'none'}
                ),
                # html.Div(image_components, style={'maxWidth': '800px', 'margin': 'auto'})
            ]),

            dcc.Tab(label='Processing & Triple-barrier Target', children=[
                
            ]),

            dcc.Tab(label='Some OOS Testing', children=[
                
            ]),
            
        ])
    ])


def render_eq1(df: pd.DataFrame) -> html.Div:
    return None

