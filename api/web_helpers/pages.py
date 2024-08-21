import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html
from api.web_helpers.utils import create_experience_card, create_professional_timeline


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

def render_risk(df: pd.DataFrame) -> html.Div:
    return html.Div([
        dbc.Container([
            html.H1("What is risk and why do we care?", className="my-4"),
            
            html.Div([
                html.H3("What is Risk?"),
                html.P("In finance, risk refers to the possibility that an investment's actual return will differ from its expected return. "
                    "It represents the potential for loss or underperformance. Understanding and managing risk is what ultimately gives investors confidence to invest.")
            ], className="mb-4"),
            
            html.Div([
                html.H3("Volatility: A Measure of Risk"),
                html.P("Volatility is a statistical measure of the dispersion of returns for a given security or market index. "
                    "It quantifies the amount of uncertainty or risk associated with the size of changes in a security's value. "
                    "Higher volatility means that a security's value can potentially be spread out over a larger range of values -- high variance, "
                    "indicating higher risk.")
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Volatility Calculation"),
                    dbc.Input(id="ticker-input", placeholder="Enter stock ticker (e.g., AAPL)", type="text", className="mb-2"),
                    dbc.Button("Calculate Volatility", id="calc-volatility", color="primary", className="mb-2"),
                    html.Div(id="volatility-output")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="price-volatility-plot")
                ], width=6)
            ], className="mb-4"),
            
            html.Div([
                html.H3("Risk Exposure: Long vs Short Positions"),
                html.P("The direction of a trade (long or short) affects the risk exposure of an investor. "
                    "Let's examine how different positions impact potential gains and losses.")
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Long Position Simulation"),
                    dcc.Graph(id="long-position-plot")
                ], width=6),
                dbc.Col([
                    html.H4("Short Position Simulation"),
                    dcc.Graph(id="short-position-plot")
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id="trade-risk-profile")
                ], width=12)
            ]) 
        ])
    ])

def render_eq1(df: pd.DataFrame) -> html.Div:
    return None

