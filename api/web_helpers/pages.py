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
def render_eda() -> html.Div:
    large_text ="""
                # Fetching Data and Creating Model Assumptions

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
        html.H1("Marco & Risk Data"),
        dcc.Tabs([
            dcc.Tab(label='Exploration', children=[
                dcc.Markdown(large_text, style={'padding': '20px'})
            ]),
            dcc.Tab(label='Notebook', children=[
                html.Iframe(
                srcDoc=notebook_to_html('/Users/hassan/Desktop/website/api/src/eda.ipynb'),
                style={'width': '85%', 'height': '1000px', 'border': 'none'}
            )
            ]),
            dcc.Tab(label='Some Interesting Plots', children=[
                html.Div(image_components, style={'maxWidth': '800px', 'margin': 'auto'})
            ]),
            
        ])
    ])


def render_eq1(df: pd.DataFrame) -> html.Div:
    return None

