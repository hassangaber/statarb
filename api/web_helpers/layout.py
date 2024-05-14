import pandas as pd
from dash import dcc, html, dash_table
import dash_dangerously_set_inner_html


def export_layout(df: pd.DataFrame) -> html.Div:

    return html.Div(
        [
            html.H1("Quantitative Strategies Dashboard: Seeking Alpha"),
            dcc.Tabs(
                id="tabs",
                children=[
                    # Intro Tab
                    dcc.Tab(
                        label="Introduction",
                        children=[
                            html.Div(
                                [
                                    html.H1("Hassan Gaber", style={"color": "#A020F0"}),
                                    html.P(
                                        "Welcome to my personal project site where I explore automated quantitative trading strategies, "
                                        "demonstrating the process of strategy development and backtesting with the objective of maximizing excess returns.",
                                        style={"margin": "20px 0"},
                                    ),
                                    html.Div(
                                        [
                                            html.H4(
                                                "Experience", style={"color": "#34495e"}
                                            ),
                                            html.P(
                                                "Data Scientist Intern at PSP Investments (January 2023 – April 2024)",
                                                style={"font-weight": "bold"},
                                            ),
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        "Supported alpha generation team managing $8.5B with predictive analytics."
                                                    ),
                                                    html.Li(
                                                        "Developed feature selector improving trading returns in derivatives markets."
                                                    ),
                                                ],
                                                style={
                                                    "list-style-type": "square",
                                                    "padding-left": "20px",
                                                },
                                            ),
                                            html.P(
                                                "Data Scientist Intern at National Bank of Canada (May 2022 – August 2022)",
                                                style={"font-weight": "bold"},
                                            ),
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        "Patented and created automatic data drift detection tool saving over $765k annually."
                                                    ),
                                                    html.Li(
                                                        "Developed drift detection framework with PyTorch, AWS Sagemaker, OpenCV."
                                                    ),
                                                ],
                                                style={
                                                    "list-style-type": "square",
                                                    "padding-left": "20px",
                                                },
                                            ),
                                            html.P(
                                                "Research Assistant - Data Scientist at McGill University Health Center (September 2021 – August 2022)",
                                                style={"font-weight": "bold"},
                                            ),
                                            html.Ul(
                                                [
                                                    html.Li(
                                                        "Innovated data computing scheme to speed up website load times by 5-fold."
                                                    ),
                                                    html.Li(
                                                        "Developed visualization methods to find new gene relationships in chronic lung diseases."
                                                    ),
                                                ],
                                                style={
                                                    "list-style-type": "square",
                                                    "padding-left": "20px",
                                                },
                                            ),
                                        ],
                                        style={"padding": "10px"},
                                    ),
                                    html.Div(
                                        [
                                            html.H4(
                                                "Education", style={"color": "#34495e"}
                                            ),
                                            html.P(
                                                "Bachelor of Engineering, Electrical Engineering, McGill University (September 2019 – April 2024)"
                                            ),
                                        ],
                                        style={
                                            "padding": "10px",
                                            "font-weight": "bold",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.A(
                                                "LinkedIn Profile",
                                                href="https://www.linkedin.com/in/hassansgaber/",
                                                target="_blank",
                                                style={
                                                    "margin-right": "15px",
                                                    "color": "#0077B5",
                                                },
                                            ),
                                            html.A(
                                                "GitHub Profile",
                                                href="https://github.com/hassangaber",
                                                target="_blank",
                                                style={"color": "#333"},
                                            ),
                                        ],
                                        style={"padding": "10px", "font-size": "64px"},
                                    ),
                                ],
                                style={"padding": "20px", "font-size": "32px"},
                            ),
                        ],
                    ),
                    # Analysis Tab
                    dcc.Tab(
                        label="Analyze Time Series",
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.P(
                                                "In this tab, all the raw market data can be viewed in the graph matrix representing the close price, stock returns frequency, volatility, and close price rate of change. \
                                The purpose of this section is to provide a tab where all the raw features and engineered features can be visualized. Mutliple stocks can be viewed together \
                                and a date range can be selected for backtesting purposes.",
                                                style={"font-size": "26px"},
                                            ),
                                            html.P(
                                                f"FEATURES:",
                                                style={"font-size": "23px"},
                                            ),
                                            html.Dl(f"{df.columns.to_list()[1:]}"),
                                            html.Hr(),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            dcc.Checklist(
                                                id="stock-checklist",
                                                options=[
                                                    {"label": i, "value": i}
                                                    for i in df["ID"].unique()
                                                ],
                                                value=[df["ID"].unique()[0]],
                                                inline=True,
                                                style={
                                                    "padding": "5px",
                                                    "width": "100%",
                                                    "display": "inline-block",
                                                },
                                            )
                                        ],
                                        style={
                                            "width": "30%",
                                            "display": "inline-block",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            dcc.Checklist(
                                                id="data-checklist",
                                                options=[
                                                    {
                                                        "label": "Close Prices",
                                                        "value": "CLOSE",
                                                    },
                                                    {
                                                        "label": "Simple Moving Averages",
                                                        "value": "SMA",
                                                    },
                                                    {
                                                        "label": "Exp-Weighted Moving Averages",
                                                        "value": "EWMA",
                                                    },
                                                    {
                                                        "label": "Normalized Returns",
                                                        "value": "RETURNS",
                                                    },
                                                ],
                                                value=["CLOSE", "RETURNS"],
                                                inline=True,
                                                style={
                                                    "padding": "5px",
                                                    "width": "100%",
                                                    "display": "inline-block",
                                                },
                                            )
                                        ],
                                        style={
                                            "width": "70%",
                                            "display": "inline-block",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            dcc.Checklist(
                                                id="ma-day-dropdown",
                                                options=[
                                                    {
                                                        "label": f"{i} Days",
                                                        "value": f"{i}",
                                                    }
                                                    for i in [
                                                        3,
                                                        9,
                                                        21,
                                                        50,
                                                        65,
                                                        120,
                                                        360,
                                                    ]
                                                ],
                                                value=["21"],
                                                inline=True,
                                                style={"display": "block"},
                                            )
                                        ],
                                        id="ma-selector",
                                        style={"display": "none"},
                                    ),
                                    dcc.DatePickerRange(
                                        id="date-range-selector",
                                        start_date=df["DATE"].min(),
                                        end_date=df["DATE"].max(),
                                        display_format="YYYY-MM-DD",
                                    ),
                                ],
                                style={"padding": "10px"},
                            ),
                            html.Div(
                                [
                                    dcc.Graph(
                                        id="stock-graph",
                                        style={
                                            "display": "inline-block",
                                            "width": "49%",
                                        },
                                    ),
                                    dcc.Graph(
                                        id="returns-graph",
                                        style={
                                            "display": "inline-block",
                                            "width": "49%",
                                        },
                                    ),
                                    dcc.Graph(
                                        id="volatility-graph",
                                        style={
                                            "display": "inline-block",
                                            "width": "49%",
                                        },
                                    ),
                                    dcc.Graph(
                                        id="roc-graph",
                                        style={
                                            "display": "inline-block",
                                            "width": "49%",
                                        },
                                    ),
                                ]
                            ),
                        ],
                    ),
                    # Monte-Carlo Portfolio Simulation Tab
                    dcc.Tab(
                        label="Monte-Carlo Portfolio Simulation",
                        children=[
                            html.P(
                                "This tab runs simulations to project future values of investment portfolios based on historical data \
                         and statistical methods. By generating a range of possible outcomes for each asset within the portfolio, \
                       the tab helps investors visualize potential risks and returns over a specified time period. Key statistics such as \
                        Value at Risk (VaR) and Conditional Value at Risk (CVaR) are calculated and displayed. VaR provides a threshold below \
                       which the portfolio value is unlikely to fall at a given confidence level, indicating the maximum expected loss under \
                       normal market conditions. CVaR, on the other hand, estimates the average loss exceeding the VaR, offering insight into \
                       potential losses in worst-case scenarios. These metrics assist investors in making informed decisions about risk management,\
                        asset allocation, and potential adjustments to their investment strategies.",
                                style={"font-size": "24px"},
                            ),
                            html.Hr(),
                            html.Div(
                                [
                                    dcc.Dropdown(
                                        id="stock-dropdown",
                                        options=[
                                            {"label": i, "value": i}
                                            for i in df.ID.unique()
                                        ],
                                        value=["AAPL", "NVDA"],
                                        multi=True,
                                        placeholder="Select stocks for your portfolio",
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                "Enter Portfolio Weights (comma separated):"
                                            ),
                                            dcc.Input(
                                                id="weights-input",
                                                type="text",
                                                value="0.5, 0.5",
                                                style={"margin": "10px"},
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Enter Number of Days:"),
                                            dcc.Input(
                                                id="num-days-input",
                                                type="number",
                                                value=100,
                                                style={"margin": "10px"},
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Label("Initial Portfolio Value ($):"),
                                            dcc.Input(
                                                id="initial-portfolio-input",
                                                type="number",
                                                value=10000,
                                                style={"margin": "10px"},
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                "Enter Target Portfolio Value ($):"
                                            ),
                                            dcc.Input(
                                                id="target-value-input",
                                                type="number",
                                                value=12000,  # Default target value, adjust as needed
                                                style={"margin": "10px"},
                                            ),
                                            html.Button(
                                                "Calculate Optimal Metrics",
                                                id="calculate-metrics-button",
                                            ),
                                            html.Div(
                                                id="optimal-metrics-output"
                                            ),  # Placeholder to display results
                                        ]
                                    ),
                                    html.Button(
                                        "Run Simulation", id="run-simulation-button"
                                    ),
                                    dcc.Graph(id="monte-carlo-simulation-graph"),
                                    html.Hr(),
                                    dcc.Graph(id="distribution-graph"),
                                    html.H3("Simulation Results"),
                                    dash_table.DataTable(id="simulation-results-table"),
                                ]
                            ),
                        ],
                    ),
                    # Backtesting with ML Models Tab
                    dcc.Tab(
                        label="Backtest with ML Models",
                        children=[
                            html.Div([
                                html.H3("Stock and Date Selection"),
                                html.Div([
                                    html.Label('Stock ID: '),
                                    dcc.Input(id="stock-id-input", type="text", value="NVDA", style={'margin-right': '10px'}),
                                    html.Label('Test Start Date: (AFTER 2024-01-01)'),
                                    dcc.Input(id="test-start-date-input", type="text", value="2024-01-02", style={'margin-right': '10px'}),
                                ], style={'margin-bottom': '20px'}),

                                html.H3("Model Parameters"),
                                html.Div([
                                    html.Label('Initial Investment: '),
                                    dcc.Input(id="initial-investment-input", type="number", value=10000, style={'margin-right': '10px'}),
                                    html.Label('Share Volume: '),
                                    dcc.Input(id="share-volume-input", type="number", value=5),
                                ], style={'margin-bottom': '20px'}),
                                html.H3("Mathematical Explanation"),
                                dash_dangerously_set_inner_html.DangerouslySetInnerHTML('''
                                    <h4>Model and Loss Function</h4>
                                    <p>The model used is a neural network with the following structure:</p>
                                    <p>\\[
                                    \\begin{align*}
                                    \\text{Layer 1:} & \\quad \\text{Input} \\rightarrow \\text{ReLU}(W_1 \\cdot \\text{Input} + b_1) \\\\
                                    \\text{Layer 2:} & \\quad \\text{ReLU}(W_2 \\cdot \\text{Layer 1 Output} + b_2) \\\\
                                    \\text{Output Layer:} & \\quad \\sigma(W_3 \\cdot \\text{Layer 2 Output} + b_3)
                                    \\end{align*}
                                    \\]</p>
                                    <p>where \\(\\sigma\\) is the sigmoid activation function.</p>

                                    <h4>Volatility Weighted Loss Function</h4>
                                    <p>The custom loss function is defined as:</p>
                                    <p>\\[
                                    L = \\text{BCE}(\\hat{y}, y) + \\lambda \\cdot \\text{Mean}((\\hat{y} - y) \\cdot \\text{volatility})
                                    \\]</p>
                                    <p>where:</p>
                                    <ul>
                                        <li>\\(\\text{BCE}(\\hat{y}, y)\\) is the binary cross-entropy loss between predictions \\(\\hat{y}\\) and targets \\(y\\).</li>
                                        <li>\\(\\lambda\\) is a weighting factor.</li>
                                        <li>The second term penalizes large deviations between predictions and targets, scaled by volatility.</li>
                                    </ul>

                                    <h4>Price Over Time</h4>
                                    <p>The price of the stock over time is represented as:</p>
                                    <p>\\[
                                    P_t = P_{t-1} + \\Delta P
                                    \\]</p>
                                    <p>where \\(\\Delta P\\) is the change in price at time \\(t\\).</p>

                                    <h4>Target Definition</h4>
                                    <p>The target variable is defined as:</p>
                                    <p>\\[
                                    y_t = 
                                    \\begin{cases} 
                                    1 & \\text{if } R_t > 0 \\\\
                                    0 & \\text{if } R_t \\leq 0 
                                    \\end{cases}
                                    \\]</p>
                                    <p>where \\(R_t\\) is the return at time \\(t\\).</p>
                                '''),
                                dcc.Store(id="stored-data"),  # Store for model data
                                html.Button("Run Model", id="run-model-button", style={'margin-bottom': '20px'}),
                                dcc.Graph(id="portfolio-value-graph"),
                                dcc.Graph(id="transaction-signals-graph"),
                                html.Div(id="table-container"),
                                html.A("Download CSV", id="download-link", download="portfolio_data.csv", href="", target="_blank")
                            ])
                        ],
                    ),
                    # Backtesting with Indictors Tab
                    dcc.Tab(
                        label="Backtest with Indictors",
                        children=[
                            html.Div(
                                [
                                    dcc.Input(
                                        id="stock-input",
                                        type="text",
                                        placeholder="Enter stock ID",
                                        value="NFLX",
                                    ),
                                    dcc.Input(
                                        id="start-date-input",
                                        type="text",
                                        placeholder="Enter start date (YYYY-MM-DD)",
                                        value="2020-03-01",
                                    ),
                                    dcc.Input(
                                        id="end-date-input",
                                        type="text",
                                        placeholder="Enter end date (YYYY-MM-DD)",
                                        value="2020-07-16",
                                    ),
                                    html.Button(
                                        "Submit", id="submit-button", n_clicks=0
                                    ),
                                ]
                            ),
                            dcc.Graph(id="trades-graph"),
                        ],
                    ),
                ],
            ),
        ]
    )
