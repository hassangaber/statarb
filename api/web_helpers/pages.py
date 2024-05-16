from dash import dcc, html, dash_table


def render_intro():
    return html.Div(
        [
            html.H1("Hassan Gaber", style={"color": "#A020F0"}),
            html.P(
                "Welcome to my personal project site where I explore automated quantitative trading strategies, "
                "demonstrating the process of strategy development and backtesting with the objective of maximizing excess returns.",
                style={"margin": "20px 0"},
            ),
            html.P(
                "I'm a new bachelor of engineering graduate with experience in data science looking to work in the Quantitative investments industry. Feel free to contact me for opportunities in data science, analytics, or quantitative development.",
                style={"margin": "20px 0"},
            ),
            html.Div(
                [
                    html.A("email", href="mailto:hassansameh90@gmail.com", target="_blank", style={"margin-right": "25px", "color": "#333"}),
                    html.A("cv", href="https://drive.google.com/file/d/1wIPUDhL86DAmzJoxc_aPlWqWzqlwahU-/view?usp=sharing", target="_blank", style={"margin-right": "25px", "color": "#333"}),
                    html.A("LinkedIn Profile", href="https://www.linkedin.com/in/hassansgaber/", target="_blank", style={"margin-right": "25px", "color": "#0077B5"}),
                    html.A("GitHub Profile", href="https://github.com/hassangaber", target="_blank", style={"color": "#333"}),
                ],
                style={"padding": "10px", "font-size": "36px"},
            ),
            html.Div(
                [
                    html.H4("Experience", style={"color": "#34495e"}),
                    html.P("Data Scientist Intern at PSP Investments (January 2023 – April 2024)", style={"font-weight": "bold"}),
                    html.Ul(
                        [
                            html.Li("Supported alpha generation team managing $8.5B with predictive analytics."),
                            html.Li("Developed feature selector improving trading returns in derivatives markets."),
                            html.Li("Wrote CNN to predict EPS surprise in PyTorch, deployed on AzureML."),
                        ],
                        style={"list-style-type": "square", "padding-left": "20px"},
                    ),
                    html.P("Data Scientist Intern at National Bank of Canada (May 2022 – August 2022)", style={"font-weight": "bold"}),
                    html.Ul(
                        [
                            html.Li("Patented and created automatic data drift detection tool saving over $765k annually."),
                            html.Li("Developed drift detection framework with PyTorch, AWS Sagemaker, OpenCV."),
                        ],
                        style={"list-style-type": "square", "padding-left": "20px"},
                    ),
                    html.P("Research Assistant - Data Scientist at McGill University Health Center (September 2021 – August 2022)", style={"font-weight": "bold"}),
                    html.Ul(
                        [
                            html.Li("Innovated data computing scheme to speed up website load times by 5-fold."),
                            html.Li("Developed visualization methods to find new gene relationships in chronic lung diseases."),
                        ],
                        style={"list-style-type": "square", "padding-left": "20px"},
                    ),
                ],
                style={"padding": "10px"},
            ),
            html.Div(
                [
                    html.H4("Education", style={"color": "#34495e"}),
                    html.P("Bachelor of Engineering, Electrical Engineering, McGill University (September 2019 – April 2024)")
                ],
                style={"padding": "10px", "font-weight": "bold"},
            ),
        ],
        style={"padding": "20px", "font-size": "30px"},
    )

def render_analyze(df):
    return html.Div(
        [
            html.Div(
                [
                    html.P(
                        "In this tab, all the raw market data can be viewed in the graph matrix representing the close price, stock returns frequency, volatility, and close price rate of change. \
                        The purpose of this section is to provide a tab where all the raw features and engineered features can be visualized. Multiple stocks can be viewed together \
                        and a date range can be selected for backtesting purposes.",
                        style={"font-size": "18px"},
                    ),
                    html.P("FEATURES:", style={"font-size": "16px"}),
                    html.Dl(f"{df.columns.to_list()[1:]}"),
                    html.Hr(),
                ]
            ),
            html.Div(
                [
                    dcc.Checklist(
                        id="stock-checklist",
                        options=[{"label": i, "value": i} for i in df["ID"].unique()],
                        value=[df["ID"].unique()[0]],
                        inline=True,
                        style={"padding": "5px", "width": "100%", "display": "inline-block"},
                    ),
                ],
                style={"width": "30%", "display": "inline-block"},
            ),
            html.Div(
                [
                    dcc.Checklist(
                        id="data-checklist",
                        options=[
                            {"label": "Close Prices", "value": "CLOSE"},
                            {"label": "Simple Moving Averages", "value": "SMA"},
                            {"label": "Exp-Weighted Moving Averages", "value": "EWMA"},
                            {"label": "Normalized Returns", "value": "RETURNS"},
                        ],
                        value=["CLOSE", "RETURNS"],
                        inline=True,
                        style={"padding": "5px", "width": "100%", "display": "inline-block"},
                    ),
                ],
                style={"width": "70%", "display": "inline-block"},
            ),
            html.Div(
                [
                    dcc.Checklist(
                        id="ma-day-dropdown",
                        options=[{"label": f"{i} Days", "value": f"{i}"} for i in [3, 9, 21, 50, 65, 120, 360]],
                        value=["21"],
                        inline=True,
                        style={"display": "block"},
                    ),
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
            html.Div(
                [
                    dcc.Graph(id="stock-graph", style={"display": "inline-block", "width": "49%"}),
                    dcc.Graph(id="returns-graph", style={"display": "inline-block", "width": "49%"}),
                    dcc.Graph(id="volatility-graph", style={"display": "inline-block", "width": "49%"}),
                    dcc.Graph(id="roc-graph", style={"display": "inline-block", "width": "49%"}),
                ]
            ),
        ],
        style={"padding": "10px"},
    )

def render_montecarlo(df):
    return html.Div(
        [
            dcc.Markdown(
                """
                ## Intro
                This tab performs Monte Carlo simulations to project future values of investment portfolios using historical data and statistical methods.
                By generating a range of possible outcomes for each asset within the portfolio, investors can visualize potential risks and returns over a specified time period.
                Key metrics such as Value at Risk (VaR), Conditional Value at Risk (CVaR), Sharpe Ratio, and Sortino Ratio are calculated and displayed: \n
                
                - **Value at Risk (VaR)**: Provides a threshold below which the portfolio value is unlikely to fall at a given confidence level, indicating the maximum expected loss under normal market conditions.\n
                - **Conditional Value at Risk (CVaR)**: Estimates the average loss exceeding the VaR, offering insight into potential losses in worst-case scenarios.\n
                - **Sharpe Ratio**: Measures the risk-adjusted return of the portfolio, with higher values indicating better risk-adjusted performance.\n
                - **Sortino Ratio**: Similar to the Sharpe Ratio but focuses only on downside risk, providing a more accurate measure of risk-adjusted performance when returns are not symmetrically distributed.\
                
                
                These metrics assist investors in making informed decisions about risk management, asset allocation, and potential adjustments to their investment strategies.

                ## Code and Params

                Monte Carlo simulation is a statistical method used to model the probability of different outcomes in a process that cannot easily be predicted due to the intervention of random variables. In the context of portfolio management, it is used to simulate the future returns of a portfolio by generating a wide range of possible outcomes based on historical data and statistical properties of asset returns.


                ### A. Define Parameters
                - **Number of Simulations (mc_sims)**: The number of simulated paths to generate.
                - **Time Horizon (T)**: The number of time periods (e.g., days) for each simulation.
                - **Portfolio Weights (weights)**: The allocation of the initial portfolio value across different assets.
                - **Mean Returns (meanReturns)**: The expected returns of the assets.
                - **Covariance Matrix (covMatrix)**: The covariance matrix of asset returns.
                - **Initial Portfolio Value (initial_portfolio)**: The starting value of the portfolio.

                ### B. Simulation Description
                The simulation uses the Cholesky decomposition of the covariance matrix to ensure that the generated random returns preserve the statistical properties of the historical data.

                - **Cholesky Decomposition (L)**: The covariance matrix is decomposed into a lower triangular matrix using Cholesky decomposition.
                - **Random Samples (Z)**: Generate random samples from a standard normal distribution.
                - **daily returns**: The daily returns are simulated by combining the mean returns with the random samples adjusted by the Cholesky matrix.
                - **portfolio values**: The portfolio values are calculated by iteratively applying the daily returns to the initial portfolio value.
                ```python
                L = np.linalg.cholesky(covMatrix)
                Z = np.random.normal(size=(T, len(weights)))
                dailyReturns = meanM + np.inner(L, Z)
                portfolio_values = np.cumprod(np.dot(weights, dailyReturns) + 1) * initial_portfolio
                ```

                """,
                style={"font-size": "18px"},
            ),
            html.Hr(),
            html.Div(
                [
                    dcc.Dropdown(
                        id="stock-dropdown",
                        options=[{"label": i, "value": i} for i in df.ID.unique()],
                        value=["AAPL", "NVDA"],
                        multi=True,
                        placeholder="Select stocks for your portfolio",
                    ),
                    html.Div(
                        [
                            html.Label("Enter Portfolio Weights (comma separated):"),
                            dcc.Input(id="weights-input", type="text", value="0.5, 0.5", style={"margin": "10px"}),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Enter Number of Days:"),
                            dcc.Input(id="num-days-input", type="number", value=100, style={"margin": "10px"}),
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
                            html.Label("Enter Number of Simulations:"),
                            dcc.Input(
                                id="num-simulations-input",
                                type="number",
                                value=2000,
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
                ],
                style={'display': 'flex', 'flex-direction': 'column', 'gap': '10px'}
            ),
            html.Hr(),
            dcc.Graph(id="monte-carlo-simulation-graph"),
            html.Hr(),
            dcc.Graph(id="distribution-graph"),
            html.H3("Simulation Results"),
            dash_table.DataTable(id="simulation-results-table"),
        ]
    )




def render_backtest_ml(df):
    return html.Div(
        [
            dcc.Markdown(
                """
                ## Intro
                Here you can backtest a trading strategy one stock at a time with variable initial investments and transaction share volume. The signals given also show the alpha of the portfolio. \
                I use a Naive fixed-time horizon target. The foundational model was trained on all data from 2010-01-01 to 2023-12-28. To generate real signals with no look-ahead bias, \
                please select a date after the last train observation date. Currently, the model is simple and the target is stochastic, which can be viewed in more detail in the Theory tab. \
                Backtesting allows us to evaluate the performance of a trading strategy using historical data. It helps determine if the strategy would have been profitable in the past, providing \
                    confidence that it might perform well in the future. A good PnL (Profit and Loss) from backtesting indicates that the strategy has potential for making money when applied to live \
                data and close prices. \
                This model is based on a neural network that tries to predict stochastic returns. Stochastic returns are inherently unpredictable, but the \
                model aims to identify patterns and generate signals that can be used to make trading decisions. By backtesting, we can see how well the model's predictions align with actual market movements.
                
                # Constructing the Target for Predicting Changes in Returns

                In supervised learning, labeling is necessary to train models to predict future changes in returns. This dataset class creates a target label for predicting changes in returns based on a dynamic threshold calculated from rolling volatility. The labels are classified into three categories:
                - **1**: Change in returns greater than the positive threshold.
                - **-1**: Change in returns less than the negative threshold.
                - **0**: Change in returns within the threshold range.

                ## Labeling Method

                ### Change in Returns Calculation
                The change in returns ($\Delta r_t$) over a specified horizon ($h$) is calculated as:

                $$ 
                \Delta r_t = r_t - r_{t-h} 
                $$

                where $r_t$ is the return at time $t$.

                ### Rolling Volatility Calculation
                Rolling volatility is computed as the square root of the sum of squared log returns over the horizon:

                $$
                V_t = \sqrt{\sum_{i=t-h+1}^{t} r_i^2} 
                $$

                ### Dynamic Threshold
                The threshold ($T$) is defined as a multiple of the rolling volatility:

                $$
                B_t = T \cdot V_t 
                $$
                

                The implementation in the `TimeSeriesDataset` class involves the following steps:

                1. **Initialize the Class**: Set the parameters and prepare the data.

                2. **Preprocess the Data**: Calculate changes in returns, rolling volatility, and assign labels.

                3. **Scale Features**: Normalize the features using `StandardScaler`.

                4. **Data Handling**: Implement methods to get the length of the dataset and retrieve individual data points.


                """,
                style={"font-size": "18px", "line-height": "1.6"}, mathjax=True
            ),
            html.Hr(),
            html.Div(
                [
                    html.H3("Stock and Date Selection"),
                    html.Div(
                        [
                            html.Label('Stock ID:'),
                        ],
                        style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'},
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="stock-id-input",
                                options=[{"label": i, "value": i} for i in df.ID.unique()],
                                value="AAPL",
                                multi=False,
                                placeholder="Select stock to backtest",
                                style={'width': '48%', 'margin-right': '4%'}
                            ),
                            html.Label('TRADING START: ', style={'margin-left': '5px'}),
                            dcc.Input(id="test-start-date-input", type="text", value="2024-01-02", style={'width': '48%'}),
                        ],
                        style={'display': 'flex', 'margin-bottom': '20px'},
                    ),
                    html.H3("Model Parameters"),
                    html.Div(
                        [
                            html.Label('Signal Model:'),
                        ],
                        style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'},
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="model-id-input",
                                options=[{"label": i, "value": i} for i in ['1_day_horizon_MLP', 'volatility_horizon_MLP']],
                                value="1_day_horizon_MLP",
                                multi=False,
                                placeholder="Select model to generate trading signal",
                                style={'width': '48%', 'margin-right': '2%'}
                            ),
                            html.Div(
                                [
                                    html.Label('Initial Investment:', style={'margin-left': '20px'}),
                                    dcc.Input(id="initial-investment-input", type="text", value="10000", style={'width': '48%'}),
                                ],
                                style={'width': '48%', 'margin-right': '2%'},
                            ),
                            html.Div(
                                [
                                    html.Label('Share Volume:', style={'margin-left': '20px'}),
                                    dcc.Input(id="share-volume-input", type="text", value="5", style={'width': '48%'}),
                                ],
                                style={'width': '48%', 'margin-right': '2%'},
                            )
                        ],
                        style={'display': 'flex', 'margin-bottom': '20px'},
                    ),
                    dcc.Store(id="stored-data"),  # Store for model data
                    html.Button("Run Model", id="run-model-button", style={'margin-bottom': '20px'}),
                    dcc.Graph(id="pnl-graph", config={'displayModeBar': False}),
                    dcc.Graph(id="transaction-signals-graph", config={'displayModeBar': False}),
                    dcc.Graph(id="portfolio-value-graph", config={'displayModeBar': False}),
                    dcc.Graph(id="cash-on-hand-graph", config={'displayModeBar': False}),
                    html.Div(id="stats-container"),
                    dcc.Graph(id="returns-distribution-graph", config={'displayModeBar': False}),
                    html.Div(id="table-container"),
                    html.A("Download CSV", id="download-link", download="portfolio_data.csv", href="", target="_blank"),
                ]
            ),
        ],
    )


# Further customization of the graphs and table can be added within the callbacks or the CSS.


def render_theory():
    return html.Div(
        [
            html.Img(src="/assets/theory.png", style={'height':'75%', 'width': '80%'}),
        ],
    )

def render_backtest_indicators(df):
    return html.Div(
        [
            dcc.Input(id="stock-input", type="text", placeholder="Enter stock ID", value="NFLX"),
            dcc.Input(id="start-date-input", type="text", placeholder="Enter start date (YYYY-MM-DD)", value="2020-03-01"),
            dcc.Input(id="end-date-input", type="text", placeholder="Enter end date (YYYY-MM-DD)", value="2020-07-16"),
            html.Button("Submit", id="submit-button", n_clicks=0),
            dcc.Graph(id="trades-graph"),
        ]
    )
