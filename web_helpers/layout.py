import pandas as pd
from dash import dcc, html, dash_table

def export_layout(df:pd.DataFrame) -> html.Div:
    
    return html.Div([
        html.H1("Quantitative Strategies Dashboard: Hassan Seeking Alpha"),
        dcc.Tabs(id="tabs", children=[
            
            # Analysis Tab
            dcc.Tab(label='Analyze Time Series', children=[
                html.Div([
                    html.Div([
                        dcc.Checklist(
                            id='stock-checklist',
                            options=[{'label': i, 'value': i} for i in df['ID'].unique()],
                            value=[df['ID'].unique()[0]],
                            inline=True,
                            style={'padding': '5px', 'width': '100%', 'display': 'inline-block'}
                        )
                    ], style={'width': '30%', 'display': 'inline-block'}),
                    html.Div([
                        dcc.Checklist(
                            id='data-checklist',
                            options=[
                                {'label': 'Close Prices', 'value': 'CLOSE'},
                                {'label': 'Simple Moving Averages', 'value': 'SMA'},
                                {'label': 'Exp-Weighted Moving Averages', 'value': 'EWMA'},
                                {'label': 'Normalized Returns', 'value': 'RETURNS'}
                            ],
                            value=['CLOSE','RETURNS'],
                            inline=True,
                            style={'padding': '5px', 'width': '100%', 'display': 'inline-block'}
                        )
                    ], style={'width': '70%', 'display': 'inline-block'}),
                    html.Div([
                        dcc.Checklist(
                            id='ma-day-dropdown',
                            options=[{'label': f'{i} Days', 'value': f'{i}'} for i in [3, 9, 21, 50, 65, 120, 360]],
                            value=['21'],
                            inline=True,
                            style={'display': 'block'}
                        )
                    ], id='ma-selector', style={'display': 'none'}),
                    dcc.DatePickerRange(
                        id='date-range-selector',
                        start_date=df['DATE'].min(),
                        end_date=df['DATE'].max(),
                        display_format='YYYY-MM-DD'
                    ),
                ], style={'padding': '10px'}),
                html.Div([
                    dcc.Graph(id='stock-graph', style={'display': 'inline-block', 'width': '49%'}),
                    dcc.Graph(id='returns-graph', style={'display': 'inline-block', 'width': '49%'}),
                    dcc.Graph(id='volatility-graph', style={'display': 'inline-block', 'width': '49%'}),
                    dcc.Graph(id='roc-graph', style={'display': 'inline-block', 'width': '49%'}),
                ])
            ]),

            # Monte-Carlo Portfolio Simulation Tab
            dcc.Tab(label='Monte-Carlo Portfolio Simulation', children=[
                html.Div([
                    dcc.Dropdown(
                        id='stock-dropdown',
                        options=[{'label': i, 'value': i} for i in df.ID.unique()],  
                        value=['AAPL', 'NVDA'],
                        multi=True,
                        placeholder='Select stocks for your portfolio'
                    ),
                    html.Div([
                        html.Label('Enter Portfolio Weights (comma separated):'),
                        dcc.Input(
                            id='weights-input',
                            type='text',
                            value='0.5, 0.5',
                            style={'margin': '10px'}
                        )
                    ]),
                    html.Div([
                        html.Label('Enter Number of Days:'),
                        dcc.Input(
                            id='num-days-input',
                            type='number',
                            value=100,
                            style={'margin': '10px'}
                        )
                    ]),
                    html.Div([
                        html.Label('Initial Portfolio Value ($):'),
                        dcc.Input(
                            id='initial-portfolio-input',
                            type='number',
                            value=10000,
                            style={'margin': '10px'}
                        )
                    ]),
                    html.Button('Run Simulation', id='run-simulation-button'),
                    dcc.Graph(id='monte-carlo-simulation-graph'),
                    html.Hr(),
                    html.H3('Simulation Results'),
                    dash_table.DataTable(id='simulation-results-table')
                ])
            ]),

            # Backtesting with ML Models Tab
            dcc.Tab(label='Backtest with ML Models', children=[
                html.Div([
                    dcc.Input(id='stock-input', type='text', placeholder='Enter stock ID', value='NFLX'),
                    dcc.Input(id='start-date-input', type='text', placeholder='Enter start date (YYYY-MM-DD)', value='2020-03-01'),
                    dcc.Input(id='end-date-input', type='text', placeholder='Enter end date (YYYY-MM-DD)', value='2020-07-16'),
                    html.Button('Submit', id='submit-button', n_clicks=0),
                ]),
            ]),

            # Backtesting with Indictors Tab
            dcc.Tab(label='Backtest with Indictors', children=[
                html.Div([
                    dcc.Input(id='stock-input', type='text', placeholder='Enter stock ID', value='NFLX'),
                    dcc.Input(id='start-date-input', type='text', placeholder='Enter start date (YYYY-MM-DD)', value='2020-03-01'),
                    dcc.Input(id='end-date-input', type='text', placeholder='Enter end date (YYYY-MM-DD)', value='2020-07-16'),
                    html.Button('Submit', id='submit-button', n_clicks=0),
                ]),
                dcc.Graph(id='trades-graph'),
            ]),
        ]),
])