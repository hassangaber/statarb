import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import random
import numpy as np
from scipy import stats

from compute import compute_signal

# Load and prepare the data
df = pd.read_csv('assets/data.csv')
df['DATE'] = pd.to_datetime(df['DATE'])

# Initialize the Dash application
app = dash.Dash(__name__)

def random_color() -> str:
    """Generates a random color in RGBA format for Plotly graphs."""
    return f'rgba({random.randint(0, 55)}, {random.randint(0, 30)}, {random.randint(0, 100)}, 0.8)'

def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Compute the kernel density estimate on a grid of x values."""
    kde = stats.gaussian_kde(x, bw_method=bandwidth, **kwargs)
    return kde.evaluate(x_grid)

# App layout
app.layout = html.Div([
    html.H1("Statistical Arbitrage Dashboard"),
    dcc.Tabs(id="tabs", children=[
        # Analysis Tab
        dcc.Tab(label='Analysis', children=[
            dcc.Checklist(
                id='stock-checklist',
                options=[{'label': i, 'value': i} for i in df['ID'].unique()],
                value=[df['ID'].unique()[0]],
                labelStyle={'display': 'inline-block'},
                inline=True
            ),
            dcc.Checklist(
                id='data-checklist',
                options=[
                    {'label': 'Close Prices', 'value': 'CLOSE'},
                    {'label': 'Moving Averages (SMA)', 'value': 'SMA'},
                    {'label': 'Moving Averages (EWMA)', 'value': 'EWMA'},
                    {'label': 'Rate of Change', 'value': 'ROC'},
                    {'label': 'Volatility', 'value': 'VOLATILITY'},
                    {'label': 'Returns', 'value': 'RETURNS'}
                ],
                value=['CLOSE'],
                labelStyle={'display': 'inline-block'}
            ),
            html.Div([
                dcc.Checklist(
                    id='ma-day-dropdown',
                    options=[{'label': f'{i} Days', 'value': f'{i}'} for i in [3, 9, 21, 50, 65, 120, 360]],
                    value=['50'],
                    labelStyle={'display': 'block'},
                    inputStyle={"margin-right": "5px"},
                    inline=True,
                    style={'display': 'none'}
                )
            ], id='ma-selector', style={'display': 'none'}),
            dcc.Graph(id='stock-graph', style={'height': '750px'}),
            dcc.Graph(id='returns-graph', style={'height': '750px'})
        ]),
        # Simulation Tab
        dcc.Tab(label='Simulation', children=[
            html.Div([
                html.H3('Simulation Space')
                # Placeholder for future simulation controls and outputs
            ])
        ]),
        # Trades Tab
        dcc.Tab(label='Trades', children=[
            html.Div([
                dcc.Input(id='stock-input', type='text', placeholder='Enter stock ID', value='NFLX'),
                dcc.Input(id='start-date-input', type='text', placeholder='Enter start date (YYYY-MM-DD)', value='2020-03-01'),
                dcc.Input(id='end-date-input', type='text', placeholder='Enter end date (YYYY-MM-DD)', value='2020-07-16'),
                html.Button('Submit', id='submit-button', n_clicks=0),
            ]),
            dcc.Graph(id='trades-graph'),
            dcc.Graph(id='trades-returns-graph'),
            html.Div(id='output-container', style={'white-space': 'pre-line'})
        ]),
    ])
])

@app.callback(
    Output('ma-selector', 'style'),
    [Input('data-checklist', 'value')]
)
def toggle_ma_selector(selected_data):
    if 'SMA' in selected_data or 'EWMA' in selected_data:
        return {'display': 'block'}
    else:
        return {'display': 'none'}

@app.callback(
    [Output('stock-graph', 'figure'),
     Output('returns-graph', 'figure')],
    [Input('stock-checklist', 'value'),
     Input('data-checklist', 'value'),
     Input('ma-day-dropdown', 'value')]
)
def update_graph(selected_ids, selected_data, selected_days):
    traces_main = []
    traces_returns = []
    for selected_id in selected_ids:
        filtered_df = df[df['ID'] == selected_id]
        
        # Add volume bars and close prices to the main graph
        if 'CLOSE' in selected_data:
            traces_main.append(go.Bar(
                x=filtered_df['DATE'],
                y=filtered_df['VOLUME'],
                name=f'{selected_id} Volume',
                marker=dict(color=random_color()),
                opacity=0.6,
                yaxis='y2'
            ))
            traces_main.append(go.Scatter(
                x=filtered_df['DATE'],
                y=filtered_df['CLOSE'],
                mode='lines',
                name=f'{selected_id} Close',
                line=dict(color=random_color())
            ))

        # Add SMA, EWMA, ROC, and Volatility traces based on user selection
        for day in selected_days:
            if 'SMA' in selected_data:
                sma_column = f'CLOSE_SMA_{day}D'
                if sma_column in filtered_df.columns:
                    traces_main.append(go.Scatter(
                        x=filtered_df['DATE'],
                        y=filtered_df[sma_column],
                        mode='lines',
                        name=f'{selected_id} SMA {day}D',
                        line=dict(color=random_color())
                    ))
            if 'EWMA' in selected_data:
                ewma_column = f'CLOSE_EWMA_{day}D'
                if ewma_column in filtered_df.columns:
                    traces_main.append(go.Scatter(
                        x=filtered_df['DATE'],
                        y=filtered_df[ewma_column],
                        mode='lines',
                        name=f'{selected_id} EWMA {day}D',
                        line=dict(color=random_color())
                    ))

        if 'ROC' in selected_data:
            roc_column = f'CLOSE_ROC_{day}D'
            if roc_column in filtered_df.columns:
                traces_main.append(go.Scatter(
                    x=filtered_df['DATE'],
                    y=filtered_df[roc_column],
                    mode='lines',
                    name=f'{selected_id} ROC {day}D',
                    line=dict(color=random_color())
                ))

        if 'VOLATILITY' in selected_data:
            traces_main.append(go.Scatter(
                x=filtered_df['DATE'],
                y=filtered_df['VOLATILITY_90D'],
                mode='lines',
                name=f'{selected_id} Volatility 90D',
                line=dict(color=random_color())
            ))

        # Histogram for Returns as a frequency distribution
        if 'RETURNS' in selected_data:
            traces_returns.append(go.Histogram(
                x=filtered_df['RETURNS'],
                name=f'{selected_id} Returns Frequency',
                marker=dict(color=random_color()),
                opacity=0.75,
                xbins=dict(
                    start=np.min(filtered_df['RETURNS']),
                    end=np.max(filtered_df['RETURNS']),
                    size=(np.max(filtered_df['RETURNS']) - np.min(filtered_df['RETURNS'])) / 40
                ),
                autobinx=False
            ))

            x_grid = np.linspace(np.min(filtered_df['RETURNS']), np.max(filtered_df['RETURNS']), 1000)
            pdf = kde_scipy(filtered_df['RETURNS'].dropna(), x_grid, bandwidth=0.2)
            
            # Adding KDE as a Scatter plot
            traces_returns.append(go.Scatter(
                x=x_grid,
                y=pdf,
                mode='lines',
                name=f'{selected_id} Returns KDE',
                line=dict(color=random_color(), width=2)
            ))

    return ({
        'data': traces_main,
        'layout': go.Layout(
            title='Stock Data',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Price and Volume'),
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right',
                showgrid=False  # Hide grid lines for volume for cleaner look
            )
        )
    }, {
        'data': traces_returns,
        'layout': go.Layout(
            title='Returns Frequency and KDE',
            xaxis=dict(title='Returns'),
            yaxis=dict(title='Frequency / Density')
        )
    })



@app.callback(
    Output('trades-graph', 'figure'),
    [Input('submit-button', 'n_clicks')],
    [State('stock-input', 'value'),
     State('start-date-input', 'value'),
     State('end-date-input', 'value')]
)
def update_trades_tab(n_clicks, stock_id, start_date, end_date):
    if n_clicks > 0:
        fig = calculate_and_plot_strategy(df, stock_id, start_date, end_date)
        return fig
    
    return go.Figure() 

def calculate_and_plot_strategy(df: pd.DataFrame, stock_id: str, start_date: str, end_date: str):
    """Calculate trading signals and plot results using Plotly."""
    # Filtering the DataFrame
    M = df.loc[df.ID == stock_id]
    M = M[(M.DATE >= pd.to_datetime(start_date)) & (M.DATE <= pd.to_datetime(end_date))]

    # Calculate signals
    M = compute_signal(M)
    
    M['system_returns'] = M['RETURNS'] * M['signal']
    M['entry'] = M.signal.diff()

    # Create figures using Plotly for interactive plots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=('Price and Signals', 'Cumulative Returns'))

    # Add price, SMA lines, and entry signals to the first plot
    fig.add_trace(go.Scatter(x=M['DATE'], y=M['CLOSE'], mode='lines', name='Close Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=M['DATE'], y=M['CLOSE_SMA_9D'], mode='lines', name='9-day SMA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=M['DATE'], y=M['CLOSE_SMA_21D'], mode='lines', name='21-day SMA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=M['DATE'][M['entry'] == 2], y=M['CLOSE'][M['entry'] == 2], mode='markers', marker=dict(color='green', size=15, symbol='triangle-up'), name='Long Entry'), row=1, col=1)
    fig.add_trace(go.Scatter(x=M['DATE'][M['entry'] == -2], y=M['CLOSE'][M['entry'] == -2], mode='markers', marker=dict(color='red', size=15, symbol='triangle-down'), name='Short Entry'), row=1, col=1)

    # Plot cumulative returns for the strategy and buy-and-hold
    fig.add_trace(go.Scatter(x=M['DATE'], y=np.exp(M['RETURNS']).cumprod(), mode='lines', name='Buy/Hold'), row=2, col=1)
    fig.add_trace(go.Scatter(x=M['DATE'], y=np.exp(M['system_returns']).cumprod(), mode='lines', name='Strategy'), row=2, col=1)

    # Update plot layouts
    fig.update_layout(height=1000, width=2000, title_text=f"Trading Strategy Analysis for {stock_id}")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Returns", row=2, col=1)

    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
