import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime as dt
from itertools import combinations
from compute import calculate_cross_correlation

# Load and prepare the DataFrame
df = pd.read_csv('data/bloomberg20240503.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)

# Generate all pairwise combinations of stocks
stock_pairs = [{'label': f'{a} - {b}', 'value': f'{a}-{b}'} for a, b in combinations(df['ID'].unique(), 2)]

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define the layout
app.layout = html.Div([
    dcc.Tabs(id="tabs", value='tab-2', children=[
        dcc.Tab(label='Time Series', value='tab-2'),
        dcc.Tab(label='Arbitrage', value='tab-3'),
    ]),
    html.Div(id='tabs-content'),
    html.Div([
        html.Label("Select maximum lag:"),
        dcc.Input(
            id='max-lag-input',
            type='number',
            value=30, 
            style={'margin': '10px'}
        )
    ])
])

# Callback to update content based on selected tab
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab-2':
        return html.Div([
            html.H3('Time Series'),
            html.Div([
                dcc.Dropdown(
                    id='stock-dropdown',
                    options=[{'label': 'All', 'value': 'All'}] + [{'label': i, 'value': i} for i in df['ID'].unique()],
                    value='All',
                    style={'width': '250px', 'display': 'inline-block'}
                ),
                dcc.DatePickerRange(
                    id='date-picker-range',
                    min_date_allowed=df.index.min(),
                    max_date_allowed=df.index.max(),
                    start_date=df.index.min(),
                    end_date=df.index.max(),
                    style={'display': 'inline-block'}
                ),
            ], style={'display': 'flex'}),
            dcc.Graph(id='time-series-chart'),
            dcc.Dropdown(
                id='pair-dropdown',
                options=stock_pairs,
                value=stock_pairs[0]['value'] if stock_pairs else None,
                style={'width': '250px', 'display': 'inline-block'}
            ),
            dcc.Graph(id='rolling-correlation-chart')
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Arbitrage Opportunities'),
            html.P("This tab's content has been cleared as requested.")
        ])

@app.callback(
    [Output('time-series-chart', 'figure'),
     Output('rolling-correlation-chart', 'figure')],
    [Input('stock-dropdown', 'value'),
     Input('pair-dropdown', 'value'),
     Input('max-lag-input', 'value'),  # Ensure this input exists in your layout for max lag
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_graphs(stock_id, selected_pair, max_lag, start_date, end_date):
    fig1 = go.Figure()
    fig2 = go.Figure()
    
    # Time Series Chart: Check for 'All' or specific stock ID selection
    if stock_id and stock_id != 'All':
        # Filter data for selected stock and date range
        filtered_df = df[(df['ID'] == stock_id) & (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
        fig1.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df['close'], mode='lines', name=f'Close Price - {stock_id}'))
        # Include volume data if available
        if 'volume' in filtered_df.columns:
            fig1.add_trace(go.Bar(x=filtered_df.index, y=filtered_df['volume'], name='Volume', yaxis='y2', opacity=1.0))
        fig1.update_layout(
            title=f'Time Series Data for {stock_id}',
            xaxis_title="Date",
            yaxis_title="Price",
            yaxis2=dict(title='Volume', overlaying='y', side='right'),
            template="plotly_dark"
        )
    elif stock_id == 'All':
        # Display all stocks if 'All' is selected
        for id in df['ID'].unique():
            id_df = df[(df['ID'] == id) & (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
            fig1.add_trace(go.Scatter(x=id_df.index, y=id_df['close'], mode='lines', name=f'Close Price - {id}'))

    # Cross-Correlation Chart
    if selected_pair and max_lag is not None:
        stock1, stock2 = selected_pair.split('-')
        cross_corr = calculate_cross_correlation(df, stock1, stock2, max_lag, start_date, end_date)
        
        if not cross_corr.empty:
            fig2.add_trace(go.Scatter(x=list(range(-max_lag, max_lag + 1)), y=cross_corr, mode='lines',
                                      name=f'Cross-Correlation: {stock1} vs {stock2}'))
            fig2.update_layout(
                title=f'Cross-Correlation between {stock1} and {stock2}',
                xaxis_title="Lag",
                yaxis_title="Correlation",
                template="plotly_dark"
            )
        else:
            fig2.add_annotation(text="No sufficient data available for correlation calculation",
                                xref="paper", yref="paper", showarrow=False,
                                font=dict(size=14, color="red"))

    return fig1, fig2


if __name__ == '__main__':
    app.run_server(debug=True)
