#!/usr/bin/env/ python3.11
import pandas as pd
import dash

from api.web_helpers.layout import export_layout
from api.web_helpers.callbacks import register_callbacks

df = pd.read_csv("assets/data.csv")
df.DATE = pd.to_datetime(df.DATE)
app = dash.Dash(__name__)
server = app.server

app.layout = export_layout(df)

register_callbacks(app, df)

if __name__ == "__main__":
    app.run_server(debug=True)
