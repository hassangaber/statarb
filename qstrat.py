#!/usr/bin/env/ python3.11
import pandas as pd
import dash
from dash import Dash

from api.web_helpers.layout import export_layout
from api.web_helpers.callbacks import register_callbacks
import dash_dangerously_set_inner_html

df = pd.read_csv("assets/data.csv")
df.DATE = pd.to_datetime(df.DATE)
MATHJAX_CDN = '''
https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/
MathJax.js?config=TeX-MML-AM_CHTML'''

external_scripts = [
                    {'type': 'text/javascript',
                     'id': 'MathJax-script',
                     'src': MATHJAX_CDN,
                     },
                    ]


app = Dash(__name__, external_scripts=external_scripts)
server = app.server

app.layout = export_layout(df)

register_callbacks(app, df)

if __name__ == "__main__":
    app.run_server(debug=True)
