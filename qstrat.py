#!/usr/bin/env/ python3.11
import pandas as pd
import dash

from api.web_helpers.layout import export_layout
from api.web_helpers.callbacks import register_callbacks
import dash_dangerously_set_inner_html

df = pd.read_csv("assets/data.csv")
df.DATE = pd.to_datetime(df.DATE)
app = dash.Dash(__name__)
server = app.server


app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <title>Dash</title>
        <script type="text/javascript" async
            src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
        </script>
        {%metas%}
        {%favicon%}
        {%css%}
    </head>
    <body>
        <div id="react-entry-point">
            {%app_entry%}
        </div>
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = export_layout(df)

register_callbacks(app, df)

if __name__ == "__main__":
    app.run_server(debug=True)
