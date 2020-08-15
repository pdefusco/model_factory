import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os

data = {'Cap' : ['A', 'B', 'C', ], 'non-Cap' : ['a','b','c', ]}
df = pd.DataFrame(data)

def generate_table(dataframe, max_rows=26):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns]) ] +
        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

app = dash.Dash(__name__, )

app.layout = html.Div(children=[
    html.H4(children='StackOverflow - Html dash table'),
    generate_table(df)
])

# Launches flask. Note the host and port details. This is specific to CML/CDSW
if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port=int(os.environ['CDSW_READONLY_PORT']))