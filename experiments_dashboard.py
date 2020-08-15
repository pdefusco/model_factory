import dash
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import pandas as pd
import os
from IPython.display import Javascript,HTML
import logging
from pandas.io.json import dumps as jsonify

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)



app = dash.Dash(__name__)

df = pd.read_csv('experiments_summary/experiments_summary.csv')


PAGE_SIZE = 5


app.layout = dash_table.DataTable(
    id='table-filtering',
    columns=[
        {"name": i, "id": i} for i in sorted(df.columns)
    ],
    page_current=0,
    page_size=PAGE_SIZE,
    page_action='custom',

    filter_action='custom',
    filter_query=''
)

operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]


def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3


@app.callback(
    Output('table-filtering', "data"),
    [Input('table-filtering', "page_current"),
     Input('table-filtering', "page_size"),
     Input('table-filtering', "filter_query")])
def update_table(page_current,page_size, filter):
    print(filter)
    filtering_expressions = filter.split(' && ')
    dff = df
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
        elif operator == 'contains':
            dff = dff.loc[dff[col_name].str.contains(filter_value)]
        elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            dff = dff.loc[dff[col_name].str.startswith(filter_value)]

    return dff.iloc[
        page_current*page_size:(page_current+ 1)*page_size
    ].to_dict('records')


#PORT = os.getenv('CDSW_APP_PORT', '8090')
# A handy way to get the link if you are running in a session.
HTML("<a href='https://{}.{}'>Open Table View</a>".format(os.environ['CDSW_ENGINE_ID'],os.environ['CDSW_DOMAIN']))

# Launches flask. Note the host and port details. This is specific to CML/CDSW
if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port=int(os.environ['CDSW_READONLY_PORT']))