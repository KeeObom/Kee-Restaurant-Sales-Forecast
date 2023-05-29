import dash
from dash import html, dcc
# from pages import Forecast, Info

app = dash.Dash(__name__, use_pages=True)
server = app.server
app.config.suppress_callback_exceptions = True



app.layout = html.Div(
    [
        # Framework of the main app
        html.Div("Sales Forecast APP", style={'fontsize':50, 'textAlign':'center'}),
        html.Div([
            dcc.Link(children=page['name']+" | ", href=page['path'])
            for page in dash.page_registry.values()
        ]),
        html.Hr(),

        # Content of each page
        dash.page_container
    ]
)


if __name__ == '__main__':
    app.run_server(debug=True)