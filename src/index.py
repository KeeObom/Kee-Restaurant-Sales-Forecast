### Import Packages ###
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
### Import Dash Instance and Pages ###
from app import app
from pages import Forecast
from pages import Info
from pages import Home

### Page container ###
page_container = html.Div(
    children=[
        # represents the URL bar, doesn't render anything
        dcc.Location(
            id='url',
            refresh=False,
        ),
        # content will be rendered in this element
        html.Div(id='page-content')
    ]
)
### Index Page Layout ###
index_layout = html.Div(
    [
        # Framework of the main app
        html.Div("Sales Forecast APP", style={'fontsize':50, 'textAlign':'center'}),
        html.Div([
            dcc.Link(
                children='Forecast',
                href='/forecast',
            ),
            html.Br(),
            dcc.Link(
                children='Info',
                href='/infocards',
            ),
        ]),
        html.Hr(),

        # Content of each page
        dash.page_container
    ]
)
### Set app layout to page container ###
app.layout = page_container

### Assemble all layouts ###
app.validation_layout = html.Div(
    children = [
        page_container,
        index_layout,
        Forecast.layout,
        Info.layout,
    ]
)

### Update Page Container ###
@app.callback(
    Output(
        component_id='page-content',
        component_property='children',
        ),
    [Input(
        component_id='url',
        component_property='pathname',
        )]
)
def display_page(pathname):
    if pathname == '/':
        return index_layout
    elif pathname == '/forecast':
        return Forecast.layout
    elif pathname == '/infocards':
        return Info.layout
    else:
        return '404'