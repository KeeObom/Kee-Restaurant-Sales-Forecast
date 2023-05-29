import dash
from dash import dcc, html

# Register dash page
dash.register_page(__name__, path='/')  # '/' is home page

layout = html.Div(
    [
        dcc.Markdown('# This page will be used for home page logo or picture')
    ]
)

