# # CARDS CARDS CARDS
# import dash_bootstrap_components as dbc
# count = "https://user-images.githubusercontent.com/72614349/194616425-107a62f9-06b3-4b84-ac89-2c42e04c00ac.png"
#
# card = dbc.Card([
#     dbc.CardImg(src=r'assets/weather2.jpg', alt='image', top=True),
#     dbc.CardBody(
#         [
#             html.H3("Count von Count", className="text-primary"),
#             html.Div("Chief Financial Officer"),
#             html.Div("Sesame Street, Inc.", className="small"),
#         ]
#     )],
#     className="shadow my-2",
#     style={"maxWidth": 350},
# )
#
#
#
# app.layout=dbc.Container(card)

import dash
from dash import dcc, html
from app import app

#dash.register_page(__name__)

layout = html.Div(
    [
        dcc.Markdown('# This page will be used for the cards')
    ]
)


# Callback
