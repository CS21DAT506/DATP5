from enum import Flag
import numpy as np
import math
import random
import json

import dash
import dash_core_components as dcc
import dash_cytoscape as cyto
import dash_html_components as html
import plotly.graph_objects as go

from dash.dependencies import Output, Input

from gekko_wrapper import Gekko
from gekko_plot import GekkoPlotter


def get_vector_with_circular_bound(max_radius):
    r = math.sqrt(random.random()) * max_radius
    a = (2 * random.random() * math.pi)
    x = math.cos(a) * r
    y = math.sin(a) * r
    return np.array([x, y])


def vector_dist(a, b):
    return np.linalg.norm(a-b)


def get_mass(min_exp, max_exp):
    return 10 ** random.uniform(min_exp, max_exp)


def is_valid_configuration(agent, planets, target, min_dist_to_target):
    for p1 in planets:
        for p2 in planets:
            if not (p1 is p2):
                if vector_dist(p1["initial_pos"], p2["initial_pos"]) < (p1["radius"] + p2["radius"]) * 2:
                    return False

        if vector_dist(p1["initial_pos"], agent["initial_pos"]) < p1["radius"] * 2:
            return False

        if vector_dist(p1["initial_pos"], target) < p1["radius"] * 2:
            return False

    return vector_dist(agent["initial_pos"], target) >= min_dist_to_target

######################################################

def create_graph(data_path):

    results = None
    with open(data_path+"//results.json") as f:
        results = json.load(f)

    return go.Figure(
        data=go.Scatter3d(
            x=results["agent_px"], y=results["agent_py"], z=results["time"],
            marker=dict(
                size=4,
                color="blue",
                colorscale='Viridis',
            ),
            line=dict(
                color='darkblue',
                width=2
            )
        )
    )

######################################################

agent = {
    "mass": 500,
    "initial_pos": np.array([0, 0]),
    "initial_velocity": np.array([-10, 0]),
}

planets = [
    {
        "mass": 100000,
        "initial_pos": np.array([30, 20]),
        "radius": 5,
        "initial_velocity": np.array([0, 0]),
    },
]


if __name__ == "__main__":
    app = dash.Dash(__name__, prevent_initial_callbacks=True)
    
    data_path_list = []
    with open(r"data.txt", "r") as f:
        data_path = f.read()
        data_path_list = data_path.split("\n")

    tab_body_content = {
        "data": html.Div([
            html.H1("Model data overview"),
            html.Button("Solve", id="solve-button"),
            html.Button("Stop solve", id="stop-solve-button"),


            html.Div(id="trainning-data", children=[
                html.P("Agent mass:"),
                dcc.Input(id="agent-mass",value=agent["mass"], type="number"),

                html.Div(children=[
                     html.P("Agent inital position X:"),
                     dcc.Input(id="agent-initial-pos-x",value=(agent["initial_pos"][0]), type="number"),
                     html.P("Agent inital position Y:"),
                     dcc.Input(id="agent-initial-pos-y",value=(agent["initial_pos"][1]), type="number"),
                     
                     html.P("Agent initial velocity X:"),
                     dcc.Input(id="agent-initial-velocity-x",value=agent["initial_velocity"][0], type="number"),
                     html.P("Agent initial velocity Y:"),
                     dcc.Input(id="agent-initial-velocity-y",value=agent["initial_velocity"][1], type="number"),
                     ])
            ]),

            html.Div(id="graphs", children=[]),
            dcc.Interval(
                id="interval-graph-updates",
                interval=1*1000,  # milliseconds
                n_intervals=0
            )
        ]),
        "plot": html.Div(id="dropdown", children=[
                dcc.Dropdown(id="datapath-dropdown",
                    options=[{'label': filename, 'value': filename} for filename in data_path_list],
                    multi=False,
                    value=data_path_list[0]
                ),
                dcc.Graph(id="data-plot", figure=create_graph(data_path_list[0])),
        ]),
    }

    app.layout = html.Div(children=[
        html.Div(
            dcc.Tabs(
                id="tabs-body",
                value="learning",
                parent_className="custom-tabs",
                children=[
                    dcc.Tab(
                        label='Data',
                        value='data',
                        className='custom-tab',
                        selected_className='custom-tab--selected'
                    ),
                    dcc.Tab(
                        label='Plot',
                        value='plot',
                        className='custom-tab',
                        selected_className='custom-tab--selected'
                    ),
                ],
            ),
        ),
        html.Div(id="tab-body-content", children=[tab_body_content["data"]]),
    ])

    #####################################################################

    @app.callback(Output('tab-body-content', 'children'),
                  Input('tabs-body', 'value'))
    def render_content(tab):
        return tab_body_content[tab]

    @app.callback(Output('data-plot', 'figure'),
                  Input('datapath-dropdown', 'value'))
    def update_graph(data_path):
        return create_graph(data_path)

    #####################################################################

    print("Starting server!")
    app.run_server(host="localhost", port=8050, debug=True)
