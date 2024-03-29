from typing import List, Optional

from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash import Input, Output
import plotly.graph_objects as go


class DashboardService:
    def __init__(self):
        self._app = JupyterDash(__name__)
        self._figures: Optional[List[go.Figure]] = None
        self._active_figure: Optional[go.Figure] = None

    def build_dashboard(self, figures: List[go.Figure]):
        self._figures = figures
        figure_mapping = {figure.layout.title.text: index for index, figure in enumerate(self._figures)}
        self._app.layout = html.Div([
            html.H1("Visual analysis"),
            html.Label([
                "plot",
                dcc.Dropdown(
                    id='plot-dropdown',
                    clearable=False,
                    value=0,
                    options=[{'label': key, 'value': value} for key, value in figure_mapping.items()])
            ]),
            dcc.Graph(id='main-graph'),
        ])

        # Define callback to update graph
        @self._app.callback(
            Output('main-graph', 'figure'),
            Input("plot-dropdown", "value")
        )
        def display_figure(figure_title: int):
            self._active_figure = self._figures[figure_title]
            return self._active_figure

    def show_dashboard(self, external: bool = False):
        # Run app and display result inline in the notebook
        self._app.run_server(debug=True, mode='external' if external else "internal", height=750)
