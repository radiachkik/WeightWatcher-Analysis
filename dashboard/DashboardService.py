import chart_studio.dashboard_objs as dashboard

import IPython.display
from IPython.display import Image


class DashboardService:
    def __init__(self):
        self._dashboard = dashboard.Dashboard()

    def add_plot(self, data, filename, auto_open):
        self._dashboard.get_preview()
