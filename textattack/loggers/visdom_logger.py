import socket

from visdom import Visdom

from textattack.shared.utils import html_table_from_rows

from .logger import Logger


def port_is_open(port_num, hostname="127.0.0.1"):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((hostname, port_num))
    sock.close()
    if result == 0:
        return True
    return False


class VisdomLogger(Logger):
    """Logs attack results to Visdom."""

    def __init__(self, env="main", port=8097, hostname="localhost"):
        if not port_is_open(port, hostname=hostname):
            raise socket.error(f"Visdom not running on {hostname}:{port}")
        self.vis = Visdom(port=port, server=hostname, env=env)
        self.env = env
        self.port = port
        self.hostname = hostname
        self.windows = {}
        self.sample_rows = []

    def __getstate__(self):
        state = {i: self.__dict__[i] for i in self.__dict__ if i != "vis"}
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.vis = Visdom(port=self.port, server=self.hostname, env=self.env)

    def log_attack_result(self, result):
        text_a, text_b = result.diff_color(color_method="html")
        result_str = result.goal_function_result_str(color_method="html")
        self.sample_rows.append([result_str, text_a, text_b])

    def log_summary_rows(self, rows, title, window_id):
        self.table(rows, title=title, window_id=window_id)

    def flush(self):
        self.table(
            self.sample_rows,
            title="Sample-Level Results",
            window_id="sample_level_results",
        )

    def log_hist(self, arr, numbins, title, window_id):
        self.bar(arr, numbins=numbins, title=title, window_id=window_id)

    def text(self, text_data, title=None, window_id="default"):
        if window_id and window_id in self.windows:
            window = self.windows[window_id]
            self.vis.text(text_data, win=window)
        else:
            new_window = self.vis.text(text_data, opts=dict(title=title))
            self.windows[window_id] = new_window

    def table(self, rows, window_id=None, title=None, header=None, style=None):
        """Generates an HTML table."""

        if not window_id:
            window_id = title  # Can provide either of these,
        if not title:
            title = window_id  # or both.
        table = html_table_from_rows(rows, title=title, header=header, style_dict=style)
        self.text(table, title=title, window_id=window_id)

    def bar(self, X_data, numbins=10, title=None, window_id=None):
        window = None
        if window_id and window_id in self.windows:
            window = self.windows[window_id]
            self.vis.bar(X=X_data, win=window, opts=dict(title=title, numbins=numbins))
        else:
            new_window = self.vis.bar(X=X_data, opts=dict(title=title, numbins=numbins))
            if window_id:
                self.windows[window_id] = new_window

    def hist(self, X_data, numbins=10, title=None, window_id=None):
        window = None
        if window_id and window_id in self.windows:
            window = self.windows[window_id]
            self.vis.histogram(
                X=X_data, win=window, opts=dict(title=title, numbins=numbins)
            )
        else:
            new_window = self.vis.histogram(
                X=X_data, opts=dict(title=title, numbins=numbins)
            )
            if window_id:
                self.windows[window_id] = new_window
