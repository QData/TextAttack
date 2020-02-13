import socket
from visdom import Visdom

from .logger import Logger

def style_from_dict(style_dict):
    """ Turns
            { 'color': 'red', 'height': '100px'}
        into
            style: "color: red; height: 100px"
    """
    style_str = ''
    for key in style_dict:
        style_str += key + ': ' + style_dict[key] + ';'
    return 'style="{}"'.format(style_str)

def port_is_open(port_num, hostname='127.0.0.1'):
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  result = sock.connect_ex((hostname, port_num))
  sock.close()
  if result == 0: return True
  return False
  
class VisdomLogger(Logger):
    def __init__(self, env='main', port=8097, hostname='localhost'):
        if not port_is_open(port, hostname=hostname):
            raise socket.error(f'Visdom not running on {hostname}:{port}')
        self.vis = Visdom(port=port, server=hostname, env=env)
        self.windows = {}
        self.sample_rows = []

    def log_attack_result(self, result):
        text_a, text_b = result.diff_color(color_method='html')
        result_str = result.result_str(color_method='html')
        self.sample_rows.append([result_str,text_a,text_b])

    def log_rows(self, rows, title, window_id):
        self.table(rows, title=title, window_id=window_id)

    def flush(self):
        self.table(self.sample_rows, title='Sample-Level Results', window_id='sample_level_results')

    def log_hist(self, arr, numbins, title, window_id):
        self.bar(arr, numbins=numbins, title=title, window_id=window_id)

    def text(self, text_data, title=None, window_id='default'):
        if window_id and window_id in self.windows:
            window = self.windows[window_id]
            self.vis.text(text_data, win=window)
        else:
            new_window = self.vis.text(text_data, 
                opts=dict(
                        title=title
                    )
                )
            self.windows[window_id] = new_window

    def table(self, rows, window_id=None, title=None, header=None, style=None):
        """ Generates an HTML table. """
        
        if not window_id:   window_id = title    # Can provide either of these,
        if not title:       title = window_id    # or both.

        # Stylize the container div.
        if style:
            table_html = '<div {}>'.format(style_from_dict(style))
        else:
            table_html = '<div>'
        # Print the title string.
        if title:
            table_html += '<h1>{}</h1>'.format(title)

        # Construct each row as HTML.
        table_html = '<table class="table">'
        if header:
            table_html += '<tr>'
            for element in header:
                table_html += '<th>'
                table_html += str(element)
                table_html += '</th>'
            table_html += '</tr>'
        for row in rows:
            table_html += '<tr>'
            for element in row:
                table_html += '<td>'
                table_html += str(element)
                table_html += '</td>'
            table_html += '</tr>'

        # Close the table and print to screen.
        table_html += '</table></div>'
        self.text(table_html, title=title, window_id=window_id)

    def bar(self, X_data, numbins=10, title=None, window_id=None):
        window = None
        if window_id and window_id in self.windows:
            window = self.windows[window_id]
            self.vis.bar(
                X=X_data,
                win=window,
                opts=dict(
                    title=title,
                    numbins=numbins
                )
            )
        else:
            new_window = self.vis.bar(
                X=X_data,
                opts=dict(
                    title=title,
                    numbins=numbins
                )
            )
            if window_id:
                self.windows[window_id] = new_window

    def hist(self, X_data, numbins=10, title=None, window_id=None):
        window = None
        if window_id and window_id in self.windows:
            window = self.windows[window_id]
            self.vis.histogram(
                X=X_data,
                win=window,
                opts=dict(
                    title=title,
                    numbins=numbins
                )
            )
        else:
            new_window = self.vis.histogram(
                X=X_data,
                opts=dict(
                    title=title,
                    numbins=numbins
                )
            )
            if window_id:
                self.windows[window_id] = new_window

