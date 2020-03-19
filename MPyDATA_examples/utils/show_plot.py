import tempfile, os
from IPython.display import FileLink, display
from ipywidgets import HTML, VBox, Output
from matplotlib import pyplot


def show_plot():
    pyplot.legend()
    tempfile_fd, tempfile_path = tempfile.mkstemp(
        dir=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output'),
        suffix='.pdf'
    )
    pyplot.savefig(tempfile_path, type='pdf')

    output = Output()
    with output:
        pyplot.show()
    link = HTML()
    filename = str(os.path.join('../utils/output', os.path.basename(tempfile_path)))
    link.value = FileLink(filename)._format_path()
    display(VBox([output, link]))
