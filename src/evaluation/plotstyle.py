"""Shared figure output settings.

Everything is written as PDF for the thesis: matplotlib's PDF backend keeps text,
axes, lines and borders as vector, so they stay sharp at any zoom. Photographs
drawn with imshow are embedded as raster inside that PDF, which is unavoidable
and fine, the source scans are only 224x224 to begin with.

A PNG copy goes out alongside each one, purely for quick viewing.
"""

import os

import matplotlib
import matplotlib.pyplot as plt

# 42 is TrueType. matplotlib defaults to Type 3, which is still vector but is not
# reliably selectable or searchable and some LaTeX/journal pipelines reject it
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# keep text as text rather than letting the backend outline it into paths
matplotlib.rcParams['svg.fonttype'] = 'none'

# only affects rasterised content (the scans). vector parts ignore it.
# the source scans are 224x224 and panels are roughly 2.3 inches wide, so native
# resolution is about 97 dpi. 300 just interpolates and triples the file size
RASTER_DPI = 150


def save_figure(fig, name, out_dir, dpi=RASTER_DPI, also_png=True):
    """Write `name` as a vector PDF, plus an optional PNG for previewing."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(name)[0]

    pdf_path = os.path.join(out_dir, stem + '.pdf')
    fig.savefig(pdf_path, format='pdf', dpi=dpi, bbox_inches='tight', facecolor='white')

    if also_png:
        png_path = os.path.join(out_dir, stem + '.png')
        fig.savefig(png_path, format='png', dpi=170, bbox_inches='tight', facecolor='white')

    plt.close(fig)
    print(f'  wrote {pdf_path}')
    return pdf_path
