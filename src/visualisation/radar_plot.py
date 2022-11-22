import pandas as pd
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import plotly.graph_objects as go
import plotly.offline as pyo
plt.matplotlib.use("WebAgg")

dataset = "data_sim_cbf"
# Define filepath

with open(os.path.join(os.getcwd(), 'path.json')) as f:
    paths = json.loads(f.read())
final_datapath= os.path.join(paths[dataset], "final_data")

csv_file = glob.glob(os.path.join(final_datapath, "*.csv"))

df_list = []
for path in csv_file:
    df_s = pd.read_csv(path)
    df_list.append(df_s)
data_full = pd.concat(df_list, axis = 0, ignore_index= True)    
print("appending data:", data_full.describe())
print(data_full.columns)
print(data_full.isnull().values.any())
data_full = data_full.dropna()

data_columns = list(data_full.columns.values)
# data_columns.remove("subject")
# data_columns.remove("severity")


features_charite = data_full[['stride_time_avg_right', 'stance_time_avg_right',
       'stance_ratio_avg_right', 'cadence_avg_right', 'speed_avg_right',
       'clearance_CV_right', 'stride_time_CV_right', 'swing_time_CV_right',
       'cadence_CV_right', 'clearance_avg_left', 'swing_time_avg_left',
       'stance_ratio_avg_left', 'stride_length_CV_left', 'clearance_CV_left',
       'swing_time_CV_left', 'stance_ratio_CV_left', 'clearance_SI',
       'stride_time_SI', 'stance_time_SI', 'speed_SI', 'severity']]

print(features_charite.describe())



categories = ['stride_time_avg_right', 'stance_time_avg_right',
       'stance_ratio_avg_right', 'cadence_avg_right', 'speed_avg_right',
       'clearance_CV_right', 'stride_time_CV_right', 'swing_time_CV_right',
       'cadence_CV_right', 'clearance_avg_left', 'swing_time_avg_left',
       'stance_ratio_avg_left', 'stride_length_CV_left', 'clearance_CV_left',
       'swing_time_CV_left', 'stance_ratio_CV_left', 'clearance_SI',
       'stride_time_SI', 'stance_time_SI', 'speed_SI']
categories = [*categories, categories[0]]

visit_1 = features_charite.loc[features_charite["severity"] == "visit1"]
visit_2 = features_charite.loc[features_charite["severity"] == "visit2"]
#visit_1 = [*visit_1, visit_1[0]]
#visit_2 = [*visit_2, visit_2[0]]


fig = go.Figure(
    data=[
        go.Scatterpolar(r=visit_1, theta=categories, name='Visit 1'),
        go.Scatterpolar(r=visit_2, theta=categories, name='Visit 2'),
    ],
    layout=go.Layout(
        title=go.layout.Title(text='Visit comparisons'),
        polar={'radialaxis': {'visible': True}},
        showlegend=True
    )
)

fig.show()
#pyo.plot(fig)


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta