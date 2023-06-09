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
plt.matplotlib.use("WebAgg")
from math import pi
from visualize import exclude_outlier


dataset = "data_charite"
subject = "imu0013"
windowed = False
# Define filepath

with open(os.path.join(os.getcwd(), 'path.json')) as f:
    paths = json.loads(f.read())
final_datapath= os.path.join(paths[dataset], "final_data")
final_datapath_full_data = os.path.join(paths[dataset], "one_window_final_data")

if windowed == True:
    csv_file = glob.glob(os.path.join(final_datapath, "*.csv"))

if windowed == False:
    csv_file = glob.glob(os.path.join(final_datapath_full_data, "*.csv"))
df_list = []

for path in csv_file:
    df_s = pd.read_csv(path)
    df_list.append(df_s)
data_full = pd.concat(df_list, axis = 0, ignore_index= True)    
print("appending data:", data_full.describe())
#data_descr = data_full[data_full["severity"]=="visit1"].describe()
data_descr = data_full.describe()
data_descr.to_csv(final_datapath + "summary.csv", sep=',')
print(data_full.columns)
print(data_full.isnull().values.any())
data_full = data_full.dropna()

data_columns = list(data_full.columns.values)



data_subject = data_full[data_full["subject"] == subject]


features_charite_all = data_subject[['stride_length_avg_left', 'clearance_avg_left', 'stride_time_avg_left',
       'swing_time_avg_left', 'stance_time_avg_left', 'stance_ratio_avg_left',
       'cadence_avg_left', 'speed_avg_left', 'stride_length_CV_left',
       'clearance_CV_left', 'stride_time_CV_left', 'swing_time_CV_left',
       'stance_time_CV_left', 'stance_ratio_CV_left', 'cadence_CV_left',
       'speed_CV_left', 'severity', 'stride_length_avg_right',
       'clearance_avg_right', 'stride_time_avg_right', 'swing_time_avg_right',
       'stance_time_avg_right', 'stance_ratio_avg_right', 'cadence_avg_right',
       'speed_avg_right', 'stride_length_CV_right', 'clearance_CV_right',
       'stride_time_CV_right', 'swing_time_CV_right', 'stance_time_CV_right',
       'stance_ratio_CV_right', 'cadence_CV_right', 'speed_CV_right',
       'stride_length_SI', 'clearance_SI', 'stride_time_SI', 'swing_time_SI',
       'stance_time_SI', 'stance_ratio_SI', 'cadence_SI', 'speed_SI']]

features_charite = data_subject[['stride_length_avg_left', 'clearance_avg_left', 'stride_time_avg_left',
       'swing_time_avg_left', 'stance_time_avg_left', 'stance_ratio_avg_left',
       'cadence_avg_left', 'speed_avg_left', 'severity', 'stride_length_avg_right',
       'clearance_avg_right', 'stride_time_avg_right', 'swing_time_avg_right',
       'stance_time_avg_right', 'stance_ratio_avg_right', 'cadence_avg_right',
       'speed_avg_right']]

print("features descr.", features_charite.describe())



categories_all = ['stride_length_avg_left', 'clearance_avg_left', 'stride_time_avg_left',
       'swing_time_avg_left', 'stance_time_avg_left', 'stance_ratio_avg_left',
       'cadence_avg_left', 'speed_avg_left', 'stride_length_CV_left',
       'clearance_CV_left', 'stride_time_CV_left', 'swing_time_CV_left',
       'stance_time_CV_left', 'stance_ratio_CV_left', 'cadence_CV_left',
       'speed_CV_left', 'stride_length_avg_right',
       'clearance_avg_right', 'stride_time_avg_right', 'swing_time_avg_right',
       'stance_time_avg_right', 'stance_ratio_avg_right', 'cadence_avg_right',
       'speed_avg_right', 'stride_length_CV_right', 'clearance_CV_right',
       'stride_time_CV_right', 'swing_time_CV_right', 'stance_time_CV_right',
       'stance_ratio_CV_right', 'cadence_CV_right', 'speed_CV_right',
       'stride_length_SI', 'clearance_SI', 'stride_time_SI', 'swing_time_SI',
       'stance_time_SI', 'stance_ratio_SI', 'cadence_SI', 'speed_SI']

categories = ['stride length left', 'clearance left', 'stride time left',
       'swing time left', 'stance time left', 'stance ratio left',
       'cadence left', 'speed left', 'stride length right',
       'clearance right', 'stride time right', 'swing time right',
       'stance time right', 'stance ratio right', 'cadence right',
       'speed right']

visit_1 = features_charite[features_charite["severity"] == "visit1"]
visit_1 = visit_1.reset_index()
visit_1.drop("severity",inplace = True, axis = 1)
visit_1.drop("index", inplace = True, axis = 1)
print("visit_1", visit_1.describe())

visit_2 = features_charite[features_charite["severity"] == "visit2"]
visit_2 = visit_2.reset_index()
visit_2.drop("severity",inplace = True, axis = 1)
visit_2.drop("index", inplace = True, axis = 1)
print("visit_2", visit_2.describe())

#for check of radarplots
diff_in_values = visit_1 - visit_2
print("diff in values", diff_in_values.transpose())

visit_1_norm = visit_1.div(visit_1).reset_index()
visit_1_norm.dropna(inplace=True)
print("VISIT1 NORM", visit_1_norm.describe())

visit1_mean = visit_1.mean()
print("Mean V1", visit1_mean)
visit_2_norm = visit_2.div(visit1_mean).reset_index()
visit_2_norm.replace([np.inf, -np.inf], np.nan, inplace=True)
visit_2_norm.dropna(inplace=True)
visit_1_norm.drop("index", inplace = True, axis = 1)
visit_2_norm.drop("index", inplace = True, axis = 1)
print("VISIT2 NORM", visit_2_norm)

### remove outliers ?
#exclude_outlier(visit_2_norm, categories)
#exclude_outlier(visit_1_norm, categories)


v_2_norm_mean = visit_2_norm.mean()
v_1_norm_mean = visit_1_norm.mean()
print("Mean V2", v_2_norm_mean)
print("VISIT 2 LOC",visit_2_norm.transpose())
# print("VISIT 1 LOC",visit_1_norm.transpose())
#visit_1_norm = visit_1_norm.transpose()
#visit_2_norm = visit_2_norm.transpose()

######### normalize data
# trans = MinMaxScaler()
# visit_1_scaled = DataFrame(trans.fit_transform(visit_1), columns = visit_1.columns)
# visit_2_scaled = DataFrame(trans.fit_transform(visit_2), columns = visit_2.columns)
# print(visit_1_scaled.describe())

features_v1v2 = features_charite.transpose()

print("visit2!", visit_2)
print("visit2! transposed", visit_2.transpose())


visit_2_trans = visit_2_norm.transpose()
visit_1_trans = visit_1_norm.transpose()
# ------- PART 1: Create background
 
# number of variable
#categories=list(df)[1:]
N = len(categories)
print(N)
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories)
ax.set_xticklabels(categories, fontsize = 8)
# Draw ylabels
ax.set_rlabel_position(0)
if windowed == True:
    max_val = v_2_norm_mean.max()
    print("max_val", max_val)
    plt.yticks(np.arange(0, max_val, step=0.5), color="grey", size=7)  
    plt.ylim(0,max_val) 
elif windowed == False:
    max_val = visit_2_trans.max()
    print("max_val", max_val)
    plt.yticks(np.arange(0, max_val.max(), step=0.5), color="grey", size=7)   
    plt.ylim(0,max_val.max())   

# ------- PART 2: Add plots
 
# Plot each individual = each line of the data
# Ind1
if windowed == True:
    values1 = v_1_norm_mean.values.flatten().tolist()
elif windowed == False:
    values1 = visit_1_trans.values.flatten().tolist()

print("VALUES1", values1)
values1 += values1[:1]
ax.plot(angles, values1, linewidth=1, linestyle='solid', label="Visit 1")
ax.fill(angles, values1, 'b', alpha=0.1)
 
# Ind2
if windowed == True:
    values2= v_2_norm_mean.values.flatten().tolist()
elif windowed == False:
    values2= visit_2_trans.values.flatten().tolist()
print("VALUES2", values2)
values2 += values2[:1]
ax.plot(angles, values2, linewidth=1, linestyle='solid', label="Visit 2")
ax.fill(angles, values2, 'r', alpha=0.1)
 
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# Add title
BLUE = "#2a475e"
fig.suptitle(
    "Average spatiotemporal features for both visits normalized on visit 1 - subject " + subject,
    x = 0.07,
    y = 0.95,
    ha="left",
    fontsize=10,
    fontname="DejaVu Sans",
    color=BLUE,
    weight="bold",    
)

if windowed ==True:
    wd = "_windowed"
else:
    wd = ""
# Show the graph
plt.show()
#plt.savefig("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/figures/"+ dataset +"_" + subject +"_" + "radar_plot_avg"+ wd +".png")
plt.close()



#### Plotly ##########
# fig = go.Figure(
#     data=[
#         go.Scatterpolar(r=visit_1, theta=categories, name='Visit 1'),
#         go.Scatterpolar(r=visit_2, theta=categories, name='Visit 2'),
#     ],
#     layout=go.Layout(
#         title=go.layout.Title(text='Visit comparisons'),
#         polar={'radialaxis': {'visible': True}},
#         showlegend=True
#     )
# )
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

#print("VISIT1 Values", visit_1.values)
### data 
# if __name__ == '__main__':
#     N = len(categories)
#     theta = radar_factory(N, frame='polygon')

#     data = visit_1
#     spoke_labels = data.pop(0)

#     fig, axs = plt.subplots(figsize=(20, 20), nrows=2, ncols=2,
#                             subplot_kw=dict(projection='radar'))
#     fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

#     colors = ['b', 'g']
#     # Plot the four cases from the example data on separate axes
#     for ax, (title, case_data) in zip(axs.flat, data):
#         ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
#         ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
#                      horizontalalignment='center', verticalalignment='center')
#         for d, color in zip(case_data, colors):
#             ax.plot(theta, d, color=color)
#             ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
#         ax.set_varlabels(spoke_labels)

#     # add legend relative to top-left plot
#     labels = data_r["severity"]
#     legend = axs[0, 0].legend(labels, loc=(0.9, .95),
#                               labelspacing=0.1, fontsize='small')

#     fig.text(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios',
#              horizontalalignment='center', color='black', weight='bold',
#              size='large')

#     plt.show()