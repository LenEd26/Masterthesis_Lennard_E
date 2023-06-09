import os
import pandas as pd 
import plotly.express as px
import plotly.subplots as sp

# import data
IMU_features =  pd.read_excel("/dhc/home/lennard.ekrod/Masterthesis_Lennard_E/Excel_Data/most_used_IMU_features.xlsx")
print(IMU_features)
IMU_features = IMU_features.rename(columns= {"N of Papers" : "count"})
print(IMU_features)

def two_level_sunburst(df, lev1, lev2='disease', subplots=False, swap_levels=False, color_map=None):
    """
    Plot and save sunburst plots from the DataFrame with the extracted data.
    
    Parameters
    ----------
    df : DataFrame
        Preprocessed DataFrame with extracted data from literature review.
    lev1 : str
        Column name of the first (outer) level in the sunburst plot.
    lev2 : str
        Column name of the second (inner) level in the sunburst plot.
        'disease' by default.
    subplots : bool
        Switch if a subplot should be generated with lev1 and lev2 in regular
        order for the left, and lev1 and lev2 in reversed order on the right
        plot.
    swap_levels : bool
        SWitch of lev1 and lev2 should be swapped.
    color_map : dict
        Optional dictionary to map the disorder name to the color to be
        assigned in the sunburst plot.
    """
    
    ### Preprocess the raw DataFrame for the sunburst plots
    #df_sb = get_sunburst_df(df, lev1)

    if subplots:
        fig1 = px.sunburst(df, path=[lev1, lev2], values="count", color=lev1)
        # In final version: only use color map for plot with disorder inside
        # to have consistent colors with other plots
        fig2 = px.sunburst(df, path=[lev2, lev1], values="count", color=lev2, color_discrete_map=color_map)

        fig = sp.make_subplots(rows=1, cols=2, 
        specs=[
            [{"type": "sunburst"}, {"type": "sunburst"}]
        ],
        subplot_titles=('A',  'B'),
        horizontal_spacing=0.05)

        fig.add_trace(fig1.data[0], row=1, col=1)
        fig.add_trace(fig2.data[0], row=1, col=2)
        fig.update_layout(
            font_size=10,
            width=600,
            height=300,
            margin=dict(l=5,r=5,b=10,t=20)
        )

        fig.write_image(
            os.path.join(
                'figures',
                f'sunburst_{lev1}_subplots.png'  
            )
        )
        fig.write_image(
            os.path.join(
                'figures',
                f'sunburst_{lev1}_subplots.pdf'
            )
        )

    else:
        if swap_levels:
            fig = px.sunburst(df, path=[lev1, lev2], values="count", color=lev1, color_discrete_map=color_map)
        else:
            fig = px.sunburst(df, path=[lev2, lev1], values="count", color=lev2, color_discrete_map=color_map)
        fig.update_layout(
            font_size=20,
            width=800,
            height=800,
            margin=dict(l=5,r=5,b=10,t=10)
        )

        fig.write_image(
            os.path.join(
                'figures',
                f'sunburst_{lev1}.png'  
            )
        )
        fig.write_image(
            os.path.join(
                'figures',
                f'sunburst_{lev1}.pdf'
            )
        )

    fig.show()


###Main

two_level_sunburst(IMU_features, lev1= "Feature", lev2= "Feature Group")