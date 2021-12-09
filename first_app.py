import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import seaborn as sns
import os
import plotly.express as px
from matplotlib.backends.backend_agg import RendererAgg
import plotly.graph_objs as go


entete = st.container()
table = st.container()
widgets = st.container()
conclusion = st.container


matplotlib.use("agg")
_lock = RendererAgg.lock


with entete:
    row0_1, row0_2, row0_3, = st.columns((3, 3, 3))
    with row0_1:
        img = Image.open("image1.PNG")
        st.image(img, width=150)
    with row0_2:
        img = Image.open("youtube.PNG")
        st.image(img, width=150)
        #st.write("")
    with row0_3:
        img = Image.open("QRC_Lin.PNG")
        st.image(img, width=150)

    st.title('APPLICATION TO VISUALIZE MY YOUTUBE DATA')
    st.markdown("""This application is built to view my watch history on Youtube since I use this application.""")

with table:
    st.subheader('Gloabal Table of my watch-history')
#loading data and visualize 
    path = 'C:/Users/asus vivobook/OneDrive/Bureau/Doc_M1_Sem7/data_visua/watch-history.csv'
    data = pd.read_csv(path, delimiter = ',')   
# rename of the data's columns
    data.columns = ['header','title','title_Url', 'title_author' , 'subtiles_Url' , 'time' , 'products' , 'details_name'] 
# Drop of unusfull colunmsst.subheader('Table of my watch-history with new columns name')
    data = data[['header','time' ,'title', 'title_author']]   
# Convertion of the time into hours, day, weekday, month and year
    from datetime import datetime
    data['time']=pd.to_datetime(data['time'])
    data['hour'] = pd.DatetimeIndex(data['time']).hour
    data['day'] = pd.DatetimeIndex(data['time']).day
    data['weekday'] = pd.DatetimeIndex(data['time']).weekday
    data['month'] = pd.DatetimeIndex(data['time']).month
    data['year'] = pd.DatetimeIndex(data['time']).year
    data['day_name'] = data["time"].apply(lambda x: x.day_name())
    data['Count'] = 1
    st.write(data)



with widgets:
    numeric_columns = list(data.select_dtypes(['float', 'int', 'object']).columns)
    chart_select = st.sidebar.selectbox(
        label = "Select the chart type",
        options = ['scatterplots', 'histogram', 'bar', 'heatmap']
    )

    if chart_select == 'scatterplots':
        st.sidebar.subheader("scatter plot setting")
        try:
            X_values = st.sidebar.selectbox('x axis', options = numeric_columns)
            Y_values = st.sidebar.selectbox('y axis', options = numeric_columns)
            plot = px.scatter(data_frame = data.iloc[0:50,:], x=X_values, y=Y_values)
            st.plotly_chart(plot)
        except Exception as e:
            print(e)
    if chart_select == 'histogram':
        st.sidebar.subheader("histogram plot setting")
        try:
            X_values = st.sidebar.selectbox('x axis', options = numeric_columns)
            Y_values = st.sidebar.selectbox('y axis', options = numeric_columns)
            plot = px.histogram(data_frame = data.iloc[0:50,:], x=X_values, y=Y_values)
            st.plotly_chart(plot)
        except Exception as e:
            print(e)
    if chart_select == 'bar':
        st.sidebar.subheader("bar plot setting")
        try:
            X_values = st.sidebar.selectbox('x axis', options = numeric_columns)
            Y_values = st.sidebar.selectbox('y axis', options = numeric_columns)
            plot = px.bar(data_frame = data.iloc[0:50,:], x=X_values, y=Y_values)
            st.plotly_chart(plot)
        except Exception as e:
            print(e)
    if chart_select == 'heatmap':
        st.sidebar.subheader("heatmap setting")
        try:
            X_values = st.sidebar.selectbox('x axis', options = numeric_columns)
            Y_values = st.sidebar.selectbox('y axis', options = numeric_columns)
            plot = px.density_heatmap(data_frame = data.iloc[0:50,:], x=X_values, y=Y_values)
            st.plotly_chart(plot)
        except Exception as e:
            print(e)


    st.subheader('Table filtered according to the frequency of view of the channels')
    data1 = data.loc[:,'title_author']
    st.sidebar.header('Select what to display')
    nb_artist = data['title_author'].value_counts()
    nb_time = st.sidebar.slider("Appearence of a channel",
    int(nb_artist.min()), int(nb_artist.max()), 
    (int(nb_artist.min()), int(nb_artist.max())), 1)#is an array
 # with the minimum and the maximum number of appearence of a channel.
#st.write(data)
    Artists = data['title_author'].unique().tolist()
    Artist_selected = st.sidebar.multiselect('Channels'
    , Artists,Artists)#is a list of artist we wish to keep

#creates masks from the sidebar selection widgets
    mask_Artists = data['title_author'].isin(Artist_selected)
#get the parties with a number of members in the range of nb_mbrs
    mask_mbrs = data['title_author'].value_counts().between(nb_time[0], nb_time[1]).to_frame()
    mask_mbrs= mask_mbrs[mask_mbrs['title_author'] == 1].index.to_list()
    mask_mbrs= data['title_author'].isin(mask_mbrs)

    data_Artists_filtered = data[mask_Artists & mask_mbrs]
    st.write(data_Artists_filtered)

    

import random
number_of_colors = len(data_Artists_filtered)
color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
#st.write(color)

matplotlib.use("agg")
_lock = RendererAgg.lock

Arts = data_Artists_filtered['title_author'].value_counts()
#merge the two dataframe to get a column with the color
data = pd.merge(pd.DataFrame(Arts), data, left_index=True, right_on='title_author')
colors = color

st.header("Pie chart of channels by number of views")
row1, row2, row3 = st.columns([1,2,1])
with row2:
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(Arts, labels=(Arts.index + ' (' + Arts.map(str)
    + ')'), wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white'
    }, colors=colors)
    #display a white circle in the middle of the pie chart
    p = plt.gcf()
    p.gca().add_artist(plt.Circle( (0,0), 0.7, color='white'))
    st.pyplot(fig)

#numeric_columns = list(data_Artists_filtered.Select.dtypes(['float', 'int']).columns)


st.header('Exploration, Analysis and Visualization of my dataframe')
if st.checkbox('Lines charts of  watch-history '):


    col1, col2 = st.columns([2,2])

    with col1:
        st.subheader('Line Chart watch-history: Frequency by Hour')
        data_Artists_filtered = pd.DataFrame(
            np.random.randn(20, 1),
            columns=['hour'])
        st.line_chart(data_Artists_filtered)

    with col2:
        st.subheader('Line Chart watch-history: Frequency by Day')
        data_Artists_filtered = pd.DataFrame(
            np.random.randn(20, 1),
            columns=['day'])
        st.line_chart(data_Artists_filtered)

    col1, col2 = st.columns([2,2])

    with col1:
        st.subheader('Line Chart watch-history: Frequency by weekday')
        data_Artists_filtered = pd.DataFrame(
            np.random.randn(20, 1),
            columns=['weekday'])
        st.line_chart(data_Artists_filtered)

    with col2:
        st.subheader('Line Chart watch-history: Frequency by month')
        data_Artists_filtered = pd.DataFrame(
            np.random.randn(20, 1),
            columns=['month'])
        st.line_chart(data_Artists_filtered)
    
    
#if st.checkbox('Line Chart watch-history: Frequency by Weekday and month'):
    #row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((0.2, 1, .1, 1, .2)) 
    
    

if st.checkbox('Histogram of watch-history'):
    col1, col2 = st.columns([2,2])

    with col1:
        st.subheader('Histogram of watch-history by hour')
        hist_values = np.histogram(
            data_Artists_filtered['time'].dt.hour, bins=24, range=(0,24))[0]
        st.bar_chart(hist_values)

    with col2:
        st.subheader('Histogram of watch-history by day')
        hist_values = np.histogram(
            data_Artists_filtered['time'].dt.day, bins=31, range=(0,31))[0]
        st.bar_chart(hist_values)

    col1, col2 = st.columns([2,2])

    with col1:
        st.subheader('Histogram of watch-history by weekday')
        hist_values = np.histogram(
            data_Artists_filtered['time'].dt.weekday, bins=10, range=(0,7))[0]
        st.bar_chart(hist_values)

    with col2:
        st.subheader('Histogram of watch-history by month')
        hist_values = np.histogram(
            data_Artists_filtered['time'].dt.month, bins=10, range=(0,12))[0]
        st.bar_chart(hist_values)
    col1, col2 = st.columns([2,2])
    with col1:
        st.subheader('Histogram of watch-history by year')
        hist_values = np.histogram(
            data_Artists_filtered['time'].dt.year, bins=10, range=(2013,2021))[0]
        st.bar_chart(hist_values)

    row1, row2, row3 = st.columns([1,2,1])
    with row2:
        hist_x = st.selectbox("Histogram variable", options=data_Artists_filtered.columns, index=data_Artists_filtered.columns.get_loc("day"))
        hist_bins = st.slider(label="Histogram bins", min_value=5, max_value=50, value=25, step=1)
        hist_fig = px.histogram(data_Artists_filtered, x=hist_x, nbins=hist_bins, title="Histogram of " + hist_x,
                        template="plotly_white")
        st.write(hist_fig)




row0_1, row0_2, row0_3 = st.columns((2, 1, 2))
with row0_1:
    st.write("")
with row0_2:
    image = Image.open('thanksyou.JPG')
    st.image(image, caption='',width=200)
with row0_3:
    st.write("")
   

        





