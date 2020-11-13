# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:32:11 2020

@author: wilhelm.vermelin
"""

import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from wordcloud import WordCloud
import re
st.set_option('deprecation.showfileUploaderEncoding', False)

#%%
#data_file = st.sidebar.selectbox('Välj tabell-data', [i for i in os.listdir('./data') if i.endswith('.xlsx')])
#data_file = st.sidebar.file_uploader('Välj fil', type = ['xlsx'])
data_file = st.sidebar.file_uploader('Ladda upp RSProduction tabell-data', type = ['xlsx', 'csv'])
@st.cache
def load_data(data_file):
    if data_file is None:
        data_file = './data/NolatoPlastteknik1YearStops.xlsx'
    df = pd.read_excel(data_file, sheet_name = 'Worksheet1')
    df = df.iloc[:-1]
    df['Total stopptid in hours'] = df['Total stopptid in hours'].astype(float)
    df['Day'] = df.Starttid.apply(lambda x: x.strftime('%A'))
    return df

df = load_data(data_file)

#file = st.file_uploader('Ladda upp RSProduction tabell-data')
stop_df = df[['Total stopptid in hours', 'Stopporsak', 'Mätpunkt', 'Skift']].groupby(['Stopporsak', 'Mätpunkt', 'Skift']).mean().reset_index().rename(columns={'Total stopptid in hours':'Mean stopptid in hours'})
stop_df['Total stopptid in hours'] = df[['Total stopptid in hours', 'Stopporsak', 'Mätpunkt', 'Skift']].groupby(['Stopporsak', 'Mätpunkt', 'Skift']).sum().reset_index()['Total stopptid in hours']
stop_df['Deviation'] = df[['Total stopptid in hours', 'Stopporsak', 'Mätpunkt', 'Skift']].groupby(['Stopporsak', 'Mätpunkt', 'Skift']).std()['Total stopptid in hours'].values
stop_df['Coef. of variation'] = df[['Total stopptid in hours', 'Stopporsak', 'Mätpunkt', 'Skift']].groupby(['Stopporsak', 'Mätpunkt', 'Skift']).apply(lambda x: np.std(x)/np.mean(x))['Total stopptid in hours'].values
stop_df['Count'] = df[['Total stopptid in hours', 'Stopporsak', 'Mätpunkt', 'Skift']].groupby(['Stopporsak', 'Mätpunkt', 'Skift']).count()['Total stopptid in hours'].values
time_df = df.copy()
time_df.Starttid = time_df.Starttid.dt.time
time_df.Sluttid = time_df.Sluttid.dt.time

#%%
st.title('RSProduction data')

st.header('Översikt')

st.text('Stopporsaker')
st.dataframe(stop_df.style.highlight_max(axis=0))

#%%
st.header('Filtrerad data')
stop_code = st.sidebar.selectbox('Välj stoppkod',
            options = np.sort(list(df.Stopporsak.unique()) + ['Alla']))
measure_point = st.sidebar.selectbox('Välj mätpunkt',
            options = np.sort(list(df['Mätpunkt'].unique()) + ['Alla']))

shift = st.sidebar.selectbox('Välj arbetsskift',
            options = np.sort(list(df['Skift'].unique()) + ['Alla']))
day = st.sidebar.selectbox('Välj dag', options = ['Alla', 'Måndag', 'Tisdag', 
                                                  'Onsdag', 'Torsdag', 'Fredag',
                                                  'Lördag', 'Söndag'])

translation_dict = {'Måndag':'Monday', 'Tisdag':'Tuesday', 'Onsdag':'Wednesday',
                    'Torsdag':'Thursday', 'Fredag':'Friday', 'Lördag':'Saturday',
                    'Söndag':'Sunday'}

st.text('Distribution över tidsåtgång')

filter_idx = [True]*len(df)

if stop_code != 'Alla':
    filter_idx = [True if i == stop_code and f else False for i, f in zip(df['Stopporsak'], filter_idx)]

if measure_point != 'Alla':
    filter_idx = [True if i == measure_point and f else False for i, f in zip(df['Mätpunkt'], filter_idx)]

if shift != 'Alla':
    filter_idx = [True if i == shift and f else False for i, f in zip(df['Skift'], filter_idx)]

if day != 'Alla':
    filter_idx = [True if i == translation_dict[day] and f else False for i, f in zip(df['Day'], filter_idx)]
#filter_idx = [True if i == stop_code and j == measure_point and k == shift else False for i, j, k in zip(df['Stopporsak'], df['Mätpunkt'], df['Skift'])]

filter_df = df.loc[filter_idx, :]
filter_time_df = time_df.loc[filter_idx, :]

fig, ax = plt.subplots()
ax.hist(filter_df['Total stopptid in hours'], bins = 50)
ax.tick_params(axis = 'x', labelsize = 8)
ax.set_xlabel('Tid (timmar)')
ax.set_ylabel('Antal')
ax.set_title(f'Distr. stoppkod/mätpunkt/skift/dag: {stop_code}/{measure_point}/{shift}/{day}')

st.pyplot(fig)

st.text('Distribution över dygnet')
time_lims = [pd.to_datetime(str(s) + ':00:00', format='%H:%M:%S').time() for s in range(24)]
pd.plotting.register_matplotlib_converters()
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(filter_time_df['Starttid'], [0]*len(filter_time_df), marker='o', linestyle='None')
ax.set_xlim(time_lims[0], pd.to_datetime('23:59:59', format='%H:%M:%S').time())
ax.set_xticks(time_lims)
ax.set_xticklabels([str(t.hour) for t in time_lims])
ax.set_xlabel('Tid på dygnet')
ax.set_ylabel('Total stopptid i timmar')
ax.hist(filter_time_df['Starttid'], bins = 24)
ax.set_title(f'Stoppkod/mätpunkt/skift/dag: {stop_code}/{measure_point}/{shift}/{day}')
st.pyplot(fig)

st.text('Stopptid per dag')

fig, ax = plt.subplots(figsize=(10,5))
try:
    sum_df = filter_df.groupby(filter_df['Starttid'].dt.to_period('D')).sum()
    ax = sum_df.plot(style = '-x')
    fig = ax.get_figure()
    # if len(sum_df.values) > 40:
    #     filter_n = max(2, len(sum_df.values)//40)
    #     ax.set_xticklabels(labels = [date if i % filter_n == 0 else None for i, date in enumerate(sum_df.keys().tolist())])
    # else:
    #     ax.set_xticklabels(labels = sum_df.keys().tolist())
except ValueError:
    pass
ax.tick_params(axis = 'x', labelsize = 10)
ax.set_xlabel('Tid')
ax.set_ylabel('Timmar')
ax.set_title(f'Stoppkod/mätpunkt/skift/dag: {stop_code}/{measure_point}/{shift}/{day}')
st.pyplot(fig)
#%%
st.header('Korrelationsanalys')
mode = st.selectbox('Välj stopporsak eller mätpunkt', ['Stopporsak', 'Mätpunkt'])
date_df = df.copy()
date_df.Starttid = date_df.Starttid.dt.date
date_df.Sluttid = date_df.Sluttid.dt.date

day_df = pd.DataFrame([[0]*len(date_df[mode].unique())]*len(date_df.Starttid.unique()), columns = np.sort(date_df[mode].unique()))
day_df.index = np.sort(date_df.Starttid.unique())

for col in day_df.columns:
    tmp_df = df.loc[df[mode] == col, 'Total stopptid in hours'].groupby(df['Starttid'].dt.to_period('D')).sum()
    day_df.loc[[True if i in tmp_df.index else False for i in day_df.index], col] = tmp_df.values.astype(float)

day_df['Total/dag'] = day_df.sum(axis = 1)

st.text('Stoppkod per dag data')
st.write(day_df)
#%%
corrmat = day_df.corr()
st.text('Korrelationsmatris')
cor_data = (corrmat.stack()
            .reset_index()     # The stacking results in an index on the correlation values, we need the index as normal columns for Altair
            .rename(columns={0: 'correlation', 'level_0': 'variable', 'level_1': 'variable2'}))
cor_data['correlation_label'] = cor_data['correlation'].map('{:.2f}'.format)  # Round to 2 decimal

var_sel_cor = alt.selection_single(fields=['variable', 'variable2'], clear=False,
                                  init={'variable': 'Total/dag', 'variable2': stop_code})

base = alt.Chart(cor_data).encode(
    x='variable2:O',
    y='variable:O',
    tooltip = ['variable', 'variable2', 'correlation']
)
# Text layer with correlation labels
# Colors are for easier readability
text = base.mark_text().encode(
    text='correlation_label',
    color=alt.condition(
        (alt.datum.correlation < -0.4) | (alt.datum.correlation > 0.4),
        alt.value('#ff7f0e'),
        alt.value('black')
    )
)


# The correlation heatmap itself
cor_plot = base.mark_rect().encode(
    color=alt.condition(var_sel_cor, alt.value('pink'), 'correlation:Q')
).add_selection(var_sel_cor)

st.altair_chart(cor_plot + text, use_container_width=True)
#%%
st.text('Plotta två kolumner i data mot varandra.')
cols = []
cols.append(st.selectbox('Välj första kolumn i korrelationsmatris', options = [i for i in np.sort(day_df.columns)]))
cols.append(st.selectbox('Välj andra kolumn i korrelationsmatris', options = [i for i in np.sort(day_df.columns)]))
if len(cols) > 2:
    st.warning('Välj endast två kolumner, plottar två första valen.')

if len(cols) >= 2:
    c = alt.Chart(day_df).mark_circle(size=60).encode(
        x= cols[0],
        y= cols[1],
        tooltip=[i for i in day_df.columns]
    ).configure_axis(
    labelFontSize=10,
    titleFontSize=20
).interactive()
    
st.altair_chart(c)

#%%
st.header('Heatmap')
cols = ['']*2
cols[0] = st.selectbox('Välj x-axel', options = [i for i in df.columns])
cols[1] = st.selectbox('Välj y-axel', options = [i for i in df.columns], index = 10)
exclude = st.multiselect('Välj bort data', options = df[cols[1]].unique(), default = [])
mode = st.radio('Välj antal tillfällen/timmar/rank', options = ['Timmar', 'Tillfällen'])
if mode == 'Tillfällen':
    tmp_df = df.loc[[True if val not in exclude else False for val in df[cols[1]]], cols + ['Total stopptid in hours']].groupby(by = cols, as_index = False).count()
else:
    tmp_df = df.loc[[True if val not in exclude else False for val in df[cols[1]]], cols + ['Total stopptid in hours']].groupby(by = cols, as_index = False).sum()
tmp_df = tmp_df.reset_index(drop = True)
norm = st.checkbox('Normalisera?', value = False)
if norm:
    pv_table = tmp_df.pivot(index = cols[1], columns = cols[0], values = 'Total stopptid in hours').apply(lambda x: (x - np.nanmin(x))/(np.nanmax(x) - np.nanmin(x)), axis = 1).reset_index()
    pv_table_melt = pv_table.melt(id_vars = [cols[1]], value_name = 'NormalizedVals').dropna(axis = 0, how = 'any').reset_index(drop = True)
    for ix, c1, c2 in zip(range(len(pv_table_melt)), pv_table_melt[cols[0]], pv_table_melt[cols[1]]):
        idx = [i for i in range(len(tmp_df)) if tmp_df.loc[i, cols[0]] == c1 and tmp_df.loc[i, cols[1]] == c2]
        tmp_df.loc[idx[0], 'NormalizedVals'] = pv_table_melt.loc[ix, 'NormalizedVals']
tmp_df.reset_index(drop=True)

x = [i for i in tmp_df[cols[0]]]
y = [i for i in tmp_df[cols[1]]]
z = [i for i in tmp_df['Total stopptid in hours']]

if norm:
    w = [i for i in tmp_df['NormalizedVals']]
    
    source = pd.DataFrame({'x': x,
                       'y': y,
                       'Total stopptid': z,
                       'Rank': w})

    hm = alt.Chart(source).mark_rect().encode(
        x='y:O',
        y='x:O',
        color='Rank:Q',
        tooltip = ['x', 'y', 'Total stopptid', 'Rank']
    ).interactive()
else:
    source = pd.DataFrame({'x': x,
                       'y': y,
                       'Total stopptid': z})

    hm = alt.Chart(source).mark_rect().encode(
        x='y:O',
        y='x:O',
        color='Total stopptid:Q',
        tooltip = ['x', 'y', 'Total stopptid']
    ).configure_axis(
        labelFontSize=10,
        titleFontSize=20
    ).interactive()


st.altair_chart(hm, use_container_width = True)

#%%
st.header('Ordmoln av kommentarer')

# if os.path.isfile(f'./plots/{data_file.replace("xlsx", "png")}'):
#     from PIL import Image

#     image = Image.open(f'./plots/{data_file.replace("xlsx", "png")}')
#     fig, ax = plt.subplots(figsize = (20,10))
    
#     ax.imshow(image)
#     ax.axis('off')
#     fig.tight_layout(pad=0)
    
#     st.pyplot(fig)
# else:
n_words = st.slider('Antal ord', 50, 500, value = 200, step = 50)
@st.cache
def create_wordcloud(n_words):
    stopwords = {'på', 'till', 'och', 'från', 'med', 'av', 'en', 'så', 'igen',
                 'vid', 'efter', 'som', 'den', 'det', 'att'}
    comments = re.sub(' +', ' ', ' '.join(df['Kommentar'].dropna()))
    wc = WordCloud(width=1600, height=800, stopwords=stopwords, max_words = n_words).generate(comments)
    
    return wc

wc = create_wordcloud(n_words)
fig, ax = plt.subplots(figsize = (20,10))
    
ax.imshow(wc, interpolation = 'bilinear')
ax.axis('off')
fig.tight_layout(pad=0)

st.pyplot(fig)

#fig.savefig(f'./plots/{data_file.replace("xlsx", "png")}', facecolor='k', bbox_inches='tight')
