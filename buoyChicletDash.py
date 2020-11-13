import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

import xarray

FONT = 'Courier New'

COLORSCHEMES = ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
			'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
			'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
			'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
			'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
			'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
			'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
			'orrd', 'oryel', 'peach', 'phase', 'picnic', 'pinkyl', 'piyg',
			'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn', 'puor',
			'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu', 'rdgy',
			'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar', 'spectral',
			'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn', 'tealrose',
			'tempo', 'temps', 'thermal', 'tropic', 'turbid', 'twilight',
			'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd']

# ========================
#	   READING DATA
# ========================

# nearest
GFS_DATA = 'data/buoys/GFS_selectionPointNearest_20190925_20200701.nc'
GWES_DATA = 'data/buoys/GWES_selectionPointNearest_20190925_20200701.nc'
WW3_DATA = 'data/buoys/WW3_selectionPointNearest_20190925_20200701.nc'

# average
# GFS_DATA = 'data/buoys/GFS_selectionPointAver_20190925_20200701.nc'
# GWES_DATA = 'data/buoys/GWES_selectionPointAver_20190925_20200701.nc'
# WW3_DATA = 'data/buoys/WW3_selectionPointAver_20190925_20200701.nc'

OBSERVED_DATA = 'data/buoys/NDBC_selection_deepWaters_20190925_20200701.nc'

gfs_data = xarray.open_dataset(GFS_DATA)
gwes_data = xarray.open_dataset(GWES_DATA)
ww3_data = xarray.open_dataset(WW3_DATA)
obs_data = xarray.open_dataset(OBSERVED_DATA)

stations = gwes_data.buoyID.data # list of station names

# variables = sorted([d for d in gwes_data.data_vars if len(gwes_data[d].shape) >1]) # list of data variables names
variables = ['Dp', 'Hs', 'Tp', 'WSPD']

fctime_name = 'fctime'

# put multiple ensembles here
ensmembers= {   'GFS' :gfs_data,
				'GWES':gwes_data,
				'WW3' :ww3_data
			}

# define functions to get which data to plot (a function that receives data (xarray), variable name and station and returns the data to plot)
data_to_plot = {}

for k in ensmembers:
	if k != 'GFS': # TODO FIXME
		dx = ensmembers[k]
		if hasattr(ensmembers[k], 'nensembles'): # if there are multiple ensembleMembers
			# mean of the ensembles, if variable does not exits the function returns None
			data_to_plot[f"{k}"] = lambda var, st, data=dx: np.mean(data[variables[var]][st, 1:, :, :], axis=0).data.T if variables[var] in data else None
			# difference from observed data
			# TODO Headache incoming...
			# data_to_plot[f"Difference (Observed - {k})"] = lambda data, var, st: np.mean(ensmembers[k][variables[var]][st, 1:, :, :], axis=0).data.T - data[obs_variables[var]][st, :, :].data.T
		else:
			# specific ensemble, if variable does not exits the function returns None
			data_to_plot[f"{k}"] = lambda var, st, data=dx: data[variables[var]][st, :, :].data.T if variables[var] in data else None

# ========================
#	  DASH APP SETUP
# ========================

app = dash.Dash(__name__)

app.layout = html.Div([
	html.H2(children = 'Chiclet Plot', style={'text-align':'center'}),

	html.Div([
		html.Div(children='Station:'),
		dcc.Dropdown(
			id='station-id',
			options=[{'label':v, 'value':i} for i, v in enumerate(stations)],
			value=0, placeholder='Select station...',
			clearable=False
		)
	], style={'width':'15%', 'display':'inline-block'}),
	html.Div([
		html.Div(children='Variable: '),
		dcc.Dropdown(
			id='variable',
			options=[{'label':v, 'value':i} for i, v in enumerate(variables)],
			value=0, placeholder='Select variable...',
			clearable = False
		)
	], style={'width':'20%', 'display':'inline-block'}),
	html.Div([
		html.Div(children='Data: '),
		dcc.Dropdown(
			id='plot-data',
			options=[{'label':k, 'value':k} for k in data_to_plot],
			value=list(ensmembers.keys())[1], placeholder='Select data to plot...',
			clearable = False
		)
	], style={'width':'30%', 'display':'inline-block'}),
	html.Div([
		html.Div(children='Colorscheme: '),
		dcc.Dropdown(
			id='colorscheme',
			options=[{'label':k, 'value':k} for k in COLORSCHEMES],
			value='jet', placeholder='Select Colorscheme to use...',
			clearable = False
		)
	], style={'width':'15%', 'display':'inline-block'}),
	html.Div([
		html.Div(children='Reversed: '),
		dcc.Checklist(
			id='reversed',
			options=[{'label':'', 'value':0}],
			value=[]
		)
	], style={'display':'inline-block', 'vertical-align': '-140%', 'padding-left':'5px'}),

	dcc.Graph(id='timeseries-graph'),

], style={'font-family':FONT})

@app.callback(
		Output('timeseries-graph', 'figure'),
		[Input('station-id', 'value'),
		Input('plot-data', 'value'),
		Input('variable', 'value'),
		Input('colorscheme', 'value'),
		Input('reversed', 'value')])
def update_graph(station, dname, variable, colorscheme, rev):
	if variable == 'WSPD':
		fig = go.Figure()
	else:
		fig = go.Figure(
				go.Heatmap(
					x = [str(t) for t in ensmembers[list(ensmembers.keys())[0]].time.data],
					y = ensmembers[list(ensmembers.keys())[0]][fctime_name].data /3600 /24,
					z = data_to_plot[dname](variable, station),
					colorscale = colorscheme if not rev else colorscheme + '_r',
					zsmooth = 'best'
				))

	fig.update_xaxes(title = 'Time')
	fig.update_yaxes(title = "Forecast Time (days)")

	fig.update_layout(title=f'Chiclet Plot: {dname}')

	return fig

if __name__ == '__main__':
	app.run_server(host='0.0.0.0', port=8888, debug=True)

