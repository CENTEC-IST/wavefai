import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

import xarray

FONT = 'Courier New'

# COLORSCHEMES = ['Greys','YlGnBu','Greens','YlOrR d','Bluered','RdBu','Reds','Blues','Picnic','Rainbow','Portland','Jet','H ot','Blackbody','Earth','Electric','Viridis','Cividis']
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

DATA_FILE = 'data/ECMWFifs_and_Obsv_StationPos_2017111300_2020082300.nc'

data = xarray.open_dataset(DATA_FILE)

stations = data.stationID.data # list of station names
datetime = [str(d) for d in data.cycletime.data] # list of datetime strings

variables = sorted([d for d in data.data_vars if len(data[d].shape) >1]) # list of data variables names
obs_variables = variables[len(variables)//2:] # names of observed variables
ens_variables = variables[:len(variables)//2] # names of ensemble variables

# define functions to get which data to plot (a function that receives data (xarray), variable name and station and returns the data to plot)
data_to_plot = {'Observed':
					lambda data, var, st: data[obs_variables[var]][st, :, :].data.T,
				'Ensemble Mean':
					lambda data, var, st: np.nanmean(data[ens_variables[var]][st, :, :, :].data, axis=0).T,
				'Ensemble Spread':
					lambda data, var, st: np.std(data[ens_variables[var]][st, :, :, :].data, axis=0).T,
				'Difference (Observed - Ensemble Mean)':
					lambda data, var, st: np.nanmean(data[ens_variables[var]][st, :, :, :].data, axis=0).T - data[obs_variables[var]][st, :, :].data.T
				}
for i, v in enumerate(data.ensmember.data):
	# specific ensemble
	data_to_plot[f"Ens {v}"] = lambda data, var, st: data[ens_variables[var]][st, i, :, :].data.T
	# specific ensemble differences
	data_to_plot[f"Difference (Observed - Ens {v})"] = lambda data, var, st: data[ens_variables[var]][st, i, :, :].data.T - data[obs_variables[var]][st, :, :].data.T

# ========================
#	  DASH APP SETUP
# ========================

app = dash.Dash(__name__)

app.layout = html.Div([
	html.H2(children = DATA_FILE, style={'text-align':'center'}),

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
			options=[{'label':v[v.index('_')+1:], 'value':i} for i, v in enumerate(obs_variables)],
			value=0, placeholder='Select variable...',
			clearable = False
		)
	], style={'width':'20%', 'display':'inline-block'}),
	html.Div([
		html.Div(children='Data: '),
		dcc.Dropdown(
			id='plot-data',
			options=[{'label':k, 'value':k} for k in data_to_plot],
			value='Observed', placeholder='Select data to plot...',
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
	fig = go.Figure(
			go.Heatmap(
				x = datetime,
				y = data.forecast_time.data /3600 /24,
				z = data_to_plot[dname](data, variable, station),
				colorscale = colorscheme if not rev else colorscheme + '_r',
				zsmooth = 'best'
			))

	fig.update_xaxes(title = 'Time')
	fig.update_yaxes(title = "Forecast Time (days)")

	fig.update_layout(title=f'Chiclet Plot: {dname}')

	return fig

if __name__ == '__main__':
	app.run_server(host='0.0.0.0', port=8888, debug=True)
