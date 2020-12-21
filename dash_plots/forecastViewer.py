# vi: foldmethod=marker

import dash
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from datetime import datetime
import glob
import os

import cartopy.feature as cfeature
import cartopy.crs as ccrs

import xarray

FONT = 'Courier New'

# ========================
#	   READING DATA
# ========================

DATA_PATH='../../downcast/data/'

TYPES = sorted([x.split('/')[-1] for x in glob.glob(DATA_PATH+'*')])

# load coordinates of the coastline
xx = []
yy = []
for p in cfeature.COASTLINE.geometries():
	xx.extend([x[0] for x in p.coords] + [np.nan])
	yy.extend([x[1] for x in p.coords] + [np.nan])

# =============================================
# keep a dictionary of open files
_CACHED_FILES = {}
def get_dataset(path):
	'''Use this function to get a new dataset'''
	if path in _CACHED_FILES:
		return _CACHED_FILES[path]
	_CACHED_FILES[path] = xarray.open_dataset(path)
	return _CACHED_FILES[path]
# =============================================

# Use this functions to access the traces

_LOADED_TRACES = {}
def add_trace(path, variable, data):
	if (path, variable) not in _LOADED_TRACES:
		_LOADED_TRACES[(path, variable)] = data

def get_trace(path, variable):
	if (path, variable) in _LOADED_TRACES:
		return _LOADED_TRACES[(path, variable)]
	return None

def del_trace(path, variable):
	if (path, variable) in _LOADED_TRACES:
		del(_LOADED_TRACES[(path, variable)])

def clear_traces():
	_LOADED_TRACES = {}

# =============================================


def str_to_dt(string):
	if len(string) == 10:
		return datetime.strptime(string, '%Y%m%d%H')
	elif len(string) == 8:
		return datetime.strptime(string, '%Y%m%d')
	else:
		raise ValueError(f"No way to process this date: {string}")

def get_fc_actual_index(idx, fc_size):
	'''Returns the correct index from the slider since it is reversed'''
	return fc_size - 1 - idx

# ========================
#	  DASH LAYOUT SETUP
# ========================

app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])

# APP LAYOUT DEFINITION

app.layout = html.Div([
	# HEADER {{{
	html.Div([
			html.H2("Forecast Data Viewer", style={"margin-bottom":"0px"}),
		],
		id="header",
		className="row flex-display",
	), # HEADER }}}
	html.Div([ # MAIN ROW
		# FILTER MENU {{{
		html.Div([
			html.P( "Type", className="control_label"),
			dcc.Dropdown(
				id='type',
				options=[{'label':t, 'value':t} for t in TYPES],
				value=TYPES[0], placeholder='Select type...',
				className="dcc_control"
			),
			html.P( "Date", className="control_label"),
			dcc.DatePickerSingle(
				id='date',
				display_format="D MMM YYYY",
				style={'display':'block', "border": "0px solid black"},
				placeholder="Select date...",
				className="dcc_control",
			),
			html.P("File", className="control_label"),
			dcc.Dropdown(
				id='file',
				placeholder='Select file...',
				className="dcc_control"
			),
			html.P( "Variable", className="control_label"),
			dcc.Dropdown(
				id='variable',
				placeholder='Select variable...',
				className="dcc_control"
			),
			html.P(id='display_forecast_time', className="control_label"),
			html.Hr(),
			html.H5("Saved Traces", className="control_label", style={'margin-bottom':'7px'}),
			# html.Button(
			# 	'Save Trace',
			# 	id='save_trace',
			# 	n_clicks=0,
			# 	style={'width':'-moz-available', 'margin-bottom':'7px'},
			# ),
			dcc.Dropdown(
				id='traces',
				placeholder='',
				multi=True,
				className="dcc_control"
			),
		],
		id="filter_options",
		className="pretty_container three columns",
		), # FILTER MENU }}}
		# MAP GRAPHS {{{
		html.Div([
			html.Div([
				dcc.RangeSlider(
					id='forecast_time_slider',
					min=0,
					step=1,
					vertical=True,
					verticalHeight=1000, # BAM, make it BIG BOY!!
				),
				dcc.Graph(id='map'),
				], className="row flex-display"),
			],
			className="pretty_container five columns",
		),
		# }}}
		html.Div(
			[dcc.Graph(id='timeseries')],
			className="pretty_container six columns",
		),
	],
	className="row flex-display",
	),
], id="mainContainer", style={'font-family':FONT})

# =========================
# MAIN PLOT UPDATE CALLBACK

@app.callback(
		Output('map', 'figure'),
		[Input('type', 'value'),
		Input('date', 'date'),
		Input('file', 'value'),
		Input('variable', 'value'),
		Input('forecast_time_slider', 'value')])
def update_graph(type, date, file, var, forecast_time):
	fig = go.Figure()

	fig.add_trace(go.Scatter(
		x = xx,
		y = yy,
		line = go.scatter.Line(width = 1, color = '#000'),
		hoverinfo = 'skip'))

	if date:
		date = date.replace('-','') + '00'

		if os.path.isfile(f"{DATA_PATH}/{type}/{date}/{file}"):
			data = get_dataset(f"{DATA_PATH}/{type}/{date}/{file}")
			if var:
				fig.add_trace(go.Contour(
					x = data.longitude.data,
					y = data.latitude.data,
					z = data[var][get_fc_actual_index(forecast_time[0], data.time.size)].data))

	fig.update_xaxes(range=[-105, 35])
	fig.update_yaxes(range=[-82, 90])

	fig.update_layout(height=1000, showlegend=False, uirevision=True,
			margin=go.layout.Margin(l=0,r=0,t=0,b=0),
			plot_bgcolor= 'rgba(0, 0, 0, 0)', paper_bgcolor= 'rgba(0, 0, 0, 0)')

	return fig

@app.callback([
		Output('timeseries', 'figure'),
		Output('traces', 'options'),
		Output('traces', 'values')]
		[Input('type', 'value'),
		Input('date', 'date'),
		Input('file', 'value'),
		Input('variable', 'value'),
		Input('map', 'clickData')])
def update_graph(type, date, file, var, click):

	fig = go.Figure()

	if not click: return fig

	cx = click['points'][0]['x']
	cy = click['points'][0]['y']

	if date:
		date = date.replace('-','') + '00'

		if os.path.isfile(f"{DATA_PATH}/{type}/{date}/{file}"):
			data = get_dataset(f"{DATA_PATH}/{type}/{date}/{file}")
			fig.add_trace(go.Scatter(
				x = [str(t) for t in data[var].time.data],
				y = data[var].sel({'latitude':cy, 'longitude':cx}),
				mode = 'lines', name = var))

	fig.update_xaxes(title='Forecast Time')
	fig.update_yaxes(title=f'{var} ({data[var].units})')

	fig.update_layout(title='Time Series', uirevision=True,
			margin=go.layout.Margin(l=0,r=0,t=0,b=0))

	return fig

# ========================
# DATASET SELECTION CONTROLLERS {{{

@app.callback(
		Output('date', 'min_date_allowed'),
		[Input('type', 'value')])
def update_available_dates(type):
	return str_to_dt(sorted([x.split('/')[-1] for x in glob.glob(DATA_PATH+type+'/*')])[0])

@app.callback(
		Output('date', 'max_date_allowed'),
		[Input('type', 'value')])
def update_available_dates2(type):
	return str_to_dt(sorted([x.split('/')[-1] for x in glob.glob(DATA_PATH+type+'/*')])[-1])

@app.callback(
		Output('file', 'options'),
		[Input('type', 'value'),
		Input('date', 'date')])
def update_available_files(type, date):
	if date:
		date = date.replace('-','') + '00'
		return [{'label':d, 'value':d} for d in [x.split('/')[-1] for x in glob.glob(f"{DATA_PATH}/{type}/{date}/*")]]
	return []

@app.callback(
		Output('variable', 'options'),
		[Input('type', 'value'),
		Input('date', 'date'),
		Input('file', 'value')])
def update_available_variables(type, date, file):
	if date:
		date = date.replace('-','') + '00'
		if os.path.isfile(f"{DATA_PATH}/{type}/{date}/{file}"):
			return [{'label':d, 'value':d} for d in list(get_dataset(f"{DATA_PATH}/{type}/{date}/{file}").data_vars)]
	return []

@app.callback(
		Output('display_forecast_time', 'children'),
		[Input('type', 'value'),
		Input('date', 'date'),
		Input('file', 'value'),
		Input('forecast_time_slider', 'value')])
def update_forecast_time_display(type, date, file, forecast_time):
	if date:
		date = date.replace('-','') + '00'
		if os.path.isfile(f"{DATA_PATH}/{type}/{date}/{file}"):
			f = get_dataset(f"{DATA_PATH}/{type}/{date}/{file}")
			return 'Forecast Time: ' + str(f.time[get_fc_actual_index(forecast_time[0], f.time.size)].dt.strftime('%Y.%m.%d %Hh').data)
	return 'Forecast Time: '

# }}}

# ===========================
# FORECAST TIME CONTROLLER {{{

@app.callback([
		Output('forecast_time_slider', 'max'),
		Output('forecast_time_slider', 'value'),
		Output('forecast_time_slider', 'marks')],
		[ Input('type', 'value'),
		Input('date', 'date'),
		Input('file', 'value')])
def update_forecast_slider(type, date, file):
	if date:
		date = date.replace('-','') + '00'
		if os.path.isfile(f"{DATA_PATH}/{type}/{date}/{file}"):
			f = get_dataset(f"{DATA_PATH}/{type}/{date}/{file}")
			# TODO fix this messy shit
			# marks = {i: str(f.time[i].dt.dayofyear.data) for i in reversed(range(len(f.time))) if f.time[i].dt.hour==0}
			marks = {i: str(f.time[get_fc_actual_index(i, f.time.size)].dt.dayofyear.data) for i in reversed(range(len(f.time))) if f.time[get_fc_actual_index(i, f.time.size)].dt.hour==0}
			fc_time_size = f.time.size-1
			return fc_time_size, [fc_time_size], marks

	return 0, [0], {}


# }}}

@app.callback

if __name__ == '__main__':
	app.run_server(host='0.0.0.0', port=8888, debug=True)
