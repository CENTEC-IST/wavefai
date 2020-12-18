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

# =====================
# Other

# keep a dictionary of open files
_CACHED_FILES = {}
def get_dataset(path):
	'''Use this function to get a new dataset'''
	if path in _CACHED_FILES:
		return _CACHED_FILES[path]
	_CACHED_FILES[path] = xarray.open_dataset(path)
	return _CACHED_FILES[path]

def str_to_dt(string):
	if len(string) == 10:
		return datetime.strptime(string, '%Y%m%d%H')
	elif len(string) == 8:
		return datetime.strptime(string, '%Y%m%d')
	else:
		raise ValueError(f"No way to process this date: {string}")

# ========================
#	  DASH APP SETUP
# ========================

app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])

# APP LAYOUT DEFINITION

app.layout = html.Div([
	html.Div([ # HEADER {{{
			html.H2("Forecast Data Viewer", style={"margin-bottom":"0px"}),
		],
		id="header",
		className="row flex-display",
		style={"margin-bottom": "15px"},
	), # HEADER }}}

	html.Div([ # FILTER MENU {{{
			html.Div([
				html.P( "Type", className="control_label"),
				dcc.Dropdown(
					id='type',
					options=[{'label':t, 'value':t} for t in TYPES],
					value=TYPES[0], placeholder='Select type...',
					className="dcc_control"
				),
			], style={'display':'inline-block', 'width':'40%', 'vertical-align':'middle'}),
			html.Div([
				html.P( "Date", className="control_label"),
				dcc.DatePickerSingle(
					id='date',
					display_format="D MM YYYY",
					style={"border": "0px solid black"},
					className="dcc_control"
				),
			], style={'display':'inline-block', 'vertical-align':'middle'}),
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
			dcc.Slider(
				id='forecast_time_slider',
				min=0,
				max=0, # this will be overwritten when the file is loaded
				value=0, # this will be overwritten on button callback
				step=1,
				className="dcc_control"
			)
		],
		id="filter_options",
		className="pretty_container three columns", # defined in the .css file
	), # FILTER MENU }}}

	html.Div([ # GRAPHS {{{
			html.Div(
				[dcc.Graph(id='map')],
				className="pretty_container seven columns",
			),
			html.Div(
				[dcc.Graph(id='timeseries')],
				className="pretty_container seven columns",
			),
		],
		className="row flex-display",
	), # GRAPHS }}}

], id="mainContainer", style={'font-family':FONT,"display": "flex", "flex-direction": "column"})

# =========================
# Main plot update callback

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
					z = data[var][forecast_time].data))

	fig.update_xaxes(autorange=False, range=[-110, 40])
	fig.update_yaxes()

	fig.update_layout(height=800, showlegend=False, autosize=True, margin=go.layout.Margin(l=0,r=0,t=0,b=0),
			plot_bgcolor= 'rgba(0, 0, 0, 0)', paper_bgcolor= 'rgba(0, 0, 0, 0)')

	return fig

@app.callback(
		Output('timeseries', 'figure'),
		[Input('type', 'value'),
		Input('date', 'date'),
		Input('file', 'value'),
		Input('map', 'clickData')])
def update_graph(type, date, file, click):

	fig = go.Figure()

	if not click: return fig

	cx = click['points'][0]['x']
	cy = click['points'][0]['y']

	if date:
		date = date.replace('-','') + '00'

		if os.path.isfile(f"{DATA_PATH}/{type}/{date}/{file}"):
			data = get_dataset(f"{DATA_PATH}/{type}/{date}/{file}")
			for v in list(data.data_vars):
				fig.add_trace(go.Scatter(
					x = [str(t) for t in data[v].time.data],
					y = data[v].sel({'latitude':cy, 'longitude':cx}),
					mode = 'lines', name = v))

	fig.update_xaxes()
	fig.update_yaxes()

	fig.update_layout(title='Time Series', uirevision=True)

	return fig

# TOP ROW DROPDOWN UPDATES
# ========================

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

# ===========================

@app.callback(
		Output('forecast_time_slider', 'max'),
		[Input('type', 'value'),
		Input('date', 'date'),
		Input('file', 'value')])
def update_forecast_slider_size(type, date, file):
	if date:
		date = date.replace('-','') + '00'
		if os.path.isfile(f"{DATA_PATH}/{type}/{date}/{file}"):
			return get_dataset(f"{DATA_PATH}/{type}/{date}/{file}").time.size -1
	return 0

@app.callback(
		Output('forecast_time_slider', 'marks'),
		[Input('type', 'value'),
		Input('date', 'date'),
		Input('file', 'value')])
def update_forecast_slider_size(type, date, file):
	if date:
		date = date.replace('-','') + '00'
		if os.path.isfile(f"{DATA_PATH}/{type}/{date}/{file}"):
			f = get_dataset(f"{DATA_PATH}/{type}/{date}/{file}")
			return {i: str(i) for i,x in enumerate(f.time)}
	return {}

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
			return 'Forecast Time: ' + str(f.time[forecast_time].data)
	return 'Forecast Time: '

if __name__ == '__main__':
	app.run_server(host='0.0.0.0', port=8888, debug=True)
