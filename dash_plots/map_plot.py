import dash
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
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

# ========================
#	  DASH APP SETUP
# ========================

app = dash.Dash(__name__)

app.layout = html.Div([
	html.H2(children = 'Ensemble Comparison', style={'text-align':'center'}),

	html.Div([
		html.Div(children='Type:'),
		dcc.Dropdown(
			id='type',
			options=[{'label':t, 'value':t} for t in TYPES],
			value=TYPES[0], placeholder='Select type...'
		)
	], style={'width':'24%', 'display':'inline-block'}),
	html.Div([
		html.Div(children='Date: '),
		dcc.Dropdown(
			id='date',
			placeholder='Select date...'
		)
	], style={'width':'24%', 'display':'inline-block'}),
	html.Div([
		html.Div(children='File: '),
		dcc.Dropdown(
			id='file',
			placeholder='Select file...'
		)
	], style={'width':'24%', 'display':'inline-block'}),
	html.Div([
		html.Div(children='Variable: '),
		dcc.Dropdown(
			id='variable',
			placeholder='Select variable...'
		)
	], style={'width':'24%', 'display':'inline-block'}),

	dcc.Graph(id='map'),

	html.Div(id='display_forecast_time'),

	html.Div([
		html.Div(children='Days:', style={'display':'inline-block', 'vertical-align':'50%'}),
		html.Div([dcc.Slider(
			id='forecast_time_slider',
			min=0,
			max=0, # this will be overwritten when the file is loaded
			value=0, # this will be overwritten on button callback
			step=1
			)], style={'width':'90%', 'display':'inline-block'})
	]),

	html.Div([
		html.Div(children='Step:', style={'display':'inline-block', 'padding-right':'15px'}),
		html.Button(
			id='step-forecast-backward',
			children='⮜',
			style={'width':'40px', 'height':'25px', 'display':'inline-block', 'border-block-style':'solid', 'border-radius':'4px'}),
		html.Button(
			id='step-forecast-forward',
			children='⮞',
			style={'width':'40px', 'height':'25px', 'display':'inline-block', 'border-block-style':'solid', 'border-radius':'4px'})
	])

], style={'font-family':FONT})


@app.callback(
		Output('map', 'figure'),
		[Input('type', 'value'),
		Input('date', 'value'),
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

	if os.path.isfile(f"{DATA_PATH}/{type}/{date}/{file}"):
		# TODO create a file cache to avoid this
		data = get_dataset(f"{DATA_PATH}/{type}/{date}/{file}")
		if var:
			fig.add_trace(go.Contour(
				x = data.longitude.data,
				y = data.latitude.data,
				z = data[var][forecast_time].data))

	fig.update_xaxes()
	fig.update_yaxes()

	fig.update_layout(title='SOME TITLE', width=1500, height=750, uirevision=True)

	return fig

# TOP ROW DROPDOWN UPDATES
# ========================

@app.callback(
		Output('date', 'options'),
		[Input('type', 'value')])
def update_available_dates(type):
	return [{'label':d, 'value':d} for d in [x.split('/')[-1] for x in glob.glob(DATA_PATH+type+'/*')]]

@app.callback(
		Output('file', 'options'),
		[Input('type', 'value'),
		Input('date', 'value')])
def update_available_variables(type, date):
	return [{'label':d, 'value':d} for d in [x.split('/')[-1] for x in glob.glob(f"{DATA_PATH}/{type}/{date}/*")]]

@app.callback(
		Output('variable', 'options'),
		[Input('type', 'value'),
		Input('date', 'value'),
		Input('file', 'value')])
def update_available_variables(type, date, file):
	if os.path.isfile(f"{DATA_PATH}/{type}/{date}/{file}"):
		return [{'label':d, 'value':d} for d in list(get_dataset(f"{DATA_PATH}/{type}/{date}/{file}").data_vars)]
	else:
		return []

# ===========================

@app.callback(
		Output('forecast_time_slider', 'max'),
		[Input('type', 'value'),
		Input('date', 'value'),
		Input('file', 'value')])
def update_forecast_slider_size(type, date, file):
	if os.path.isfile(f"{DATA_PATH}/{type}/{date}/{file}"):
		return get_dataset(f"{DATA_PATH}/{type}/{date}/{file}").time.size -1
	return 0

@app.callback(
		Output('forecast_time_slider', 'marks'),
		[Input('type', 'value'),
		Input('date', 'value'),
		Input('file', 'value')])
def update_forecast_slider_size(type, date, file):
	if os.path.isfile(f"{DATA_PATH}/{type}/{date}/{file}"):
		f = get_dataset(f"{DATA_PATH}/{type}/{date}/{file}")
		return {i: str(i) for i,x in enumerate(f.time)}
	return {}

@app.callback(
		Output('display_forecast_time', 'children'),
		[Input('type', 'value'),
		Input('date', 'value'),
		Input('file', 'value'),
		Input('forecast_time_slider', 'value')])
def update_forecast_time_display(type, date, file, forecast_time):
	if os.path.isfile(f"{DATA_PATH}/{type}/{date}/{file}"):
		f = get_dataset(f"{DATA_PATH}/{type}/{date}/{file}")
		return 'Forecast Time: ' + str(f.time[forecast_time].data)
	return 'Forecast Time: '

@app.callback(
		Output('forecast_time_slider', 'value'),
		[Input('step-forecast-forward', 'n_clicks'),
		Input('step-forecast-backward', 'n_clicks')],
		[State('forecast_time_slider', 'value')])
def step_forecast(forward_clicks, backward_clicks, val):
	cid = [p['prop_id'] for p in dash.callback_context.triggered][0]
	if 'step-forecast-forward' in cid:
		return val + 1 if val + 1 < 100 else val # TODO replace 100 with maximum allowed forecast time
	elif 'step-forecast-backward' in cid:
		return val - 1 if val > 0 else val
	else:
		return 0

if __name__ == '__main__':
	app.run_server(host='0.0.0.0', port=8888, debug=True)
