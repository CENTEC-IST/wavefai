import dash
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

import xarray

FONT = 'Courier New'

# ========================
#	   READING DATA
# ========================

# nearest
# GFS_DATA = '../data/buoys/GFS_selectionPointNearest_20190925_20200701.nc'
# GWES_DATA = '../data/buoys/GWES_selectionPointNearest_20190925_20200701.nc'
# WW3_DATA = '../data/buoys/WW3_selectionPointNearest_20190925_20200701.nc'

# average
GFS_DATA = '../data/buoys/GFS_selectionPointAver_20190925_20200701.nc'
GWES_DATA = '../data/buoys/GWES_selectionPointAver_20190925_20200701.nc'
WW3_DATA = '../data/buoys/WW3_selectionPointAver_20190925_20200701.nc'

OBSERVED_DATA = '../data/buoys/NDBC_selection_deepWaters_20190925_20200701.nc'

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

# ========================
#	  DASH APP SETUP
# ========================

app = dash.Dash(__name__)

app.layout = html.Div([
	html.H2(children = 'Ensemble Comparison', style={'text-align':'center'}),

	html.Div([
		html.Div(children='Station:'),
		dcc.Dropdown(
			id='station-id',
			options=[{'label':v, 'value':i} for i, v in enumerate(stations)],
			value=0, placeholder='Select station...'
		)
	], style={'width':'33%', 'display':'inline-block'}),
	html.Div([
		html.Div(children='Variable: '),
		dcc.Dropdown(
			id='variable',
			options=[{'label':v, 'value':i} for i, v in enumerate(variables)],
			value=0, placeholder='Select variable...'
		)
	], style={'width':'33%', 'display':'inline-block'}),
	html.Div([
		html.Div(children='Ensemble: '),
		dcc.Dropdown(
			id='ensemble',
			options=[{'label':ens, 'value':ens} for ens in ensmembers],
			value=['GWES'], multi=True, placeholder='Select ensembles...'
		)
	], style={'width':'33%', 'display':'inline-block'}),

	dcc.Graph(id='timeseries-graph'),

	html.Div(id='display-forecast-time'),

	html.Div([
		html.Div(children='Days:', style={'display':'inline-block', 'vertical-align':'50%'}),
		html.Div([dcc.Slider(
			id='forecast-time-slider',
			min=0,
			max=ensmembers[list(ensmembers.keys())[0]][fctime_name].size-1,
			value=0, # this will be overwritten on button callback
			# this creates marks only when the number of seconds represents a full day
			# i*4 represents the index of that time so that it correctly selects this value
			marks={i*8: str(int(x/(3600*24))) for i,x in enumerate(ensmembers[list(ensmembers.keys())[0]][fctime_name].isel({fctime_name:slice(0,180,8)}))},
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
		Output('timeseries-graph', 'figure'),
		[Input('station-id', 'value'),
		Input('ensemble', 'value'),
		Input('variable', 'value'),
		Input('forecast-time-slider', 'value')])
def update_graph(station, ensemble, var, forecast_time):
	fig = go.Figure()

	if (variables[var] in obs_data):
		# grab the time from the ensembles
		fc = ensmembers[list(ensmembers.keys())[0]][fctime_name][forecast_time]
		actual_time = (ensmembers[list(ensmembers.keys())[0]].datetime + fc/3600/24).data
		mask = np.in1d(obs_data.datetime, actual_time)
		qobs = np.nanpercentile(obs_data[variables[var]][station, :][mask].data, range(1, 100))

		for e in ensemble:
			if variables[var] == 'WSPD' and 'U10m' in ensmembers[e] and 'V10m' in ensmembers[e]:
				qensdata = np.sqrt(ensmembers[e]['U10m'][station, :, forecast_time].data**2 + ensmembers[e]['V10m'][station, :, forecast_time].data**2)
				qens = np.nanpercentile(qensdata, range(1, 100))

				fig.add_trace(go.Scatter( x = qobs, y = qens, mode = 'lines+markers', name = e))

			elif variables[var] in ensmembers[e]:
				if len(ensmembers[e][variables[var]].shape) == 4: # if there are multiple ensembleMembers
					qensdata = np.mean(ensmembers[e][variables[var]][station, :, :, forecast_time], axis=0).data
				else:
					qensdata = ensmembers[e][variables[var]][station, :, forecast_time].data
				qens = np.nanpercentile(qensdata, range(1, 100))

				fig.add_trace(go.Scatter( x = qobs, y = qens, mode = 'lines+markers', name = e))

		# Draw Truth line
		fig.add_trace(go.Scatter(
				x = qobs,
				y = qobs,
				mode = 'lines', name = 'TRUTH',
				marker_color='#000'))

	fig.update_xaxes(title = 'Measurements', fixedrange=True)
	fig.update_yaxes(title = 'Model', fixedrange=True)
	fig.update_layout(title='QQ Plot', uirevision=True)

	return fig

@app.callback(
		Output('display-forecast-time', 'children'),
		[Input('forecast-time-slider', 'value')])
def update_forecast_time_display(forecast_time):
	day_hour = [(int(x//3600//24), int(x//3600%24)) for x in ensmembers[list(ensmembers.keys())[0]][fctime_name].data]
	return f"Forecast time: {'%d dias %d horas' % day_hour[forecast_time]}. Indice {forecast_time}."

@app.callback(
		Output('forecast-time-slider', 'value'),
		[Input('step-forecast-forward', 'n_clicks'),
		Input('step-forecast-backward', 'n_clicks')],
		[State('forecast-time-slider', 'value')])
def step_forecast(forward_clicks, backward_clicks, val):
	cid = [p['prop_id'] for p in dash.callback_context.triggered][0]
	if 'step-forecast-forward' in cid:
		return val + 1 if val + 1 < ensmembers[list(ensmembers.keys())[0]][fctime_name].size else val
	elif 'step-forecast-backward' in cid:
		return val - 1 if val > 0 else val
	else:
		return 0


if __name__ == '__main__':
	app.run_server(host='0.0.0.0', port=8888, debug=True)
