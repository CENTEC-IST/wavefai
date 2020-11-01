import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

import xarray

# ========================
#	   READING DATA
# ========================

DATA_FILE = 'ECMWFifs_and_Obsv_StationPos_2017111300_2020082300.nc'

data = xarray.open_dataset(DATA_FILE)

stations = data.stationID.data
datetime = [str(d) for d in data.cycletime.data]

variables = sorted([d for d in data.data_vars if len(data[d].shape) >1])
obs_variables = variables[len(variables)//2:]
ens_variables = variables[:len(variables)//2]

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
			value=0, placeholder='Select station...'
		)
	], style={'width':'33%', 'display':'inline-block'}),
	html.Div([
		html.Div(children='Variable: '),
		dcc.Dropdown(
			id='variable',
			options=[{'label':v[v.index('_')+1:], 'value':i} for i, v in enumerate(obs_variables)],
			value=0, placeholder='Select variable...'
		)
	], style={'width':'33%', 'display':'inline-block'}),
	html.Div([
		html.Div(children='Ensemble: '),
		dcc.Dropdown(
			id='ensemble',
			options=[{'label':v, 'value':i} for i, v in enumerate(data.ensmember.data)],
			value=[0], multi=True, placeholder='Select ensembles...'
		)
	], style={'width':'33%', 'display':'inline-block'}),

	dcc.Graph(id='timeseries-graph'),

	html.Div(id='display-forecast-time'),
	dcc.Slider(
		id='forecast-time-slider',
		min=0,
		max=data.forecast_time.size-1,
		value=0,
		# this creates marks only when the number of seconds represents a full day
		# i*4 represents the index of that time so that it correctly selects this value
		marks={i*4: str(int(x/(3600*24))) for i,x in enumerate(data.forecast_time.isel({'forecast_time':slice(0,180,4)}))},
		step=1
	)

])

@app.callback(
		Output('timeseries-graph', 'figure'),
		[Input('station-id', 'value'),
		Input('ensemble', 'value'),
		Input('variable', 'value'),
		Input('forecast-time-slider', 'value')])
def update_graph(station, ensemble, variable, forecast_time):
	fig = go.Figure()
	fig.add_trace(go.Scatter(
			x = datetime,
			y = data[obs_variables[variable]][station, :, forecast_time].data,
			mode = 'lines+markers', name = 'Observed'))

	for e in ensemble:
		fig.add_trace(go.Scatter(
				x = datetime,
				y = data[ens_variables[variable]][station, e, :, forecast_time].data,
				mode = 'lines', name = f'Ensemble {e+1}'))
	var_name = variables[variable][variables[variable].index('_')+1:]
	fig.update_xaxes(title = 'Tempo')
	fig.update_yaxes(title = f"{var_name} ({data[obs_variables[variable]].units})")

	return fig

@app.callback(
		Output('display-forecast-time', 'children'),
		[Input('forecast-time-slider', 'value')])
def update_forecast_time_display(forecast_time):
	day_hour = [(int(x//3600//24), int(x//3600%24)) for x in data.forecast_time.data]
	return f"Forecast time: {'%d dias %d horas' % day_hour[forecast_time]}. Indice {forecast_time}."


if __name__ == '__main__':
	app.run_server(host='0.0.0.0', port=8888, debug=True)
