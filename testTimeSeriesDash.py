import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

import xarray

# ========================
#	   READING DATA
# ========================

DATA_FILE = 'ECMWFifs_and_Obsv_StationPos_2017111300_2020082300.nc'

data = xarray.open_dataset(DATA_FILE)

stations = data.stationID.data
datetime = data.date_time.data

# ========================
#	  DASH APP SETUP
# ========================

app = dash.Dash(__name__)

app.layout = html.Div([
	html.H2(children = DATA_FILE),

	# TODO make another dropdown to select variables
	# TODO make another dropdown to select ensemblers to use
	dcc.Dropdown(
		id='station-id',
		options=[{'label':v, 'value':i} for i, v in enumerate(stations)],
		# multi=True,
		value=0
	),
	dcc.Graph(id='timeseries-graph'),

	# TODO make a div with the slider to show the actual forecast being selected
	dcc.Slider(
		id='forecast-time-slider',
		min=0,
		max=data.forecast_time.size,
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
		Input('forecast-time-slider', 'value')])
def update_graph(station, forecast_time):
	fig = px.scatter(
			# TODO find a way to make this work....
			# data[
			# dict(
				x = datetime,
				y = data.omega_atmp[station, :, forecast_time].data)
				# name = 'Observed',
				# marker = dict(color='rgb(55, 83, 109)'))])
			# dict(
			# 	x = datetime,
			# 	y = data.ecmwf_atmp[station, 0, :, forecast_time].data,
			# 	name = 'Ensemble',
			# 	marker = dict(color='rgb(55, 83, 109)'))
			# ])

	fig.update_xaxes(title = 'Tempo')
	fig.update_yaxes(title = 'atmp')

	return fig


if __name__ == '__main__':
	app.run_server(host='0.0.0.0', port=8888, debug=True)
