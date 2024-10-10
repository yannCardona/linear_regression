import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np

# Instantiate the model
model = LinearModel(lr=0.01, learning_iterations=1000)
model.train('data.csv')

# Create the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Linear Regression Dashboard'),

    # Input for mileage
    html.Label('Enter Mileage:'),
    dcc.Input(id='mileage-input', type='number', value=50000),
    html.Div(id='price-output'),

    # Graphs
    html.Div([
        html.H2('Data and Regression Line'),
        html.Img(id='data-plot'),
    ]),
    html.Div([
        html.H2('Sum of Squared Residuals Surface Plot'),
        html.Img(id='ssr-plot'),
    ])
])

@app.callback(
    Output('price-output', 'children'),
    Input('mileage-input', 'value')
)
def update_price_output(mileage):
    estimated_price = model.estimate_price(mileage)
    return f'Estimated Price: {estimated_price:.2f}'

@app.callback(
    Output('data-plot', 'src'),
    Input('mileage-input', 'value')
)
def update_data_plot(mileage):
    # Use the existing plot_data method
    return model.plot_data()

@app.callback(
    Output('ssr-plot', 'src'),
    Input('mileage-input', 'value')
)
def update_ssr_plot(mileage):
    # Use the existing plot_ssr method
    return model.plot_ssr()

if __name__ == '__main__':
    app.run_server(debug=True)
