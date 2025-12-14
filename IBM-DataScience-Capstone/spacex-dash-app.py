# python3.11 -m pip install pandas dash
# wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv"
# wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/t4-Vy4iOU19i8y6E3Px_ww/spacex-dash-app.py"


# Import required libraries
import pandas as pd
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px

# Read the airline data into pandas dataframe
spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Create a dash application
app = dash.Dash(__name__)

site_options = [
    {'label': site, 'value': site}
    for site in spacex_df['Launch Site'].unique()
]
site_options.insert(0, {'label': 'All Sites', 'value': 'ALL'})

# Create an app layout
app.layout = html.Div(children=[html.H1('SpaceX Launch Records Dashboard',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
                                # TASK 1: Add a dropdown list to enable Launch Site selection
                                # The default select value is for ALL sites
                                # dcc.Dropdown(id='site-dropdown',...)
                                dcc.Dropdown(
                                    id='site-dropdown',
                                    options=site_options,
                                    value='ALL',
                                    placeholder="Launch Site selection",
                                    searchable=True
                                ),
                                html.Br(),

                                # TASK 2: Add a pie chart to show the total successful launches count for all sites
                                # If a specific launch site was selected, show the Success vs. Failed counts for the site
                                html.Div(dcc.Graph(id='success-pie-chart')),
                                html.Br(),

                                html.P("Payload range (Kg):"),
                                # TASK 3: Add a slider to select payload range
                                dcc.RangeSlider(
                                    id='payload-slider',
                                    min=0,
                                    max=10000,
                                    step=1000,
                                    value=[min_payload, max_payload],
                                    marks={
                                        0: '0',
                                        2500: '2500',
                                        5000: '5000',
                                        7500: '7500',
                                        10000: '10000'
                                    }
                                ),

                                # TASK 4: Add a scatter chart to show the correlation between payload and launch success
                                html.Div(dcc.Graph(id='success-payload-scatter-chart')),
                                ])

# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output
@app.callback( Output(component_id='success-pie-chart', component_property='figure'),
               Input(component_id='site-dropdown', component_property='value'))

# Add computation to callback function and return graph
def get_graph(site_dropdown):
    if site_dropdown == 'ALL':
        # Total successful launches count for all sites (successes per site)
        df_all = spacex_df.groupby('Launch Site', as_index=False)['class'].sum()
        fig = px.pie(
            df_all,
            names='Launch Site',
            values='class',
            title='Total Successful Launches by Site'
        )
        return fig
    else:
        # Success vs Failed counts for the selected site
        df_site = spacex_df[spacex_df['Launch Site'] == site_dropdown]

        # Count occurrences of Class=1 and Class=0
        counts = df_site['class'].value_counts().rename_axis('class').reset_index(name='Count')

        # Map 0/1 to labels for readability
        counts['Outcome'] = counts['class'].map({1: 'Success', 0: 'Failure'})

        fig = px.pie(
            counts,
            names='Outcome',
            values='Count',
            title=f'Success vs Failure for site {site_dropdown}'
        )
        return fig

# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output
@app.callback( Output(component_id='success-payload-scatter-chart', component_property='figure'),
               [Input(component_id='site-dropdown', component_property='value'), Input(component_id='payload-slider', component_property='value')])
def get_graph(selected_site, selected_payload):

    # payload range from the slider
    low, high = selected_payload

    # filter by payload first
    df_filtered = spacex_df[
        (spacex_df['Payload Mass (kg)'] >= low) &
        (spacex_df['Payload Mass (kg)'] <= high)
    ]

    fig = px.scatter(
        df_filtered,
        x='Payload Mass (kg)',
        y='class',
        color='Booster Version Category',
        title=(
            'Correlation between Payload and Success for all Sites'
            if selected_site == 'ALL'
            else f'Correlation between Payload and Success for {selected_site}'
        )
    )

    # Optional: make the y-axis easier to read
    fig.update_yaxes(title='class (0 = Failure, 1 = Success)', tickmode='linear', dtick=1)

    return fig

# Run the app
if __name__ == '__main__':
    app.run()
