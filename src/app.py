import base64
import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import networkx as nx
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import pandas as pd
import io
import itertools
from dash.dependencies import ALL

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Initial Bayesian Network setup (without edges)
model = BayesianNetwork()
model_fitted = False  # Global variable to track whether model needs to be refitted
nodes = []
df = None

# Function to recursively get all descendants (children, grandchildren, etc.) of the selected variable
def get_descendants_for_selected_variable(model, selected_var):
    descendants = set()
    children = model.get_children(selected_var)
    
    for child in children:
        descendants.add(child)
        # Recursively add descendants of the child
        descendants.update(get_descendants_for_selected_variable(model, child))
    
    return descendants

# Function to perform inference and return DataFrame for selected variable and its descendants
def inference_for_selected_variable_and_descendants(evidence, selected_var):
    inference = VariableElimination(model)

    # Get all descendants of the selected variable (children, grandchildren, etc.)
    descendant_vars = get_descendants_for_selected_variable(model, selected_var)

    # Include the selected variable itself in the list of variables to infer
    variables_to_infer = [selected_var] + list(descendant_vars)

    # Filter the evidence to exclude both the selected variable and its descendants
    filtered_evidence = {k: v for k, v in evidence.items() if k != selected_var and k not in descendant_vars}

    results = []

    # Perform inference for each variable (selected variable and its descendants)
    for var in variables_to_infer:
        query_result = inference.query(variables=[var], evidence=filtered_evidence)
        states = query_result.state_names[var]
        probabilities = query_result.values

        # Create a DataFrame for each variable (selected or descendant)
        result_df = pd.DataFrame({
            'Variable': [var] * len(states),
            'State': states,
            'Probability': probabilities
        })

        results.append(result_df)

    # Concatenate results from all variables into one DataFrame
    final_df = pd.concat(results, ignore_index=True)
    return final_df

# Function to visualize the Bayesian Network as a graph with arrows
def create_network_graph(edges):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G, k=0.6)

    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                     line=dict(width=2, color='black'),
                                     hoverinfo='none', mode='lines'))

    node_trace = go.Scatter(x=[pos[node][0] for node in G.nodes()],
                            y=[pos[node][1] for node in G.nodes()],
                            mode='markers+text',
                            text=[node for node in G.nodes()],
                            textposition="bottom center",
                            hoverinfo='text',
                            marker=dict(color='blue', size=10, line_width=2))

    layout = go.Layout(
        showlegend=False,
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        hovermode='closest',
        height=400,
    )

    return go.Figure(data=edge_trace + [node_trace], layout=layout)

# Function to dynamically generate evidence dropdowns, excluding the selected child variable
def generate_evidence_dropdowns(child_variable):
    evidence_dropdowns = []
    
    for node in nodes:
        dropdown = dcc.Dropdown(
            id={'type': 'dynamic_input', 'index': node},  # Dynamic input matching
            options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
            value=1,
            clearable=False,
            disabled=(node == child_variable)  # Disable the child variable dropdown
        )
        evidence_dropdowns.append(
            dbc.Col([
                html.Label(f"Select Evidence for '{node}':", style={'font-weight': 'bold'}),
                dropdown
            ], width=6)
        )
    
    return evidence_dropdowns




# Define the layout of the app
app.layout = dbc.Container([
    html.H1("Dynamic Bayesian Network Editor", style={'textAlign': 'center', 'margin-bottom': '30px', 'font-size': '36px'}),

    # File upload component
    dcc.Upload(
        id='upload_data',
        children=html.Div(['Drag and Drop or ', html.A('Select a CSV File')]),
        style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
               'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin-bottom': '20px'}
    ),

    # Placeholder for file upload feedback
    html.Div(id='upload_status', style={'margin-bottom': '20px'}),

    # Buttons for clearing and resetting network and evidence
    dbc.Row([
        dbc.Col(dbc.Button('Clear Network', id='clear_network_btn', color='danger', style={'width': '100%'}), width=4),
        dbc.Col(dbc.Button('Reset Evidence', id='reset_evidence_btn', color='secondary', style={'width': '100%'}), width=4)
    ], style={'margin-bottom': '20px'}),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='network_graph', figure=create_network_graph([]))
        ], width=6),

        dbc.Col([

            # Dropdowns for selecting source and target nodes to add/remove edges
            dbc.Row([
                dbc.Col([
                    html.Label("Select Parent Node:", style={'font-weight': 'bold'}),
                    dcc.Dropdown(id='parent_node', clearable=False)
                ], width=6),
                dbc.Col([
                    html.Label("Select Child Node:", style={'font-weight': 'bold'}),
                    dcc.Dropdown(id='child_node', clearable=False)
                ], width=6)
            ], style={'margin-bottom': '20px'}),

            # Action dropdown and update button
            dbc.Row([ 
                dbc.Col([
                    html.Label("Action:", style={'font-weight': 'bold'}),
                    dcc.Dropdown(id='action', options=[
                        {'label': 'Add Edge', 'value': 'add'},
                        {'label': 'Remove Edge', 'value': 'remove'}
                    ], value='add', clearable=False)
                ], width=6),
                dbc.Col([
                    dbc.Button('Update Network', id='update_network_btn', color='primary', style={'width': '100%', 'margin-top': '25px'})
                ], width=6)
            ], style={'margin-bottom': '20px'}),

            # Evidence dropdowns will be dynamically generated
            dbc.Row(id='evidence_dropdowns', style={'margin-bottom': '20px'}),

            dbc.Row([
                dbc.Col([
                    html.Label("Select Child Variable:", style={'font-weight': 'bold'}),
                    dcc.Dropdown(id='child_variable', clearable=False)
                ], width=6),
                dbc.Col([
                    dbc.Button('Perform Inference', id='perform_inference_btn', color='success', style={'width': '100%', 'margin-top': '25px'})
                ], width=6)
            ], style={'margin-bottom': '20px'}),

            # Inference result (table will be displayed here)
            dbc.Row([
                dbc.Col([
                    html.H3("Inference Results", style={'margin-top': '20px', 'font-weight': 'bold'}),
                    dash_table.DataTable(id='inference_output', 
                                         style_table={'margin': '0 auto', 'width': '80%'},
                                         style_cell={'textAlign': 'center'},
                                         style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}),
                    html.Div(id='inference_status', style={'marginTop': '20px', 'color': 'red'})  # For displaying status messages
                ])
            ])
        ], width=6)
    ])
])

# Callback to handle network updates, file upload, and dynamic evidence dropdown generation
@app.callback(
    [Output('network_graph', 'figure'),
     Output('evidence_dropdowns', 'children'),
     Output('upload_status', 'children'),
     Output('parent_node', 'options'),
     Output('child_node', 'options'),
     Output('child_variable', 'options')],
    [Input('update_network_btn', 'n_clicks'),
     Input('child_variable', 'value'),
     Input('upload_data', 'contents'),
     Input('clear_network_btn', 'n_clicks'),
     Input('reset_evidence_btn', 'n_clicks')],
    [State('upload_data', 'filename'),
     State('parent_node', 'value'),
     State('child_node', 'value'),
     State('action', 'value')],
    prevent_initial_call=True
)
def handle_callbacks(n_clicks_update, child_variable, contents, n_clicks_clear, n_clicks_reset, filename, parent_node, child_node, action):
    global df, nodes, model_fitted, model

    triggered_event = callback_context.triggered[0]['prop_id'].split('.')[0]
    
    # File upload handling
    if triggered_event == 'upload_data' and contents:
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            # Convert data to binary if necessary
            if not set(df.values.flatten()).issubset({0, 1}):
                df = df.applymap(lambda x: 0 if x <= 8 else 1)
            
            nodes = df.columns.tolist()
            node_options = [{'label': node, 'value': node} for node in nodes]

            # Generate evidence dropdowns excluding the current child variable (reset on upload)
            evidence_dropdowns = generate_evidence_dropdowns(child_variable)

            return create_network_graph([]), evidence_dropdowns, f"{filename} uploaded successfully.", node_options, node_options, node_options
        except Exception as e:
            return create_network_graph([]), [], f"Error processing {filename}: {str(e)}", [], [], []

    # Handle network update (add/remove edges)
    if triggered_event == 'update_network_btn' and parent_node and child_node and action:
        if action == 'add':
            model.add_edge(parent_node, child_node)
            model_fitted = False
        elif action == 'remove':
            model.remove_edge(parent_node, child_node)
            model_fitted = False
        return create_network_graph(list(model.edges())), dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Clear the network
    if triggered_event == 'clear_network_btn':
        model = BayesianNetwork()
        return create_network_graph([]), dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Reset evidence (resets all dropdowns)
    if triggered_event == 'reset_evidence_btn':
        return dash.no_update, generate_evidence_dropdowns(child_variable), dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Handle child variable change
    if triggered_event == 'child_variable':
        # Dynamically update dropdowns when child variable changes
        evidence_dropdowns = generate_evidence_dropdowns(child_variable)
        return dash.no_update, evidence_dropdowns, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

# Updated callback to handle the inference and pass the correct evidence dynamically
@app.callback(
    [Output('inference_output', 'data'),
     Output('inference_output', 'columns'),
     Output('inference_status', 'children')],
    Input('perform_inference_btn', 'n_clicks'),
    [State('child_variable', 'value')],
    [State({'type': 'dynamic_input', 'index': ALL}, 'value')],
    prevent_initial_call=True
)
@app.callback(
    [Output('inference_output', 'data', allow_duplicate=True),
     Output('inference_output', 'columns', allow_duplicate=True),
     Output('inference_status', 'children', allow_duplicate=True)],
    Input('perform_inference_btn', 'n_clicks'),
    [State('child_variable', 'value')],
    [State({'type': 'dynamic_input', 'index': ALL}, 'value')],
    prevent_initial_call=True
)
def perform_inference(n_clicks, child_variable, evidence_values):
    global model_fitted

    # Ensure that nodes and child_variable are valid
    if not nodes or not child_variable:
        return [], [], "No nodes or child variable available for inference"

    # Collect evidence nodes excluding the child variable
    evidence_nodes = [node for node in nodes if node != child_variable]

    # Ensure we are excluding the child variable from evidence values
    filtered_evidence_values = evidence_values[:len(evidence_nodes)]

    # Debugging step to ensure lengths match
    print(f"Evidence Nodes: {evidence_nodes}")
    print(f"Filtered Evidence Values: {filtered_evidence_values}")

    # Check if evidence values are correctly captured and match the length
    if len(filtered_evidence_values) != len(evidence_nodes):
        return [], [], "Mismatch in evidence input length"

    # Construct the evidence dictionary
    evidence = {node: filtered_evidence_values[i] for i, node in enumerate(evidence_nodes)}

    # Fit the model if necessary
    if not model_fitted:
        try:
            model.fit(df, estimator=MaximumLikelihoodEstimator)
            model_fitted = True
        except Exception as e:
            return [], [], f"Error fitting model: {e}"

    try:
        # Perform inference on the selected child variable and its descendants
        result_df = inference_for_selected_variable_and_descendants(evidence, child_variable)
        columns = [{"name": i, "id": i} for i in result_df.columns]
        data = result_df.to_dict('records')
        return data, columns, ""
    except Exception as e:
        return [], [], f"Error during inference: {e}"
    
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
