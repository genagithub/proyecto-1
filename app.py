import pandas as pd
import numpy as np
import plotly.express as px
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier


df  = pd.read_csv("data/customer_invoices.zip", encoding="ISO-8859-1")

df = df.dropna(subset=["Description", "CustomerID"])
df["CustomerID"] = df["CustomerID"].astype(int)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

date_cut = pd.to_datetime("2011-09-01")

df_obs = df[df["InvoiceDate"] < date_cut].copy()
df_pred = df[df["InvoiceDate"] >= date_cut].copy()

max_date = df["InvoiceDate"].max()

last_contact = df_pred.groupby("CustomerID")["InvoiceDate"].max().reset_index()
last_contact.columns = ["CustomerID", "LastDatePred"]
last_contact["RecencyTarget"] = (max_date - last_contact["LastDatePred"]).dt.days

customers_to_retain = pd.DataFrame({"CustomerID": df_obs["CustomerID"].unique()})

target_df = pd.merge(customers_to_retain, last_contact, on="CustomerID", how="left")

last_date_obs = df_obs.groupby("CustomerID")["InvoiceDate"].max().reset_index()
last_date_obs.columns = ["CustomerID", "LastDateObs"]
target_df = pd.merge(target_df, last_date_obs, on="CustomerID", how="left")

target_df["RecencyTarget"] = target_df["RecencyTarget"].fillna((max_date - target_df["LastDateObs"]).dt.days)

p75 = target_df["RecencyTarget"].quantile(0.75)
p90 = target_df["RecencyTarget"].quantile(0.90)

def assign_class(recency):
    if recency <= p75:
        return 0  
    elif recency <= p90:
        return 1  
    else:
        return 2  

target_df["Target"] = target_df["RecencyTarget"].apply(assign_class)
features_df = pd.DataFrame({"CustomerID": df_obs["CustomerID"].unique()})

finances = df_obs.groupby("CustomerID").agg(
    TotalExpenditure=("Revenue", "sum"),
    TicketAverage=("Revenue", "mean"),
    ExpenditureVariance=("Revenue", "std"),
    TotalQuantities=("Quantity", "sum")
).reset_index()

finances["ExpenditureVariance"] = finances["ExpenditureVariance"].fillna(0)
features_df = pd.merge(features_df, finances, on="CustomerID", how="left")

frecuency = df_obs.groupby("CustomerID")["InvoiceNo"].nunique().reset_index()
frecuency.columns = ["CustomerID", "FrecuencyHist"]
features_df = pd.merge(features_df, frecuency, on="CustomerID", how="left")

skus = df_obs.groupby("CustomerID")["StockCode"].nunique().reset_index()
skus.columns = ["CustomerID", "DiversitySKUs"]
features_df = pd.merge(features_df, skus, on="CustomerID", how="left")

df_obs["Cancellations"] = df_obs["InvoiceNo"].astype(str).str.startswith("C").astype(int)
cancellations = df_obs.groupby("CustomerID")["Cancellations"].mean().reset_index()
cancellations.columns = ["CustomerID", "CancellationsRate"]
features_df = pd.merge(features_df, cancellations, on="CustomerID", how="left")

date_cut_obs = df_obs["InvoiceDate"].max()
recency_obs = df_obs.groupby("CustomerID")["InvoiceDate"].max().reset_index()
recency_obs["RecencyObs"] = (date_cut_obs - recency_obs["InvoiceDate"]).dt.days
features_df = pd.merge(features_df, recency_obs[["CustomerID", "RecencyObs"]], on="CustomerID", how="left")

buys_date = df_obs.groupby(["CustomerID", "InvoiceNo"])["InvoiceDate"].min().reset_index()
buys_date = buys_date.sort_values(by=["CustomerID", "InvoiceDate"])
buys_date["DaysBetweenBuys"] = buys_date.groupby("CustomerID")["InvoiceDate"].diff().dt.days

purchasing_pace = buys_date.groupby("CustomerID").agg(
    AverageTBP=("DaysBetweenBuys", "mean"),
    VarianceTBP=("DaysBetweenBuys", "var")
).reset_index()

purchasing_pace = pd.merge(purchasing_pace, features_df[["CustomerID", "RecencyObs"]], on="CustomerID", how="left")
purchasing_pace["AverageTBP"] = purchasing_pace["AverageTBP"].fillna(purchasing_pace["RecencyObs"] + 1)
purchasing_pace["VarianceTBP"] = purchasing_pace["VarianceTBP"].fillna(0)
purchasing_pace["AverageTBP"] = purchasing_pace["AverageTBP"].replace(0, 0.5)

features_df = pd.merge(features_df, purchasing_pace[["CustomerID", "AverageTBP", "VarianceTBP"]], on="CustomerID", how="left")
features_df["ExcessivePace"] = features_df["RecencyObs"] / features_df["AverageTBP"]

features_df["PurchasingStability"] = np.where(
    features_df["AverageTBP"] > 0, 
    np.sqrt(features_df["VarianceTBP"]) / features_df["AverageTBP"], 
    0
)

geography = df_obs.groupby("CustomerID")["Country"].first().reset_index()
geography["IsUK"] = (geography["Country"] == "United Kingdom").astype(int)
features_df = pd.merge(features_df, geography[["CustomerID", "IsUK"]], on="CustomerID", how="left")

df_model = pd.merge(features_df, target_df[["CustomerID", "Target"]], on="CustomerID", how="inner")

X_train, X_test, y_train, y_test = train_test_split(df_model.drop(columns=["CustomerID", "Target"]), 
                                                    df_model["Target"], 
                                                    test_size=0.20, 
                                                    random_state=42, 
                                                    stratify=df_model["Target"])

cart_model = DecisionTreeClassifier(criterion="entropy",
                                    max_depth=10,
                                    min_samples_leaf=10,  
                                    class_weight="balanced", 
                                    random_state=42)

cart_model.fit(X_train, y_train)

y_pred = cart_model.predict(X_test)
probabilities = cart_model.predict_proba(X_test)

analysis_roi = X_test.copy()
analysis_roi["RealTarget"] = y_test
analysis_roi["ModelPredict"] = y_pred
analysis_roi["MidProb"] = probabilities[:, 1]
analysis_roi["HighProb"] = probabilities[:, 2]
analysis_roi["TotalRiskProb"] = analysis_roi["MidProb"] + analysis_roi["HighProb"]
analysis_roi["CustomerID"] = df_model.loc[X_test.index, "CustomerID"]

CAMPAIGN_COST = 15.0      
CAMPAIGN_EFFECTIVENESS = 0.30 

analysis_roi["VEN"] = (analysis_roi["TotalExpenditure"] * analysis_roi["TotalRiskProb"] * CAMPAIGN_EFFECTIVENESS) - CAMPAIGN_COST
analysis_roi["CampaignPrescription"] = np.where(analysis_roi["VEN"] > 0, "APROBADO: Desplegar campaña", "RECHAZADO: No invertir")

total_customers_predicted_risk = (analysis_roi["ModelPredict"].isin([1, 2])).sum()
approved_campaigns = (analysis_roi["CampaignPrescription"] == "APROBADO: Desplegar campaña").sum()
budget_savings = (total_customers_predicted_risk  - approved_campaigns) * CAMPAIGN_COST

top_5_customers = analysis_roi.loc[(analysis_roi["CampaignPrescription"] == "APROBADO: Desplegar campaña") & (analysis_roi["TotalRiskProb"] >= 0.65),:]
top_5_customers = top_5_customers.sort_values(by="VEN", ascending=False).head(5)

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(id="body", className="e1_body", children=[
html.H1("Análisis prescriptivo de Churn Rate", id="title", className="e1_title"),
html.Div(id="dashboard", className="e1_dashboard", children=[
    html.Div(id="graph_div_1", className="e1_graph_div", children=[
        html.Div(id="dropdown_div_1", className="e1_dropdown_div", style={"justify-content":"flex-start"}, children=[
            dcc.Dropdown(id="dropdown_1", className="e1_dropdown",
                        options = [
                            {"label":"Gasto total","value":"TotalExpenditure"},
                            {"label":"Frecuencia histórica","value":"FrecuencyHist"},
                            {"label":"Tasa de cancelaciones","value":"CancellationsRate"},
                            {"label":"TBT promedio","value":"AverageTBT"}
                        ],
                        value="TotalExpenditure",
                        multi=False,
                        clearable=False)
        ]),
        dcc.Graph(id="histogram", className="e1_graph", figure={})
    ]),
    html.Div(id="graph_div_2", className="e1_graph_div", children=[
        html.Div(id="dropdown_div_2", className="e1_dropdown_div", style={"justify-content":"center"}, children=[
            dcc.Dropdown(id="dropdown_2", className="e1_dropdown", style={"padding-right":"5px"},
                        options = [
                            {"label":"Recencia","value":"RecencyObs"}, 
                            {"label":"Ritmo excedido","value":"ExcessivePace"}, 
                            {"label":"TBT promedio","value":"AverageTBT"}
                        ],
                        value="RecencyObs",
                        multi=False,
                        clearable=False),
            dcc.Dropdown(id="dropdown_3", className="e1_dropdown", style={"padding-left":"5px"},
                        options = [
                            {"label":"Gasto total","value":"TotalExpenditure"},
                            {"label":"Ticket promedio","value":"AverageTicket"}, 
                            {"label":"Cantidades totales","value":"TotalQuantities"}
                        ],
                        value="TotalExpenditure",
                        multi=False,
                        clearable=False)
        ]),
        dcc.Graph(id="scatterplot", className="e1_graph", figure={})
    ]),
]),
   html.H2("Optimización de presupuesto publicitario", id="H2", className="e1_title"),  
   html.Div(id="div_prescritive", className="e1_div_prescriptive", children=[
       html.Div(f"Total de clientes detectados en riesgo por el modelo: {total_customers_predicted_risk}", id="total_customers", className="e1_txt"),
       html.Div(f"Campañas de pago estratégicamente APROBADAS por ROI: {approved_campaigns}", id="aprove_campaigns", className="e1_txt"),
       html.Div(f"Campañas RECHAZADAS (Se ahorra pauta o pasa a canal gratuito): {total_customers_predicted_risk - approved_campaigns}", id="reject_campagins", className="e1_txt"),
       html.Div(f"Dinero directo RESCATADO / AHORRADO en presupuesto publicitario: ${budget_savings}", id="ROI", className="e1_txt"),
       html.H3("Top 5 clientes a fidelizar", id="H3", style={"font-family":"sans-serif","font-weight":"bold"}),
       html.Div(id="matrix", className="e1_matrix", children=[
            html.Div([html.B("ID", className="e1_header")], id="col_1"),
            html.Div([html.B("Gasto total", className="e1_header")], id="col_2"),
            html.Div([html.B("Probabilidad de Churn", className="e1_header")], id="col_3"),
            html.Div([html.B("Valor Esperado Neto", className="e1_header")], id="col_4"),
            *sum([
                [
                    html.Div(str(row["CustomerID"]), className="e1_cell"),
                    html.Div(f"${row["TotalExpenditure"]:,.2f}", className="e1_cell"),
                    html.Div(f"{row["HighProb"]:.2%}", className="e1_cell"),
                    html.Div(f"${row["VEN"]:,.2f}", className="e1_cell")
                ] for _, row in top_5_customers.iterrows()
            ], [])
       ])
  ])
])


@app.callback(
    [Output(component_id="histogram",component_property="figure"),
    Output(component_id="scatterplot",component_property="figure")],
    [Input(component_id="dropdown_1",component_property="value"),
    Input(component_id="dropdown_2",component_property="value"),
    Input(component_id="dropdown_3",component_property="value")]
)

def update_dashboard(slct_var_histogram, slct_var_X, slct_var_Y):

    histogram = px.histogram(
        df_model, 
        x=slct_var_histogram, 
        color="Target", 
        barmode="group",
        nbins=30,
        title=f"Distribución de clientes por {slct_var_histogram} y nivel de riesgo",
        color_discrete_map={0: "#2ecc71", 1: "#f1c40f", 2: "#e74c3c"}, 
        labels={"Target": "Estado de riesgo"}
    )
    
    histogram.update_layout(
        template="plotly_white",
        xaxis_title=slct_var_histogram,
        yaxis_title="Cantidad de clientes (volumen)",
        legend_title="Riesgo real"
    )

    scatterplot = px.scatter(
        df_model, 
        x=slct_var_X, 
        y=slct_var_Y,
        color="HighProb", 
        color_continuous_scale=px.colors.sequential.Reds, 
        title=f"Correlación: {slct_var_X} vs {slct_var_Y} (mapeo de probabilidad de Churn)",
        labels={"HighProb": "Probabilidad de Churn"},
        hover_data=[
            slct_var_X,
            slct_var_Y,
            "HighProb",
            "TotalExpenditure",
            "FrecuencyHist",
            "CustomerID"
        ]
    )

    scatterplot.update_traces(hovertemplate=None)
    
    scatterplot.update_layout(
        template="plotly_white",
        xaxis_title=slct_var_X,
        yaxis_title=slct_var_Y,
        coloraxis_colorbar=dict(title="Probabilidad<br>de Churn", tickformat=".2%")
    )

    return histogram, scatterplot

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050)) 
    app.run_server(host='0.0.0.0', port=port)
