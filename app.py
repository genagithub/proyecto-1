import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder 
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn import tree


df  = pd.read_csv("data/employee.csv")

df_encoded = df.copy()
df_encoded[["Education","City","Gender","EverBenched"]] = OrdinalEncoder().fit_transform(df[["Education","City","Gender","EverBenched"]])

map = {
    0:"permanece",
    1:"sale"
}

df["LeaveOrNot_txt"] = df["LeaveOrNot"].apply(lambda x : map.get(x))


x_train, x_test, y_train, y_test = train_test_split(
                                                   df_encoded[df_encoded.columns[:-1]],
                                                   df_encoded["LeaveOrNot"], 
                                                   test_size=0.25)

tree_decision_clf = tree.DecisionTreeClassifier(criterion="entropy",
                                           max_depth=6, 
                                           min_samples_split=6, 
                                           min_samples_leaf=5, 
                                           ccp_alpha=0.003)  

model = tree_decision_clf.fit(x_train, y_train)

class_predicts = model.predict(x_test)
class_real = y_test.values

matrix_confusion = confusion_matrix(class_real,class_predicts)
TP = matrix_confusion[0,0]
FP = matrix_confusion[0,1]
FN = matrix_confusion[1,0]
TN = matrix_confusion[1,1]

accuracy = accuracy_score(class_real, class_predicts)
color_accuracy = "green"
if accuracy < 0.6:
    color_accuracy = "red"
accuracy_str = str(accuracy)

recall = recall_score(class_real, class_predicts)
color_recall = "green"
if recall < 0.6:
    color_recall = "red"
recall_str = str(recall)

precision = precision_score(class_real, class_predicts)
color_precision = "green"
if precision < 0.6:
    color_precision = "red"
precision_str = str(precision)

F1_score = f1_score(class_real, class_predicts)
color_f1 = "green"
if F1_score < 0.6:
    color_f1 = "red"
F1_score_str = str(F1_score)

predict_leave_percentage = class_predicts.flatten().mean() * 100
predict_not_leave_percentage = 100 - predict_leave_percentage

real_leave_percentage = class_real.flatten().mean() * 100
real_not_leave_percentage = 100 - real_leave_percentage

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(id="body",className="e1_body",children=[
html.A(href="https://github.com/genagithub/proyecto-1/blob/main/an%C3%A1lisis_predictivo_de_rotaci%C3%B3n_de_personal.ipynb",children=[html.H1("Análisis predictivo de rotación de personal",id="title",className="e1_title")]),
html.Div(className="e1_dashboards",children=[
    html.Div(id="graph_div_1",className="e1_graph_div",children=[
        html.Div(id="dropdown_div_1",className="e1_dropdown_div",children=[
            dcc.Dropdown(id="dropdown_1",className="e1_dropdown",
                        options = [
                            {"label":"Educación","value":"Education"},
                            {"label":"Año de incorporación","value":"JoiningYear"},
                            {"label":"Ciudad","value":"City"},
                            {"label":"Género","value":"Gender"},
                            {"label":"Siempre en banca","value":"EverBenched"}
                        ],
                        value="Education",
                        multi=False,
                        clearable=False)
        ]),
        dcc.Graph(id="piechart",className="e1_graph",figure={})
    ]),
    html.Div(id="graph_div_2",className="e1_graph_div",children=[
        html.Div(id="dropdown_div_2",className="e1_dropdown_div",children=[
            dcc.Dropdown(id="dropdown_2",className="e1_dropdown",
                        options = [
                            {"label":"Edad","value":"Age"},
                            {"label":"Nivel de pago","value":"PaymentTier"},
                            {"label":"Experiencia en el dominio","value":"ExperienceInCurrentDomain"}
                        ],
                        value="Age",
                        multi=False,
                        clearable=False)
        ]),
        dcc.Graph(id="bar",className="e1_graph",figure={})
    ]),
]),
    
    html.Div(className="e1_div", children=[
        html.Div(id="performance", className="e1_performance",children=[
            html.P([html.B("Clases reales", style={"color":"blue"}),"   vs.   ",html.B("Predicciones",style={"color":"red"})], style={"text-align":"center","font-family":"sans-serif"}),
            html.P(f"{round(predict_not_leave_percentage)}% permanece | {round(predict_leave_percentage)}% sale", className="e1_predicts"),
            html.P("-----------------------------------------------------------------",style={"margin":"0"}),
            html.P(f"{round(real_not_leave_percentage)}% permanece | {round(real_leave_percentage)}% sale", className="e1_real_class")
        ]),
        html.Div(id="metrics", className="e1_metrics", children=[
                html.P("Matriz de confusión", style={"font-size":"0.92em","text-align":"center","font-family":"sans-serif","font-weigth":"bold","margin-top":"15px"}),
                html.Div(id="matrix", className="e1_matrix", children=[
                html.Div([html.B(TP,style={"color":"green","font-family":"sans-serif"})],id="TP",className="e1_successes"), 
                html.Div([html.B(FP,style={"color":"red","font-family":"sans-serif"})],id="FP",className="e1_mistakes"),
                html.Div([html.B(FN,style={"color":"red","font-family":"sans-serif"})],id="FN",className="e1_mistakes"),
                html.Div([html.B(TN,style={"color":"green","font-family":"sans-serif"})],id="TN",className="e1_successes")
                ]),
                html.Div(id="scores",children=[
                html.Ul(id="list",children=[
                html.Li([f"Accuracy: ",html.B(accuracy_str[:4],style={"color":f"{color_accuracy}"})],id="accuracy",className="e1_score"),
                html.Li([f"Recall: ",html.B(recall_str[:4],style={"color":f"{color_recall}"})],id="recall",className="e1_score"),
                html.Li([f"Precision: ",html.B(precision_str[:4],style={"color":f"{color_precision}"})],id="precision",className="e1_score"),
                html.Li([f"F1 Score: ",html.B(F1_score_str[:4],style={"color":f"{color_f1}"})],id="f1_score",className="e1_score")
                ])
                
            ])
        ])
    ])
])


@app.callback(
    [Output(component_id="piechart",component_property="figure"),
    Output(component_id="bar",component_property="figure")],
    [Input(component_id="dropdown_1",component_property="value"),
    Input(component_id="dropdown_2",component_property="value")]
)

def update_graph(slct_var_cat, slct_var_num):
    
    df_stats = df.groupby(slct_var_cat).agg(exit_probability=("LeaveOrNot", "mean"), sample_size=("LeaveOrNot", "count")).reset_index()
    df_stats["probability_pct"] = (df_stats["exit_probability"] * 100).round(1)
    df_stats["display_label"] = (df_stats[slct_var_cat].astype(str) + " (n=" + df_stats["sample_size"].astype(str) + ")")

    exist_risk_bar = px.bar(
        df_stats,
        x="display_label",
        y="probability_pct",
        text="probability_pct",
        title="Riesgo de salida",
        labels={"probability_pct": "Probabilidad de Salida (%)", "display_label": slct_var_cat},
        color="probability_pct",
        color_continuous_scale="Reds"
    )

    exist_risk_bar.update_traces(texttemplate="%{text}%", textposition="outside")
    exist_risk_bar.update_layout(xaxis={"categoryorder": "total descending"}, yaxis_range=[0, 100], coloraxis_showscale=False, margin=dict(t=50, l=25, r=25, b=25))
    
    df_mean = df.groupby("LeaveOrNot_txt")[slct_var_num].mean().reset_index()
    
    mean_comparison_bar = px.bar(
        df_mean, 
        x="LeaveOrNot_txt", 
        y=slct_var_num,
        text=slct_var_num, 
        title="Comparación de medias",
        color="LeaveOrNot_txt",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    
    mean_comparison_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    mean_comparison_bar.update_layout(xaxis_title=" ", yaxis_title=" ",   yaxis_range=[0, 50], showlegend=False, margin=dict(t=50, l=25, r=25, b=25))
    
    return exist_risk_bar, mean_comparison_bar

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050)) 
    app.run_server(host='0.0.0.0', port=port)















