import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

st.set_page_config(page_title="F1 Paddock Analytics", page_icon="🏁", layout="wide", initial_sidebar_state="expanded")

# Final 2024 drivers' championship standings. Results and points verified against Formula1.com.
DATA = [
    (1,"Max Verstappen","Red Bull Racing",9,437),
    (2,"Lando Norris","McLaren",4,374),
    (3,"Charles Leclerc","Ferrari",3,356),
    (4,"Oscar Piastri","McLaren",2,292),
    (5,"Carlos Sainz","Ferrari",2,290),
    (6,"George Russell","Mercedes",2,245),
    (7,"Lewis Hamilton","Mercedes",2,223),
    (8,"Sergio Perez","Red Bull Racing",0,152),
    (9,"Fernando Alonso","Aston Martin",0,70),
    (10,"Pierre Gasly","Alpine",0,42),
    (11,"Nico Hulkenberg","Haas",0,41),
    (12,"Yuki Tsunoda","RB",0,30),
    (13,"Lance Stroll","Aston Martin",0,24),
    (14,"Esteban Ocon","Alpine",0,23),
    (15,"Kevin Magnussen","Haas",0,16),
    (16,"Alexander Albon","Williams",0,12),
    (17,"Daniel Ricciardo","RB",0,12),
    (18,"Oliver Bearman","Haas",0,7),
    (19,"Franco Colapinto","Williams",0,5),
    (20,"Zhou Guanyu","Kick Sauber",0,4),
    (21,"Liam Lawson","RB",0,4),
    (22,"Valtteri Bottas","Kick Sauber",0,0),
    (23,"Logan Sargeant","Williams",0,0),
    (24,"Jack Doohan","Alpine",0,0),
]
df = pd.DataFrame(DATA, columns=["Position","Driver","Team","Wins","Points"])
df["Championship Group"] = np.where(df["Position"] <= 10, "Top 10", "Outside Top 10")
TEAMS = sorted(df["Team"].unique())
TEAM_COLORS = {
    "McLaren":"#ff8000", "Ferrari":"#e10600", "Red Bull Racing":"#3671c6",
    "Mercedes":"#27f4d2", "Aston Martin":"#229971", "Alpine":"#ff87bc",
    "Haas":"#b6babd", "RB":"#6692ff", "Williams":"#64c4ff", "Kick Sauber":"#52e252"
}
RED = "#e10600"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800;900&family=Inter:wght@400;500;600;700&display=swap');
:root { --f1red:#e10600; --carbon:#15151e; --panel:#20202b; }
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background: radial-gradient(ellipse at 92% 0%, rgba(225,6,0,.16), transparent 32%), #0d0d12; color:#f4f4f6; }
[data-testid="stHeader"] { background:rgba(13,13,18,.86); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#191922,#101016); border-right:1px solid #30303b; }
h1,h2,h3 { font-family:'Barlow Condensed',sans-serif !important; letter-spacing:.3px; }
.hero { padding:1.4rem 1.5rem 1.3rem; border:1px solid #393944; border-radius:18px; background:linear-gradient(110deg,rgba(225,6,0,.2),rgba(29,29,39,.95) 48%,rgba(30,30,40,.82)); position:relative; overflow:hidden; }
.hero:after { content:'🏁'; position:absolute; right:22px; top:10px; font-size:4.6rem; opacity:.12; transform:rotate(-8deg); }
.eyebrow { color:#ff5148; text-transform:uppercase; letter-spacing:2.4px; font-size:.72rem; font-weight:800; }
.hero-title { font-family:'Barlow Condensed',sans-serif; font-size:3.2rem; line-height:1; font-weight:900; margin:.3rem 0 .55rem; }
.hero-sub { color:#c5c5ce; font-size:.95rem; max-width:760px; }
.metric-card { background:linear-gradient(145deg,#20202b,#17171f); border:1px solid #343440; border-radius:14px; padding:16px 18px; min-height:112px; }
.metric-label { color:#a9a9b6; font-size:.76rem; text-transform:uppercase; letter-spacing:1.2px; font-weight:700; }
.metric-value { font-family:'Barlow Condensed',sans-serif; font-size:2.1rem; font-weight:800; margin-top:4px; }
.metric-note { color:#ff6b63; font-size:.78rem; }
.section-kicker { color:#ff5148; font-size:.72rem; text-transform:uppercase; letter-spacing:2px; font-weight:800; margin-bottom:.15rem; }
.pill { display:inline-block; border:1px solid #454552; color:#e5e5eb; border-radius:99px; padding:5px 10px; margin:3px 5px 3px 0; font-size:.75rem; background:#1b1b24; }
.stTabs [data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid #383843; }
.stTabs [data-baseweb="tab"] { background:#1a1a23; border-radius:8px 8px 0 0; padding:10px 15px; }
.stTabs [aria-selected="true"] { border-bottom:3px solid #e10600 !important; }
div[data-testid="stMetric"] { background:#1b1b24; padding:12px 14px; border:1px solid #343440; border-radius:12px; }
.stButton button { background:#e10600; color:white; border:0; border-radius:8px; font-weight:800; }
.stButton button:hover { background:#ff2b20; color:white; border:0; }
div[data-testid="stDataFrame"] { border:1px solid #33333d; border-radius:10px; overflow:hidden; }
small { color:#a5a5b1; }
hr { border-color:#353540; }
</style>
""", unsafe_allow_html=True)

def plot_theme(fig, height=390):
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#ededf2"),
        margin=dict(l=18,r=18,t=45,b=18), legend_title_text=""
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.08)", zerolinecolor="rgba(255,255,255,.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.08)", zerolinecolor="rgba(255,255,255,.12)")
    return fig

def metric_card(label, value, note):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🏁 Paddock Menu")
    page = st.radio("Navigate", ["Race Control", "Driver Grid", "Data Garage", "Points Lab", "Top 10 Predictor", "Methodology"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="eyebrow">Season under review</div><h2 style="margin-top:0">2024</h2><span class="pill">24 drivers</span><span class="pill">10 teams</span><span class="pill">24 rounds</span>', unsafe_allow_html=True)
    st.caption("Final drivers’ championship classification")
    st.markdown("---")
    st.markdown("**Data source**")
    st.markdown("[Formula 1 — 2024 Drivers’ Standings](https://www.formula1.com/en/results/2024/drivers)")

st.markdown('<div class="hero"><div class="eyebrow">Formula 1 · Season Review · 2024</div><div class="hero-title">F1 PADDOCK ANALYTICS</div><div class="hero-sub">From the starting grid to the chequered flag — explore driver performance, statistical patterns and machine-learning predictions in one race-inspired dashboard.</div><div style="margin-top:13px"><span class="pill">📊 Statistical analysis</span><span class="pill">⚙️ Machine learning</span><span class="pill">🏆 Championship insights</span></div></div>', unsafe_allow_html=True)
st.write("")

if page == "Race Control":
    st.markdown('<div class="section-kicker">Race control · Championship snapshot</div>', unsafe_allow_html=True)
    cols=st.columns(4)
    with cols[0]: metric_card("World champion","Max Verstappen","9 race wins")
    with cols[1]: metric_card("Top points","437","Championship points")
    with cols[2]: metric_card("Most wins","9","Across the 2024 season")
    with cols[3]: metric_card("Drivers classified","24","Final season dataset")
    st.write("")
    left,right=st.columns([1.45,1])
    with left:
        st.subheader("🏆 Championship podium")
        podium=df.head(3).copy()
        fig=go.Figure()
        fig.add_trace(go.Bar(x=podium["Points"],y=podium["Driver"],orientation="h",
                             marker_color=["#e6c15a","#bfc7d5","#bd7b48"],
                             text=podium["Points"],textposition="outside",hovertemplate="%{y}<br>%{x} points<extra></extra>"))
        fig.update_layout(yaxis=dict(autorange="reversed"),xaxis_title="Championship points",showlegend=False)
        st.plotly_chart(plot_theme(fig,330),use_container_width=True)
    with right:
        st.subheader("🚦 Points by team")
        team_pts=df.groupby("Team",as_index=False)["Points"].sum().sort_values("Points",ascending=True)
        fig=px.bar(team_pts,x="Points",y="Team",orientation="h",color="Team",color_discrete_map=TEAM_COLORS)
        fig.update_layout(showlegend=False)
        st.plotly_chart(plot_theme(fig,330),use_container_width=True)
    st.subheader("⚡ The season at a glance")
    a,b,c=st.columns(3)
    with a:
        st.markdown("**Pole-to-flag consistency**")
        st.write("Explore how race wins and points align across the final driver standings.")
    with b:
        st.markdown("**Team battle**")
        st.write("Compare combined driver points across the ten constructors.")
    with c:
        st.markdown("**Predictive pit wall**")
        st.write("Compare five regression models and five Top 10 classifiers.")

elif page == "Driver Grid":
    st.markdown('<div class="section-kicker">The grid · Final classification</div>', unsafe_allow_html=True)
    st.subheader("🪪 Driver standings")
    q=st.text_input("Search a driver or team", placeholder="Try Norris, Ferrari, Mercedes…")
    filtered=df[df["Driver"].str.contains(q,case=False,na=False)|df["Team"].str.contains(q,case=False,na=False)] if q else df
    st.dataframe(filtered, hide_index=True, use_container_width=True, height=530,
                 column_config={"Position":st.column_config.NumberColumn("Pos.",format="%d"),
                                "Points":st.column_config.ProgressColumn("Points",min_value=0,max_value=437,format="%d"),
                                "Wins":st.column_config.NumberColumn("Wins",format="%d")})
    st.subheader("🏎️ Driver performance map")
    fig=px.scatter(df,x="Wins",y="Points",size="Points",color="Team",hover_name="Driver",
                   color_discrete_map=TEAM_COLORS, size_max=25, custom_data=["Position"])
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>Wins: %{x}<br>Points: %{y}<extra></extra>")
    st.plotly_chart(plot_theme(fig,430),use_container_width=True)

elif page == "Data Garage":
    st.markdown('<div class="section-kicker">Telemetry · Explore the dataset</div>', unsafe_allow_html=True)
    st.subheader("🔧 Data inspection")
    m1,m2,m3=st.columns(3)
    m1.metric("Rows",len(df)); m2.metric("Fields",len(df.columns)); m3.metric("Missing values",int(df.isna().sum().sum()))
    tab1,tab2,tab3=st.tabs(["Data view","Statistical pit stop","Visual telemetry"])
    with tab1:
        st.dataframe(df,hide_index=True,use_container_width=True)
        st.download_button("⬇️ Download driver data (CSV)",df.to_csv(index=False).encode("utf-8"),"f1_2024_driver_standings.csv","text/csv")
    with tab2:
        st.dataframe(df[["Position","Wins","Points"]].describe().T.round(2),use_container_width=True)
        st.markdown("**Data types**")
        st.dataframe(pd.DataFrame({"Column":df.columns,"Data type":[str(t) for t in df.dtypes]}),hide_index=True,use_container_width=True)
    with tab3:
        x=st.selectbox("Choose the x-axis",["Wins","Position","Points"],index=0)
        y=st.selectbox("Choose the y-axis",["Points","Wins","Position"],index=0)
        fig=px.scatter(df,x=x,y=y,color="Team",hover_name="Driver",size="Points",size_max=22,color_discrete_map=TEAM_COLORS)
        st.plotly_chart(plot_theme(fig,420),use_container_width=True)
        fig2=px.histogram(df,x="Points",nbins=12,color_discrete_sequence=[RED],title="Distribution of championship points")
        st.plotly_chart(plot_theme(fig2,330),use_container_width=True)

elif page == "Points Lab":
    st.markdown('<div class="section-kicker">Machine learning · Regression garage</div>', unsafe_allow_html=True)
    st.subheader("🎯 Predict championship points")
    st.write("Compare five regression algorithms trained on the season’s **race wins** and **team**.")
    reg_df=df[["Wins","Team","Points"]].copy()
    X=reg_df[["Wins","Team"]]; y=reg_df["Points"]
    numeric=["Wins"]; categorical=["Team"]
    prep=ColumnTransformer([
        ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),
        ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)
    ])
    models={
        "Multiple Linear Regression":LinearRegression(),
        "Ridge Regression":Ridge(alpha=1.0),
        "Lasso Regression":Lasso(alpha=.1,max_iter=10000),
        "Decision Tree":DecisionTreeRegressor(max_depth=3,random_state=42),
        "Random Forest":RandomForestRegressor(n_estimators=200,max_depth=4,random_state=42)
    }
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42)
    results=[]; pipes={}
    for name,model in models.items():
        pipe=Pipeline([("prep",prep),("model",model)])
        pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
        cv=KFold(n_splits=4,shuffle=True,random_state=42)
        cv_mae=-cross_val_score(pipe,Xtr,ytr,cv=cv,scoring="neg_mean_absolute_error").mean()
        results.append({"Model":name,"Test MAE":mean_absolute_error(yte,pred),
                        "Test RMSE":np.sqrt(mean_squared_error(yte,pred)),"Test R²":r2_score(yte,pred),
                        "CV MAE":cv_mae})
        pipes[name]=pipe
    rdf=pd.DataFrame(results).sort_values("CV MAE")
    best=rdf.iloc[0]["Model"]
    col1,col2=st.columns([1.3,1])
    with col1:
        st.markdown("**Model lap times** *(lower error is better)*")
        fig=px.bar(rdf.sort_values("CV MAE",ascending=True),x="CV MAE",y="Model",orientation="h",
                   color="CV MAE",color_continuous_scale=["#ff6b61","#e10600","#7e0000"],text_auto=".1f")
        st.plotly_chart(plot_theme(fig,350),use_container_width=True)
    with col2:
        st.markdown("**Best cross-validation score**")
        st.markdown(f'<div class="metric-card"><div class="metric-label">Selected model</div><div class="metric-value" style="font-size:1.65rem">{best}</div><div class="metric-note">Lowest CV MAE</div></div>',unsafe_allow_html=True)
        st.caption("MAE and RMSE measure prediction error. R² describes explained variation.")
    st.dataframe(rdf.round(3),hide_index=True,use_container_width=True)
    st.divider()
    st.markdown("**🧪 Try a points estimate**")
    a,b=st.columns(2)
    with a:
        team=st.selectbox("Team",TEAMS,key="reg_team")
        wins=st.slider("Race wins",0,24,2,key="reg_wins")
    with b:
        model_choice=st.selectbox("Model",list(models.keys()),index=list(models.keys()).index(best),key="reg_model")
        st.write("The estimate is based on historical 2024 driver records.")
        if st.button("🏁 Run points prediction",key="reg_predict"):
            val=float(pipes[model_choice].predict(pd.DataFrame({"Wins":[wins],"Team":[team]}))[0])
            st.success(f"Estimated championship points: {max(0,val):.1f}")

elif page == "Top 10 Predictor":
    st.markdown('<div class="section-kicker">Machine learning · Classification garage</div>', unsafe_allow_html=True)
    st.subheader("🏁 Who makes the Top 10?")
    st.write("Choose a team, race wins and points. The classifier estimates whether a driver profile resembles a **Top 10** or **Outside Top 10** finisher.")
    st.markdown('<span class="pill">Logistic Regression</span><span class="pill">Decision Tree</span><span class="pill">Random Forest</span><span class="pill">KNN</span><span class="pill">SVM</span>',unsafe_allow_html=True)
    # Points is included as an input because wins/team alone cannot distinguish drivers with the same win count.
    cdf=df[["Position","Team","Wins","Points"]].copy()
    cdf["Result"]=np.where(cdf["Position"]<=10,"Top 10","Outside Top 10")
    Xc=cdf[["Wins","Points","Team"]]; yc=cdf["Result"]
    prep_c=ColumnTransformer([
        ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),["Wins","Points"]),
        ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),["Team"])
    ])
    clfs={
        "Logistic Regression":LogisticRegression(max_iter=3000,class_weight="balanced"),
        "Decision Tree":DecisionTreeClassifier(max_depth=4,class_weight="balanced",random_state=42),
        "Random Forest":RandomForestClassifier(n_estimators=300,max_depth=4,class_weight="balanced",random_state=42),
        "KNN":KNeighborsClassifier(n_neighbors=3),
        "SVM":SVC(kernel="rbf",probability=True,class_weight="balanced",random_state=42)
    }
    Xct,Xcv,yct,ycv=train_test_split(Xc,yc,test_size=.25,random_state=42,stratify=yc)
    cr=[]; cpipes={}
    for name,clf in clfs.items():
        pipe=Pipeline([("prep",prep_c),("model",clf)])
        pipe.fit(Xct,yct); pred=pipe.predict(Xcv)
        cv=StratifiedKFold(n_splits=4,shuffle=True,random_state=42)
        f1cv=cross_val_score(pipe,Xct,yct,cv=cv,scoring="f1_macro").mean()
        cr.append({"Model":name,"Accuracy":accuracy_score(ycv,pred),
                   "Precision (macro)":precision_score(ycv,pred,average="macro",zero_division=0),
                   "Recall (macro)":recall_score(ycv,pred,average="macro",zero_division=0),
                   "F1 (macro)":f1_score(ycv,pred,average="macro",zero_division=0),
                   "CV F1 (macro)":f1cv})
        cpipes[name]=pipe
    crdf=pd.DataFrame(cr).sort_values("CV F1 (macro)",ascending=False)
    bestc=crdf.iloc[0]["Model"]
    topcol, chartcol=st.columns([1,1.25])
    with topcol:
        st.markdown("**Prediction inputs**")
        with st.form("classifier_form"):
            in_team=st.selectbox("Team",TEAMS,index=TEAMS.index("Ferrari"))
            in_wins=st.number_input("Race wins",min_value=0,max_value=24,value=0,step=1)
            in_points=st.number_input("Championship points",min_value=0,max_value=600,value=25,step=1)
            in_model=st.selectbox("Classifier",list(clfs.keys()),index=list(clfs.keys()).index(bestc))
            submitted=st.form_submit_button("🚦 Predict championship group")
        if submitted:
            sample=pd.DataFrame({"Wins":[int(in_wins)],"Points":[int(in_points)],"Team":[in_team]})
            pipe=cpipes[in_model]
            pred=pipe.predict(sample)[0]
            probs=pipe.predict_proba(sample)[0]
            classes=list(pipe.named_steps["model"].classes_)
            prob=float(probs[classes.index(pred)])
            if pred=="Top 10":
                st.success(f"🏆 Predicted group: {pred}")
            else:
                st.info(f"🏁 Predicted group: {pred}")
            st.progress(prob,text=f"Model confidence for predicted class: {prob:.0%}")
            st.caption("This is a model prediction, not a guaranteed race outcome.")
    with chartcol:
        st.markdown("**Classifier qualifying table**")
        fig=px.bar(crdf,x="CV F1 (macro)",y="Model",orientation="h",color="CV F1 (macro)",
                   color_continuous_scale=["#5a1010","#e10600","#ff817a"],text_auto=".2f")
        st.plotly_chart(plot_theme(fig,340),use_container_width=True)
    st.subheader("📋 Head-to-head results")
    st.dataframe(crdf.round(3),hide_index=True,use_container_width=True)
    st.subheader("🔲 Confusion matrix · Best model")
    cm_pipe=cpipes[bestc]; cm_pred=cm_pipe.predict(Xcv)
    labels=["Top 10","Outside Top 10"]
    cm=confusion_matrix(ycv,cm_pred,labels=labels)
    fig=px.imshow(cm,x=labels,y=labels,text_auto=True,color_continuous_scale=[[0,"#21151a"],[.5,"#8c1510"],[1,"#e10600"]],aspect="auto")
    fig.update_layout(xaxis_title="Predicted class",yaxis_title="Actual class")
    st.plotly_chart(plot_theme(fig,310),use_container_width=True)
    with st.expander("View classification report"):
        st.code(classification_report(ycv,cm_pred,labels=labels,zero_division=0))

elif page == "Methodology":
    st.markdown('<div class="section-kicker">Behind the visor · Project notes</div>', unsafe_allow_html=True)
    st.subheader("🧰 What powers this dashboard?")
    steps=[
        ("01","Source","Final 2024 Formula 1 drivers’ championship standings."),
        ("02","Prepare","Create a structured DataFrame, inspect columns, and check missing values and duplicates."),
        ("03","Explore","Calculate descriptive statistics and visualize points, wins, drivers and teams."),
        ("04","Model","Compare five regression algorithms and five classification algorithms."),
        ("05","Evaluate","Use MAE, RMSE and R² for regression; accuracy, precision, recall, F1 and a confusion matrix for classification.")
    ]
    for n,title,desc in steps:
        st.markdown(f'<div class="metric-card" style="display:flex;gap:16px;align-items:center;margin-bottom:10px"><div style="font-family:Barlow Condensed;font-size:2rem;color:#e10600;font-weight:900">{n}</div><div><b>{title}</b><div style="color:#b9b9c5;margin-top:4px">{desc}</div></div></div>',unsafe_allow_html=True)
    st.subheader("📚 Official season references")
    st.markdown("- [Formula 1 — 2024 Drivers’ Standings](https://www.formula1.com/en/results/2024/drivers)\n- [FIA — 2024 Classifications](https://www.fia.com/events/fia-formula-one-world-championship/season-2024/2024-classifications-0)")
    st.caption("The dashboard focuses on the final 2024 drivers’ championship. Model results are calculated from the included dataset when the app runs.")

st.markdown("---")
st.markdown('<div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;color:#8f8f9b;font-size:.75rem"><span>🏁 F1 PADDOCK ANALYTICS</span><span>2024 season · Statistical analysis & machine learning</span><span>Lights out. Data on.</span></div>',unsafe_allow_html=True)
