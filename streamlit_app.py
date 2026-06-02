import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import openai






st.set_page_config(page_title="AI Driven CLV & Retention Dashboard", layout="wide")

# -------------------------------
# PROFESSIONAL DASHBOARD THEME
# -------------------------------
st.markdown("""
<style>

/* ============================= */
/* MAIN APP BACKGROUND */
/* ============================= */

.stApp {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb, #90caf9);;
}
h1 {color:#0e2a47; text-align: centre;}
h2,h3{color:#1c3d63;}            
/* ============================= */
/* TEXT VISIBILITY FIX */
/* ============================= */

h1, h2, h3, h4, h5 {
    color: #111827 !important;
    font-weight: 700;
}

p, span, label, div {
    color: #1f2937 !important;
    font-size: 14px;
}

/* ============================= */
/* CONTAINER */
/* ============================= */

.block-container {
    padding: 1rem 1.5rem;
    max-width: 100%;
}

/* ============================= */
/* SIDEBAR */
/* ============================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #4A8DDC, #6b9cff);
    color: white;
}

/* ============================= */
/* DASHBOARD FRAME */
/* ============================= */

.dashboard-frame {
    background: #ffffff;
    border-radius: 16px;
    border: 2px solid #d1d5db;
    padding: 15px;
    margin-bottom: 20px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
}

/* ============================= */
/* DASHBOARD TITLE */
/* ============================= */

.dashboard-title {
    font-size: 24px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 15px;
    color: #111827;
}

/* ============================= */
/* KPI CARDS */
/* ============================= */

.kpi-card {
    background: linear-gradient(135deg, #4A8DDC, #6ea8fe);
    color: white !important;
    padding: 14px;
    border-radius: 12px;
    text-align: center;
    font-weight: 600;
    box-shadow: 0px 3px 8px rgba(0,0,0,0.15);
}
  /* REMOVE EMPTY SPACE ABOVE CHARTS */
.element-container:has(.js-plotly-plot) {
    margin-top: -10px;
}
/* ============================= */
/* CHART BOX (VERY IMPORTANT) */
/* ============================= */

.chart-box {
    background: #ffffff;
    border: 1.5px solid #e5e7eb;
    border-radius: 12px;
    padding: 8px;
    height: Auto;
    overflow: hidden;
}
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 0.5rem !important;
}
/* ============================= */
/* PLOTLY FIX */
/* ============================= */

.js-plotly-plot, .plotly, .stPlotlyChart {
    width: 100% !important;
    height: 100% !important;
}
 
/* ============================= */
/* REMOVE EXTRA SPACE */
/* ============================= */

[data-testid="stHorizontalBlock"] {
    gap: 0.8rem;
}

/* ============================= */
/* METRIC (STREAMLIT DEFAULT) */
/* ============================= */

[data-testid="stMetric"] {
    background: white;
    border-radius: 10px;
    padding: 10px;
    border: 1px solid #e5e7eb;
}

/* ============================= */
/* BUTTONS */
/* ============================= */

.stButton>button {
    background-color: #4A8DDC;
    color: white;
    border-radius: 8px;
    border: none;
}

.stButton>button:hover {
    background-color: #3b73c5;
}
       /* ============================= */
/* FILE UPLOADER FIX */
/* ============================= */

/* Main upload box */
[data-testid="stFileUploader"] {
    background-color: #ffffff !important;
    border: 2px dashed #d1d5db !important;
    border-radius: 12px !important;
    padding: 15px !important;
}

/* Upload text */
[data-testid="stFileUploader"] span {
    color: #1f2937 !important;
}

/* Browse button */
[data-testid="stFileUploader"] button {
    background-color: #4A8DDC !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 6px 12px !important;
}

/* Hover effect */
[data-testid="stFileUploader"] button:hover {
    background-color: #3b73c5 !important;
}

/* Drag & drop text fix */
[data-testid="stFileUploader"] div {
    color: #374151 !important;
}
/* ============================= */
/* RESPONSIVE FIX */
/* ============================= */

@media (max-width: 200px) {
    section[data-testid="stSidebar"] {
        width: 200px;
    }
    .dashboard-frame {
        margin-left: 220px;
    }
}

</style>
""", unsafe_allow_html=True)
# -------------------------------
# DASHBOARD FRAME FUNCTIONS
# -------------------------------

def start_dashboard(title):
    st.markdown(
        f"""
        <div class="dashboard-frame">
        <div class="dashboard-title">{title}</div>
        """,
        unsafe_allow_html=True
    )

def end_dashboard():
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------

st.title("AI Driven CLV & Retention Analytics System")

## 1. INIT
if "data" not in st.session_state:
    st.session_state["data"] = None
# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Upload Dataset",
        "Customer Overview Dashboard",
        "Customer Behavior Dashboard",
        "Churn Analysis Dashboard",
        "CLV Prediction Dashboard",
        "Retention Insights",
        "Customer Segmentation ",
        "Executive Dashboard",
        "Market Analysis Dashboard",
        "AI Insights Dashboard",
        "Analytical Dashboard",
        "AI Chatbot",
        "Interactive Customer Analytical Dashboard",
        "Project Workflow",
    ]
)
st.sidebar.success("Select 'Project Workflow' to see full pipeline")
# -------------------------------
# DATASET UPLOAD
# -------------------------------

if page == "Upload Dataset":

    st.header("Upload Customer Dataset")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        if "Revenue" not in df.columns:
            df["Revenue"] = df["MonthlySpend"]

        st.session_state["data"] = df

        st.success("Dataset Uploaded Successfully")

        st.write(df.head())
        st.dataframe(df.head())

# Load dataset
df = None
if "data" in st.session_state:
    df = st.session_state["data"]

# -----------------------------
# CUSTOMER OVERVIEW DASHBOARD
# -----------------------------
if page == "Customer Overview Dashboard" and "data" in st.session_state:

    import plotly.express as px

    df = st.session_state["data"]
    def style_chart(fig):
     fig.update_layout(
        height=H,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",

        # MAIN FIX
        font=dict(
            color="black",   
            size=13          
        ),

        title_font=dict(
            size=16,
            color="black"
        ),

        # AXIS FIX (IMPORTANT)
        xaxis=dict(
            title_font=dict(size=14, color="black"),
            tickfont=dict(size=12, color="black"),
            showgrid=True,
            gridcolor="#e6e6e6"
        ),

        yaxis=dict(
            title_font=dict(size=14, color="black"),
            tickfont=dict(size=12, color="black"),
            showgrid=True,
            gridcolor="#e6e6e6"
        ),

        margin=dict(l=10, r=10, t=40, b=10)
    )
     return fig

    # ================= STYLE =================
    st.markdown("""
    <style>

    .main { background: #f4f6f9; }
    .dashboard-frame {
    background-color: #f8fafc;
    padding: 20px;
    border-radius: 12px;
}

    .header {
        font-size: 26px;
        font-weight: bold;
        color: #000000;
        padding: 10px 0;
        margin-bottom: 15px;
        border-bottom: 2px solid #e5e7eb;
    }

    /* KPI COLORS */
    .kpi1 {background:#5b8ff9;}
    .kpi2 {background:#61d9a7;}
    .kpi3 {background:#f6bd16;}
    .kpi4 {background:#e8684a;}

    .kpi-card {
        padding: 12px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 15px;
    }

    .chart-card {
        background: white;
        padding: 8px;
        border-radius: 10px;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.15);
    }

    .right-panel {
        background: #2e75b6;
        padding: 15px;
        border-radius: 10px;
        color: white;
    }

    </style>
    """, unsafe_allow_html=True)

    # ================= OUTER FRAME =================
    st.markdown('<div class="dashboard-frame">''<div class="header">CUSTOMER OVERVIEW DASHBOARD</div>', unsafe_allow_html=True)

    # HEADER
    st.markdown('<div class="header">CUSTOMER OVERVIEW DASHBOARD</div>', unsafe_allow_html=True)

    # THEN ALL CONTENT
    st.markdown('<div class="content">', unsafe_allow_html=True)
    
    # LAYOUT
    left, right = st.columns([1,4])

    # ================= FILTER =================
    with left:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)

        st.markdown("### 📂 Category")
        segment = st.multiselect("", df["Segment"].unique(), default=df["Segment"].unique())

        st.markdown("### 📅 Year")
        tenure = st.slider("", int(df["Tenure"].min()), int(df["Tenure"].max()),
                           (int(df["Tenure"].min()), int(df["Tenure"].max())))

        st.markdown("### 📦 Type")
        sub = st.multiselect("", df["SubscriptionType"].unique(), default=df["SubscriptionType"].unique())

        st.markdown('</div>', unsafe_allow_html=True)

    # FILTER DATA
    df = df[
        (df["Segment"].isin(segment)) &
        (df["SubscriptionType"].isin(sub)) &
        (df["Tenure"].between(tenure[0], tenure[1]))
    ]

    # ================= RIGHT =================
    with right:

        # KPI ROW
        k1,k2,k3 = st.columns(3)
        k1.markdown(f'<div class="kpi">💰 Sales<br>{round(df["TotalSpend"].sum(),0)}</div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi">📈 Profit<br>{round(df["CLV"].sum(),0)}</div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi">👥 Customers<br>{df.shape[0]}</div>', unsafe_allow_html=True)

        H = 230   # reduced height to REMOVE SCROLL

        # ===== ROW 1 =====
        c1,c2,c3 = st.columns(3)

        with c1:
            st.markdown('<div class="chart">', unsafe_allow_html=True)
            fig = px.bar(df.groupby("Segment")["TotalSpend"].sum().reset_index(),
                         x="Segment", y="TotalSpend", title="Sales by Category")
            fig = style_chart(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="chart">', unsafe_allow_html=True)
            fig = px.pie(df, names="Segment", title="Customer Distribution")
            fig = style_chart(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="chart">', unsafe_allow_html=True)
            fig = px.histogram(df, x="CLV", color="Segment", title="Profit Distribution")
            fig = style_chart(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ===== ROW 2 =====
        c1,c2 = st.columns(2)

        with c1:
           st.markdown('<div class="chart">', unsafe_allow_html=True)
           fig = px.box(df, x="Segment", y="MonthlySpend", title="Monthly Spend")
           fig = style_chart(fig)
           st.plotly_chart(fig, use_container_width=True)
           st.markdown('</div>', unsafe_allow_html=True)

        with c2:
           st.markdown('<div class="chart">', unsafe_allow_html=True)
           fig = px.violin(df, x="Segment", y="CLV", title="CLV Spread")
           fig = style_chart(fig)
           st.plotly_chart(fig, use_container_width=True)
           st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


    end_dashboard()       
# -----------------------------
# Customer Behavior Dashboard
# -----------------------------
if page == "Customer Behavior Dashboard" and df is not None:

    start_dashboard("Customer Behavior Dashboard")

    import plotly.express as px

    # ================= STYLE =================
    st.markdown("""
    <style>

    .main { background: #f4f6f9; }

    .header {
        font-size: 26px;
        font-weight: bold;
        color: #2e75b6;
        padding: 10px;
    }

    /* KPI COLORS */
    .kpi1 {background:#5b8ff9;}
    .kpi2 {background:#61d9a7;}
    .kpi3 {background:#f6bd16;}
    .kpi4 {background:#e8684a;}

    .kpi-card {
        padding: 12px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 15px;
    }

    .chart-card {
        background: white;
        padding: 8px;
        border-radius: 10px;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.15);
    }

    .right-panel {
        background: #2e75b6;
        padding: 15px;
        border-radius: 10px;
        color: white;
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="header">📊 Customer Behaviour Dashboard</div>', unsafe_allow_html=True)

    # ================= MAIN LAYOUT =================
    left, right = st.columns([4,1])

    # ================= LEFT SIDE =================
    with left:

        # KPI
        c1,c2,c3,c4 = st.columns(4)

        c1.markdown(f'<div class="kpi-card kpi1">Avg Spend<br>{round(df["MonthlySpend"].mean(),2)}</div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="kpi-card kpi2">Avg Tenure<br>{round(df["Tenure"].mean(),2)}</div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="kpi-card kpi3">Avg Age<br>{round(df["Age"].mean(),2)}</div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="kpi-card kpi4">Revenue<br>{round(df["TotalSpend"].sum(),2)}</div>', unsafe_allow_html=True)

        H = 210

        def style(fig):
            fig.update_layout(template="plotly_white", height=H,
                              margin=dict(l=10,r=10,t=40,b=10))
            return fig

        # ===== ROW 1 =====
        c1,c2,c3 = st.columns(3)

        with c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig = px.scatter(df, x="Age", y="MonthlySpend", color="Segment",
                             title="Age vs Spend")
            st.plotly_chart(style(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig = px.box(df, x="Segment", y="MonthlySpend",
                         title="Spend by Segment")
            st.plotly_chart(style(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig = px.histogram(df, x="Tenure", color="Segment",
                               title="Tenure Dist")
            st.plotly_chart(style(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ===== ROW 2 =====
        c1,c2 = st.columns(2)

        with c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig = px.violin(df, x="Segment", y="MonthlySpend",
                            title="Spending Pattern")
            st.plotly_chart(style(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig = px.scatter(df, x="Tenure", y="CLV", color="Segment",
                             size="MonthlySpend", title="Tenure vs CLV")
            st.plotly_chart(style(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ===== ROW 3 (NEW MAP + TREND) =====
        c1,c2 = st.columns(2)

        # MAP (if Country exists)
        with c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            if "Country" in df.columns:
                fig = px.choropleth(df,
                                    locations="Country",
                                    locationmode="country names",
                                    color="Revenue",
                                    title="Revenue by Country")
                st.plotly_chart(style(fig), use_container_width=True)
           
            st.markdown('</div>', unsafe_allow_html=True)

        # TREND
        with c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            trend = df.groupby("Tenure")["TotalSpend"].sum().reset_index()
            fig = px.line(trend, x="Tenure", y="TotalSpend",
                          title="Revenue Trend")
            st.plotly_chart(style(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ================= RIGHT PANEL =================
    with right:

        st.markdown('<div class="right-panel">', unsafe_allow_html=True)

        st.markdown("### Filters")

        segment = st.multiselect("Segment", df["Segment"].unique(), default=df["Segment"].unique())
        tenure = st.slider("Tenure", int(df["Tenure"].min()), int(df["Tenure"].max()),
                           (int(df["Tenure"].min()), int(df["Tenure"].max())))

        # APPLY FILTER
        df = df[
            (df["Segment"].isin(segment)) &
            (df["Tenure"].between(tenure[0], tenure[1]))
        ]

        st.markdown("---")
        st.markdown("Insights Panel")
        st.write("• High spenders in Segment 2")
        st.write("• CLV increases with tenure")

        st.markdown('</div>', unsafe_allow_html=True)

    end_dashboard()

# -----------------------------
# CHURN ANALYSIS DASHBOARD
# -----------------------------

if page == "Churn Analysis Dashboard" and df is not None:

    start_dashboard("Churn Analysis Dashboard")
    import plotly.express as px

    # ================= STYLE =================
    st.markdown("""
    <style>
    .main { background: #eef2f7; }

    .header {
        font-size: 26px;
        font-weight: bold;
        color: #2e75b6;
        text-align: center;
        padding: 10px;
    }

    .kpi {
        background: #2e75b6;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        color: white;
        font-weight: bold;
    }

    .kpi-value {
        font-size: 22px;
        color: black;
        background: white;
        padding: 6px;
        border-radius: 5px;
        margin-top: 5px;
    }

    .chart {
        background: white;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.2);
    }

    .filter-box {
        background: #dbe5f1;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="header">Customer Churn Analysis</div>', unsafe_allow_html=True)

    # ================= LAYOUT =================
    left, right = st.columns([1,4])

    # ================= FILTER =================
    with left:
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)

        segment = st.multiselect("Segment", df["Segment"].unique(), default=df["Segment"].unique())

        sub = st.multiselect("Subscription Type",
                             df["SubscriptionType"].unique(),
                             default=df["SubscriptionType"].unique())

        tenure = st.slider("Tenure",
                           int(df["Tenure"].min()),
                           int(df["Tenure"].max()),
                           (int(df["Tenure"].min()), int(df["Tenure"].max())))

        st.markdown('</div>', unsafe_allow_html=True)

    # APPLY FILTER
    df = df[
        (df["Segment"].isin(segment)) &
        (df["SubscriptionType"].isin(sub)) &
        (df["Tenure"].between(tenure[0], tenure[1]))
    ]

    # ================= RIGHT =================
    with right:

        # KPI
        total = df.shape[0]
        churned = df[df["Churn"] == 1].shape[0]
        churn_rate = round((churned / total) * 100, 1) if total > 0 else 0

        k1,k2,k3 = st.columns(3)

        k1.markdown(f'<div class="kpi">Total Customers<div class="kpi-value">{total}</div></div>', unsafe_allow_html=True)
        k2.markdown(f'<div class="kpi">Churn Customers<div class="kpi-value">{churned}</div></div>', unsafe_allow_html=True)
        k3.markdown(f'<div class="kpi">Churn Rate<div class="kpi-value">{churn_rate}%</div></div>', unsafe_allow_html=True)

        H = 220

        def style(fig):
            fig.update_layout(
                template="plotly_white",
                height=H,
                margin=dict(l=10,r=10,t=40,b=10),
                font=dict(color="black")
            )
            return fig

        # ===== ROW 1 =====
        c1,c2,c3 = st.columns(3)

        with c1:
            st.markdown('<div class="chart">', unsafe_allow_html=True)
            fig = px.bar(df.groupby("Segment")["Churn"].sum().reset_index(),
                         x="Segment", y="Churn", title="Churn by Segment")
            st.plotly_chart(style(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="chart">', unsafe_allow_html=True)
            fig = px.histogram(df, x="Age", color="Churn", title="Churn by Age")
            st.plotly_chart(style(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="chart">', unsafe_allow_html=True)
            fig = px.scatter(df, x="Tenure", y="CLV",
                             color="Churn", title="Tenure vs CLV")
            st.plotly_chart(style(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ===== ROW 2 =====
        c1,c2,c3 = st.columns(3)

        with c1:
            st.markdown('<div class="chart">', unsafe_allow_html=True)
            fig = px.pie(df, names="SubscriptionType", title="Subscription Distribution")
            st.plotly_chart(style(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="chart">', unsafe_allow_html=True)
            fig = px.box(df, x="Churn", y="MonthlySpend", title="Spend vs Churn")
            st.plotly_chart(style(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="chart">', unsafe_allow_html=True)
            fig = px.violin(df, x="Churn", y="CLV", title="CLV Spread")
            st.plotly_chart(style(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    end_dashboard()

# -----------------------------
# CLV PREDICTION DASHBOARD
# -----------------------------

if page == "CLV Prediction Dashboard" and df is not None:

    start_dashboard("CLV Prediction Dashboard")

    # ---------- PREMIUM CSS ----------
    st.markdown("""
    <style>

    .main {
        background: linear-gradient(135deg, #1f2c44, #2a3f5f);
    }

    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0rem;
    }

    /* KPI CARDS */
    .kpi-card {
        background: linear-gradient(135deg, #6a11cb, #2575fc);
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
    }

    /* CHART CARDS */
    .chart-card {
        background: rgba(255,255,255,0.05);
        padding: 10px;
        border-radius: 12px;
        margin-bottom: 10px;
    }

    /* RIGHT PANEL */
    .right-panel {
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 12px;
    }

    </style>
    """, unsafe_allow_html=True)

    st.title("📊 CLV Prediction Dashboard")

    # ---------- KPI ROW ----------
    k1, k2, k3, k4 = st.columns(4)

    k1.markdown(f"<div class='kpi-card'>Total Customers<br>{len(df)}</div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi-card'>Avg CLV<br>₹ {int(df['CLV'].mean())}</div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi-card'>Max CLV<br>₹ {int(df['CLV'].max())}</div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi-card'>Total Revenue<br>₹ {int(df['Revenue'].sum())}</div>", unsafe_allow_html=True)

    # ---------- MAIN + RIGHT PANEL ----------
    left, right = st.columns([3,1])

    with right:
        st.markdown("<div class='right-panel'>", unsafe_allow_html=True)

        segment = st.multiselect(
            "Segment",
            df["Segment"].unique(),
            default=df["Segment"].unique()
        )

        tenure = st.slider("Tenure", int(df["Tenure"].min()), int(df["Tenure"].max()), (1, 10))

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- FILTER ----------
    filtered_df = df[
        (df["Segment"].isin(segment)) &
        (df["Tenure"].between(tenure[0], tenure[1]))
    ]

    with left:

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            fig = px.histogram(filtered_df, x="CLV", nbins=30, title="CLV Distribution")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            fig = px.bar(filtered_df, x="Segment", y="CLV", title="CLV by Segment")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        c3, c4 = st.columns(2)

        with c3:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            fig = px.scatter(filtered_df, x="Tenure", y="CLV", color="Segment",
                             title="Tenure vs CLV")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c4:
            st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
            fig = px.pie(filtered_df, values="Revenue", names="Segment",
                         title="Revenue Contribution")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    end_dashboard()

# -----------------------------
# RETENTION INSIGHTS
# -----------------------------

st.set_page_config(layout="wide")

# =========================
# CSS 
# =========================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    height: 100vh;
    overflow: hidden;
}

.block-container {
    padding-top: 0.5rem;
    padding-bottom: 0rem;
    max-height: 100vh;
    overflow: auto;
}

.kpi-card {
    padding: 12px;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-size: 14px;
    margin-bottom: 6px;
}

.kpi-card h2 {
    font-size: 18px;
    margin: 5px 0;
}

.chart-card {
    background: rgba(255,255,255,0.05);
    padding: 10px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

if page == "Retention Insights" and df is not None:

    start_dashboard("Retention Insights Dashboard")

    # =========================
    # KPI LEFT PANEL (Mimic layout)
    # =========================
    col1, col2 = st.columns([1, 4])

    with col1:
        # Left panel KPIs (mimicking Power BI left panel)
        st.markdown('<div class="kpi-card" style="background:#4A8DDC;">Highest Segment<br><h2>{}</h2></div>'.format(
            df.groupby("Segment")["CLV"].sum().idxmax()), unsafe_allow_html=True)

        st.markdown('<div class="kpi-card" style="background:#2D9CDB;">Highest City<br><h2>{}</h2></div>'.format(
            df["City"].mode()[0] if "City" in df.columns else "N/A"), unsafe_allow_html=True)

        st.markdown('<div class="kpi-card" style="background:#F63E6D;">Best Customer<br><h2>{}</h2><br>{}</div>'.format(
            df.loc[df["CLV"].idxmax(), "CLV"], df.loc[df["CLV"].idxmax(), "CustomerID"]), unsafe_allow_html=True)

        st.markdown('<div class="kpi-card" style="background:#FF9F43;">Best State<br><h2>{}</h2></div>'.format(
            df["State"].mode()[0] if "State" in df.columns else "N/A"), unsafe_allow_html=True)

    with col2:
        # =========================
        # TOP KPI ROW (Retention Insights)
        # =========================
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="kpi-card">Retention Rate<br><h2>{:.2f}%</h2></div>'.format(100 - df["Churn"].mean()*100), unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="kpi-card">Churn Rate<br><h2>{:.2f}%</h2></div>'.format(df["Churn"].mean()*100), unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="kpi-card">Avg CLV<br><h2>{:.0f}</h2></div>'.format(df["CLV"].mean()), unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="kpi-card">Total Customers<br><h2>{}</h2></div>'.format(df.shape[0]), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # =========================
        # ROW 1 (3 CHARTS)
        # =========================
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            fig1 = px.pie(df, names="Segment", title="Customer Segments")
            st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            fig2 = px.bar(df.groupby("Segment")["CLV"].mean().reset_index(),
                          x="Segment", y="CLV", title="Avg CLV by Segment", color="Segment")
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            fig3 = px.bar(df.groupby("Segment")["Churn"].mean().reset_index(),
                          x="Segment", y="Churn", title="Churn by Segment", color="Segment")
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        # =========================
        # ROW 2 (2 CHARTS)
        # =========================
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            fig4 = px.line(df, x="Tenure", y="CLV", color="Segment",
                           title="CLV Trend by Tenure")
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            fig5 = px.histogram(df, x="CLV", color="Segment",
                                title="CLV Distribution")
            st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        # =========================
        # ROW 3 (FULL WIDTH)
        # =========================
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        fig6 = px.bar(df.groupby("Segment")["Revenue"].sum().reset_index(),
                      x="Segment", y="Revenue",
                      title="Revenue by Segment", color="Segment")
        st.plotly_chart(fig6, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    end_dashboard()


# -----------------------------
# CUSTOMER SEGMENTATION 
# -----------------------------

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(layout="wide")

# =========================
# CSS 
# =========================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    height: 100vh;
    overflow: hidden;
}

.block-container {
    padding-top: 0.5rem;
    padding-bottom: 0rem;
    max-height: 100vh;
    overflow: auto;
}

.kpi-card {
    padding: 12px;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-size: 14px;
    margin-bottom: 6px;
}

.kpi-card h2 {
    font-size: 18px;
    margin: 5px 0;
}

.chart-card {
    background: rgba(255,255,255,0.05);
    padding: 10px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# DASHBOARD
# =========================
if page == "Customer Segmentation " and df is not None:

    start_dashboard("AI Customer Segmentation")

    st.title("AI Customer Segmentation Dashboard")

    # =========================
    # ML MODEL
    # =========================
    features = df[["Age","Tenure","MonthlySpend","CLV"]]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=4, random_state=42)
    df["Cluster"] = kmeans.fit_predict(scaled)

    # =========================
    # KPIs (NEW UNIQUE ONES)
    # =========================
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f'<div class="kpi-card" style="background:#6C5CE7;">Total Segments<br><h2>{df["Cluster"].nunique()}</h2></div>', unsafe_allow_html=True)

    with k2:
        st.markdown(f'<div class="kpi-card" style="background:#00B894;">Avg Monthly Spend<br><h2>{df["MonthlySpend"].mean():.0f}</h2></div>', unsafe_allow_html=True)

    with k3:
        st.markdown(f'<div class="kpi-card" style="background:#E17055;">High Value Customers<br><h2>{df[df["CLV"] > df["CLV"].mean()].shape[0]}</h2></div>', unsafe_allow_html=True)

    with k4:
        st.markdown(f'<div class="kpi-card" style="background:#0984E3;">Avg Tenure<br><h2>{df["Tenure"].mean():.1f}</h2></div>', unsafe_allow_html=True)

    # =========================
    # GRID LAYOUT (2 ROWS)
    # =========================
    

    # -------- ROW 1 --------
    c1, c2, c3 = st.columns(3)

    with c1:
        fig1 = px.scatter(df, x="CLV", y="MonthlySpend", color=df["Cluster"].astype(str),
                          title="Customer Value vs Spend")
        st.plotly_chart(fig1, use_container_width=True, height=220)

    with c2:
        fig2 = px.box(df, x="Cluster", y="CLV",
                      title="CLV Distribution by Cluster")
        st.plotly_chart(fig2, use_container_width=True, height=220)

    with c3:
        fig3 = px.histogram(df, x="Age", color="Cluster",
                            title="Age Distribution by Cluster")
        st.plotly_chart(fig3, use_container_width=True, height=220)

    # -------- ROW 2 --------
    c4, c5, c6 = st.columns(3)

    with c4:
        fig4 = px.bar(df.groupby("Cluster")["MonthlySpend"].mean().reset_index(),
                      x="Cluster", y="MonthlySpend",
                      title="Avg Spend per Cluster")
        st.plotly_chart(fig4, use_container_width=True, height=220)

    with c5:
        fig5 = px.line(df.sort_values("Tenure"),
                       x="Tenure", y="CLV", color="Cluster",
                       title="CLV Trend by Tenure")
        st.plotly_chart(fig5, use_container_width=True, height=220)

    with c6:
        fig6 = px.density_heatmap(df, x="Tenure", y="MonthlySpend",
                                 title="Tenure vs Spend Heatmap")
        st.plotly_chart(fig6, use_container_width=True, height=220)

    end_dashboard()


# -----------------------------
# DOWNLOAD REPORT
# -----------------------------

if page == "Download Report" and df is not None:

    st.header("Download Analytics Report")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV Report",
        csv,
        "customer_analytics_report.csv",
        "text/csv"
    )

# -----------------------------
# EXECUTIVE DASHBOARD
# -----------------------------


if page == "Executive Dashboard" and df is not None:

    start_dashboard("Executive Business Intelligence Dashboard")
    st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    height: 100vh;
    overflow: hidden;
}

.block-container {
    padding-top: 0.5rem;
    padding-bottom: 0rem;
    max-height: 100vh;
    overflow: auto;
}

.kpi-card {
    padding: 12px;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-size: 14px;
    margin-bottom: 6px;
}

.kpi-card h2 {
    font-size: 18px;
    margin: 5px 0;
}

.chart-card {
    background: rgba(255,255,255,0.05);
    padding: 10px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

    st.title("Executive Dashboard")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Customers", df.shape[0])
    col2.metric("Revenue", round(df["Revenue"].sum(),2))
    col3.metric("Avg CLV", round(df["CLV"].mean(),2))
    col4.metric("Churn %", round(df["Churn"].mean()*100,2))

    st.divider()

    col1,col2 = st.columns(2)

    col1.plotly_chart(px.histogram(df,x="CLV"),use_container_width=True, height=300)
    col2.plotly_chart(px.pie(df,names="Segment"),use_container_width=True, height=300)

    col1,col2 = st.columns(2)

    col1.plotly_chart(px.scatter(df,x="Tenure",y="Revenue",color="Segment"),use_container_width=True, height=300)
    col2.plotly_chart(px.box(df,x="Churn",y="CLV"),use_container_width=True, height=300)

    end_dashboard()
# -----------------------------
# Market Analysis Dashboard
# -----------------------------
if page == "Market Analysis Dashboard" and df is not None:

    st.header("Market Trends Dashboard")
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    }

    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 0rem;
    }

    /* KPI CARDS */
    .kpi-card {
        background: linear-gradient(135deg, #ff512f, #dd2476);
        padding: 18px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
        transition: 0.3s;
    }
    .kpi-card:hover {
        transform: scale(1.05);
    }

    /* CHART CARDS */
    .chart-card {
        background: rgba(255,255,255,0.05);
        padding: 10px;
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Advanced Market Analysis Dashboard")

    # KPI CARDS
    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Total Customers", df.shape[0])
    col2.metric("Average CLV", round(df["CLV"].mean(),2))
    col3.metric("Total Revenue", round(df["Revenue"].sum(),2))
    col4.metric("Churn Rate", f"{df['Churn'].mean()*100:.2f}%")

    st.divider()

    # ROW 1
    col1,col2,col3 = st.columns(3)

    with col1:
        fig = px.pie(df, names="Segment", hole=0.5,
                     title="Customer Segments Distribution")
        st.plotly_chart(fig,use_container_width=True, height=300)

    with col2:
        fig = px.scatter(df,
                         x="Age",
                         y="CLV",
                         color="Segment",
                         size="MonthlySpend",
                         title="Age vs CLV Analysis")
        st.plotly_chart(fig,use_container_width=True, height=300)

    with col3:
        revenue_segment = df.groupby("Segment")["Revenue"].sum().reset_index()

        fig = px.bar(revenue_segment,
                     x="Segment",
                     y="Revenue",
                     title="Revenue Contribution by Segment")
        st.plotly_chart(fig,use_container_width=True, height=300)

    # ROW 2
    col1,col2,col3 = st.columns(3)

    with col1:
        churn_segment = df.groupby("Segment")["Churn"].mean().reset_index()

        fig = px.bar(churn_segment,
                     x="Segment",
                     y="Churn",
                     title="Churn Rate by Segment")
        st.plotly_chart(fig,use_container_width=True, height=300)

    with col2:
        fig = px.treemap(df,
                         path=["Segment"],
                         values="Revenue",
                         title="Revenue Treemap")
        st.plotly_chart(fig,use_container_width=True,  height=300)

    with col3:
        fig = px.box(df,
                     x="Segment",
                     y="CLV",
                     title="CLV Distribution by Segment")
        st.plotly_chart(fig,use_container_width=True, height=300)

    # ROW 3

    col1,col2 = st.columns(2)

    with col1:
        fig = px.histogram(df,
                           x="MonthlySpend",
                           color="Segment",
                           title="Monthly Spend Distribution")
        st.plotly_chart(fig,use_container_width=True, height=300)

    with col2:
        fig = px.line(df,
                      y="Revenue",
                      title="Revenue Trend")
        st.plotly_chart(fig,use_container_width=True, height=300)
# -----------------------------
# AI Insights Dashboard
# -----------------------------
if page == "AI Insights Dashboard" and df is not None:

    st.header("AI Insights Dashboard")

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    st.divider()

    # HISTOGRAMS FOR ALL NUMERIC COLUMNS
    st.subheader("Histogram Analysis")

    for col in numeric_cols:
        fig = px.histogram(df, x=col, title=f"{col} Distribution")
        st.plotly_chart(fig, use_container_width=True,  height=300)

    # BOXPLOTS
    st.subheader("Box Plot Analysis")

    for col in numeric_cols:
        fig = px.box(df, y=col, title=f"{col} Outlier Detection")
        st.plotly_chart(fig, use_container_width=True, height=300)

    # PIE CHARTS FOR CATEGORICAL
    st.subheader("Categorical Distribution")

    for col in categorical_cols:
        fig = px.pie(df, names=col, title=f"{col} Distribution")
        st.plotly_chart(fig, use_container_width=True, height=300)

    # BAR CHARTS
    st.subheader("Category Frequency")

    for col in categorical_cols:
        count_df = df[col].value_counts().reset_index()

        fig = px.bar(count_df,
                     x="index",
                     y=col,
                     title=f"{col} Frequency")
        st.plotly_chart(fig, use_container_width=True, height=300)

    # CORRELATION HEATMAP
    st.subheader("Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    st.pyplot(fig)

    # SCATTER MATRIX
    st.subheader("Scatter Matrix")

    if len(numeric_cols) >= 2:
        fig = px.scatter_matrix(df, dimensions=numeric_cols)
        st.plotly_chart(fig, use_container_width=True,  height=300)
# -----------------------------
# Analytical Dashboard
# -----------------------------
if page == "Analytical Dashboard" and df is not None:

    st.header("Analytical Dashboard")

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    st.subheader("Bubble Chart")

    if len(numeric_cols) >= 3:
        fig = px.scatter(
            df,
            x=numeric_cols[0],
            y=numeric_cols[1],
            size=numeric_cols[2],
            color=categorical_cols[0] if len(categorical_cols)>0 else None,
            title="Bubble Chart Analysis"
        )
        st.plotly_chart(fig, use_container_width=True, height=300)

    st.subheader("Sunburst Chart")

    if len(categorical_cols) >= 1:
        fig = px.sunburst(df, path=[categorical_cols[0]])
        st.plotly_chart(fig, use_container_width=True, height=300)

    st.subheader("Density Heatmap")

    if len(numeric_cols) >= 2:
        fig = px.density_heatmap(
            df,
            x=numeric_cols[0],
            y=numeric_cols[1]
        )
        st.plotly_chart(fig, use_container_width=True, height=300)

    st.subheader("Area Chart")

    if len(numeric_cols) >= 1:
        fig = px.area(df, y=numeric_cols[0])
        st.plotly_chart(fig, use_container_width=True, height=300)

    st.subheader("Radar Chart")

    if len(numeric_cols) >= 3:
        radar_data = df[numeric_cols[:5]].mean().reset_index()
        radar_data.columns = ["Feature","Value"]

        fig = px.line_polar(
            radar_data,
            r="Value",
            theta="Feature",
            line_close=True
        )
        st.plotly_chart(fig, use_container_width=True, height=300)  
# -----------------------------
# AI chatbot
# -----------------------------
# ============================================
# AI CHATBOT 
# ============================================
def smart_ai_chatbot(question, df):
    import pandas as pd
    import numpy as np

    q = question.lower()

    # =============================
    # GPT-LIKE RESPONSE STYLE
    # =============================
    def gpt_style(text):
        return f"AI Analyst:\n\n{text}"

    # =============================
    # BASIC UNDERSTANDING
    # =============================
    if "rows" in q:
        return gpt_style(f"Your dataset contains **{df.shape[0]} records**, which is a good size for analysis.")

    if "columns" in q:
        return gpt_style(f"The dataset includes the following features:\n\n{', '.join(df.columns)}")

    if "summary" in q:
        return gpt_style("Here’s a statistical summary:\n\n" + df.describe().to_string())

    # =============================
    # SMART TREND DETECTION
    # =============================
    if "trend" in q or "analysis" in q:
        insights = []
        for col in df.select_dtypes(include=np.number).columns:
            mean = df[col].mean()
            growth = df[col].iloc[-1] - df[col].iloc[0]

            trend = "increasing 📈" if growth > 0 else "decreasing 📉"

            insights.append(f"• {col} shows a {trend} trend with average {round(mean,2)}")

        return gpt_style("\n".join(insights[:6]))

    # =============================
    # CLV ANALYSIS
    # =============================
    if "clv" in q:
        if "CLV" in df.columns:
            avg = df["CLV"].mean()
            high = df["CLV"].max()

            return gpt_style(f"""
Customer Lifetime Value analysis shows:

• Average CLV is **{round(avg,2)}**
• Highest customer value is **{high}**

This indicates strong revenue potential from top customers.
""")
        else:
            return gpt_style("CLV is not available yet. Please run preprocessing.")

    # =============================
    # CHURN PREDICTION MODEL
    # =============================
    if "predict" in q or "model" in q:

        if "Churn" not in df.columns:
            return gpt_style("Churn column not found. Cannot build prediction model.")

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        X = df.select_dtypes(include=['int64','float64']).drop(columns=["Churn"], errors='ignore')
        y = df["Churn"]

        if len(X.columns) == 0:
            return gpt_style("No numeric features available for modeling.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        return gpt_style(f"""
 Machine Learning Model Built Successfully!

• Algorithm: Random Forest
• Accuracy: **{round(acc*100,2)}%**

The model can now predict customer churn behavior.
""")

    # =============================
    # AUTO INSIGHTS
    # =============================
    if "insight" in q:
        insights = []

        if "CLV" in df.columns:
            high_value = df[df["CLV"] > df["CLV"].mean()].shape[0]
            insights.append(f"💰 {high_value} high-value customers identified")

        if "Churn" in df.columns:
            churn_rate = df["Churn"].mean()*100
            insights.append(f"📉 Churn rate is {round(churn_rate,2)}%")

        return gpt_style("\n".join(insights))

    # =============================
    # BUSINESS RECOMMENDATIONS
    # =============================
    if "recommend" in q or "business" in q:
        return gpt_style("""
Here are strategic recommendations:

• Focus on retaining high-value customers
• Introduce loyalty programs
• Target churn-risk users with offers
• Use personalized marketing campaigns

These actions can significantly improve revenue.
""")

    # =============================
    # DEFAULT GPT RESPONSE
    # =============================
    return gpt_style("I can help analyze your dataset. Try asking about trends, predictions, churn, or insights.")


# ============================================
# AI CHATBOT 
# ============================================
if page == "AI Chatbot":

    st.title("AI Data Analyst Assistant")

    # CHAT MEMORY
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # CHECK DATA
    if "data" not in st.session_state:
        st.warning("Upload dataset first")
    else:
        df = st.session_state.data

       # INPUT
         # DEFINE INPUT FIRST
    user_input = st.text_input("Ask anything about your data...")

    # THEN USE IT
    if user_input:
        answer = smart_ai_chatbot(user_input, df)

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("AI", answer))

    # DISPLAY CHAT
    for role, msg in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(msg)
        


# -----------------------------
# Interactive Customer Analytical Dashboard
# -----------------------------
if page == "Interactive Customer Analytical Dashboard" and df is not None:

    st.header("Interactive Customer Analytics Dashboard")

    # Sidebar Filters
    st.sidebar.subheader("Dashboard Filters")

    # Segment Filter
    segment_filter = st.sidebar.multiselect(
        "Select Segment",
        options=df["Segment"].unique(),
        default=df["Segment"].unique()
    )

    # Age Filter
    age_range = st.sidebar.slider(
        "Select Age Range",
        int(df["Age"].min()),
        int(df["Age"].max()),
        (int(df["Age"].min()), int(df["Age"].max())),
        key="age_filter"
    )

    # Tenure Filter
    tenure_range = st.sidebar.slider(
        "Select Tenure Range",
        int(df["Tenure"].min()),
        int(df["Tenure"].max()),
        (int(df["Tenure"].min()), int(df["Tenure"].max())),
        key="tenure_filter"
    )

    # Apply Filters
    filtered_df = df[
        (df["Segment"].isin(segment_filter)) &
        (df["Age"].between(age_range[0], age_range[1])) &
        (df["Tenure"].between(tenure_range[0], tenure_range[1]))
    ]

    st.subheader("Filtered Dataset Overview")
    st.dataframe(filtered_df.head())

    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Customers", filtered_df.shape[0])
    col2.metric("Avg CLV", round(filtered_df["CLV"].mean(),2))
    col3.metric("Total Revenue", round(filtered_df["Revenue"].sum(),2))
    col4.metric("Churn Rate", f"{filtered_df['Churn'].mean()*100:.2f}%")

    st.divider()

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(filtered_df, x="CLV", color="Segment",
                           title="CLV Distribution")
        st.plotly_chart(fig, use_container_width=True, height=300)

    with col2:
        fig = px.scatter(filtered_df,
                         x="Age",
                         y="Revenue",
                         color="Segment",
                         size="MonthlySpend",
                         title="Age vs Revenue")
        st.plotly_chart(fig, use_container_width=True, height=300)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(filtered_df,
                     x="Segment",
                     y="Revenue",
                     title="Revenue by Segment")
        st.plotly_chart(fig, use_container_width=True, height=300)

    with col2:
        fig = px.pie(filtered_df,
                     names="Churn",
                     title="Churn Distribution")
        st.plotly_chart(fig, use_container_width=True, height=300)

# ------------------------------------
# -----------Project Workflow----------
# ------------------------------------
if page == "Project Workflow":

    # ===============================
    # STEP STATE
    # ===============================
    if "step" not in st.session_state:
        st.session_state.step = 1

    st.title("Complete Project Workflow")

    # ===============================
    # NAVIGATION BUTTONS
    # ===============================
    col1, col2, col3 = st.columns([1,2,1])

    with col1:
        if st.button("⬅️ Previous"):
            if st.session_state.step > 1:
                st.session_state.step -= 1

    with col3:
        if st.button("Next ➡️"):
            if st.session_state.step < 7:
                st.session_state.step += 1

    # ===============================
    # DATA CHECK
    # ===============================
    if "data" not in st.session_state or st.session_state["data"] is None:
        st.warning("Please upload dataset first from 'Upload Dataset' page")
    
    else:
        df = st.session_state["data"]
        step = st.session_state.step

        st.success(f"Step {step} of 7")

        import plotly.express as px

        # ===============================
        # STEP-WISE DISPLAY
        # ===============================
        if step == 1:
            st.header("Data Collection")
            st.write("Dataset loaded successfully")
            st.dataframe(df.head())
            st.info("This step shows how data is collected and loaded into the system.")

        elif step == 2:
            st.header("Data Cleaning")
            st.write("Missing Values:")
            st.write(df.isnull().sum())

            st.write("Duplicate Rows:")
            st.write(df.duplicated().sum())

            st.info("Handling missing values and removing duplicates.")

        elif step == 3:
             st.header("Data Preprocessing")

             st.subheader("1. Handling Missing Values")
             missing = df.isnull().sum()
             st.write(missing)

             if missing.sum() > 0:
              df = df.fillna(0)
              st.success("Missing values handled by filling with 0")

             st.subheader("2. Removing Duplicates")
             duplicates = df.duplicated().sum()
             st.write(f"Duplicate Rows: {duplicates}")

             if duplicates > 0:
                df = df.drop_duplicates()
                st.success("Duplicate rows removed")

                st.subheader("3. Data Type Conversion")
                st.write(df.dtypes)

                st.subheader("4. Feature Engineering (CLV Calculation)")

             if "TotalSpend" in df.columns and "Frequency" in df.columns:
              df["CLV"] = df["TotalSpend"] * df["Frequency"]
              st.success("CLV Feature Created Successfully")
              st.dataframe(df[["TotalSpend", "Frequency", "CLV"]].head())
             else:
                st.info("Columns not found, skipping CLV calculation")

                st.subheader("5. Final Processed Data Preview")
                st.dataframe(df.head())

                st.success("Data preprocessing completed successfully")

        elif step == 4:
            st.header("Exploratory Data Analysis")

            if "CLV" in df.columns:
                fig = px.histogram(df, x="CLV", title="CLV Distribution")
                st.plotly_chart(fig, width='stretch')

            if "Churn" in df.columns:
                fig2 = px.pie(df, names="Churn", title="Churn Distribution")
                st.plotly_chart(fig2, width='stretch')

            st.info("Understanding patterns and trends in data.")

        elif step == 5:
            st.header("Model Building")

            if "Churn" in df.columns:
                from sklearn.model_selection import train_test_split
                from sklearn.ensemble import RandomForestClassifier

                X = df.select_dtypes(include=['int64','float64']).drop(columns=["Churn"], errors='ignore')
                y = df["Churn"]

                if len(X) > 0:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                    model = RandomForestClassifier()
                    model.fit(X_train, y_train)

                    # STORE MODEL
                    st.session_state.model = model
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test

                    st.success("Random Forest Model Trained")
                else:
                    st.warning("No numeric columns available")
            else:
                st.warning("Churn column not found")

            st.info("Machine learning model is built here.")

        elif step == 6:
            st.header("Testing & Validation")

            if "model" in st.session_state:
                model = st.session_state.model
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test

                y_pred = model.predict(X_test)

                from sklearn.metrics import classification_report
                st.text(classification_report(y_test, y_pred))
            else:
                st.warning("Run Model step first")

            st.info("Model performance evaluation.")

        elif step == 7:
            st.header("Dashboard & AI")

            st.write("Navigate to other pages to explore dashboards and AI insights.")
            st.success("Dashboards: Customer Overview, CLV, Churn")
            st.info("Final visualization and AI insights layer.")