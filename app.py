# app.py
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import plotly.express as px
import plotly.graph_objects as go

# ------------------ Config ------------------
st.set_page_config(page_title="COVID-19 Data Mining & Analytics", page_icon="🦠", layout="wide")
COVID_PATH = "covid_19_data.csv"
LINE_PATH = "COVID19_line_list_data_modified.csv"


# ------------------ Cached Load ------------------
@st.cache_data(show_spinner=False)
def load_data(covid_path=COVID_PATH, line_path=LINE_PATH):
    required = [covid_path, line_path]
    missing = [p for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files: {', '.join(missing)}. "
            f"Place them in the repo root or update the path."
        )

    covid = pd.read_csv(covid_path, encoding="latin-1", encoding_errors="ignore")
    line = pd.read_csv(line_path, encoding="latin-1", encoding_errors="ignore")
    return covid, line


# ------------------ Helpers ------------------
def clean_covid(df):
    needed = ["ObservationDate", "Country/Region", "Confirmed", "Deaths", "Recovered"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Covid_19.csv: {missing}")

    # Types
    df["ObservationDate"] = pd.to_datetime(df["ObservationDate"], errors="coerce")
    for c in ["Confirmed", "Deaths", "Recovered"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "Province/State" in df.columns:
        df["Province/State"] = df["Province/State"].fillna("Unknown")

    # Latest totals per country
    latest_date = df.groupby("Country/Region")["ObservationDate"].max().reset_index()
    latest_date = latest_date.rename(columns={"ObservationDate": "LatestDate"})
    merged = df.merge(latest_date, on="Country/Region", how="left")
    latest_rows = merged[merged["ObservationDate"] == merged["LatestDate"]]

    country_latest = (
        latest_rows.groupby("Country/Region", as_index=False)[["Confirmed", "Deaths", "Recovered"]].sum()
    )

    return df, country_latest


def clean_line_list(df):
    df = df.copy()

    # Gender
    if "gender" in df.columns:
        df["gender"] = df["gender"].fillna("Unknown").astype(str).str.title()
        df.loc[~df["gender"].isin(["Male", "Female"]), "gender"] = "Unknown"
    else:
        df["gender"] = "Unknown"

    # Age numeric
    if "age" in df.columns:
        df["age_numeric"] = (
            df["age"].astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)
        )
    else:
        df["age_numeric"] = np.nan

    # Age groups
    bins = [0, 18, 30, 50, 70, 120]
    labels = ["0-17", "18-29", "30-49", "50-69", "70+"]
    df["age_group"] = pd.cut(df["age_numeric"], bins=bins, labels=labels, include_lowest=True)

    # Death flag
    if "death" in df.columns:
        s = df["death"].astype(str).str.lower()
        df["death_binary"] = np.where(
            s.isin(["1", "yes", "true", "death", "died", "deceased"]), 1, 0
        )
    else:
        df["death_binary"] = 0

    return df


def global_latest_totals(covid_df):
    ts = (
        covid_df.groupby("ObservationDate")[["Confirmed", "Deaths", "Recovered"]]
        .sum()
        .reset_index()
        .sort_values("ObservationDate")
    )
    if ts.empty:
        return None
    latest = ts.iloc[-1]
    ratio = (latest["Deaths"] / latest["Recovered"]) if latest["Recovered"] > 0 else np.inf
    return {
        "date": latest["ObservationDate"],
        "confirmed": int(latest["Confirmed"]),
        "deaths": int(latest["Deaths"]),
        "recovered": int(latest["Recovered"]),
        "ratio": ratio,
        "ts": ts,
    }


def highest_two(country_latest):
    ranked = country_latest.sort_values("Confirmed", ascending=False).reset_index(drop=True)
    if ranked.empty:
        return None, None
    top1 = ranked.iloc[0]
    top2 = ranked.iloc[1] if len(ranked) > 1 else None
    return top1, top2


def demographics(line_df):
    gender_counts = line_df["gender"].value_counts(dropna=False)
    age_stats = line_df["age_numeric"].describe()
    ag_counts = line_df["age_group"].value_counts().sort_index()
    return gender_counts, age_stats, ag_counts


def mortality_by_age(line_df):
    d = line_df.dropna(subset=["age_group"])
    if d.empty:
        return pd.DataFrame()
    out = d.groupby("age_group").agg(
        deaths=("death_binary", "sum"),
        total=("death_binary", "count")
    ).reset_index()
    out["mortality_rate"] = np.where(out["total"] > 0, 100.0 * out["deaths"] / out["total"], 0.0)
    return out


def cluster_countries(country_df, k=4):
    df = country_df.copy()
    if df.empty:
        df["Mortality_Rate"] = []
        df["Recovery_Rate"] = []
        df["Cluster"] = []
        df["PCA1"] = []
        df["PCA2"] = []
        return df

    df["Mortality_Rate"] = np.where(df["Confirmed"] > 0, df["Deaths"] / df["Confirmed"], 0.0)
    df["Recovery_Rate"] = np.where(df["Confirmed"] > 0, df["Recovered"] / df["Confirmed"], 0.0)

    X = df[["Confirmed", "Deaths", "Recovered", "Mortality_Rate", "Recovery_Rate"]].astype(float).values
    X_scaled = StandardScaler().fit_transform(X)

    # Safe, dynamic clusters
    n_samples = len(df)
    n_clusters = min(max(1, k), n_samples)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # PCA projection for visualization
    pca = PCA(n_components=2, random_state=42)
    p = pca.fit_transform(X_scaled)
    df["PCA1"], df["PCA2"] = p[:, 0], p[:, 1]
    return df


# ------------------ Compute-All ------------------
def compute_all(covid_raw, line_raw):
    covid_df, country_latest = clean_covid(covid_raw)
    line_df = clean_line_list(line_raw)

    # Global totals + time series
    gl = global_latest_totals(covid_df)

    # Highest affected
    top1, top2 = highest_two(country_latest)

    # Demographics
    g_counts, age_stats, ag_counts = demographics(line_df)
    gender_df = pd.DataFrame({"Gender": g_counts.index, "Count": g_counts.values})
    ag_counts_df = pd.DataFrame({"Age Group": ag_counts.index.astype(str), "Count": ag_counts.values})

    # Mortality by age group
    m_age = mortality_by_age(line_df)

    # Clustering (dynamic k up to 4)
    cluster_df = cluster_countries(country_latest, k=4)

    # Top 10 countries (latest)
    top10 = country_latest.sort_values("Confirmed", ascending=False).head(10)

    return (
        covid_df,
        country_latest,
        line_df,
        gl,            # dict with date, confirmed, deaths, recovered, ratio, ts
        top1,
        top2,
        gender_df,
        age_stats,
        ag_counts_df,
        m_age,
        cluster_df,
        top10,
    )


# ------------------ Main UI ------------------
st.title("COVID-19 Data Mining & Analytics System")

try:
    covid_raw, line_raw = load_data()
    (
        covid_df,
        country_latest,
        line_df,
        gl,
        top1,
        top2,
        gender_df,
        age_stats,
        ag_counts_df,
        m_age,
        cluster_df,
        top10,
    ) = compute_all(covid_raw, line_raw)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.exception(e)
    st.stop()

menu = st.sidebar.radio(
    "Choose Analysis",
    [
        "Highest Affected Countries",
        "Mortality vs Recovery (Global)",
        "Demographics",
        "Mortality by Age Group",
        "Country Clusters",
        "Top 10 Countries (Latest)",
        "Global Trends Over Time",
        "Geographic Distribution (Latest)",
    ],
)

# ------------- Views -------------
if menu == "Highest Affected Countries":
    st.subheader("Highest Affected Countries (Latest)")
    st.dataframe(country_latest.sort_values("Confirmed", ascending=False).reset_index(drop=True), use_container_width=True)

    c1, c2 = st.columns(2)
    if top1 is not None:
        with c1:
            st.info("Highest Affected")
            st.write(f"Country: {top1['Country/Region']}")
            st.write(f"Confirmed: {int(top1['Confirmed']):,}")
            st.write(f"Deaths: {int(top1['Deaths']):,}")
            st.write(f"Recovered: {int(top1['Recovered']):,}")
    if top2 is not None:
        with c2:
            st.info("Second Highest")
            st.write(f"Country: {top2['Country/Region']}")
            st.write(f"Confirmed: {int(top2['Confirmed']):,}")
            st.write(f"Deaths: {int(top2['Deaths']):,}")
            st.write(f"Recovered: {int(top2['Recovered']):,}")

elif menu == "Mortality vs Recovery (Global)":
    st.subheader("Mortality vs Recovery Ratio (Global Latest Totals)")
    if gl:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.metric("Date", gl["date"].date().isoformat())
        with c2: st.metric("Confirmed", f"{gl['confirmed']:,}")
        with c3: st.metric("Deaths", f"{gl['deaths']:,}")
        with c4: st.metric("Recovered", f"{gl['recovered']:,}")
        with c5:
            ratio_str = f"{gl['ratio']:.4f}" if np.isfinite(gl["ratio"]) else "∞"
            st.metric("Deaths/Recovered", ratio_str)

        st.write("Global Trends Over Time")
        ts = gl["ts"]
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=ts["ObservationDate"], y=ts["Confirmed"], mode="lines", name="Confirmed"))
        fig_ts.add_trace(go.Scatter(x=ts["ObservationDate"], y=ts["Deaths"], mode="lines", name="Deaths"))
        fig_ts.add_trace(go.Scatter(x=ts["ObservationDate"], y=ts["Recovered"], mode="lines", name="Recovered"))
        fig_ts.update_layout(title="Global COVID-19 Trends", xaxis_title="Date", yaxis_title="Count", hovermode="x unified")
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("No global time series available.")

elif menu == "Demographics":
    st.subheader("Demographic Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Gender Distribution")
        if not gender_df.empty:
            fig_g = px.pie(gender_df, values="Count", names="Gender", title="Gender Distribution")
            st.plotly_chart(fig_g, use_container_width=True)
        else:
            st.info("No gender data available.")

    with col2:
        st.write("Age Statistics")
        if age_stats.get("count", 0) > 0:
            st.write(f"Mean: {age_stats['mean']:.1f}")
            st.write(f"Median: {line_df['age_numeric'].median():.1f}")
            st.write(f"Std: {age_stats['std']:.1f}")
            st.write(f"Range: {age_stats['min']:.1f} - {age_stats['max']:.1f}")
        else:
            st.info("No age data available.")

    st.write("Age Group Distribution")
    if not ag_counts_df.empty:
        fig_ag = px.bar(
            ag_counts_df,
            x="Age Group", y="Count", title="Age Group Distribution", text="Count"
        )
        fig_ag.update_traces(textposition="outside")
        st.plotly_chart(fig_ag, use_container_width=True)
        st.dataframe(ag_counts_df, use_container_width=True)
    else:
        st.info("No age group data available.")

elif menu == "Mortality by Age Group":
    st.subheader("Mortality Rate by Age Group")
    if not m_age.empty:
        fig_m = px.bar(
            m_age.assign(age_group=m_age["age_group"].astype(str)),
            x="age_group", y="mortality_rate",
            title="Mortality Rate by Age Group",
            labels={"age_group": "Age Group", "mortality_rate": "Mortality Rate (%)"},
            text=m_age["mortality_rate"].round(2)
        )
        fig_m.update_traces(textposition="outside")
        st.plotly_chart(fig_m, use_container_width=True)
        st.dataframe(m_age.assign(mortality_rate=m_age["mortality_rate"].round(2)), use_container_width=True)
    else:
        st.info("Not enough data to compute mortality by age group.")

elif menu == "Country Clusters":
    st.subheader("Clustering of Countries (KMeans up to 4 clusters)")
    st.caption("Features: Confirmed, Deaths, Recovered, Mortality_Rate, Recovery_Rate")
    if not cluster_df.empty:
        fig = px.scatter(
            cluster_df,
            x="PCA1", y="PCA2",
            color=cluster_df["Cluster"].astype(str),
            hover_name="Country/Region",
            hover_data=["Confirmed", "Deaths", "Recovered"],
            title="Clusters (PCA Projection)"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cluster_df.sort_values(["Cluster", "Confirmed"], ascending=[True, False]).reset_index(drop=True), use_container_width=True)
    else:
        st.info("No data available for clustering.")

elif menu == "Top 10 Countries (Latest)":
    st.subheader("Top 10 Countries by Latest Confirmed")
    if not top10.empty:
        fig_top = px.bar(
            top10,
            x="Country/Region", y=["Confirmed", "Deaths", "Recovered"],
            barmode="group",
            title="Top 10 Countries by Cases"
        )
        st.plotly_chart(fig_top, use_container_width=True)
        st.dataframe(top10.reset_index(drop=True), use_container_width=True)
    else:
        st.info("No country data available.")

elif menu == "Global Trends Over Time":
    st.subheader("Global Trends Over Time")
    gl_local = global_latest_totals(covid_df)
    if gl_local:
        ts = gl_local["ts"]
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=ts["ObservationDate"], y=ts["Confirmed"], mode="lines", name="Confirmed"))
        fig_ts.add_trace(go.Scatter(x=ts["ObservationDate"], y=ts["Deaths"], mode="lines", name="Deaths"))
        fig_ts.add_trace(go.Scatter(x=ts["ObservationDate"], y=ts["Recovered"], mode="lines", name="Recovered"))
        fig_ts.update_layout(title="Global COVID-19 Trends", xaxis_title="Date", yaxis_title="Count", hovermode="x unified")
        st.plotly_chart(fig_ts, use_container_width=True)
        st.dataframe(ts, use_container_width=True)
    else:
        st.info("No global time series available.")

elif menu == "Geographic Distribution (Latest)":
    st.subheader("Geographic Distribution (Latest)")
    if not country_latest.empty:
        fig_map = px.choropleth(
            country_latest,
            locations="Country/Region",
            locationmode="country names",
            color="Confirmed",
            hover_data=["Deaths", "Recovered"],
            color_continuous_scale="Reds",
            title="COVID-19 Cases by Country"
        )
        st.plotly_chart(fig_map, use_container_width=True)
        st.dataframe(country_latest.sort_values("Confirmed", ascending=False).reset_index(drop=True), use_container_width=True)
    else:
        st.info("No latest country data available.")
