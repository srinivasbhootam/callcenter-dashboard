## Description: Streamlit dashboard for analyzing call center data

import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ── Upload & Load ─────────────────────────────────────────────────────────────────
st.sidebar.header("🔄 Upload Cleaned Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Choose cleaned_snead_data.csv", type="csv"
)
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["Timestamp"])
else:
    st.sidebar.info("Please upload your cleaned CSV to proceed.")
    st.stop()

df.columns = [col.strip() for col in df.columns]
df = df.rename(columns={
    "Agent Name:":        "Agent Name",
    "Reason for Calling:":"Reason for Calling",
    "New patient?":       "New patient?"
})

# ── Standardize “Reason for Calling” ──────────────────────────────────────────────
df["Reason for Calling"] = (
    df["Reason for Calling"]
      .astype(str)
      .str.strip()
      .replace({"Follow Up": "Follow-up"})
)

# ── Feature Engineering ───────────────────────────────────────────────────────────
df = df.sort_values(["Agent Name", "Timestamp"])
df["Call Date"]       = df["Timestamp"].dt.date
df["Day of Week"]     = df["Timestamp"].dt.day_name()
df["Week Start"]      = df["Timestamp"].dt.to_period("W").apply(lambda r: r.start_time)
df["Idle Time (min)"] = (
    df.groupby("Agent Name")["Timestamp"]
      .diff()
      .dt.total_seconds() / 60
)

# ── KPI Threshold ─────────────────────────────────────────────────────────────────
call_counts_excl_kim = (
    df["Agent Name"]
      .value_counts()
      .drop("Kim Villafuerte", errors="ignore")
)
threshold_75 = 0.75 * call_counts_excl_kim.max()

# ── Summary by Agent ──────────────────────────────────────────────────────────────
summary_by_agent = df.groupby("Agent Name").agg(
    Total_Calls       = ("Timestamp", "count"),
    Avg_Idle_Time_min = ("Idle Time (min)", "mean"),
    New_Bookings      = ("New patient?", lambda x: (x=="Yes").sum())
).reset_index()

summary_by_agent["Avg_Idle_Time_min"] = (
    summary_by_agent["Avg_Idle_Time_min"]
      .round()
      .astype(int)
      .astype(str)
    + " mins"
)

# ── Weekday Order ─────────────────────────────────────────────────────────────────
WEEKDAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday"]

# ── Sidebar Filters ──────────────────────────────────────────────────────────────
st.sidebar.header("🔎 Filter Options")
agent_filter  = st.sidebar.multiselect("Agent(s)", df["Agent Name"].unique())
reason_filter = st.sidebar.multiselect("Reason(s)", df["Reason for Calling"].unique())
date_filter   = st.sidebar.date_input("Date range", [])

if agent_filter:
    df = df[df["Agent Name"].isin(agent_filter)]
if reason_filter:
    df = df[df["Reason for Calling"].isin(reason_filter)]
if isinstance(date_filter, list) and len(date_filter)==2:
    start, end = date_filter
    df = df[(df["Call Date"] >= start) & (df["Call Date"] <= end)]

# ── Tabs ──────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Client View", "👤 Agent View"])

# ── Client View ──────────────────────────────────────────────────────────────────
with tab1:
    st.title("📊 Client View Dashboard")

    # 1️⃣ Daily Call Volume
    daily = df.groupby("Call Date").size().reset_index(name="Calls")
    fig1 = px.line(daily, x="Call Date", y="Calls", title="Daily Call Volume")
    fig1.update_layout(yaxis=dict(tickformat=","))
    st.plotly_chart(fig1, use_container_width=True)

    # 2️⃣ Weekly Call Volume
    weekly = df.groupby("Week Start").size().reset_index(name="Calls")
    st.subheader("📅 Weekly Call Volume")
    fig2 = px.bar(
        weekly,
        x="Week Start",
        y="Calls",
        labels={"Week Start":"Week Starting","Calls":"# Calls"},
        title="Weekly Call Volume"
    )
    fig2.update_layout(xaxis_tickangle=45, yaxis=dict(tickformat=","))
    st.plotly_chart(fig2, use_container_width=True)

    # 3️⃣ Call Volume by Day of Week (Mon–Fri)
    dow_counts = (
        df[df["Day of Week"].isin(WEEKDAYS)]
          .groupby("Day of Week")
          .size()
          .reindex(WEEKDAYS, fill_value=0)
          .reset_index(name="Calls")
    )
    st.subheader("📆 Call Volume by Day of Week (Mon–Fri)")
    fig3 = px.bar(
        dow_counts,
        x="Day of Week",
        y="Calls",
        category_orders={"Day of Week": WEEKDAYS},
        title="Call Volume by Day (Mon–Fri)"
    )
    fig3.update_layout(yaxis=dict(tickformat=","))
    st.plotly_chart(fig3, use_container_width=True)

    # 4️⃣ Top 10 Reasons for Calling (Percentage View)
    reasons_pct = (
        df["Reason for Calling"]
          .value_counts(normalize=True)
          .mul(100)
          .head(10)
          .reset_index()
    )
    reasons_pct.columns = ["Reason", "Percent"]
    reasons_pct["Count"] = df["Reason for Calling"].value_counts().head(10).values
    st.subheader("📋 Top 10 Reasons for Calling (by % of Total)")
    fig4 = px.bar(
        reasons_pct,
        x="Percent",
        y="Reason",
        orientation="h",
        text="Percent",
        labels={"Percent":"% of Calls","Reason":""}
    )
    fig4.update_traces(
        texttemplate='%{text:.1f}%  (%{customdata[0]})',
        customdata=reasons_pct[["Count"]].values,
        textposition='inside'
    )
    fig4.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig4, use_container_width=True)

    # 5️⃣ New vs Established Appointments
    df["Booking Type"] = (
        df["New patient?"]
          .fillna("No")
          .map({"Yes":"New","No":"Established"})
    )
    booking = df["Booking Type"].value_counts().reset_index()
    booking.columns = ["Booking Type","Count"]
    st.subheader("🗓️ New vs Established Appointments")
    fig5 = px.pie(booking, names="Booking Type", values="Count")
    st.plotly_chart(fig5, use_container_width=True)

# ── Agent View ───────────────────────────────────────────────────────────────────
with tab2:
    st.title("👤 Agent Performance View")

    # Leaderboard (exclude team lead, add rank)
    lb = df.groupby("Agent Name").agg(
        Calls     = ("Timestamp","count"),
        New_Books = ("New patient?", lambda x:(x=="Yes").sum())
    ).reset_index()
    lb = lb[lb["Agent Name"] != "Zayra Lopez"]
    lb["Meets 75%"] = lb["Calls"] >= threshold_75
    lb = lb.sort_values("Calls", ascending=False).reset_index(drop=True)
    lb.insert(0, "Rank", lb.index + 1)

    st.subheader("🏆 Leaderboard")
    st.dataframe(lb)

    # Calls by Agent
    st.subheader("📊 Calls by Agent")
    fig_agent = px.bar(lb, x="Agent Name", y="Calls", title="Calls by Agent")
    fig_agent.update_layout(yaxis=dict(tickformat=","))
    st.plotly_chart(fig_agent, use_container_width=True)

    # Underperformers
    st.subheader("⚠️ Underperformers (<75% of Top)")
    under = lb[~lb["Meets 75%"]]
    if not under.empty:
        st.table(under[["Agent Name","Calls","New_Books"]])
    else:
        st.success("All agents meet the 75% threshold!")

    # Heatmap (Mon–Fri)
    st.subheader("🗓️ Call Activity Heatmap (Mon–Fri)")
    hm = (
        df[df["Day of Week"].isin(WEEKDAYS)]
          .groupby(["Agent Name","Day of Week"])
          .size()
          .unstack(fill_value=0)
          .reindex(columns=WEEKDAYS, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(hm, annot=True, fmt="d", cmap="Reds", ax=ax)
    ax.set_title("All Agents: Calls by Day (Mon–Fri)")
    ax.set_xlabel("")
    ax.set_ylabel("Agent Name")
    st.pyplot(fig)

    # Std Dev of Daily Calls (exclude team lead)
    st.subheader("📉 STD of Daily Call Volume (All Agents)")
    daily_counts = (
        df.groupby(["Agent Name","Call Date"])
          .size()
          .unstack(fill_value=0)
          .drop(index="Zayra Lopez", errors="ignore")
    )
    std_all = daily_counts.std(axis=1).reset_index(name="STD_Daily_Calls")
    std_all["STD_Daily_Calls"] = std_all["STD_Daily_Calls"].round(2)
    st.dataframe(std_all)

    # Detailed Summary by Agent (exclude team lead)
    st.subheader("✅ Detailed Summary by Agent")
    summary_display = summary_by_agent[summary_by_agent["Agent Name"]!="Zayra Lopez"] \
                         .set_index("Agent Name")
    st.dataframe(summary_display)
    st.info(
        "ℹ️ Kim Villafuerte is a special case with both inbound and outbound calls."
    )
    st.warning(
        "⚠️ Zayra Lopez is the team lead and she is not included in performance metrics comparison."
    )
