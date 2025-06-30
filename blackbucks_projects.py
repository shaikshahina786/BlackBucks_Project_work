import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import speech_recognition as sr
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ⚠️ DO NOT use pyttsx3 on Streamlit Cloud
# import pyttsx3 ← removed

st.set_page_config(page_title="Smart Market Analyzer", layout="wide")

# 🔒 Voice Output Removed for Deployment
# if 'engine' not in st.session_state:
#     st.session_state.engine = pyttsx3.init()

# def speak_safe(text):
#     try:
#         st.session_state.engine.say(text)
#         st.session_state.engine.runAndWait()
#     except RuntimeError:
#         pass

def listen_to_microphone():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Please speak your question clearly.")
        audio = recognizer.listen(source, timeout=5)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError:
            return "Could not request results from Google Speech Recognition."

st.title("🛍️ Smart Market Analyzer with AI Assistant")
st.markdown("""
Welcome to **Smart Market Analyzer**, an intelligent data exploration platform for:
- 🧠 Customer Segmentation using Clustering (KMeans)
- 🛒 Market Basket Analysis using Apriori Algorithm
- 📊 Interactive Data Visualization
- 🤖 AI Assistant with Data Analysis Capability
""")

if 'df' not in st.session_state:
    st.session_state['df'] = None

st.sidebar.title("🤖 AI Assistant")
user_query = st.sidebar.text_input("Ask a question about your data or the analysis:")

mic_button = st.sidebar.button("🎙️ Use Microphone")
if mic_button:
    mic_query = listen_to_microphone()
    if mic_query:
        user_query = mic_query
        st.sidebar.success(f"You said: {mic_query}")

def advanced_bot_reply(query, df):
    query = query.lower()
    if any(word in query for word in ["what", "how", "task", "purpose", "use", "feature", "action"]) and ("website" in query or "platform" in query):
        return ("This website, Smart Market Analyzer, allows users to: \n"
                "- 🧠 Perform customer segmentation using KMeans clustering\n"
                "- 🛒 Analyze frequently purchased item sets using Market Basket Analysis\n"
                "- 📊 Visualize your data with over 10 different interactive charts\n"
                "- 🤖 Ask questions to an AI assistant trained to analyze your uploaded dataset and explain insights.")
    if any(word in query for word in ["owner", "who invented", "who made", "created this", "developer", "whose project"]):
        return ("This website was created by Shahina Sheikh, a B.Tech undergraduate student from Chalavadi Mallikharjuna Engineering College in Vijayawada. Her native place is Nidamanuru.")
    if df is None or df.empty:
        return "No dataset uploaded. Please upload a dataset to ask data-specific questions."
    if "market" in query or "analysis" in query:
        return "Market Basket Analysis identifies product combinations frequently bought together using Apriori algorithm."
    if "tell me about the dataset" in query or "describe dataset" in query:
        summary = f"Your dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n"
        summary += "Here are a few column insights:\n"
        for col in df.columns:
            if df[col].dtype == 'object':
                summary += f"- '{col}': Text column with {df[col].nunique()} unique values.\n"
            else:
                summary += f"- '{col}': Numeric column with mean = {df[col].mean():.2f}, min = {df[col].min()}, max = {df[col].max()}.\n"
        return summary
    if "attribute" in query or "columns" in query:
        return "The dataset has the following columns:\n" + ", ".join(df.columns.tolist())
    if "missing values" in query:
        nulls = df.isnull().sum()
        return nulls[nulls > 0].to_string() if not nulls.empty else "No missing values."
    return "I'm here to assist with your uploaded dataset and this website's features. Please ask questions related to your data or the tool."

if user_query:
    df = st.session_state.get('df')
    st.session_state['bot_response'] = advanced_bot_reply(user_query, df)
    st.sidebar.success(f"Bot: {st.session_state['bot_response']}")

# 🔊 Voice output disabled
# read_button = st.sidebar.button("🔊 Read Bot Response Aloud")
# if read_button and 'bot_response' in st.session_state:
#     speak_safe(st.session_state['bot_response'])

option = st.sidebar.selectbox(
    "Choose Analysis Module",
    ("Upload Dataset", "Data Visualization", "Customer Segmentation (Clustering)", "Market Basket Analysis")
)

if option == "Upload Dataset":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
        st.success("File uploaded successfully!")
        st.subheader("📄 Preview of Data")
        st.dataframe(df.head())
        auto_summary = advanced_bot_reply("describe dataset", df)
        st.session_state['bot_response'] = auto_summary
        st.sidebar.info(f"Bot: {auto_summary}")

elif option == "Data Visualization":
    df = st.session_state.get('df')
    if df is not None:
        st.subheader("📊 Data Visualization Options")
        chart_type = st.selectbox("Select a Chart Type", [
            "Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Histogram",
            "Box Plot", "Violin Plot", "Area Chart", "Heatmap", "Pair Plot"
        ])
        col_options = df.columns.tolist()
        x_axis = st.selectbox("Select X-axis", col_options)
        y_axis = st.selectbox("Select Y-axis", col_options)
        if chart_type == "Bar Chart":
            fig = px.bar(df, x=x_axis, y=y_axis)
            st.plotly_chart(fig)
        elif chart_type == "Line Chart":
            fig = px.line(df, x=x_axis, y=y_axis)
            st.plotly_chart(fig)
        elif chart_type == "Pie Chart":
            fig = px.pie(df, names=x_axis, values=y_axis)
            st.plotly_chart(fig)
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_axis, y=y_axis)
            st.plotly_chart(fig)
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_axis)
            st.plotly_chart(fig)
        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_axis, y=y_axis)
            st.plotly_chart(fig)
        elif chart_type == "Violin Plot":
            fig = px.violin(df, x=x_axis, y=y_axis, box=True, points="all")
            st.plotly_chart(fig)
        elif chart_type == "Area Chart":
            fig = px.area(df, x=x_axis, y=y_axis)
            st.plotly_chart(fig)
        elif chart_type == "Heatmap":
            st.write("### Correlation Heatmap")
            corr = df.corr(numeric_only=True)
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        elif chart_type == "Pair Plot":
            st.write("### Pair Plot")
            fig = sns.pairplot(df.select_dtypes(include=np.number))
            st.pyplot(fig)
    else:
        st.warning("Please upload a dataset first.")

elif option == "Customer Segmentation (Clustering)":
    df = st.session_state.get('df')
    if df is not None:
        st.subheader("🧠 Customer Segmentation Using KMeans Clustering")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        features = st.multiselect("Select features for clustering", numeric_cols, default=numeric_cols)
        if len(features) >= 2:
            k = st.slider("Select number of clusters (K)", 2, 10, 3)
            scaled_data = StandardScaler().fit_transform(df[features])
            kmeans = KMeans(n_clusters=k, random_state=42)
            df['Cluster'] = kmeans.fit_predict(scaled_data)
            st.success("Clustering completed!")
            cluster_summary = df.groupby('Cluster')[features].mean().round(2)
            st.write("### Cluster Centers (Mean Values):")
            st.dataframe(cluster_summary)
            fig = px.scatter(df, x=features[0], y=features[1], color='Cluster', title="Customer Segments")
            st.plotly_chart(fig)
            st.session_state['df'] = df
        else:
            st.warning("Please select at least 2 numeric features.")
    else:
        st.warning("Please upload a dataset first.")

elif option == "Market Basket Analysis":
    df = st.session_state.get('df')
    if df is not None:
        st.subheader("🛒 Market Basket Analysis Using Apriori Algorithm")
        object_cols = df.select_dtypes(include='object').columns.tolist()
        item_col = st.selectbox("Select column that contains item lists or transactions", object_cols)
        try:
            df[item_col] = df[item_col].astype(str).apply(lambda x: re.split(r'[,&]| and ', x))
            df[item_col] = df[item_col].apply(lambda items: [i.strip().title() for i in items if i.strip()])
            te = TransactionEncoder()
            te_ary = te.fit(df[item_col]).transform(df[item_col])
            df_trans = pd.DataFrame(te_ary, columns=te.columns_)
            min_support = st.slider("Minimum Support (%)", 1, 100, 10) / 100
            frequent_items = apriori(df_trans, min_support=min_support, use_colnames=True)
            if frequent_items.empty:
                st.warning("No frequent itemsets found. Try lowering support.")
            else:
                rules = association_rules(frequent_items, metric="confidence", min_threshold=0.5)
                st.write("### Frequent Itemsets")
                st.dataframe(frequent_items)
                st.write("### Association Rules")
                if not rules.empty:
                    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
                    rule = rules.iloc[0]
                    st.info(f"If a customer buys **{', '.join(list(rule['antecedents']))}**, they may also buy **{', '.join(list(rule['consequents']))}** (Confidence: {rule['confidence']:.2f}, Lift: {rule['lift']:.2f})")
                else:
                    st.warning("No strong association rules found.")
        except Exception as e:
            st.error(f"Error processing market basket analysis: {e}")
    else:
        st.warning("Please upload a dataset first.")
