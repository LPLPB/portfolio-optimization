import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

# La fonction mise en cache
@st.cache_data
def run_simulation(log_ret, num_ports, num_assets):
    all_weights = np.zeros((num_ports, num_assets))
    all_returns = np.zeros((num_ports))
    all_volatilities = np.zeros((num_ports))
    all_sharpe_ratios = np.zeros((num_ports))

    for ind in range(num_ports):
        weights = np.array(np.random.random(num_assets))
        weights = weights / np.sum(weights)
        all_weights[ind, :] = weights
        all_returns[ind] = np.sum((log_ret.mean() * weights) * 252)
        all_volatilities[ind] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
        all_sharpe_ratios[ind] = all_returns[ind] / all_volatilities[ind]
    
    return all_weights, all_returns, all_volatilities, all_sharpe_ratios


st.set_page_config(
    page_title="Optimiseur de Portefeuille",
    page_icon="📊",
    layout="wide"
)

st.sidebar.header("Paramètres de l'Optimisation")

tickers_string = st.sidebar.text_input(
    "Entrez les Tickers des actions (séparés par une virgule)",
    "DCAM.PA, TSLA"
)
tickers = [t.strip().upper() for t in tickers_string.split(',')]

period = st.sidebar.text_input(
    "Période (ex: '504d', '2y', '5y')",
    "504d"
)

num_ports = st.sidebar.slider(
    "Nombre de portefeuilles à simuler",
    1000, 20000, 10000, 1000
)

st.sidebar.header("Votre Portefeuille Actuel (Optionnel)")
use_current_portfolio = st.sidebar.checkbox("Comparer avec un portefeuille actuel")

current_inputs = []
input_mode = None
monetary_values = []

if use_current_portfolio and tickers:
    input_mode = st.sidebar.radio(
        "Comment voulez-vous entrer vos actifs ?",
        ("Par Montant (€/$)", "Par Nombre d'Actions")
    )

    if input_mode == "Par Montant (€/$)":
        st.sidebar.write("Entrez le montant investi pour chaque action :")
        for ticker in tickers:
            amount = st.sidebar.number_input(f"Montant pour {ticker}", min_value=0.0, value=1000.0, step=10.0)
            current_inputs.append(amount)
    
    elif input_mode == "Par Nombre d'Actions":
        st.sidebar.write("Entrez le nombre d'actions que vous possédez :")
        for ticker in tickers:
            shares = st.sidebar.number_input(f"Nombre d'actions {ticker}", min_value=0.0, value=10.0, step=0.1)
            current_inputs.append(shares)

run_button = st.sidebar.button("Lancer l'Optimisation")

st.title("📊 Optimiseur de Portefeuille (Modèle Markowitz)")
st.markdown("Créé par **Léo-Paul Laisne** | [Profil LinkedIn](https://www.linkedin.com/in/leopaullaisne)")

if not run_button:
    st.info("Veuillez entrer les tickers et cliquer sur 'Lancer l'Optimisation' dans la barre latérale.")
    st.stop()

if not tickers
