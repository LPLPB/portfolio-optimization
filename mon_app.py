import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# ---- Traductions simples ----
T = {
    'app_title': "Portfolio Optimization Dashboard",
    'ticker_section': "1️⃣ Sélection des tickers",
    'ticker_label': "Choisissez des tickers :",
    'custom_label': "Ou ajoutez des tickers manuellement (séparés par des virgules) :",
    'validate_button': "Valider la sélection",
    'ticker_error': "Veuillez sélectionner ou entrer au moins un ticker valide.",
    'compare_label': "Compare with my portfolio",
    'input_mode_label': "Mode d’entrée",
    'mode_amount': "By Amount",
    'mode_shares': "By Shares",
    'current_header': "Current Portfolio",
    'amount_label': "Montant investi pour {ticker}",
    'shares_label': "Nombre d’actions pour {ticker}",
    'sim_params_header': "2️⃣ Paramètres de simulation",
    'period_label': "Période d'historique (ex : 504d, 5y...)",
    'ports_label': "Nombre de portefeuilles simulés",
    'opt_button': "Lancer l’optimisation",
    'results_header': "3️⃣ Résultats de l’optimisation",
    'loading_error': "Erreur lors du chargement des données : {e}"
}

# ---- Fonctions principales ----
@st.cache_data
def run_simulation(log_ret, num_ports, num_assets):
    all_weights = np.zeros((num_ports, num_assets))
    all_returns = np.zeros(num_ports)
    all_vols = np.zeros(num_ports)
    all_sharpes = np.zeros(num_ports)

    for i in range(num_ports):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        all_weights[i, :] = weights

        portfolio_return = np.sum(log_ret.mean() * weights) * 252
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))

        all_returns[i] = portfolio_return
        all_vols[i] = portfolio_vol
        all_sharpes[i] = all_returns[i] / all_vols[i]

    return all_weights, all_returns, all_vols, all_sharpes

# ---- Étape 1 : Sélection des tickers ----
st.sidebar.header(T['ticker_section'])

if "locked_tickers" not in st.session_state:
    st.session_state.locked_tickers = []
if "step" not in st.session_state:
    st.session_state.step = 1

available_tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "XOM"]
selected_tickers_multi = st.sidebar.multiselect(T['ticker_label'], available_tickers)

custom_tickers_string = st.sidebar.text_input(T['custom_label'], key="custom_tickers_input")
validate_button = st.sidebar.button(T['validate_button'], key="validate_btn")

if validate_button:
    custom_tickers = [t.strip().upper() for t in custom_tickers_string.split(',') if t.strip()]

    if st.session_state.locked_tickers:
        tickers = sorted(list(set(st.session_state.locked_tickers + selected_tickers_multi + custom_tickers)))
    else:
        tickers = sorted(list(set(selected_tickers_multi + custom_tickers)))

    if not tickers:
        st.sidebar.error(T['ticker_error'])
    else:
        st.session_state.locked_tickers = tickers
        st.session_state.step = 2
        st.rerun()

# ---- Étape 2 : Paramètres de simulation ----
if st.session_state.step >= 2:
    use_current_portfolio = st.sidebar.checkbox(T['compare_label'], key='compare_checkbox')

    input_mode = None
    if use_current_portfolio:
        input_mode = st.sidebar.radio(
            T['input_mode_label'],
            (T['mode_amount'], T['mode_shares']),
            key='input_mode_radio'
        )

    with st.sidebar.form(key='params_form'):
        st.sidebar.subheader(T['sim_params_header'])
        period = st.text_input(T['period_label'], "504d")
        num_ports = st.slider(T['ports_label'], 1000, 20000, 10000, 1000)

        tickers = st.session_state.locked_tickers
        current_inputs = []

        if use_current_portfolio and input_mode:
            st.sidebar.header(T['current_header'])

            if input_mode == T['mode_amount']:
                for ticker in tickers:
                    amount = st.number_input(
                        T['amount_label'].format(ticker=ticker),
                        min_value=0.0, value=1000.0, step=10.0,
                        key=f"amount_{ticker}"
                    )
                    current_inputs.append(amount)
            elif input_mode == T['mode_shares']:
                for ticker in tickers:
                    shares = st.number_input(
                        T['shares_label'].format(ticker=ticker),
                        min_value=0.0, value=10.0, step=0.1,
                        key=f"shares_{ticker}"
                    )
                    current_inputs.append(shares)

        submitted = st.form_submit_button(T['opt_button'])

    # ---- Étape 3 : Lancer l’optimisation ----
    if submitted:
        try:
            data = yf.download(tickers, period=period, progress=False)

            # ✅ Gérer Adj Close / Close proprement
            if isinstance(data.columns, pd.MultiIndex):
                if 'Adj Close' in data.columns.get_level_values(0):
                    data = data['Adj Close']
                else:
                    data = data['Close']
            else:
                if 'Adj Close' in data.columns:
                    data = data[['Adj Close']]
                elif 'Close' in data.columns:
                    data = data[['Close']]
                else:
                    raise ValueError("No 'Close' or 'Adj Close' column found in data")

            # ✅ Nettoyage : suppression des colonnes vides
            data = data.dropna(axis=1, how='all')

            # ✅ Vérifie les tickers réellement présents
            valid_tickers = list(data.columns)
            missing_tickers = [t for t in tickers if t not in valid_tickers]

            if len(valid_tickers) == 0:
                raise ValueError("No valid data found for selected tickers.")

            if missing_tickers:
                st.warning(f"⚠️ No data found for: {', '.join(missing_tickers)}")

            tickers = valid_tickers
            log_ret = np.log(data / data.shift(1)).dropna()

            # ---- Simulation ----
            all_weights, all_returns, all_vols, all_sharpes = run_simulation(log_ret, num_ports, len(tickers))

            max_sr_idx = all_sharpes.argmax()
            max_sr_ret = all_returns[max_sr_idx]
            max_sr_vol = all_vols[max_sr_idx]
            best_weights = all_weights[max_sr_idx, :]

            st.subheader(T['results_header'])
            col1, col2 = st.columns(2)
            col1.metric("Expected Annual Return", f"{max_sr_ret:.2%}")
            col2.metric("Expected Volatility", f"{max_sr_vol:.2%}")
            st.write("Optimal Weights:")
            df_weights = pd.DataFrame({'Ticker': tickers, 'Weight': best_weights})
            st.dataframe(df_weights)

            # ---- Graphique ----
            fig, ax = plt.subplots()
            scatter = ax.scatter(all_vols, all_returns, c=all_sharpes)
            ax.scatter(max_sr_vol, max_sr_ret, c='red', s=50)
            ax.set_xlabel("Volatility")
            ax.set_ylabel("Return")
            ax.set_title("Simulated Portfolios")
            st.pyplot(fig)

        except Exception as e:
            st.error(T['loading_error'].format(e=str(e)))
