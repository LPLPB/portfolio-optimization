import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# CONFIGURATION DE LA PAGE
# ---------------------------
st.set_page_config(page_title="Portfolio Optimizer", page_icon="📊", layout="wide")

# ---------------------------
# TRADUCTIONS (EN & FR)
# ---------------------------
TRANSLATIONS = {
    'en': {
        'title': "Portfolio Optimizer",
        'subtitle': "Based on Modern Portfolio Theory (Markowitz)",
        'created_by': "Created by **Léo-Paul Laisne**",
        'sidebar_header': "Optimization Parameters",
        'lang_select': "Language",
        'ticker_select_label': "1. Select or Search Assets",
        'ticker_manual_label': "Or add tickers manually (comma separated)",
        'ticker_validate_button': "Add Manual Tickers",
        'ticker_global_validate': "Validate Selection",
        'tickers_locked': "Selected Assets:",
        'tickers_modify_button': "Modify Selection",
        'ticker_error': "Please select at least one ticker.",
        'sim_params_header': "2. Simulation Parameters",
        'period_label': "Period (e.g., '504d', '2y', '5y')",
        'ports_label': "Number of portfolios to simulate",
        'current_header': "3. Your Current Portfolio (Optional)",
        'compare_label': "Compare with a current portfolio",
        'input_mode_label': "How to enter your assets?",
        'mode_amount': "By Amount (€/$)",
        'mode_shares': "By Number of Shares",
        'amount_label': "Enter amount for {ticker}",
        'shares_label': "Number of shares for {ticker}",
        'run_button': "Run Optimization",
        'run_info': "Please select your assets in the sidebar and run the optimization.",
        'loading_data': "Loading data for {tickers}...",
        'loading_error': "Error loading data: {e}",
        'optimal_header': "Optimized Portfolio Results",
        'optimal_subheader': "Optimal portfolio (best Sharpe Ratio) found from {num_ports} simulations.",
        'optimal_return': "Optimal Annual Return",
        'optimal_risk': "Optimal Annual Risk",
        'optimal_sharpe': "Best Sharpe Ratio",
        'optimal_alloc_header': "Optimal Portfolio Allocation",
        'alloc_chart_title': "Allocation (Weights) in %",
        'frontier_chart_title': "Efficient Frontier",
        'legend_optimal': "Optimal Portfolio",
    },
    'fr': {
        'title': "Optimiseur de Portefeuille",
        'subtitle': "Basé sur la Théorie Moderne du Portefeuille (Markowitz)",
        'created_by': "Créé par **Léo-Paul Laisne**",
        'sidebar_header': "Paramètres de l'Optimisation",
        'lang_select': "Langue",
        'ticker_select_label': "1. Sélectionnez ou recherchez des actifs",
        'ticker_manual_label': "Ou ajoutez des tickers manuellement (séparés par des virgules)",
        'ticker_validate_button': "Ajouter les tickers manuels",
        'ticker_global_validate': "Valider la sélection",
        'tickers_locked': "Actifs sélectionnés :",
        'tickers_modify_button': "Modifier la sélection",
        'ticker_error': "Veuillez sélectionner au moins un ticker.",
        'sim_params_header': "2. Paramètres de simulation",
        'period_label': "Période (ex : '504d', '2y', '5y')",
        'ports_label': "Nombre de portefeuilles à simuler",
        'current_header': "3. Votre portefeuille actuel (optionnel)",
        'compare_label': "Comparer avec un portefeuille actuel",
        'input_mode_label': "Comment voulez-vous entrer vos actifs ?",
        'mode_amount': "Par montant (€/$)",
        'mode_shares': "Par nombre d’actions",
        'amount_label': "Montant pour {ticker}",
        'shares_label': "Nombre d’actions pour {ticker}",
        'run_button': "Lancer l’optimisation",
        'run_info': "Veuillez sélectionner vos actifs dans la barre latérale et lancer l’optimisation.",
        'loading_data': "Téléchargement des données pour {tickers}...",
        'loading_error': "Erreur lors du chargement des données : {e}",
        'optimal_header': "Résultats du portefeuille optimal",
        'optimal_subheader': "Portefeuille optimal (meilleur ratio de Sharpe) trouvé parmi {num_ports} simulations.",
        'optimal_return': "Rendement annuel optimal",
        'optimal_risk': "Risque annuel optimal",
        'optimal_sharpe': "Meilleur ratio de Sharpe",
        'optimal_alloc_header': "Allocation du portefeuille optimal",
        'alloc_chart_title': "Répartition (poids en %)",
        'frontier_chart_title': "Frontière efficiente",
        'legend_optimal': "Portefeuille optimal",
    }
}

PREDEFINED_TICKERS = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet",
    "AMZN": "Amazon", "META": "Meta", "NVDA": "NVIDIA",
    "TSLA": "Tesla", "JPM": "JPMorgan", "V": "Visa", "PG": "Procter & Gamble"
}

@st.cache_data
def run_simulation(log_ret, num_ports, num_assets):
    all_weights = np.zeros((num_ports, num_assets))
    all_returns = np.zeros(num_ports)
    all_vols = np.zeros(num_ports)
    all_sharpes = np.zeros(num_ports)
    for i in range(num_ports):
        w = np.random.random(num_assets)
        w /= np.sum(w)
        all_weights[i, :] = w
        all_returns[i] = np.sum((log_ret.mean() * w) * 252)
        all_vols[i] = np.sqrt(np.dot(w.T, np.dot(log_ret.cov() * 252, w)))
        all_sharpes[i] = all_returns[i] / all_vols[i]
    return all_weights, all_returns, all_vols, all_sharpes

# ---------------------------
# BARRE LATÉRALE
# ---------------------------
st.sidebar.header("Portfolio Optimizer")

lang_choice = st.sidebar.selectbox("Language / Langue", ['English', 'Français'])
lang = 'en' if lang_choice == 'English' else 'fr'
T = TRANSLATIONS[lang]

if 'step' not in st.session_state:
    st.session_state.step = 1
if 'locked_tickers' not in st.session_state:
    st.session_state.locked_tickers = []
if 'run_simulation' not in st.session_state:
    st.session_state.run_simulation = False

# --- ÉTAPE 1 ---
if st.session_state.step == 1:
    st.sidebar.subheader(T['ticker_select_label'])
    selected_tickers_multi = st.sidebar.multiselect(
        "Predefined tickers:",
        options=sorted(list(set(list(PREDEFINED_TICKERS.keys()) + st.session_state.locked_tickers))),
        default=st.session_state.locked_tickers,
        format_func=lambda t: f"{t} - {PREDEFINED_TICKERS.get(t, 'Custom')}"
    )

    custom_tickers_string = st.sidebar.text_input(T['ticker_manual_label'])
    add_button = st.sidebar.button(T['ticker_validate_button'])
    validate_all_button = st.sidebar.button(T['ticker_global_validate'])

    if add_button and custom_tickers_string:
        custom_tickers = [t.strip().upper() for t in custom_tickers_string.split(',') if t.strip()]
        new_list = sorted(list(set(st.session_state.locked_tickers + selected_tickers_multi + custom_tickers)))
        st.session_state.locked_tickers = new_list
        st.sidebar.success(f"✅ Added tickers: {', '.join(custom_tickers)}")

    if validate_all_button:
        tickers = sorted(list(set(st.session_state.locked_tickers + selected_tickers_multi)))
        if not tickers:
            st.sidebar.error(T['ticker_error'])
        else:
            st.session_state.locked_tickers = tickers
            st.session_state.step = 2
            st.rerun()

# --- ÉTAPE 2 ---
elif st.session_state.step == 2:
    st.sidebar.subheader(T['tickers_locked'])
    st.sidebar.info(", ".join(st.session_state.locked_tickers))
    if st.sidebar.button(T['tickers_modify_button']):
        st.session_state.step = 1
        st.session_state.run_simulation = False
        st.rerun()

    use_current_portfolio = st.sidebar.checkbox(T['compare_label'])
    input_mode = None
    if use_current_portfolio:
        input_mode = st.sidebar.radio(T['input_mode_label'], (T['mode_amount'], T['mode_shares']))

    with st.sidebar.form('params_form'):
        st.sidebar.subheader(T['sim_params_header'])
        period = st.text_input(T['period_label'], "504d")
        num_ports = st.slider(T['ports_label'], 1000, 20000, 10000, 1000)

        # 🧩 Ajout des champs de saisie selon le mode choisi
        if use_current_portfolio and input_mode:
            st.sidebar.header(T['current_header'])
            for ticker in st.session_state.locked_tickers:
                if input_mode == T['mode_amount']:
                    st.number_input(
                        T['amount_label'].format(ticker=ticker),
                        min_value=0.0, value=1000.0, step=10.0,
                        key=f"amount_{ticker}"
                    )
                else:
                    st.number_input(
                        T['shares_label'].format(ticker=ticker),
                        min_value=0.0, value=10.0, step=0.1,
                        key=f"shares_{ticker}"
                    )

        run_button = st.form_submit_button(T['run_button'])
        if run_button:
            st.session_state.run_simulation = True

# ---------------------------
# CORPS PRINCIPAL
# ---------------------------
st.title(T['title'])
st.caption(T['subtitle'])

if st.session_state.run_simulation:
    tickers = st.session_state.locked_tickers
    st.info(T['loading_data'].format(tickers=", ".join(tickers)))
    try:
        data = yf.download(tickers, period=period)

        # ✅ Correction complète du bug 'Adj Close'
        if isinstance(data.columns, pd.MultiIndex):
            if 'Adj Close' in data.columns.get_level_values(0):
                data = data['Adj Close']
            else:
                data = data['Close']
        else:
            if 'Adj Close' in data.columns:
                data = data['Adj Close']
            elif 'Close' in data.columns:
                data = data['Close']
            else:
                raise ValueError("No 'Close' or 'Adj Close' column found in data")

        log_ret = np.log(data / data.shift(1)).dropna()
    except Exception as e:
        st.error(T['loading_error'].format(e=str(e)))
        st.stop()

    all_weights, all_returns, all_vols, all_sharpes = run_simulation(log_ret, num_ports, len(tickers))
    max_sharpe_idx = np.argmax(all_sharpes)
    opt_weights = all_weights[max_sharpe_idx]
    opt_return, opt_vol, opt_sharpe = all_returns[max_sharpe_idx], all_vols[max_sharpe_idx], all_sharpes[max_sharpe_idx]

    st.header(T['optimal_header'])
    st.subheader(T['optimal_subheader'].format(num_ports=num_ports))
    c1, c2, c3 = st.columns(3)
    c1.metric(T['optimal_return'], f"{opt_return*100:.2f}%")
    c2.metric(T['optimal_risk'], f"{opt_vol*100:.2f}%")
    c3.metric(T['optimal_sharpe'], f"{opt_sharpe:.2f}")

    df_opt = pd.DataFrame({'Asset': tickers, 'Weight (%)': opt_weights * 100})
    st.subheader(T['optimal_alloc_header'])
    st.dataframe(df_opt)

    fig = px.pie(df_opt, values='Weight (%)', names='Asset', title=T['alloc_chart_title'])
    st.plotly_chart(fig, use_container_width=True)

    frontier_fig = px.scatter(x=all_vols, y=all_returns, color=all_sharpes, title=T['frontier_chart_title'])
    frontier_fig.add_trace(go.Scatter(x=[opt_vol], y=[opt_return], mode='markers',
                                      marker=dict(color='white', size=15), name=T['legend_optimal']))
    st.plotly_chart(frontier_fig, use_container_width=True)
else:
    st.info(T['run_info'])
