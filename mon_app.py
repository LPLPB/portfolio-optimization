import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# CONFIGURATION DE LA PAGE
# ---------------------------
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📊",
    layout="wide"
)

# ---------------------------
# TRADUCTIONS (EN & FR)
# ---------------------------
TRANSLATIONS = {
    'en': {
        'title': "Portfolio Optimizer",
        'subtitle': "Based on Modern Portfolio Theory (Markowitz)",
        'created_by': "Created by **Léo-Paul Laisne**",
        'linkedin_profile': "LinkedIn Profile",
        'sidebar_header': "Optimization Parameters",
        'lang_select': "Language",
        'ticker_select_label': "1. Select or Search Assets",
        'ticker_manual_label': "Or add tickers manually (comma separated)",
        'ticker_validate_button': "Validate Tickers",
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
        'input_error': "Error: The number of entries ({entries}) does not match the number of tickers ({tickers}).",
        'value_error': "Your portfolio's total value is 0. Cannot calculate weights.",
        'running_sim': "Running simulation for {num_ports} portfolios...",
        'current_analysis_header': "Your Current Portfolio Analysis",
        'current_total_value': "Total Portfolio Value",
        'current_return': "Current Annual Return",
        'current_risk': "Current Annual Risk",
        'current_sharpe': "Current Sharpe Ratio",
        'current_weights_header': "Your Calculated Portfolio Weights",
        'col_action': "Asset",
        'col_weight': "Weight (%)",
        'col_amount': "Amount (€/$)",
        'optimal_header': "Optimized Portfolio Results",
        'optimal_subheader': "Optimal portfolio (best Sharpe Ratio) found from {num_ports} simulations.",
        'optimal_return': "Optimal Annual Return",
        'optimal_risk': "Optimal Annual Risk",
        'optimal_sharpe': "Best Sharpe Ratio",
        'optimal_alloc_header': "Optimal Portfolio Allocation",
        'col_amount_optimal': "Optimal Amount (€/$)",
        'alloc_chart_title': "Allocation (Weights) in %",
        'frontier_header': "Efficient Frontier & Comparison",
        'frontier_subheader': "Each point is a simulated portfolio. Compare your portfolio (⭐) to the optimal (⚪).",
        'frontier_chart_title': "Efficient Frontier (Curve for 2 assets, Cloud for 3+)",
        'frontier_xaxis': "Annual Volatility (Risk)",
        'frontier_yaxis': "Expected Annual Return",
        'legend_title': "Legend",
        'legend_optimal': "Optimal Portfolio",
        'legend_current': "My Current Portfolio",
        'extra_charts_header': "Historical Price Action & Daily Returns",
        'prices_chart_title': "Adjusted Close Price History",
        'prices_chart_yaxis': "Price",
        'prices_chart_xaxis': "Date",
        'prices_chart_legend': "Asset",
        'corr_header': "Correlation Matrix",
        'corr_company': "Company",
        'conclusion_header': "Conclusion & Action Plan",
        'conclusion_subheader': "To rebalance your {value:,.2f} (€/$) portfolio to the optimal allocation:",
        'action_header': "Recommended Actions:",
        'col_current_pos': "Current Position",
        'col_optimal_pos': "Optimal Position",
        'col_action_req': "Action Required",
        'action_buy': "🟢 BUY {diff:,.2f}",
        'action_sell': "🔴 SELL {abs_diff:,.2f}",
        'action_hold': "⚪ HOLD"
    },
    'fr': {
        'title': "Optimiseur de Portefeuille",
        'subtitle': "Basé sur la Théorie Moderne du Portefeuille (Markowitz)",
        'created_by': "Créé par **Léo-Paul Laisne**",
        'linkedin_profile': "Profil LinkedIn",
        'sidebar_header': "Paramètres de l'Optimisation",
        'lang_select': "Langue",
        'ticker_select_label': "1. Sélectionnez ou recherchez des actifs",
        'ticker_manual_label': "Ou ajoutez des tickers manuellement (séparés par virgule)",
        'ticker_validate_button': "Valider les Actifs",
        'tickers_locked': "Actifs Sélectionnés :",
        'tickers_modify_button': "Modifier la Sélection",
        'ticker_error': "Veuillez sélectionner au moins un ticker.",
        'sim_params_header': "2. Paramètres de simulation",
        'period_label': "Période (ex: '504d', '2y', '5y')",
        'ports_label': "Nombre de portefeuilles à simuler",
        'current_header': "3. Votre Portefeuille Actuel (Optionnel)",
        'compare_label': "Comparer avec un portefeuille actuel",
        'input_mode_label': "Comment voulez-vous entrer vos actifs ?",
        'mode_amount': "Par Montant (€/$)",
        'mode_shares': "Par Nombre d'Actions",
        'amount_label': "Montant pour {ticker}",
        'shares_label': "Nombre d'actions {ticker}",
        'run_button': "Lancer l'Optimisation",
        'run_info': "Veuillez sélectionner vos actifs dans la barre latérale et lancer l'optimisation.",
        'loading_data': "Téléchargement des données pour {tickers}...",
        'loading_error': "Erreur lors du téléchargement des données : {e}",
        'input_error': "Erreur : Le nombre d'entrées ({entries}) ne correspond pas au nombre de tickers ({tickers}).",
        'value_error': "La valeur totale de votre portefeuille est 0. Impossible de calculer les poids.",
        'running_sim': "Simulation de {num_ports} portefeuilles...",
        'current_analysis_header': "Analyse de Votre Portefeuille Actuel",
        'current_total_value': "Valeur Totale de Votre Portefeuille",
        'current_return': "Rendement Annuel Actuel",
        'current_risk': "Risque Annuel Actuel",
        'current_sharpe': "Ratio de Sharpe Actuel",
        'current_weights_header': "Poids calculés de votre portefeuille",
        'col_action': "Actif",
        'col_weight': "Poids (%)",
        'col_amount': "Montant (€/$)",
        'optimal_header': "Résultats de l'Optimisation (Portefeuille Optimal)",
        'optimal_subheader': "Portefeuille optimal (meilleur Ratio de Sharpe) trouvé parmi {num_ports} simulations.",
        'optimal_return': "Rendement Optimal Annuel",
        'optimal_risk': "Volatilité Optimale",
        'optimal_sharpe': "Meilleur Ratio de Sharpe",
        'optimal_alloc_header': "Allocation du Portefeuille Optimal",
        'col_amount_optimal': "Montant Optimal (€/$)",
        'alloc_chart_title': "Allocation (Poids) en %",
        'frontier_header': "Frontière Efficiente & Comparaison",
        'frontier_subheader': "Chaque point est un portefeuille simulé. Comparez votre portefeuille (⭐) à l'optimal (⚪).",
        'frontier_chart_title': "Frontière Efficiente (Courbe si 2 actifs, Nuage si 3+)",
        'frontier_xaxis': "Volatilité Annuelle (Risque)",
        'frontier_yaxis': "Rendement Annuel Espéré",
        'legend_title': "Légende",
        'legend_optimal': "Portefeuille Optimal",
        'legend_current': "Mon Portefeuille Actuel",
        'extra_charts_header': "Évolution historique des prix et retours journaliers",
        'prices_chart_title': "Évolution des prix de clôture (Ajustés)",
        'prices_chart_yaxis': "Prix",
        'prices_chart_xaxis': "Date",
        'prices_chart_legend': "Actif",
        'corr_header': "Matrice de Corrélation",
        'corr_company': "Compagnie",
        'conclusion_header': "Conclusion & Plan d'Action",
        'conclusion_subheader': "Pour rééquilibrer votre portefeuille (valeur : {value:,.2f} €/$) vers l'allocation optimale :",
        'action_header': "Actions Recommandées :",
        'col_current_pos': "Position Actuelle",
        'col_optimal_pos': "Position Optimale",
        'col_action_req': "Action Requise",
        'action_buy': "🟢 ACHETER {diff:,.2f}",
        'action_sell': "🔴 VENDRE {abs_diff:,.2f}",
        'action_hold': "⚪ CONSERVER"
    }
}

# ---------------------------
# TICKERS PAR DÉFAUT
# ---------------------------
PREDEFINED_TICKERS = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet",
    "AMZN": "Amazon", "META": "Meta", "NVDA": "NVIDIA",
    "TSLA": "Tesla", "JPM": "JPMorgan", "V": "Visa", "PG": "Procter & Gamble"
}

# ---------------------------
# SIMULATION
# ---------------------------
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
lang_choice = st.sidebar.selectbox("Language", ['English', 'Français'])
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
    with st.sidebar.form('ticker_form'):
        selected_tickers_multi = st.multiselect(
            T['ticker_select_label'],
            options=sorted(list(set(list(PREDEFINED_TICKERS.keys()) + st.session_state.locked_tickers))),
            default=st.session_state.locked_tickers,
            format_func=lambda t: f"{t} - {PREDEFINED_TICKERS.get(t, 'Custom')}"
        )
        custom_tickers_string = st.text_input(T['ticker_manual_label'])
        validate_button = st.form_submit_button(T['ticker_validate_button'])

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

        current_inputs = []
        tickers_in_form = st.session_state.locked_tickers
        if use_current_portfolio and input_mode:
            st.sidebar.header(T['current_header'])
            if input_mode == T['mode_amount']:
                for t in tickers_in_form:
                    val = st.number_input(f"{T['amount_label'].format(ticker=t)}", 0.0, value=1000.0, step=10.0)
                    current_inputs.append(val)
            else:
                for t in tickers_in_form:
                    val = st.number_input(f"{T['shares_label'].format(ticker=t)}", 0.0, value=10.0, step=0.1)
                    current_inputs.append(val)

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
        data = yf.download(tickers, period=period)['Adj Close']
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
    st.metric(T['optimal_return'], f"{opt_return*100:.2f}%")
    st.metric(T['optimal_risk'], f"{opt_vol*100:.2f}%")
    st.metric(T['optimal_sharpe'], f"{opt_sharpe:.2f}")

    df_opt = pd.DataFrame({'Asset': tickers, 'Weight (%)': opt_weights * 100})
    st.subheader(T['optimal_alloc_header'])
    st.dataframe(df_opt)

    fig = px.pie(df_opt, values='Weight (%)', names='Asset', title=T['alloc_chart_title'])
    st.plotly_chart(fig, use_container_width=True)

    frontier_fig = px.scatter(x=all_vols, y=all_returns, color=all_sharpes, title=T['frontier_chart_title'])
    frontier_fig.add_trace(go.Scatter(x=[opt_vol], y=[opt_return], mode='markers', marker=dict(color='white', size=15), name=T['legend_optimal']))
    st.plotly_chart(frontier_fig, use_container_width=True)

else:
    st.info(T['run_info'])
