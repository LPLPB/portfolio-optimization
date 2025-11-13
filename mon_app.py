import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Portfolio Optimizer", page_icon="📊", layout="wide")

# --- 2. DICTIONNAIRE DE TRADUCTION (EN/FR) ---
# (Conservé du Script 1, car il est complet)
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
        'ticker_validate_button': "Add Manual Tickers", # Clé utilisée par la nouvelle Étape 1
        'ticker_global_validate': "Validate Selection", # Clé utilisée par la nouvelle Étape 1
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
        'prices_table_start': "Start of Period Values (First 5 days)",
        'prices_table_end': "End of Period Values (Last 5 days)",
        'returns_table_start': "Start of Period Daily Returns (First 5 days, %)",
        'returns_table_end': "End of Period Daily Returns (Last 5 days, %)",
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
        'ticker_manual_label': "Ou ajoutez des tickers manuellement (séparés par des virgules)",
        'ticker_validate_button': "Ajouter les tickers manuels", # Clé utilisée par la nouvelle Étape 1
        'ticker_global_validate': "Valider la sélection", # Clé utilisée par la nouvelle Étape 1
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
        'optimal_header': "Résultats du portefeuille optimal",
        'optimal_subheader': "Portefeuille optimal (meilleur ratio de Sharpe) trouvé parmi {num_ports} simulations.",
        'optimal_return': "Rendement annuel optimal",
        'optimal_risk': "Risque annuel optimal",
        'optimal_sharpe': "Meilleur ratio de Sharpe",
        'optimal_alloc_header': "Allocation du portefeuille optimal",
        'col_amount_optimal': "Montant Optimal (€/$)",
        'alloc_chart_title': "Répartition (poids en %)",
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
        'prices_table_start': "Valeurs de Début de Période (5 premiers jours)",
        'prices_table_end': "Valeurs de Fin de Période (5 derniers jours)",
        'returns_table_start': "Retours de Début de Période (5 premiers jours, %)",
        'returns_table_end': "Retours de Fin de Période (5 derniers jours, %)",
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

# --- GRANDE LISTE DE TICKERS (Conservée du Script 1) ---
PREDEFINED_TICKERS = {
    # ETFs Principaux
    'SPY': 'ETF - S&P 500 (SPDR)', 'QQQ': 'ETF - Nasdaq 100 (Invesco)', 'DIA': 'ETF - Dow Jones (SPDR)',
    'URTH': 'ETF - MSCI World (iShares)', 'EEM': 'ETF - MSCI Emerging Markets (iShares)',
    'EWJ': 'ETF - MSCI Japan (iShares)', 'EWG': 'ETF - MSCI Germany (iShares)',
    'EWU': 'ETF - MSCI UK (iShares)', 'GLD': 'ETF - Or (SPDR Gold Shares)', 'SLV': 'ETF - Argent (iShares Silver Trust)',
    # US Tech
    'AAPL': 'Apple (NASDAQ)', 'MSFT': 'Microsoft (NASDAQ)', 'GOOG': 'Alphabet (Google) (NASDAQ)',
    'AMZN': 'Amazon (NASDAQ)', 'TSLA': 'Tesla (NASDAQ)', 'NVDA': 'NVIDIA (NASDAQ)',
    'META': 'Meta Platforms (NASDAQ)', 'ORCL': 'Oracle (NYSE)', 'ADBE': 'Adobe (NASDAQ)',
    'CRM': 'Salesforce (NYSE)', 'INTC': 'Intel (NASDAQ)', 'AMD': 'AMD (NASDAQ)', 'CSCO': 'Cisco (NASDAQ)',
    # US Finance
    'JPM': 'JPMorgan Chase (NYSE)', 'BAC': 'Bank of America (NYSE)', 'WFC': 'Wells Fargo (NYSE)',
    'GS': 'Goldman Sachs (NYSE)', 'MS': 'Morgan Stanley (NYSE)', 'V': 'Visa (NYSE)',
    'MA': 'Mastercard (NYSE)', 'AXP': 'American Express (NYSE)',
    # US Santé
    'JNJ': 'Johnson & Johnson (NYSE)', 'UNH': 'UnitedHealth Group (NYSE)', 'PFE': 'Pfizer (NYSE)',
    'LLY': 'Eli Lilly (NYSE)', 'MRK': 'Merck & Co. (NYSE)',
    # US Consommation
    'WMT': 'Walmart (NYSE)', 'PG': 'Procter & Gamble (NYSE)', 'KO': 'Coca-Cola (NYSE)',
    'PEP': 'PepsiCo (NASDAQ)', 'NKE': 'Nike (NYSE)', 'MCD': 'McDonald\'s (NYSE)',
    'DIS': 'Disney (NYSE)', 'COST': 'Costco (NASDAQ)',
    # Euronext Paris
    'MC.PA': 'LVMH (Euronext Paris)', 'OR.PA': 'L\'Oréal (Euronext Paris)', 'RMS.PA': 'Hermès (Euronext Paris)',
    'DCAM.PA': 'Amundi (Euronext Paris)', 'TTE.PA': 'TotalEnergies (Euronext Paris)', 'SAN.PA': 'Sanofi (Euronext Paris)',
    'AIR.PA': 'Airbus (Euronext Paris)', 'BNP.PA': 'BNP Paribas (Euronext Paris)',
    'SAF.PA': 'Safran (Euronext Paris)', 'KER.PA': 'Kering (Euronext Paris)',
    'AI.PA': 'Air Liquide (Euronext Paris)', 'EL.PA': 'EssilorLuxottica (Euronext Paris)',
    # Asie (Exemples)
    '7203.T': 'Toyota Motor (Tokyo)', '6758.T': 'Sony (Tokyo)', 'BABA': 'Alibaba (NYSE)', 'TM': 'Toyota Motor (NYSE)',
    # Crypto
    'BTC-USD': 'Bitcoin (Crypto)', 'ETH-USD': 'Ethereum (Crypto)', 'SOL-USD': 'Solana (Crypto)', 'XRP-USD': 'Ripple (Crypto)',
}


# --- FONCTION DE SIMULATION OPTIMISÉE (Conservée du Script 1) ---
@st.cache_data
def run_simulation(log_ret, num_ports, num_assets):
    all_weights = np.zeros((num_ports, num_assets))
    all_returns = np.zeros(num_ports)
    all_vols = np.zeros(num_ports)
    all_sharpes = np.zeros(num_ports)
    
    # On convertit les objets pandas en arrays numpy UNE SEULE FOIS
    mean_returns = log_ret.mean().values * 252
    cov_matrix = log_ret.cov().values * 252
    
    for i in range(num_ports):
        w = np.random.random(num_assets)
        w /= np.sum(w)
        all_weights[i, :] = w
        
        # Les calculs sont maintenant 100% numpy
        all_returns[i] = np.sum(mean_returns * w)
        all_vols[i] = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        all_sharpes[i] = all_returns[i] / all_vols[i]
        
    return all_weights, all_returns, all_vols, all_sharpes
# --- FIN DE LA CORRECTION ---

# ---------------------------
# BARRE LATÉRALE (Structure "Wizard" 2 Étapes)
# ---------------------------
lang_choice = st.sidebar.selectbox("Language / Langue", ['English', 'Français'])
lang = 'en' if lang_choice == 'English' else 'fr'
T = TRANSLATIONS[lang] 

st.sidebar.header(T['sidebar_header'])

if 'step' not in st.session_state:
    st.session_state.step = 1
if 'locked_tickers' not in st.session_state:
    st.session_state.locked_tickers = []
if 'run_simulation' not in st.session_state:
    st.session_state.run_simulation = False
if 'current_inputs' not in st.session_state:
    st.session_state.current_inputs = []

# --- ÉTAPE 1 : Sélection (MODIFIÉE SELON LE SCRIPT 2) ---
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
        # Note : Le message suivant est codé en dur (venant du script 2)
        st.sidebar.success(f"✅ Added tickers: {', '.join(custom_tickers)}") 

    if validate_all_button:
        tickers = sorted(list(set(st.session_state.locked_tickers + selected_tickers_multi)))
        if not tickers:
            st.sidebar.error(T['ticker_error'])
        else:
            st.session_state.locked_tickers = tickers
            st.session_state.step = 2
            st.rerun()

# --- ÉTAPE 2 : Paramètres (Conservée du Script 1) ---
elif st.session_state.step == 2:
    st.sidebar.subheader(T['tickers_locked'])
    st.sidebar.info(", ".join(st.session_state.locked_tickers))
    if st.sidebar.button(T['tickers_modify_button']):
        st.session_state.step = 1
        st.session_state.run_simulation = False # On réinitialise
        st.rerun()

    # Case à cocher interactive (HORS DU FORMULAIRE)
    use_current_portfolio = st.sidebar.checkbox(T['compare_label'], key='compare_checkbox')
    
    # Logique pour les boutons "By Amount" / "By Share" (HORS DU FORMULAIRE)
    input_mode = None
    if use_current_portfolio: # N'affiche les boutons que si la case est cochée
        input_mode = st.sidebar.radio(
            T['input_mode_label'],
            (T['mode_amount'], T['mode_shares']),
            key='input_mode_radio'
        )
    
    with st.sidebar.form('params_form'):
        st.sidebar.subheader(T['sim_params_header'])
        period = st.text_input(T['period_label'], "504d")
        num_ports = st.slider(T['ports_label'], 1000, 20000, 10000, 1000)

        current_inputs = []
        monetary_values = []
        
        tickers_in_form = st.session_state.locked_tickers

        if use_current_portfolio: # Utilise la valeur de la checkbox
            st.sidebar.header(T['current_header'])
            
            if input_mode == T['mode_amount']:
                st.write(T['mode_amount'] + ":")
                for ticker in tickers_in_form:
                    amount = st.number_input(
                        T['amount_label'].format(ticker=ticker),
                        min_value=0.0, value=1000.0, step=10.0,
                        key=f"amount_{ticker}"
                    )
                    current_inputs.append(amount)
            
            elif input_mode == T['mode_shares']:
                st.write(T['mode_shares'] + ":")
                for ticker in tickers_in_form:
                    shares = st.number_input(
                        T['shares_label'].format(ticker=ticker),
                        min_value=0.0, value=10.0, step=0.1,
                        key=f"shares_{ticker}"
                    )
                    current_inputs.append(shares)

        run_button = st.form_submit_button(T['run_button'])
        if run_button:
            st.session_state.run_simulation = True # On sauvegarde le fait qu'on a cliqué
            st.session_state.current_inputs = current_inputs
            st.session_state.input_mode = input_mode
            st.session_state.use_current_portfolio = use_current_portfolio


# ---------------------------
# CORPS PRINCIPAL (Conservé du Script 1)
# ---------------------------

col_img, col_titre = st.columns([1, 4])
with col_img:
    st.image(
        "markowitz.jpg", # Charge le fichier local
        width=150,
        caption="Harry Markowitz"
    )
with col_titre:
    st.title(f"📊 {T['title']}")
    st.markdown(f"{T['created_by']} | [LinkedIn](https://www.linkedin.com/in/leopaullaisne)")
    st.markdown(f"*{T['subtitle']}*")


if not st.session_state.run_simulation:
    st.info(T['run_info'])
    st.stop()

# Le code s'exécute si on a cliqué sur "Run"
tickers = st.session_state.locked_tickers

if not tickers:
    st.error(T['ticker_error'])
    st.stop()

try:
    with st.spinner(T['loading_data'].format(tickers=", ".join(tickers))):
        stocks = yf.download(tickers, period=period, auto_adjust=True)['Close']
        if isinstance(stocks, pd.Series):
            stocks = stocks.to_frame(name=tickers[0])
    
    log_ret = np.log(stocks / stocks.shift(1)).dropna()
    num_assets = len(tickers)
    current_prices = stocks.iloc[-1]

except Exception as e:
    st.error(T['loading_error'].format(e=e))
    st.stop()

# --- CALCULS DU PORTEFEUILLE ACTUEL (Conservé du Script 1) ---
current_return, current_risk, current_sharpe = None, None, None
current_weights_np = None
total_portfolio_value = 0
monetary_values = []
use_current_portfolio = st.session_state.use_current_portfolio
input_mode = st.session_state.input_mode
current_inputs = st.session_state.current_inputs

if use_current_portfolio and current_inputs:
    if len(current_inputs) != num_assets:
        st.error(T['input_error'].format(entries=len(current_inputs), tickers=num_assets))
        st.stop()

    if input_mode == T['mode_amount']:
        monetary_values = current_inputs
    
    elif input_mode == T['mode_shares']:
        for i, ticker in enumerate(tickers):
            price = current_prices[ticker] if num_assets > 1 else current_prices[tickers[0]]
            shares = current_inputs[i]
            monetary_values.append(shares * price)
    
    total_portfolio_value = sum(monetary_values)

    if total_portfolio_value == 0:
        st.error(T['value_error'])
        st.stop()
    
    current_weights = [value / total_portfolio_value for value in monetary_values]
    current_weights_np = np.array(current_weights)

    current_return = np.sum((log_ret.mean().values * current_weights_np) * 252)
    current_risk = np.sqrt(np.dot(current_weights_np.T, np.dot(log_ret.cov().values * 252, current_weights_np)))
    current_sharpe = current_return / current_risk if current_risk != 0 else 0
# --- FIN DES CALCULS ---


with st.spinner(T['running_sim'].format(num_ports=num_ports)):
    all_weights, all_returns, all_vols, all_sharpes = run_simulation(log_ret, num_ports, num_assets)

max_sharpe_idx = np.argmax(all_sharpes)
opt_weights = all_weights[max_sharpe_idx]
opt_return, opt_vol, opt_sharpe = all_returns[max_sharpe_idx], all_vols[max_sharpe_idx], all_sharpes[max_sharpe_idx]

# --- AFFICHAGE COMPLET (Conservé du Script 1) ---

if use_current_portfolio and current_return is not None:
    st.header(T['current_analysis_header'])
    
    st.metric(T['current_total_value'], f"{total_portfolio_value:,.2f} (devise de l'action)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric(T['current_return'], f"{(current_return*100):.2f}%")
    col2.metric(T['current_risk'], f"{(current_risk*100):.2f}%")
    col3.metric(T['current_sharpe'], f"{current_sharpe:.4f}")
    
    st.subheader(T['current_weights_header'])
    current_weights_df = pd.DataFrame({
        T['col_action']: tickers,
        T['col_weight']: [w * 100 for w in current_weights],
        T['col_amount']: monetary_values
    })
    st.dataframe(current_weights_df.set_index(T['col_action']).style.format({
        T['col_weight']: '{:.2f}%',
        T['col_amount']: '{:,.2f}'
    }))
    st.divider()

st.header(T['optimal_header'])
st.write(T['optimal_subheader'].format(num_ports=num_ports))

col1, col2, col3 = st.columns(3)
col1.metric(T['optimal_return'], f"{(opt_return*100):.2f}%")
col2.metric(T['optimal_risk'], f"{(opt_vol*100):.2f}%")
col3.metric(T['optimal_sharpe'], f"{opt_sharpe:.4f}")

st.subheader(T['optimal_alloc_header'])
weights_data = {T['col_action']: tickers, T['col_weight']: opt_weights * 100}

if use_current_portfolio and total_portfolio_value > 0:
    weights_data[T['col_amount_optimal']] = opt_weights * total_portfolio_value

weights_df = pd.DataFrame(weights_data)

format_dict = {T['col_weight']: '{:.2f}%'}
if T['col_amount_optimal'] in weights_df.columns:
     format_dict[T['col_amount_optimal']] = '{:,.2f}'

st.dataframe(weights_df.set_index(T['col_action']).style.format(format_dict))

fig_weights = px.bar(weights_df, x=T['col_action'], y=T['col_weight'],
                       title=T['alloc_chart_title'],
                       text=weights_df[T['col_weight']].apply(lambda x: f'{x:.2f}%')
                      )
fig_weights.update_layout(template='plotly_dark')
st.plotly_chart(fig_weights, use_container_width=True)

st.header(T['frontier_header'])
st.write(T['frontier_subheader'])

df_plot = pd.DataFrame({
    'Return': all_returns,
    'Risk': all_vols,
    'Sharpe': all_sharpes
})
fig_scatter = px.scatter(df_plot, x="Risk", y="Return", color="Sharpe",
               color_continuous_scale='RdYlGn',
               labels={'Sharpe': 'Ratio de Sharpe'},
               hover_data={'Risk': ':.4f', 'Return': ':.4f', 'Sharpe': ':.4f'}
              )
fig_scatter.update_layout(
    title=T['frontier_chart_title'],
    xaxis_title=T['frontier_xaxis'],
    yaxis_title=T['frontier_yaxis'],
    template='plotly_dark',
    legend=dict(
        title=T['legend_title'],
        yanchor="bottom", y=1.02,  # Ajusté pour être au-dessus de la courbe
        xanchor="right", x=1,     # Ajusté pour être en haut à droite
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="white", borderwidth=1
    )
)
fig_scatter.add_shape(type='line', x0=0, y0=0,
                      x1=opt_vol, y1=opt_return,
                      line=dict(color="lime", width=2, dash="dot"))

fig_scatter.add_trace(go.Scatter(
    x=[opt_vol], 
    y=[opt_return],
    mode='markers',
    marker=dict(color='white', size=10, line=dict(color='black', width=2)),
    name=T['legend_optimal']
))

if use_current_portfolio and current_return is not None:
    fig_scatter.add_trace(go.Scatter(
        x=[current_risk], 
        y=[current_return],
        mode='markers',
        marker=dict(color='cyan', size=12, symbol='star', line=dict(color='black', width=1)),
        name=T['legend_current']
    ))

st.plotly_chart(fig_scatter, use_container_width=True)

with st.expander(T['extra_charts_header']):
    
    st.subheader(T['prices_chart_title'])
    fig_prices = px.line(stocks[tickers], title=T['prices_chart_title'])
    fig_prices.update_layout(template='plotly_dark', yaxis_title=T['prices_chart_yaxis'], xaxis_title=T['prices_chart_xaxis'], legend_title=T['prices_chart_legend'])
    st.plotly_chart(fig_prices, use_container_width=True)

    st.subheader(T['prices_table_start'])
    st.dataframe(stocks[tickers].head(5).style.format("{:.2f}"))

    st.subheader(T['prices_table_end'])
    st.dataframe(stocks[tickers].tail(5).style.format("{:.2f}"))
    st.divider()

    st.subheader(T['returns_table_start'])
    daily_pct_change = stocks[tickers].pct_change().dropna() * 100
    st.dataframe(daily_pct_change.head(5).style.format("{:.2f}%"))
    
    st.subheader(T['returns_table_end'])
    st.dataframe(daily_pct_change.tail(5).style.format("{:.2f}%"))

with st.expander(T['corr_header']):
    df_corr = log_ret[tickers].corr()
    fig_heatmap = px.imshow(df_corr, text_auto=True, color_continuous_scale='Mint',
                             labels=dict(y=T['corr_company'], x=T['corr_company']))
    fig_heatmap.update_layout(template='plotly_dark')
    st.plotly_chart(fig_heatmap, use_container_width=True)

if use_current_portfolio and current_return is not None:
if use_current_portfolio and current_return is not None:
    st.header(T['conclusion_header'])
    st.write(T['conclusion_subheader'].format(value=total_portfolio_value))

    optimal_values = opt_weights * total_portfolio_value
    
    st.subheader(T['action_header'])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.write(f"**{T['col_action']}**")
    col2.write(f"**{T['col_current_pos']}**")
    col3.write(f"**{T['col_optimal_pos']}**")
    col4.write(f"**{T['col_action_req']}**")
    st.divider()

    for i, ticker in enumerate(tickers):
        current_val = monetary_values[i]
        optimal_val = optimal_values[i]
        diff = optimal_val - current_val
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**{ticker}**")
        with col2:
            st.write(f"{current_val:,.2f}")
        with col3:
            st.write(f"{optimal_val:,.2f}")
        with col4:
            if diff > 0.01:
                st.success(T['action_buy'].format(diff=diff))
            elif diff < -0.01:
                st.error(T['action_sell'].format(abs_diff=abs(diff)))
            else:
                st.info(T['action_hold'])
