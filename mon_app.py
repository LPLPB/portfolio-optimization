import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Optimiseur de Portefeuille",
    page_icon="📊",
    layout="wide"
)

# --- DONNÉES PRÉDÉFINIES (Pour la recherche) ---
PREDEFINED_TICKERS = {
    'AAPL': 'Apple (NASDAQ)',
    'MSFT': 'Microsoft (NASDAQ)',
    'GOOG': 'Alphabet (Google) (NASDAQ)',
    'AMZN': 'Amazon (NASDAQ)',
    'TSLA': 'Tesla (NASDAQ)',
    'NVDA': 'NVIDIA (NASDAQ)',
    'META': 'Meta Platforms (NASDAQ)',
    'JPM': 'JPMorgan Chase (NYSE)',
    'JNJ': 'Johnson & Johnson (NYSE)',
    'V': 'Visa (NYSE)',
    'SPY': 'ETF - S&P 500 (SPDR)',
    'QQQ': 'ETF - Nasdaq 100 (Invesco)',
    'URTH': 'ETF - MSCI World (iShares)',
    'EEM': 'ETF - MSCI Emerging Markets (iShares)',
    'MC.PA': 'LVMH (Euronext Paris)',
    'OR.PA': 'L\'Oréal (Euronext Paris)',
    'RMS.PA': 'Hermès (Euronext Paris)',
    'DCAM.PA': 'Amundi (Euronext Paris)',
    'TTE.PA': 'TotalEnergies (Euronext Paris)',
    'SAN.PA': 'Sanofi (Euronext Paris)',
    'AIR.PA': 'Airbus (Euronext Paris)',
    'BNP.PA': 'BNP Paribas (Euronext Paris)',
    'BTC-USD': 'Bitcoin (Crypto)',
    'ETH-USD': 'Ethereum (Crypto)',
}

def add_ticker(ticker):
    if ticker not in st.session_state.selected_tickers:
        st.session_state.selected_tickers.append(ticker)
    st.rerun()

# --- FONCTION CACHÉE POUR LA SIMULATION ---
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

# --- 2. BARRE LATÉRALE (Inputs) ---
st.sidebar.header("Paramètres de l'Optimisation")

if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = ['DCAM.PA', 'TSLA']

# Le sélecteur "intelligent"
selected_tickers_multi = st.sidebar.multiselect(
    "1. Sélectionnez ou recherchez des actions",
    options=list(PREDEFINED_TICKERS.keys()),
    default=st.session_state.selected_tickers,
    format_func=lambda ticker: f"{ticker} - {PREDEFINED_TICKERS.get(ticker, 'Ticker personnalisé')}"
)

# --- NOUVEAU : Ajout manuel de Tickers ---
custom_tickers_string = st.sidebar.text_input("Ou ajoutez des tickers manuellement (ex: TCKR1, TCKR2)")
custom_tickers = [t.strip().upper() for t in custom_tickers_string.split(',') if t.strip()]

# On combine les deux listes et on enlève les doublons
tickers = sorted(list(set(selected_tickers_multi + custom_tickers)))
st.session_state.selected_tickers = tickers # On sauvegarde la liste combinée
# --- FIN NOUVEAU ---

st.sidebar.subheader("Ou ajoutez via les catégories")
with st.sidebar.expander("ETFs Populaires"):
    if st.button("S&P 500 (SPY)", use_container_width=True): add_ticker('SPY')
    if st.button("Nasdaq 100 (QQQ)", use_container_width=True): add_ticker('QQQ')
    if st.button("MSCI World (URTH)", use_container_width=True): add_ticker('URTH')

with st.sidebar.expander("Actions US Tech (NASDAQ)"):
    if st.button("Apple (AAPL)", use_container_width=True): add_ticker('AAPL')
    if st.button("Microsoft (MSFT)", use_container_width=True): add_ticker('MSFT')
    if st.button("Google (GOOG)", use_container_width=True): add_ticker('GOOG')
    if st.button("Tesla (TSLA)", use_container_width=True): add_ticker('TSLA')

with st.sidebar.expander("Actions Françaises (CAC40)"):
    if st.button("LVMH (MC.PA)", use_container_width=True): add_ticker('MC.PA')
    if st.button("L'Oréal (OR.PA)", use_container_width=True): add_ticker('OR.PA')
    if st.button("Hermès (RMS.PA)", use_container_width=True): add_ticker('RMS.PA')

st.sidebar.divider()
st.sidebar.subheader("2. Paramètres de simulation")

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
            amount = st.sidebar.number_input(f"Montant pour {ticker}", min_value=0.0, value=1000.0, step=10.0, key=f"amount_{ticker}")
            current_inputs.append(amount)
    
    elif input_mode == "Par Nombre d'Actions":
        st.sidebar.write("Entrez le nombre d'actions que vous possédez :")
        for ticker in tickers:
            shares = st.sidebar.number_input(f"Nombre d'actions {ticker}", min_value=0.0, value=10.0, step=0.1, key=f"shares_{ticker}")
            current_inputs.append(shares)

run_button = st.sidebar.button("Lancer l'Optimisation")


# --- 3. CORPS PRINCIPAL DE L'APPLICATION ---

# --- NOUVEAU : Correction Photo Markowitz ---
col_img, col_titre = st.columns([1, 4])
with col_img:
    st.image(
        "markowitz.jpg", # Charge le fichier local que tu as uploadé sur GitHub
        width=150,
        caption="Harry Markowitz"
    )
with col_titre:
    st.title("📊 Optimiseur de Portefeuille")
    st.markdown("Créé par **Léo-Paul Laisne** | [Profil LinkedIn](https://www.linkedin.com/in/leopaullaisne)")
    st.markdown("*Basé sur la Théorie Moderne du Portefeuille (Markowitz)*")
# --- FIN NOUVEAU ---


if not run_button:
    st.info("Veuillez sélectionner vos actions et cliquer sur 'Lancer l'Optimisation' dans la barre latérale.")
    st.stop()

if not tickers:
    st.error("Veuillez sélectionner au moins un ticker.")
    st.stop()

try:
    with st.spinner(f"Téléchargement des données pour {', '.join(tickers)}..."):
        stocks = yf.download(tickers, period=period, auto_adjust=True)['Close']
        if len(tickers) == 1:
            stocks = stocks.to_frame(name=tickers[0])
    
    log_ret = np.log(stocks / stocks.shift(1)).dropna()
    num_assets = len(tickers)
    current_prices = stocks.iloc[-1]

except Exception as e:
    st.error(f"Erreur lors du téléchargement des données : {e}")
    st.stop()

current_return, current_risk, current_sharpe = None, None, None
current_weights_np = None
total_portfolio_value = 0

if use_current_portfolio and current_inputs:
    if len(current_inputs) != num_assets:
        st.error(f"Erreur : Le nombre d'entrées ({len(current_inputs)}) ne correspond pas au nombre de tickers ({num_assets}).")
        st.stop()

    if input_mode == "Par Montant (€/$)":
        monetary_values = current_inputs
    
    elif input_mode == "Par Nombre d'Actions":
        for i, ticker in enumerate(tickers):
            price = current_prices if num_assets == 1 else current_prices[ticker]
            shares = current_inputs[i]
            monetary_values.append(shares * price)
    
    total_portfolio_value = sum(monetary_values)

    if total_portfolio_value == 0:
        st.error("La valeur totale de votre portefeuille est 0. Impossible de calculer les poids.")
        st.stop()
    
    current_weights = [value / total_portfolio_value for value in monetary_values]
    current_weights_np = np.array(current_weights)

    current_return = np.sum((log_ret.mean() * current_weights_np) * 252)
    current_risk = np.sqrt(np.dot(current_weights_np.T, np.dot(log_ret.cov() * 252, current_weights_np)))
    current_sharpe = current_return / current_risk if current_risk != 0 else 0

with st.spinner(f"Simulation de {num_ports} portefeuilles..."):
    all_weights, all_returns, all_volatilities, all_sharpe_ratios = run_simulation(log_ret, num_ports, num_assets)

max_sr_index = all_sharpe_ratios.argmax()
max_sr_ret = all_returns[max_sr_index]
max_sr_vol = all_volatilities[max_sr_index]
best_sharpe_ratio = all_sharpe_ratios[max_sr_index]
best_weights = all_weights[max_sr_index]

if use_current_portfolio and current_return is not None:
    st.header("Analyse de Votre Portefeuille Actuel")
    
    st.metric("Valeur Totale de Votre Portefeuille", f"{total_portfolio_value:,.2f} (devise de l'action)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Rendement Actuel", f"{(current_return*100):.2f}%")
    col2.metric("Risque Actuel", f"{(current_risk*100):.2f}%")
    col3.metric("Sharpe Ratio Actuel", f"{current_sharpe:.4f}")
    
    st.subheader("Poids calculés de votre portefeuille")
    current_weights_df = pd.DataFrame({
        'Action': tickers,
        'Poids (%)': [w * 100 for w in current_weights],
        'Montant (€/$)': monetary_values
    })
    st.dataframe(current_weights_df.set_index('Action').style.format({
        'Poids (%)': '{:.2f}%',
        'Montant (€/$)': '{:,.2f}'
    }))
    st.divider()

st.header("Résultats de l'Optimisation (Portefeuille Optimal)")
st.write(f"Portefeuille optimal (meilleur Sharpe) trouvé parmi {num_ports} simulations.")

col1, col2, col3 = st.columns(3)
col1.metric("Rendement Optimal", f"{(max_sr_ret*100):.2f}%")
col2.metric("Volatilité Optimale", f"{(max_sr_vol*100):.2f}%")
col3.metric("Meilleur Sharpe Ratio", f"{best_sharpe_ratio:.4f}")

st.subheader("Allocation du Portefeuille Optimal")
weights_data = {'Action': tickers, 'Poids (%)': best_weights * 100}

if use_current_portfolio and total_portfolio_value > 0:
    weights_data['Montant Optimal (€/$)'] = best_weights * total_portfolio_value

weights_df = pd.DataFrame(weights_data)

format_dict = {'Poids (%)': '{:.2f}%'}
if 'Montant Optimal (€/$)' in weights_df.columns:
     format_dict['Montant Optimal (€/$)'] = '{:,.2f}'

st.dataframe(weights_df.set_index('Action').style.format(format_dict))

fig_weights = px.bar(weights_df, x='Action', y='Poids (%)',
                     title="Allocation (Poids) en %",
                     text=weights_df['Poids (%)'].apply(lambda x: f'{x:.2f}%')
                    )
fig_weights.update_layout(template='plotly_dark')
st.plotly_chart(fig_weights, use_container_width=True)

st.header("Frontière Efficiente & Comparaison")
st.write("Chaque point est un portefeuille simulé. Comparez votre portefeuille (⭐) à l'optimal (⚪).")

df_plot = pd.DataFrame({
    'Return': all_returns,
    'Risk': all_volatilities,
    'Sharpe': all_sharpe_ratios
})
fig_scatter = px.scatter(df_plot, x="Risk", y="Return", color="Sharpe",
                 color_continuous_scale='RdYlGn',
                 labels={'Sharpe': 'Ratio de Sharpe'},
                 hover_data={'Risk': ':.4f', 'Return': ':.4f', 'Sharpe': ':.4f'}
                )
fig_scatter.update_layout(
    title='Frontière Efficiente (Courbe si 2 actifs, Nuage si 3+)',
    xaxis_title='Volatilité Annuelle (Risque)',
    yaxis_title='Rendement Annuel Espéré',
    template='plotly_dark',
    legend=dict(
        title="Légende",
        yanchor="top", y=0.98,
        xanchor="left", x=0.01,
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="white", borderwidth=1
    )
)
fig_scatter.add_shape(type='line', x0=0, y0=0,
                      x1=max_sr_vol, y1=max_sr_ret,
                      line=dict(color="lime", width=2, dash="dot"))

fig_scatter.add_trace(go.Scatter(
    x=[max_sr_vol], 
    y=[max_sr_ret],
    mode='markers',
    marker=dict(color='white', size=10, line=dict(color='black', width=2)),
    name='Portefeuille Optimal'
))

if use_current_portfolio and current_return is not None:
    fig_scatter.add_trace(go.Scatter(
        x=[current_risk], 
        y=[current_return],
        mode='markers',
        marker=dict(color='cyan', size=12, symbol='star', line=dict(color='black', width=1)),
        name='Mon Portefeuille Actuel'
    ))

st.plotly_chart(fig_scatter, use_container_width=True)

with st.expander("Afficher l'évolution historique des prix et retours journaliers"):
    
    st.subheader(f"Prix de clôture (sur {period})")
    fig_prices = px.line(stocks[tickers], title="Évolution des prix de clôture (Ajustés)")
    fig_prices.update_layout(template='plotly_dark', yaxis_title="Prix ($)", xaxis_title="Date", legend_title="Action")
    st.plotly_chart(fig_prices, use_container_width=True)

    st.subheader("Valeurs de Début de Période (5 premiers jours)")
    st.dataframe(stocks[tickers].head(5).style.format("{:.2f}"))

    st.subheader("Valeurs de Fin de Période (5 derniers jours)")
    st.dataframe(stocks[tickers].tail(5).style.format("{:.2f}"))
    st.divider()

    st.subheader("Retours de Début de Période (5 premiers jours, en %)")
    daily_pct_change = stocks[tickers].pct_change().dropna() * 100
    st.dataframe(daily_pct_change.head(5).style.format("{:.2f}%"))
    
    st.subheader("Retours de Fin de Période (5 derniers jours, en %)")
    st.dataframe(daily_pct_change.tail(5).style.format("{:.2f}%"))

with st.expander("Afficher la Matrice de Corrélation"):
    df_corr = log_ret[tickers].corr()
    fig_heatmap = px.imshow(df_corr, text_auto=True, color_continuous_scale='Mint',
                            labels=dict(y='Compagny', x='Compagny'))
    fig_heatmap.update_layout(template='plotly_dark')
    st.plotly_chart(fig_heatmap, use_container_width=True)

if use_current_portfolio and current_return is not None:
    st.header("Conclusion & Plan d'Action")
    st.write(f"Pour rééquilibrer votre portefeuille (valeur : {total_portfolio_value:,.2f} €/$) vers l'allocation optimale :")

    optimal_values = best_weights * total_portfolio_value
    
    st.subheader("Actions Recommandées :")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.write("**Action**")
    col2.write("**Position Actuelle**")
    col3.write("**Position Optimale**")
    col4.write("**Action Requise**")
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
                st.success(f"🟢 ACHETER {diff:,.2f}")
            elif diff < -0.01:
                st.error(f"🔴 VENDRE {abs(diff):,.2f}")
            else:
                st.info("⚪ CONSERVER")
