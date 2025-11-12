import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go # Important pour les marqueurs

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Optimiseur de Portefeuille",
    page_icon="📊",
    layout="wide"
)

# --- 2. BARRE LATÉRALE (Inputs) ---
st.sidebar.header("Paramètres de l'Optimisation")

# Saisie des tickers
tickers_string = st.sidebar.text_input(
    "Entrez les Tickers des actions (séparés par une virgule)",
    "DCAM.PA, TSLA" # Valeurs par défaut
)
tickers = [t.strip().upper() for t in tickers_string.split(',')]

# Saisie de la période
period = st.sidebar.text_input(
    "Période (ex: '504d', '2y', '5y')",
    "504d" # Valeur par défaut
)

# Saisie du nombre de simulations
num_ports = st.sidebar.slider(
    "Nombre de portefeuilles à simuler",
    1000, 20000, 10000, 1000 # min, max, default, step
)

# Section pour le portefeuille actuel (par Montant ou Nb d'Actions)
st.sidebar.header("Votre Portefeuille Actuel (Optionnel)")
use_current_portfolio = st.sidebar.checkbox("Comparer avec un portefeuille actuel")

current_inputs = [] # Stocke les montants OU le nombre d'actions
input_mode = None
monetary_values = [] # Stocke les valeurs en €/$

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


# Le bouton qui lance toute l'analyse
run_button = st.sidebar.button("Lancer l'Optimisation")


# --- 3. CORPS PRINCIPAL DE L'APPLICATION ---
st.title("📊 Optimiseur de Portefeuille (Modèle Markowitz)")

# --- NOUVEAU : Lien LinkedIn (mis à jour) ---
st.markdown("Créé par **Léo-Paul Laisné** | [Profil LinkedIn](https://www.linkedin.com/in/leopaullaisne)")
# --- FIN NOUVEAU ---

if not run_button:
    st.info("Veuillez entrer les tickers et cliquer sur 'Lancer l'Optimisation' dans la barre latérale.")
    st.stop()

if not tickers:
    st.error("Veuillez entrer au moins un ticker.")
    st.stop()

# --- A. TÉLÉCHARGEMENT & PRÉPARATION DES DONNÉES ---
try:
    with st.spinner(f"Téléchargement des données pour {', '.join(tickers)}..."):
        stocks = yf.download(tickers, period=period, auto_adjust=True)['Close']
        if len(tickers) == 1:
            stocks = stocks.to_frame(name=tickers[0])
    
    log_ret = np.log(stocks / stocks.shift(1)).dropna()
    num_assets = len(tickers)
    
    # Récupérer le dernier prix pour les calculs
    current_prices = stocks.iloc[-1]

except Exception as e:
    st.error(f"Erreur lors du téléchargement des données : {e}")
    st.stop()


# --- B. CALCUL DU PORTEFEILLE ACTUEL (SI DEMANDÉ) ---
current_return, current_risk, current_sharpe = None, None, None
current_weights_np = None
total_portfolio_value = 0

if use_current_portfolio and current_inputs:
    if len(current_inputs) != num_assets:
        st.error(f"Erreur : Le nombre d'entrées ne correspond pas au nombre de tickers.")
        st.stop()

    if input_mode == "Par Montant (€/$)":
        monetary_values = current_inputs
    
    elif input_mode == "Par Nombre d'Actions":
        for i, ticker in enumerate(tickers):
            # Gérer le cas d'un seul ticker où current_prices est un float
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


# --- C. SIMULATION MONTE CARLO ---
with st.spinner(f"Simulation de {num_ports} portefeuilles..."):
    all_weights = np.zeros((num_ports, num_assets))
    all_returns = np.zeros(num_ports)
    all_volatilities = np.zeros(num_ports)
    all_sharpe_ratios = np.zeros(num_ports)

    for ind in range(num_ports):
        weights = np.array(np.random.random(num_assets))
        weights = weights / np.sum(weights)
        all_weights[ind, :] = weights
        all_returns[ind] = np.sum((log_ret.mean() * weights) * 252)
        all_volatilities[ind] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
        all_sharpe_ratios[ind] = all_returns[ind] / all_volatilities[ind]

# --- D. TROUVER LE PORTEFEUILLE OPTIMAL ---
max_sr_index = all_sharpe_ratios.argmax()
max_sr_ret = all_returns[max_sr_index]
max_sr_vol = all_volatilities[max_sr_index]
best_sharpe_ratio = all_sharpe_ratios[max_sr_index]
best_weights = all_weights[max_sr_index]

# --- E. AFFICHER LE "PRODUIT FINAL" (Le Dashboard) ---

# Affichage des stats du portefeuille actuel (SI COCHÉ)
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


# Affichage des stats du portefeuille optimal
st.header("Résultats de l'Optimisation (Portefeuille Optimal)")
st.write(f"Portefeuille optimal (meilleur Sharpe) trouvé parmi {num_ports} simulations.")

col1, col2, col3 = st.columns(3)
col1.metric("Rendement Optimal", f"{(max_sr_ret*100):.2f}%")
col2.metric("Volatilité Optimale", f"{(max_sr_vol*100):.2f}%")
col3.metric("Meilleur Sharpe Ratio", f"{best_sharpe_ratio:.4f}")

# Allocation optimale en % ET en Montant
st.subheader("Allocation du Portefeuille Optimal")
weights_data = {'Action': tickers, 'Poids (%)': best_weights * 100}

# On ne peut afficher le montant que si l'utilisateur a donné son portefeuille actuel
if use_current_portfolio and total_portfolio_value > 0:
    weights_data['Montant Optimal (€/$)'] = best_weights * total_portfolio_value

weights_df = pd.DataFrame(weights_data)

# Formatter le dictionnaire pour l'affichage
format_dict = {'Poids (%)': '{:.2f}%'}
if 'Montant Optimal (€/$)' in weights_df.columns:
     format_dict['Montant Optimal (€/$)'] = '{:,.2f}'

st.dataframe(weights_df.set_index('Action').style.format(format_dict))

# Bar chart (ne montre que les Poids, c'est plus clair)
fig_weights = px.bar(weights_df, x='Action', y='Poids (%)',
                     title="Allocation (Poids) en %",
                     text=weights_df['Poids (%)'].apply(lambda x: f'{x:.2f}%')
                    )
fig_weights.update_layout(template='plotly_dark')
st.plotly_chart(fig_weights, use_container_width=True)


# --- F. AFFICHER LE GRAPHIQUE DE LA FRONTIÈRE ---
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
    legend=dict( # La correction pour la légende
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

# Point du portefeuille optimal
fig_scatter.add_trace(go.Scatter(
    x=[max_sr_vol], y=[max_sr_ret],
    mode='markers',