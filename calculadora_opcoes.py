
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

def black_scholes(S, K, T, r, sigma, tipo):
    if T <= 0:
        return max(S - K, 0) if tipo == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if tipo == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def vega_bs(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 1e-10
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)

def monte_carlo(S, K, T, r, sigma, tipo, opcao_tipo, n_sim, n_steps=252):
    np.random.seed(42)
    dt = T / n_steps
    payoffs = []
    trajetorias = []
    for i in range(n_sim):
        prices = [S]
        for _ in range(n_steps):
            Z = np.random.standard_normal()
            S_t = prices[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            prices.append(S_t)
        ST = prices[-1]
        S_media = np.mean(prices)
        if opcao_tipo == "asiatica":
            payoff = max(S_media - K, 0) if tipo == "call" else max(K - S_media, 0)
        else:
            payoff = max(ST - K, 0) if tipo == "call" else max(K - ST, 0)
        payoffs.append(payoff)
        if i < 50:
            trajetorias.append(prices)
    preco = np.exp(-r * T) * np.mean(payoffs)
    erro = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sim)
    return preco, erro, trajetorias

def arvore_binomial(S, K, T, r, sigma, tipo, opcao_tipo, n_steps):
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    ST = np.array([S * (u ** j) * (d ** (n_steps - j)) for j in range(n_steps + 1)])
    V = np.maximum(ST - K, 0) if tipo == "call" else np.maximum(K - ST, 0)
    for i in range(n_steps - 1, -1, -1):
        S_node = np.array([S * (u ** j) * (d ** (i - j)) for j in range(i + 1)])
        V = disc * (p * V[1:i+2] + (1 - p) * V[0:i+1])
        if opcao_tipo == "americana":
            V = np.maximum(V, S_node - K) if tipo == "call" else np.maximum(V, K - S_node)
    return V[0]

def vol_implicita_bissecao(S, K, T, r, preco_mercado, tipo, tol=1e-6, max_iter=1000):
    sigma_min, sigma_max = 1e-4, 10.0
    f_min = black_scholes(S, K, T, r, sigma_min, tipo) - preco_mercado
    f_max = black_scholes(S, K, T, r, sigma_max, tipo) - preco_mercado
    if f_min * f_max > 0:
        return None, "Preco de mercado fora dos limites teoricos."
    for _ in range(max_iter):
        sigma_mid = (sigma_min + sigma_max) / 2
        f_mid = black_scholes(S, K, T, r, sigma_mid, tipo) - preco_mercado
        if abs(f_mid) < tol:
            return sigma_mid, None
        if f_min * f_mid < 0:
            sigma_max = sigma_mid
            f_max = f_mid
        else:
            sigma_min = sigma_mid
            f_min = f_mid
    return (sigma_min + sigma_max) / 2, None

def vol_implicita_newton(S, K, T, r, preco_mercado, tipo, tol=1e-6, max_iter=100):
    sigma = 0.3
    for _ in range(max_iter):
        preco = black_scholes(S, K, T, r, sigma, tipo)
        v = vega_bs(S, K, T, r, sigma)
        if abs(v) < 1e-10:
            return None, "Vega muito baixo."
        sigma_novo = sigma - (preco - preco_mercado) / v
        if sigma_novo <= 0:
            return None, "Sigma negativo."
        if abs(sigma_novo - sigma) < tol:
            return sigma_novo, None
        sigma = sigma_novo
    return None, "Nao convergiu."

@st.cache_data(ttl=300)
def buscar_dados(ticker, periodo="1y"):
    try:
        dados = yf.download(ticker, period=periodo, progress=False, auto_adjust=True)
        if dados.empty:
            return None, None, None
        preco_atual = float(dados["Close"].iloc[-1])
        log_ret = np.log(dados["Close"] / dados["Close"].shift(1)).dropna()
        vol_hist = float(log_ret.std() * np.sqrt(252))
        return preco_atual, vol_hist, dados
    except:
        return None, None, None

def grafico_payoff(S, K, tipo, preco_opcao):
    S_range = np.linspace(S * 0.5, S * 1.5, 300)
    payoff = np.maximum(S_range - K, 0) if tipo == "call" else np.maximum(K - S_range, 0)
    lucro = payoff - preco_opcao
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=payoff, name="Payoff no vencimento", line=dict(color="#2196F3", width=2)))
    fig.add_trace(go.Scatter(x=S_range, y=lucro, name="Lucro/Prejuizo", line=dict(color="#4CAF50", width=2, dash="dash")))
    fig.add_hline(y=0, line_color="gray", line_dash="dot")
    fig.add_vline(x=K, line_color="red", line_dash="dot", annotation_text="K=" + str(round(K, 2)), annotation_position="top right")
    fig.add_vline(x=S, line_color="orange", line_dash="dot", annotation_text="S=" + str(round(S, 2)), annotation_position="top left")
    fig.update_layout(title="Payoff da " + tipo.upper(), xaxis_title="ST", yaxis_title="Valor (R$)", template="plotly_dark", height=400)
    return fig

def grafico_monte_carlo(trajetorias, S, T, ticker):
    fig = go.Figure()
    t_axis = np.linspace(0, T * 252, len(trajetorias[0]))
    for traj in trajetorias:
        fig.add_trace(go.Scatter(x=t_axis, y=traj, mode="lines", line=dict(width=0.8, color="rgba(100,180,255,0.3)"), showlegend=False))
    fig.add_hline(y=S, line_color="white", line_dash="dot", annotation_text="S0=" + str(round(S, 2)))
    fig.update_layout(title="Trajetorias Monte Carlo - " + ticker, xaxis_title="Dias", yaxis_title="Preco (R$)", template="plotly_dark", height=400)
    return fig

def grafico_smile(S, K, T, r, tipo):
    strikes = np.linspace(K * 0.7, K * 1.3, 30)
    moneyness = strikes / S
    vol_smile = 0.25 + 0.10 * (moneyness - 1)**2 + 0.03 * (moneyness - 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strikes, y=vol_smile * 100, name="Vol Implicita (%)", line=dict(color="#FF9800", width=2)))
    fig.add_vline(x=K, line_color="red", line_dash="dot", annotation_text="ATM", annotation_position="top right")
    fig.update_layout(title="Smile de Volatilidade", xaxis_title="Strike (K)", yaxis_title="Vol Implicita (%)", template="plotly_dark", height=380)
    return fig

def grafico_historico(dados, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dados.index, y=dados["Close"].squeeze(), mode="lines", line=dict(color="#2196F3", width=1.5), name=ticker))
    fig.update_layout(title="Historico - " + ticker, xaxis_title="Data", yaxis_title="Preco (R$)", template="plotly_dark", height=350)
    return fig

st.set_page_config(page_title="Calculadora de Opcoes", page_icon="📊", layout="wide")
st.title("📊 Calculadora de Opcoes")
st.caption("Black-Scholes · Monte Carlo · Arvore Binomial · Volatilidade Implicita")
st.divider()

with st.sidebar:
    st.header("Parametros")
    ticker = st.text_input("Ticker (Yahoo Finance)", value="PETR4.SA")
    usar_yahoo = st.checkbox("Buscar dados do Yahoo Finance", value=True)

    preco_atual_yahoo = None
    vol_hist_yahoo = None
    dados_hist = None

    if usar_yahoo and ticker:
        with st.spinner("Buscando dados..."):
            preco_atual_yahoo, vol_hist_yahoo, dados_hist = buscar_dados(ticker)
        if preco_atual_yahoo:
            st.success("Preco atual: R$ " + str(round(preco_atual_yahoo, 2)))
            st.info("Vol. Historica: " + str(round(vol_hist_yahoo * 100, 1)) + "% a.a.")
        else:
            st.warning("Ticker nao encontrado. Insira os dados manualmente.")

    st.divider()
    S_default = float(preco_atual_yahoo) if preco_atual_yahoo else 38.0
    sigma_default = float(vol_hist_yahoo) if vol_hist_yahoo else 0.30

    S = st.number_input("Preco atual do ativo (S0)", value=S_default, min_value=0.01, step=0.5, format="%.2f")
    K = st.number_input("Preco de exercicio (K)", value=round(S_default * 1.05, 0), min_value=0.01, step=0.5, format="%.2f")
    T = st.number_input("Prazo ate vencimento (anos)", value=0.5, min_value=0.01, max_value=5.0, step=0.05, format="%.2f")
    r = st.number_input("Taxa livre de risco (a.a.)", value=0.10, min_value=0.0, max_value=1.0, step=0.01, format="%.3f")
    sigma = st.number_input("Volatilidade (a.a.)", value=sigma_default, min_value=0.01, max_value=5.0, step=0.01, format="%.3f")

    st.divider()
    tipo_payoff = st.radio("Payoff", ["call", "put"], horizontal=True)
    tipo_opcao = st.radio("Estilo", ["europeia", "americana", "asiatica"], horizontal=True)

    st.divider()
    metodo = st.radio("Metodo", ["Black-Scholes", "Monte Carlo", "Arvore Binomial", "Todos"], index=3)

    n_sim = 10000
    n_steps = 100
    if metodo in ["Monte Carlo", "Todos"]:
        n_sim = st.slider("Simulacoes (Monte Carlo)", 1000, 50000, 10000, step=1000)
    if metodo in ["Arvore Binomial", "Todos"]:
        n_steps = st.slider("Passos (Arvore Binomial)", 10, 500, 100, step=10)

    st.divider()
    calcular_vi = st.checkbox("Calcular Volatilidade Implicita", value=False)
    preco_mercado_vi = 2.0
    metodo_vi = "Bissecao"
    vi = None
    if calcular_vi:
        preco_mercado_vi = st.number_input("Preco de mercado da opcao", value=2.0, min_value=0.001, step=0.1, format="%.3f")
        metodo_vi = st.radio("Metodo VI", ["Bissecao", "Newton-Raphson"], horizontal=True)

    calcular = st.button("Calcular", type="primary", use_container_width=True)

if not calcular:
    st.info("Configure os parametros na barra lateral e clique em Calcular.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Modelos**\n- Black-Scholes\n- Monte Carlo\n- Arvore Binomial")
    c2.markdown("**Funcionalidades**\n- Yahoo Finance\n- Vol. historica e implicita\n- Graficos")
    c3.markdown("**Exemplos**\n- PETR4.SA Call Europeia\n- VALE3.SA Put Americana\n- ITUB4.SA Call Asiatica")
else:
    resultados = {}
    erro_mc = None
    trajetorias = []

    if tipo_opcao == "europeia" and metodo in ["Black-Scholes", "Todos"]:
        resultados["Black-Scholes"] = black_scholes(S, K, T, r, sigma, tipo_payoff)

    if tipo_opcao in ["europeia", "asiatica"] and metodo in ["Monte Carlo", "Todos"]:
        with st.spinner("Rodando Monte Carlo..."):
            preco_mc, erro_mc, trajetorias = monte_carlo(S, K, T, r, sigma, tipo_payoff, tipo_opcao, n_sim)
        resultados["Monte Carlo"] = preco_mc

    if tipo_opcao in ["europeia", "americana"] and metodo in ["Arvore Binomial", "Todos"]:
        resultados["Arvore Binomial"] = arvore_binomial(S, K, T, r, sigma, tipo_payoff, tipo_opcao, n_steps)

    st.subheader("Resultados")
    if not resultados:
        st.warning("Combinacao nao disponivel para este tipo de opcao e metodo.")
    else:
        n_cols = len(resultados) + (1 if calcular_vi else 0)
        cols = st.columns(n_cols)
        for i, (nome, preco) in enumerate(resultados.items()):
            cols[i].metric(label=nome, value="R$ " + str(round(preco, 4)))

        if calcular_vi:
            if metodo_vi == "Bissecao":
                vi, err_vi = vol_implicita_bissecao(S, K, T, r, preco_mercado_vi, tipo_payoff)
            else:
                vi, err_vi = vol_implicita_newton(S, K, T, r, preco_mercado_vi, tipo_payoff)
            if vi:
                cols[-1].metric(label="Vol. Implicita (" + metodo_vi + ")", value=str(round(vi * 100, 2)) + "%")
            else:
                cols[-1].error("VI: " + str(err_vi))

        st.divider()
        if len(resultados) > 1:
            st.subheader("Comparacao entre Metodos")
            rows = []
            for n, p in resultados.items():
                rows.append({"Metodo": n, "Preco (R$)": round(p, 4), "Opcao": tipo_opcao, "Payoff": tipo_payoff.upper(), "S0": S, "K": K, "T": T, "r": str(round(r * 100, 1)) + "%", "sigma": str(round(sigma * 100, 1)) + "%"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if erro_mc:
                st.caption("Erro padrao Monte Carlo: +/- R$ " + str(round(erro_mc, 4)) + " (" + str(n_sim) + " simulacoes)")

        st.divider()
        tab1, tab2, tab3, tab4 = st.tabs(["Payoff", "Monte Carlo", "Smile de Volatilidade", "Historico"])
        preco_ref = list(resultados.values())[0]

        with tab1:
            st.plotly_chart(grafico_payoff(S, K, tipo_payoff, preco_ref), use_container_width=True)
            itm = (tipo_payoff == "call" and S > K) or (tipo_payoff == "put" and S < K)
            otm = (tipo_payoff == "call" and S < K) or (tipo_payoff == "put" and S > K)
            moneyness = "ITM" if itm else ("OTM" if otm else "ATM")
            st.markdown("Preco: **R$ " + str(round(preco_ref, 4)) + "** | Break-even: **R$ " + str(round(K + preco_ref, 2)) + "** | Moneyness: **" + moneyness + "**")

        with tab2:
            if "Monte Carlo" in resultados and trajetorias:
                st.plotly_chart(grafico_monte_carlo(trajetorias, S, T, ticker), use_container_width=True)
                st.markdown(str(n_sim) + " simulacoes | Preco: **R$ " + str(round(resultados["Monte Carlo"], 4)) + "** +/- R$ " + str(round(erro_mc, 4)))
            else:
                st.info("Selecione Monte Carlo para ver as trajetorias.")

        with tab3:
            st.plotly_chart(grafico_smile(S, K, T, r, tipo_payoff), use_container_width=True)
            st.markdown("Smile de volatilidade: vol. implicita tende a ser maior nas pontas (ITM e OTM).")
            if calcular_vi and vi:
                st.info("Vol. implicita para K=" + str(K) + ": **" + str(round(vi * 100, 2)) + "%** (preco de mercado = R$ " + str(round(preco_mercado_vi, 3)) + ")")

        with tab4:
            if dados_hist is not None:
                st.plotly_chart(grafico_historico(dados_hist, ticker), use_container_width=True)
                ret = np.log(dados_hist["Close"] / dados_hist["Close"].shift(1)).dropna()
                ca, cb, cc = st.columns(3)
                ca.metric("Preco Atual", "R$ " + str(round(float(dados_hist["Close"].iloc[-1]), 2)))
                cb.metric("Vol. Historica (a.a.)", str(round(float(ret.std() * np.sqrt(252)) * 100, 1)) + "%")
                cc.metric("Retorno 1 ano", str(round(float((dados_hist["Close"].iloc[-1] / dados_hist["Close"].iloc[0] - 1)) * 100, 1)) + "%")
            else:
                st.info("Ative o Yahoo Finance para ver o historico.")

        st.divider()
        st.subheader("Gregas (Black-Scholes)")
        if T > 0:
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            delta = norm.cdf(d1) if tipo_payoff == "call" else norm.cdf(d1) - 1
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega_val = S * np.sqrt(T) * norm.pdf(d1) * 0.01
            if tipo_payoff == "call":
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 252
                rho = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01
            else:
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 252
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01
            g1, g2, g3, g4, g5 = st.columns(5)
            g1.metric("Delta", str(round(delta, 4)))
            g2.metric("Gamma", str(round(gamma, 4)))
            g3.metric("Vega (+1% vol)", str(round(vega_val, 4)))
            g4.metric("Theta (por dia)", str(round(theta, 4)))
            g5.metric("Rho (+1% juros)", str(round(rho, 4)))

        st.divider()
        with st.expander("Perguntas para Discussao (do Case)"):
            st.markdown("1. **Por que Black-Scholes e mais adequado para opcoes europeias?** BS assume exercicio apenas no vencimento. Para americanas, e necessario avaliar o exercicio antecipado em cada passo.")
            st.markdown("2. **Por que opcoes americanas exigem analise de exercicio antecipado?** O detentor pode exercer antes do vencimento. Em cada no da arvore, compara-se V_continuidade com V_exercicio.")
            st.markdown("3. **Quando Monte Carlo e mais adequado?** Para opcoes exoticas cujo payoff depende de toda a trajetoria do preco, como as asiaticas.")
            st.markdown("4. **Por que opcoes asiaticas sao uteis em commodities?** Reduzem o impacto de movimentos extremos em uma unica data.")
            st.markdown("5. **O que significa uma volatilidade implicita elevada?** Maior incerteza precificada pelo mercado.")
            st.markdown("6. **Diferenca entre vol. historica e implicita?** Historica: dados passados. Implicita: extraida do preco da opcao no mercado.")
            st.markdown("7. **Quando Newton-Raphson pode falhar?** Quando o Vega e muito baixo ou o chute inicial esta longe da solucao.")
            st.markdown("8. **Por que a Bissecao e mais robusta?** Sempre converge desde que o intervalo contenha a solucao.")
            st.markdown("9. **Como o aumento da volatilidade afeta calls e puts?** Aumenta o premio de ambas.")
            st.markdown("10. **Como a calculadora poderia ser usada por uma mesa de trading?** Comparar vol. implicita com historica, montar estrategias de hedge, marcar posicoes a mercado.")
