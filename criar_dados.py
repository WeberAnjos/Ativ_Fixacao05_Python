import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- 1. GERAÇÃO DO ARQUIVO: transacoes.csv ---
datas_transacoes = pd.date_range(start='2026-09-01', periods=1000, freq='h')

df_transacoes = pd.DataFrame({
    'id_cliente': np.random.choice(['C100', 'C101', 'C102', 'C103', 'C104'], size=1000),
    'data_transacao': datas_transacoes,
    'valor': np.random.choice(
        [np.nan, 150.0, 3000.0, 7500.0, 12000.0, 50.0], 
        size=1000, 
        p=[0.05, 0.40, 0.30, 0.15, 0.05, 0.05]
    ),
    'estado_cliente': np.random.choice(['SP', 'RJ', 'MG', 'RS'], size=1000),
    'origem': np.random.choice(['web', 'mobile_app', 'atm'], size=1000),
    'Plataforma': "Mobile"
})

df_transacoes = pd.concat([df_transacoes, df_transacoes.iloc[:15]], ignore_index=True)

df_transacoes.to_csv('transacoes.csv', index=False, encoding='latin1')


# 2. GERAÇÃO DO ARQUIVO: cotacoes.csv ---
datas_cotacoes = pd.date_range(start='2026-09-01', end='2026-10-01', freq='D')

variacoes = np.random.normal(loc=0.001, scale=0.015, size=len(datas_cotacoes))
preco_inicial = 5.20
precos = preco_inicial * np.exp(np.cumsum(variacoes))

df_cotacoes = pd.DataFrame({
    'data': datas_cotacoes,
    'cotacao_usd': np.round(precos, 4),
    'volume_negociado': np.random.randint(10000, 500000, size=len(datas_cotacoes))
})

df_cotacoes.loc[5, 'cotacao_usd'] = np.nan
df_cotacoes.loc[18, 'cotacao_usd'] = np.nan

df_cotacoes.to_csv('cotacoes.csv', index=False, encoding='utf-8')

print("Arquivos 'transacoes.csv' e 'cotacoes.csv' gerados com sucesso!")






print("=== ETAPA 1: Tratando dados nulos e a mediana por estado ===")

vendas_loja = pd.read_csv("transacoes.csv")

vendas_validas = vendas_loja.dropna(subset=["valor"])
print(vendas_validas)

mediana = vendas_validas["valor"].median()
print(mediana)

print("=== ETAPA 2: Dateime ===")

vendas_validas["data_transacao"] = pd.to_datetime(vendas_validas["data_transacao"])
vendas_validas['data_transacao'] = vendas_validas['data_transacao'].dt.tz_localize('America/Sao_Paulo')
print(vendas_validas)

print("=== ETAPA 3: semana, dia e mes ")

vendas_validas["dia_semana_pt"] = vendas_validas["data_transacao"].dt.day_name()
vendas_validas["mes_transacao_pt"] = vendas_validas["data_transacao"].dt.month_name()

print(vendas_validas)

print("=== ETAPA 3: Linhas duplicadas ")

vendas_sem_duplicacao = vendas_validas.drop_duplicates(keep="first")
print(vendas_sem_duplicacao)

print("=== ETAPA 4 === ")

print("""Filtre as transações do mês de setembro que atendam às seguintes condições simultâneas: \n
Transações realizadas no estado 'SP' ou 'RJ' E \n
Valor da transação maior que R$ 5.000,00.""")

filtro_setembro = vendas_validas["data_transacao"].dt.month == 9
filtro_estado = (vendas_validas["estado_cliente"] == "SP") | (
    vendas_validas["estado_cliente"] == "RJ"
)
filtro_valor = vendas_validas["valor"] > 5000.0

vendas_filtrado = vendas_validas[filtro_setembro & filtro_estado & filtro_valor]

print(vendas_filtrado)

print("=== ETAPA 5 === ")

risco_dict = {
    "C100": "Baixo",
    "C101": "Alto",
    "C102": "Médio",
    "C103": "Alto",
    "C104": "Baixo",
}

vendas_validas["nivel_risco"] = vendas_validas["id_cliente"].map(risco_dict)

vendas_validas["mes"] = vendas_validas["data_transacao"].dt.month_name()
vendas_validas["dia_semana"] = vendas_validas["data_transacao"].dt.day_name()

tabela_dinamica = pd.pivot_table(
    data=vendas_validas,
    values="valor",
    index="mes",
    columns="nivel_risco",
    aggfunc="sum",
    margins=True,
    margins_name="Total Geral",
)

print(tabela_dinamica)

print("=== ETAPA 6 ===")

def calcular_zscore_por_estado(df: pd.DataFrame, col_valor: str = 'valor', col_grupo: str = 'estado_cliente') -> pd.DataFrame:
    df_out = df.copy()
    
    media_grupo = df_out.groupby(col_grupo)[col_valor].transform('mean')
    std_grupo = df_out.groupby(col_grupo)[col_valor].transform('std')
    
    df_out['z_score'] = (df_out[col_valor] - media_grupo) / std_grupo
    
    return df_out

df_com_zscore = calcular_zscore_por_estado(vendas_validas)

df_anomalias = df_com_zscore[df_com_zscore['z_score'] > 2.5]

print(f"Total de anomalias encontradas: {len(df_anomalias)}")
print(df_anomalias[['id_cliente', 'estado_cliente', 'valor', 'z_score']].head())


print("=== ETAPA 7 ===")

vendas_validas['data_dia'] = vendas_validas['data_transacao'].dt.floor('D')
df_diario = (
    vendas_validas.groupby('data_dia')['valor']
    .sum()
    .reset_index()
    .sort_values('data_dia')
)

df_diario['media_movel_7d'] = df_diario['valor'].rolling(window=7, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    df_diario['data_dia'], 
    df_diario['valor'], 
    label='Valor Total Diário', 
    color='#5dade2', 
    linewidth=1.5, 
    marker='o', 
    markersize=3
)

ax.plot(
    df_diario['data_dia'], 
    df_diario['media_movel_7d'], 
    label='Média Móvel (7 dias)', 
    color='#1b4f72', 
    linewidth=2.5
)


ax.set_ylim(bottom=0)

ax.set_title('Evolução do Valor Total de Transações Diárias', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Data da Transação', fontsize=11)
ax.set_ylabel('Valor Total (R$)', fontsize=11)
ax.legend(loc='upper right', frameon=True)

ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()