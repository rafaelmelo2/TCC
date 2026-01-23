"""
Script simples para baixar dados da B3 usando IbovFinancials
Baixa dados de candlestick de 15 minutos para PETR4, VALE3, ITUB4
A API tem limite de 90 dias por requisição
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

# Configurações
API_TOKEN = "ad65bc2dc74a9ce734523f7219bdc1f775d149c8"
API_URL = "https://www.ibovfinancials.com/api/ibov/historical"
TICKERS = ["PETR4", "VALE3", "ITUB4"]
TIMEFRAME = "15"  # 15 minutos

# IMPORTANTE: Ajuste o período aqui! A API só permite 90 dias por requisição
# Para baixar de 2020 até hoje, serão ~22 requisições por ticker (66 total)
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Delay entre requisições para não sobrecarregar a API
DELAY_SECONDS = 2

def criar_diretorios():
    """Cria os diretórios necessários"""
    Path("dados").mkdir(exist_ok=True)
    Path("dados/csv").mkdir(exist_ok=True)
    Path("dados/json").mkdir(exist_ok=True)

def baixar_periodo(ticker, start_date, end_date):
    """Baixa dados para um ticker em um período específico (max 90 dias)"""
    params = {
        "symbol": ticker,
        "timeframe": TIMEFRAME,
        "start_date": start_date,
        "end_date": end_date,
        "token": API_TOKEN
    }
    
    try:
        response = requests.get(API_URL, params=params, timeout=30)
        
        if response.status_code != 200:
            erro = response.text
            print(f"    ERRO {response.status_code}: {erro}")
            return []
        
        data = response.json()
        return data if isinstance(data, list) else []
        
    except Exception as e:
        print(f"    Erro na requisicao: {e}")
        return []

def dividir_periodos(start_date, end_date, dias_por_lote=90):
    """Divide o período total em lotes menores"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    periodos = []
    current = start
    
    while current < end:
        next_date = min(current + timedelta(days=dias_por_lote), end)
        periodos.append((
            current.strftime("%Y-%m-%d"),
            next_date.strftime("%Y-%m-%d")
        ))
        current = next_date + timedelta(days=1)
    
    return periodos

def baixar_dados_ticker(ticker):
    """Baixa todos os dados para um ticker"""
    print(f"\n{'='*60}")
    print(f"TICKER: {ticker}")
    print(f"{'='*60}")
    
    # Dividir em períodos de 90 dias
    periodos = dividir_periodos(START_DATE, END_DATE, 90)
    print(f"Total de periodos: {len(periodos)}")
    print(f"Total de requisicoes necessarias: {len(periodos)}")
    
    all_data = []
    
    for i, (inicio, fim) in enumerate(periodos, 1):
        print(f"\n[{i}/{len(periodos)}] {inicio} ate {fim}")
        
        dados = baixar_periodo(ticker, inicio, fim)
        
        if dados:
            all_data.extend(dados)
            print(f"    OK - {len(dados)} registros")
        else:
            print(f"    Nenhum dado neste periodo")
        
        # Delay entre requisições (exceto na última)
        if i < len(periodos):
            print(f"    Aguardando {DELAY_SECONDS}s...")
            time.sleep(DELAY_SECONDS)
    
    if not all_data:
        print(f"\nNENHUM DADO para {ticker}")
        return False
    
    # Remover duplicatas se houver (baseado em timestamp se existir)
    df = pd.DataFrame(all_data)
    
    # Salvar CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"dados/csv/{ticker}_{TIMEFRAME}min_{START_DATE}_{END_DATE}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV salvo: {csv_path}")
    print(f"Tamanho: {len(df)} registros")
    print(f"Colunas: {list(df.columns)}")
    
    # Salvar JSON
    json_path = f"dados/json/{ticker}_{TIMEFRAME}min_{START_DATE}_{END_DATE}_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"JSON salvo: {json_path}")
    
    return True

def main():
    """Função principal"""
    print("="*60)
    print("DOWNLOAD DE DADOS DA B3")
    print("="*60)
    print(f"Periodo: {START_DATE} ate {END_DATE}")
    print(f"Timeframe: {TIMEFRAME} minutos")
    print(f"Tickers: {', '.join(TICKERS)}")
    
    # Calcular total de requisições
    periodos = dividir_periodos(START_DATE, END_DATE, 90)
    total_requests = len(periodos) * len(TICKERS)
    print(f"\nTOTAL DE REQUISICOES: {total_requests}")
    print(f"Periodos por ticker: {len(periodos)}")
    
    resposta = input(f"\nDeseja continuar? (s/n): ")
    if resposta.lower() != 's':
        print("Download cancelado.")
        return
    
    # Criar diretórios
    criar_diretorios()
    
    # Baixar dados
    sucessos = 0
    for ticker in TICKERS:
        if baixar_dados_ticker(ticker):
            sucessos += 1
    
    print("\n" + "="*60)
    print(f"CONCLUIDO: {sucessos}/{len(TICKERS)} tickers processados")
    print("="*60)
    
    if sucessos > 0:
        print("\nDados salvos em:")
        print("  - dados/csv/")
        print("  - dados/json/")

if __name__ == "__main__":
    main()
