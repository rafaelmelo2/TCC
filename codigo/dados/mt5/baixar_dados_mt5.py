"""
Script para baixar dados da B3 usando MetaTrader 5
Simples, organizado e funcional
"""

import os
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta

# ==================== CONFIGURAÇÕES ====================

# Ativos para baixar
TICKERS = ["PETR4", "VALE3", "ITUB4"]

# Período dos dados
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2025, 12, 31)
# END_DATE = datetime.now()

# Timeframe
TIMEFRAME = mt5.TIMEFRAME_M15
# Opções: M1, M5, M15, M30, H1, H4, D1, W1, MN1

# Lotes (divide o período em partes menores)
DIAS_POR_LOTE = 365  # 1 ano por lote

# Caminho do MT5 (Clear, XP, Rico, etc)
MT5_PATH = r"C:\Program Files\Clear Investimentos MT5 Terminal\terminal64.exe"
# MT5_PATH = None  # Use None para MT5 padrão

# Diretório de saída
DIRETORIO_SAIDA = "dados"

# ======================================================


def inicializar_mt5():
    """Conecta ao MT5"""
    print("Conectando ao MT5...")
    
    if MT5_PATH and os.path.exists(MT5_PATH):
        if not mt5.initialize(path=MT5_PATH):
            print(f"ERRO: Não foi possível inicializar MT5")
            return False
    else:
        if not mt5.initialize():
            print(f"ERRO: Não foi possível inicializar MT5")
            return False
    
    account = mt5.account_info()
    if account:
        print(f"OK! Conta: {account.login} | Corretora: {account.company}")
    else:
        print("OK! MT5 conectado")
    
    return True


def obter_nome_timeframe(tf):
    """Converte constante do timeframe para nome legível"""
    timeframes = {
        mt5.TIMEFRAME_M1: "M1",
        mt5.TIMEFRAME_M5: "M5",
        mt5.TIMEFRAME_M15: "M15",
        mt5.TIMEFRAME_M30: "M30",
        mt5.TIMEFRAME_H1: "H1",
        mt5.TIMEFRAME_H4: "H4",
        mt5.TIMEFRAME_D1: "D1",
        mt5.TIMEFRAME_W1: "W1",
        mt5.TIMEFRAME_MN1: "MN1"
    }
    return timeframes.get(tf, "DESCONHECIDO")


def criar_estrutura_pastas(base_dir):
    """Cria estrutura de pastas organizadas"""
    org_dir = os.path.join(base_dir, "organizados")
    os.makedirs(org_dir, exist_ok=True)
    
    for tf in ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]:
        os.makedirs(os.path.join(org_dir, tf), exist_ok=True)
    
    return org_dir


def dividir_em_lotes(start, end, dias):
    """Divide período em lotes menores"""
    lotes = []
    atual = start
    
    while atual < end:
        fim = min(atual + timedelta(days=dias), end)
        lotes.append((atual, fim))
        atual = fim + timedelta(days=1)
    
    return lotes


def baixar_ticker(ticker, timeframe, start_date, end_date):
    """
    Baixa dados de um ticker
    Retorna DataFrame ou None se erro
    """
    print(f"\n{'='*60}")
    print(f"BAIXANDO: {ticker}")
    print(f"{'='*60}")
    
    dias = (end_date - start_date).days
    print(f"Periodo: {start_date.strftime('%Y-%m-%d')} ate {end_date.strftime('%Y-%m-%d')} ({dias} dias)")
    
    # Dividir em lotes se necessário
    if dias > DIAS_POR_LOTE:
        lotes = dividir_em_lotes(start_date, end_date, DIAS_POR_LOTE)
        print(f"Dividindo em {len(lotes)} lotes")
    else:
        lotes = [(start_date, end_date)]
    
    # Verificar símbolo
    simbolos_tentar = [ticker, f"{ticker}.SA", f"{ticker}_B3", f"{ticker}$"]
    simbolo = None
    
    for s in simbolos_tentar:
        info = mt5.symbol_info(s)
        if info and mt5.symbol_select(s, True):
            simbolo = s
            print(f"Simbolo encontrado: {s}")
            break
    
    if not simbolo:
        print(f"ERRO: Simbolo {ticker} nao encontrado")
        return None
    
    # Baixar dados de cada lote
    todos_dados = []
    
    for i, (inicio, fim) in enumerate(lotes, 1):
        print(f"\nLote {i}/{len(lotes)}: {inicio.strftime('%Y-%m-%d')} ate {fim.strftime('%Y-%m-%d')}")
        
        rates = mt5.copy_rates_range(simbolo, timeframe, inicio, fim)
        
        if rates is not None and len(rates) > 0:
            print(f"  OK! {len(rates)} barras baixadas")
            todos_dados.extend(rates)
        else:
            print(f"  Nenhum dado neste periodo")
    
    if not todos_dados:
        print(f"ERRO: Nenhum dado foi baixado para {ticker}")
        return None
    
    # Converter para DataFrame
    # O MT5 retorna numpy structured array - converter cada tupla em linha
    import numpy as np
    
    # Converter numpy array para DataFrame corretamente
    if len(todos_dados) > 0:
        # Criar DataFrame diretamente dos nomes dos campos do structured array
        df = pd.DataFrame(np.array(todos_dados).tolist(), 
                         columns=['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'])
    else:
        print(f"ERRO: Lista vazia para {ticker}")
        return None
    
    # Converter timestamp para datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Renomear para português
    df = df.rename(columns={
        'time': 'data',
        'open': 'abertura',
        'high': 'maxima',
        'low': 'minima',
        'close': 'fechamento',
        'tick_volume': 'volume_ticks',
        'spread': 'spread',
        'real_volume': 'volume_real'
    })
    
    # Adicionar informações
    df.insert(0, 'ticker', ticker)
    df.insert(1, 'simbolo_mt5', simbolo)
    
    # Remover duplicatas e ordenar
    df = df.drop_duplicates(subset=['data'], keep='first')
    df = df.sort_values('data').reset_index(drop=True)
    
    print(f"\nSUCESSO!")
    print(f"  Total de registros: {len(df)}")
    print(f"  Periodo: {df['data'].min()} ate {df['data'].max()}")
    
    return df


def salvar_dados(df, ticker, timeframe, start_date, end_date, output_dir):
    """Salva os dados em arquivos CSV organizados"""
    tf_name = obter_nome_timeframe(timeframe)
    filename = f"{ticker}_{tf_name}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    
    # Salvar na pasta principal
    filepath_principal = os.path.join(output_dir, filename)
    df.to_csv(filepath_principal, index=False)
    print(f"  Salvo: {filepath_principal}")
    
    # Salvar na pasta organizada
    org_dir = os.path.join(output_dir, "organizados", tf_name)
    filepath_org = os.path.join(org_dir, filename)
    df.to_csv(filepath_org, index=False)
    print(f"  Organizado: {filepath_org}")


def criar_resumo(dados_dict, timeframe, start_date, end_date, output_dir):
    """Cria arquivo resumo com todos os tickers"""
    if not dados_dict:
        return
    
    print(f"\n{'='*60}")
    print("CRIANDO RESUMO GERAL")
    print(f"{'='*60}")
    
    # Juntar todos os DataFrames
    df_total = pd.concat(dados_dict.values(), ignore_index=True)
    df_total = df_total.sort_values(['ticker', 'data']).reset_index(drop=True)
    
    # Salvar
    tf_name = obter_nome_timeframe(timeframe)
    filename = f"todos_ativos_{tf_name}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    
    filepath_principal = os.path.join(output_dir, filename)
    df_total.to_csv(filepath_principal, index=False)
    print(f"Resumo salvo: {filepath_principal}")
    
    filepath_org = os.path.join(output_dir, "organizados", tf_name, filename)
    df_total.to_csv(filepath_org, index=False)
    print(f"Resumo organizado: {filepath_org}")
    
    print(f"Total: {len(df_total)} registros de {len(dados_dict)} ativos")


def main():
    """Função principal"""
    print("="*60)
    print("DOWNLOAD DE DADOS DA B3 - METATRADER 5")
    print("="*60)
    print(f"Periodo: {START_DATE.strftime('%Y-%m-%d')} ate {END_DATE.strftime('%Y-%m-%d')}")
    print(f"Timeframe: {obter_nome_timeframe(TIMEFRAME)}")
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Lotes de: {DIAS_POR_LOTE} dias")
    print("="*60)
    
    # Conectar ao MT5
    if not inicializar_mt5():
        return
    
    # Criar estrutura de pastas
    os.makedirs(DIRETORIO_SAIDA, exist_ok=True)
    criar_estrutura_pastas(DIRETORIO_SAIDA)
    
    # Baixar dados de cada ticker
    dados = {}
    sucessos = 0
    
    for ticker in TICKERS:
        df = baixar_ticker(ticker, TIMEFRAME, START_DATE, END_DATE)
        
        if df is not None:
            salvar_dados(df, ticker, TIMEFRAME, START_DATE, END_DATE, DIRETORIO_SAIDA)
            dados[ticker] = df
            sucessos += 1
    
    # Criar arquivo resumo
    if dados:
        criar_resumo(dados, TIMEFRAME, START_DATE, END_DATE, DIRETORIO_SAIDA)
    
    # Finalizar
    mt5.shutdown()
    print("\nMT5 desconectado")
    
    # Resultado final
    print("\n" + "="*60)
    print(f"CONCLUIDO: {sucessos}/{len(TICKERS)} ativos baixados com sucesso")
    print("="*60)
    
    if sucessos > 0:
        print(f"\nArquivos salvos em:")
        print(f"  {os.path.abspath(DIRETORIO_SAIDA)}")
        print(f"  {os.path.abspath(os.path.join(DIRETORIO_SAIDA, 'organizados'))}")
    else:
        print("\nNENHUM DADO FOI BAIXADO")


if __name__ == "__main__":
    main()
