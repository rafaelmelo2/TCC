"""
Script rápido para verificar histórico disponível no MetaTrader 5
Barras de 15 minutos (M15)
Busca em lotes de 1 mês indo para trás no tempo
"""

import os
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# ==================== CONFIGURAÇÕES ====================
TICKERS = ["PETR4", "VALE3", "ITUB4"]
TIMEFRAME = mt5.TIMEFRAME_M15
DIAS_POR_LOTE = 30  # 1 mês por lote

# Caminho do MT5 (ajuste se necessário)
MT5_PATH = r"C:\Program Files\Clear Investimentos MT5 Terminal\terminal64.exe"
# MT5_PATH = None  # Use None para MT5 padrão
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
        print(f"✓ Conectado! Conta: {account.login} | Corretora: {account.company}\n")
    else:
        print("✓ Conectado ao MT5\n")
    
    return True


def verificar_historico(ticker, timeframe):
    """Verifica histórico disponível para um ticker buscando em lotes"""
    print(f"  Buscando símbolo para {ticker}...")
    
    # Tentar diferentes formatos de símbolo (mesma lógica do script original)
    simbolos_tentar = [ticker, f"{ticker}.SA", f"{ticker}_B3", f"{ticker}$"]
    simbolo = None
    
    for s in simbolos_tentar:
        info = mt5.symbol_info(s)
        if info and mt5.symbol_select(s, True):
            simbolo = s
            print(f"  ✓ Símbolo encontrado: {s}")
            break
    
    if not simbolo:
        print(f"  ✗ Símbolo não encontrado")
        return None
    
    # Estratégia: buscar em lotes de 1 mês, indo para trás no tempo
    print(f"  Buscando histórico em lotes de {DIAS_POR_LOTE} dias...")
    
    data_fim = datetime.now()
    data_inicio_teste = data_fim - timedelta(days=DIAS_POR_LOTE)
    
    todos_dados = []
    lote_num = 0
    limite_atingido = False
    
    # Primeiro, vamos para frente (mais recente) para encontrar onde começa
    while not limite_atingido and data_inicio_teste >= datetime(2000, 1, 1):
        lote_num += 1
        inicio_lote = data_inicio_teste
        fim_lote = data_fim
        
        rates = mt5.copy_rates_range(simbolo, timeframe, inicio_lote, fim_lote)
        
        if rates is not None and len(rates) > 0:
            todos_dados.extend(rates)
            print(f"    Lote {lote_num}: {inicio_lote.strftime('%Y-%m-%d')} até {fim_lote.strftime('%Y-%m-%d')} → {len(rates)} barras")
            # Avançar para o próximo lote (indo para trás)
            data_fim = inicio_lote
            data_inicio_teste = data_fim - timedelta(days=DIAS_POR_LOTE)
        else:
            # Se não encontrou dados, tenta um período maior antes de desistir
            if lote_num == 1:
                # Primeiro lote sem dados - pode ser que não tenha dados recentes
                print(f"    Lote {lote_num}: Sem dados no período recente, tentando mais atrás...")
                data_fim = data_inicio_teste
                data_inicio_teste = data_fim - timedelta(days=DIAS_POR_LOTE)
            else:
                # Já encontrou dados antes, então chegou ao limite
                limite_atingido = True
                print(f"    Limite do histórico atingido")
                break
    
    if not todos_dados:
        print(f"  ✗ Nenhum dado encontrado")
        return None
    
    # Converter para DataFrame
    df = pd.DataFrame(np.array(todos_dados).tolist(), 
                     columns=['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Remover duplicatas e ordenar
    df = df.drop_duplicates(subset=['time'], keep='first')
    df = df.sort_values('time').reset_index(drop=True)
    
    data_mais_antiga = df['time'].min()
    data_mais_recente = df['time'].max()
    total_barras = len(df)
    dias_total = (data_mais_recente - data_mais_antiga).days
    
    return {
        'simbolo': simbolo,
        'data_mais_antiga': data_mais_antiga,
        'data_mais_recente': data_mais_recente,
        'total_barras': total_barras,
        'dias_total': dias_total,
        'lotes_buscados': lote_num
    }


def main():
    """Função principal"""
    print("=" * 70)
    print("VERIFICAÇÃO DE HISTÓRICO - METATRADER 5")
    print("=" * 70)
    print(f"Timeframe: Barras de 15 minutos (M15)\n")
    
    # Conectar ao MT5
    if not inicializar_mt5():
        return
    
    # Verificar cada ticker
    resultados = {}
    
    for ticker in TICKERS:
        print(f"\n{'='*70}")
        print(f"Verificando {ticker}")
        print(f"{'='*70}")
        resultado = verificar_historico(ticker, TIMEFRAME)
        
        if resultado:
            resultados[ticker] = resultado
            print(f"\n  ✓ RESULTADO:")
            print(f"     Símbolo: {resultado['simbolo']}")
            print(f"     Data mais antiga: {resultado['data_mais_antiga'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"     Data mais recente: {resultado['data_mais_recente'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"     Total de barras: {resultado['total_barras']:,}")
            print(f"     Período total: {resultado['dias_total']} dias (~{resultado['dias_total']/365:.1f} anos)")
            print(f"     Lotes buscados: {resultado['lotes_buscados']}")
        else:
            print(f"\n  ✗ Não foi possível obter histórico para {ticker}")
    
    # Resumo geral
    if resultados:
        print("=" * 70)
        print("RESUMO GERAL")
        print("=" * 70)
        
        datas_antigas = [r['data_mais_antiga'] for r in resultados.values()]
        datas_recentes = [r['data_mais_recente'] for r in resultados.values()]
        total_barras_geral = sum(r['total_barras'] for r in resultados.values())
        
        data_mais_antiga_geral = min(datas_antigas)
        data_mais_recente_geral = max(datas_recentes)
        dias_geral = (data_mais_recente_geral - data_mais_antiga_geral).days
        
        print(f"Período disponível: {data_mais_antiga_geral.strftime('%Y-%m-%d')} até {data_mais_recente_geral.strftime('%Y-%m-%d')}")
        print(f"Total de dias: {dias_geral} dias (~{dias_geral/365:.1f} anos)")
        print(f"Total de barras (todos os ativos): {total_barras_geral:,}")
        print("=" * 70)
    
    # Finalizar
    mt5.shutdown()
    print("\nMT5 desconectado")


if __name__ == "__main__":
    main()
