"""
Módulo para auditoria completa de dados intradiários da B3.

Conforme metodologia do TCC (Seção 4.1 - Aquisição de Dados).
Realiza auditoria técnica completa: estrutura, gaps, missing values,
ajustes corporativos, timestamps e gera relatório detalhado.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import time, timedelta
from pathlib import Path

from .load_data import carregar_dados
from ..config import COLUNAS_OBRIGATORIAS, HORARIO_ABERTURA, HORARIO_FECHAMENTO, INTERVALO_BARRAS_MINUTOS

# Thresholds específicos para validação (usados apenas aqui)
THRESHOLD_ANOMALIA_PRECO = 0.15  # 15% - variação para considerar anomalia
THRESHOLD_SPLIT_DETECCAO = 0.3   # 30% - variação para considerar possível split
TOLERANCIA_GAP = 1.5  # 1.5x o intervalo esperado


def auditar_dados(
    caminho_arquivo: str,
    intervalo_esperado_minutos: int = INTERVALO_BARRAS_MINUTOS,
    threshold_split: float = THRESHOLD_SPLIT_DETECCAO,
    threshold_anomalia: float = THRESHOLD_ANOMALIA_PRECO,
    verbose: bool = True
) -> Dict:
    """
    Realiza auditoria técnica completa dos dados.
    
    Conforme metodologia do TCC (Seção 4.1). Valida estrutura, detecta gaps,
    missing values, possíveis ajustes corporativos e problemas temporais.
    
    Parâmetros:
        caminho_arquivo: Caminho para o arquivo CSV
        intervalo_esperado_minutos: Intervalo esperado entre barras (padrão: 15min)
        threshold_split: Threshold para detectar possíveis splits (variação de preço)
        threshold_anomalia: Threshold para detectar anomalias de preço
        verbose: Se True, exibe logs detalhados
        
    Retorna:
        Dicionário com resultados da auditoria:
        {
            'arquivo': str,
            'estrutura': dict,
            'periodo': dict,
            'gaps': dict,
            'missing_values': dict,
            'validacao_ohlc': dict,
            'validacao_pregão': dict,
            'anomalias_preco': dict,
            'possiveis_splits': list,
            'estatisticas': dict,
            'problemas_criticos': list,
            'warnings': list,
            'status_geral': str  # 'OK', 'WARNING', 'ERRO'
        }
    """
    if verbose:
        print("=" * 70)
        print("AUDITORIA TÉCNICA COMPLETA DE DADOS")
        print("=" * 70)
        print(f"\n[1/8] Iniciando auditoria: {os.path.basename(caminho_arquivo)}")
    
    resultado = {
        'arquivo': caminho_arquivo,
        'estrutura': {},
        'periodo': {},
        'gaps': {},
        'missing_values': {},
        'validacao_ohlc': {},
        'validacao_pregão': {},
        'anomalias_preco': {},
        'possiveis_splits': [],
        'estatisticas': {},
        'problemas_criticos': [],
        'warnings': [],
        'status_geral': 'OK'
    }
    
    # [1/8] Carregar dados
    try:
        df = carregar_dados(
            caminho_arquivo,
            validar_pregão=False,  # Vamos validar manualmente aqui
            remover_volume_zero=False,  # Queremos detectar
            remover_duplicatas=False,  # Queremos detectar
            verbose=False
        )
        if verbose:
            print(f"[OK] Dados carregados. Shape: {df.shape}")
    except Exception as e:
        resultado['problemas_criticos'].append(f"Erro ao carregar dados: {str(e)}")
        resultado['status_geral'] = 'ERRO'
        return resultado
    
    # [2/8] Auditoria de Estrutura
    if verbose:
        print("\n[2/8] Validando estrutura dos dados...")
    
    resultado['estrutura'] = {
        'shape': df.shape,
        'colunas': df.columns.tolist(),
        'colunas_obrigatorias_presentes': all(
            col in df.columns for col in COLUNAS_OBRIGATORIAS if col != 'data'
        ),
        'tipo_indice': str(type(df.index)),
        'é_datetimeindex': isinstance(df.index, pd.DatetimeIndex)
    }
    
    if not resultado['estrutura']['é_datetimeindex']:
        resultado['problemas_criticos'].append("Índice não é DatetimeIndex")
        resultado['status_geral'] = 'ERRO'
    
    # [3/8] Auditoria de Período
    if verbose:
        print("[3/8] Analisando período de cobertura...")
    
    periodo_inicio = df.index.min()
    periodo_fim = df.index.max()
    duracao = periodo_fim - periodo_inicio
    
    resultado['periodo'] = {
        'inicio': periodo_inicio,
        'fim': periodo_fim,
        'duracao_dias': duracao.days,
        'duracao_anos': duracao.days / 365.25,
        'total_barras': len(df),
        'barras_por_dia_medio': len(df) / max(duracao.days, 1)
    }
    
    # Esperado: ~26 barras por dia (10h-17h = 7h = 28 barras de 15min, menos algumas)
    barras_esperadas_por_dia = 28  # Aproximadamente
    if resultado['periodo']['barras_por_dia_medio'] < barras_esperadas_por_dia * 0.8:
        resultado['warnings'].append(
            f"Poucas barras por dia ({resultado['periodo']['barras_por_dia_medio']:.1f}). "
            f"Esperado: ~{barras_esperadas_por_dia}"
        )
        resultado['status_geral'] = 'WARNING'
    
    # [4/8] Detectar Gaps Temporais
    if verbose:
        print("[4/8] Detectando gaps temporais...")
    
    gaps = detectar_gaps_temporais(df, intervalo_esperado_minutos)
    resultado['gaps'] = {
        'total_gaps': len(gaps),
        'gaps_detalhes': gaps[:10],  # Primeiros 10 gaps
        'maior_gap_horas': max([g['duracao_horas'] for g in gaps]) if gaps else 0,
        'total_barras_faltantes_estimadas': sum([g['barras_faltantes'] for g in gaps])
    }
    
    if gaps:
        resultado['warnings'].append(
            f"Encontrados {len(gaps)} gaps temporais. "
            f"Maior gap: {resultado['gaps']['maior_gap_horas']:.1f} horas"
        )
        if resultado['status_geral'] == 'OK':
            resultado['status_geral'] = 'WARNING'
    
    # [5/8] Missing Values
    if verbose:
        print("[5/8] Verificando missing values...")
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    resultado['missing_values'] = {
        'total_missing': missing.sum(),
        'por_coluna': missing.to_dict(),
        'percentual_por_coluna': missing_pct.to_dict(),
        'colunas_com_missing': [col for col in missing.index if missing[col] > 0]
    }
    
    if missing.sum() > 0:
        resultado['warnings'].append(
            f"Encontrados {missing.sum()} missing values. "
            f"Colunas afetadas: {resultado['missing_values']['colunas_com_missing']}"
        )
        if resultado['status_geral'] == 'OK':
            resultado['status_geral'] = 'WARNING'
    
    # [6/8] Validação OHLC
    if verbose:
        print("[6/8] Validando lógica OHLC...")
    
    validacao_ohlc = validar_ohlc(df)
    resultado['validacao_ohlc'] = validacao_ohlc
    
    if validacao_ohlc['barras_invalidas'] > 0:
        resultado['problemas_criticos'].append(
            f"{validacao_ohlc['barras_invalidas']} barras com lógica OHLC inválida"
        )
        resultado['status_geral'] = 'ERRO'
    
    # [7/8] Validação Horário de Pregão
    if verbose:
        print("[7/8] Validando horário de pregão...")
    
    validacao_pregão = validar_horario_pregão(df)
    resultado['validacao_pregão'] = validacao_pregão
    
    if validacao_pregão['barras_fora_pregão'] > 0:
        resultado['warnings'].append(
            f"{validacao_pregão['barras_fora_pregão']} barras fora do horário de pregão"
        )
        if resultado['status_geral'] == 'OK':
            resultado['status_geral'] = 'WARNING'
    
    # [8/8] Detectar Anomalias e Possíveis Splits
    if verbose:
        print("[8/8] Detectando anomalias de preço e possíveis ajustes corporativos...")
    
    anomalias = detectar_anomalias_preco(df, threshold_anomalia)
    resultado['anomalias_preco'] = anomalias
    
    possiveis_splits = detectar_possiveis_splits(df, threshold_split)
    resultado['possiveis_splits'] = possiveis_splits
    
    if possiveis_splits:
        resultado['warnings'].append(
            f"Encontrados {len(possiveis_splits)} possíveis eventos de split/ajuste corporativo"
        )
    
    # Estatísticas Finais
    resultado['estatisticas'] = {
        'preco_medio': df['fechamento'].mean(),
        'preco_min': df['fechamento'].min(),
        'preco_max': df['fechamento'].max(),
        'volatilidade_media': df['fechamento'].pct_change().std(),
        'volume_medio': df['volume_real'].mean(),
        'volume_total': df['volume_real'].sum(),
        'barras_com_volume_zero': (df['volume_real'] == 0).sum(),
        'timestamps_duplicados': df.index.duplicated().sum()
    }
    
    if resultado['estatisticas']['timestamps_duplicados'] > 0:
        resultado['warnings'].append(
            f"{resultado['estatisticas']['timestamps_duplicados']} timestamps duplicados"
        )
    
    if verbose:
        print(f"\n[OK] Auditoria completa! Status: {resultado['status_geral']}")
    
    return resultado


def detectar_gaps_temporais(
    df: pd.DataFrame,
    intervalo_esperado_minutos: int = 15
) -> List[Dict]:
    """
    Detecta gaps temporais na série de dados.
    
    Parâmetros:
        df: DataFrame indexado por timestamp
        intervalo_esperado_minutos: Intervalo esperado entre barras
        
    Retorna:
        Lista de dicionários com informações sobre cada gap
    """
    gaps = []
    
    if len(df) < 2:
        return gaps
    
    # Calcular diferenças entre timestamps consecutivos
    diferencas = df.index.to_series().diff().dropna()
    intervalo_esperado = timedelta(minutes=intervalo_esperado_minutos)
    
    # Gaps são diferenças maiores que o intervalo esperado
    mask_gaps = diferencas > intervalo_esperado * TOLERANCIA_GAP
    
    for idx, is_gap in mask_gaps.items():
        if is_gap:
            gap_inicio = df.index[df.index.get_loc(idx) - 1]
            gap_fim = idx
            duracao = gap_fim - gap_inicio
            barras_faltantes = int(duracao.total_seconds() / 60 / intervalo_esperado_minutos)
            
            gaps.append({
                'inicio': gap_inicio,
                'fim': gap_fim,
                'duracao_horas': duracao.total_seconds() / 3600,
                'duracao_dias': duracao.days,
                'barras_faltantes': barras_faltantes
            })
    
    return gaps


def validar_ohlc(df: pd.DataFrame) -> Dict:
    """
    Valida lógica dos dados OHLC.
    
    Parâmetros:
        df: DataFrame com colunas OHLC
        
    Retorna:
        Dicionário com resultados da validação
    """
    problemas = []
    barras_invalidas = 0
    
    # Verificar se high >= low
    mask1 = df['maxima'] < df['minima']
    if mask1.any():
        n = mask1.sum()
        problemas.append(f"{n} barras com high < low")
        barras_invalidas += n
    
    # Verificar se high >= open
    mask2 = df['maxima'] < df['abertura']
    if mask2.any():
        n = mask2.sum()
        problemas.append(f"{n} barras com high < open")
        barras_invalidas += n
    
    # Verificar se high >= close
    mask3 = df['maxima'] < df['fechamento']
    if mask3.any():
        n = mask3.sum()
        problemas.append(f"{n} barras com high < close")
        barras_invalidas += n
    
    # Verificar se low <= open
    mask4 = df['minima'] > df['abertura']
    if mask4.any():
        n = mask4.sum()
        problemas.append(f"{n} barras com low > open")
        barras_invalidas += n
    
    # Verificar se low <= close
    mask5 = df['minima'] > df['fechamento']
    if mask5.any():
        n = mask5.sum()
        problemas.append(f"{n} barras com low > close")
        barras_invalidas += n
    
    # Verificar se preços são positivos
    mask6 = (df[['abertura', 'maxima', 'minima', 'fechamento']] <= 0).any(axis=1)
    if mask6.any():
        n = mask6.sum()
        problemas.append(f"{n} barras com preços <= 0")
        barras_invalidas += n
    
    return {
        'barras_invalidas': barras_invalidas,
        'problemas': problemas,
        'é_válido': barras_invalidas == 0
    }


def validar_horario_pregão(df: pd.DataFrame) -> Dict:
    """
    Valida se barras estão dentro do horário de pregão.
    
    Parâmetros:
        df: DataFrame indexado por timestamp
        
    Retorna:
        Dicionário com resultados da validação
    """
    hora_series = pd.Series(df.index.time, index=df.index)
    mask_pregão = (
        (hora_series >= HORARIO_ABERTURA) & 
        (hora_series <= HORARIO_FECHAMENTO)
    )
    
    barras_fora_pregão = (~mask_pregão).sum()
    percentual_fora = (barras_fora_pregão / len(df) * 100).round(2)
    
    return {
        'barras_fora_pregão': barras_fora_pregão,
        'percentual_fora_pregão': percentual_fora,
        'barras_dentro_pregão': mask_pregão.sum(),
        'percentual_dentro_pregão': 100 - percentual_fora
    }


def detectar_anomalias_preco(
    df: pd.DataFrame,
    threshold: float = 0.15
) -> Dict:
    """
    Detecta anomalias de preço (variações muito grandes entre barras).
    
    Parâmetros:
        df: DataFrame com coluna 'fechamento'
        threshold: Threshold de variação percentual para considerar anomalia
        
    Retorna:
        Dicionário com informações sobre anomalias
    """
    retornos = df['fechamento'].pct_change().abs()
    mask_anomalias = retornos > threshold
    
    anomalias = []
    for idx in df.index[mask_anomalias]:
        idx_pos = df.index.get_loc(idx)
        if idx_pos > 0:
            preco_anterior = df.iloc[idx_pos - 1]['fechamento']
            preco_atual = df.iloc[idx_pos]['fechamento']
            variacao = (preco_atual / preco_anterior - 1) * 100
            
            anomalias.append({
                'timestamp': idx,
                'preco_anterior': preco_anterior,
                'preco_atual': preco_atual,
                'variacao_pct': variacao
            })
    
    return {
        'total_anomalias': len(anomalias),
        'threshold_usado': threshold,
        'anomalias': anomalias[:20]  # Primeiras 20
    }


def detectar_possiveis_splits(
    df: pd.DataFrame,
    threshold: float = 0.3
) -> List[Dict]:
    """
    Detecta possíveis eventos de split ou ajuste corporativo.
    
    Um split geralmente causa variação grande e simétrica no preço
    (ex: split 2:1 causa queda de ~50%).
    
    Parâmetros:
        df: DataFrame com coluna 'fechamento'
        threshold: Threshold de variação para considerar possível split
        
    Retorna:
        Lista de possíveis eventos de split
    """
    retornos = df['fechamento'].pct_change()
    mask_grande_variacao = retornos.abs() > threshold
    
    possiveis_splits = []
    for idx in df.index[mask_grande_variacao]:
        idx_pos = df.index.get_loc(idx)
        if idx_pos > 0:
            preco_anterior = df.iloc[idx_pos - 1]['fechamento']
            preco_atual = df.iloc[idx_pos]['fechamento']
            variacao = (preco_atual / preco_anterior - 1) * 100
            
            # Splits geralmente causam quedas grandes e proporcionais
            # Ex: split 2:1 → queda de 50%, split 3:1 → queda de 66.7%
            if variacao < -threshold * 100:  # Queda grande
                possiveis_splits.append({
                    'timestamp': idx,
                    'preco_anterior': preco_anterior,
                    'preco_atual': preco_atual,
                    'variacao_pct': variacao,
                    'possivel_split_ratio': estimar_ratio_split(variacao)
                })
    
    return possiveis_splits


def estimar_ratio_split(variacao_pct: float) -> Optional[str]:
    """
    Estima o ratio de split baseado na variação de preço.
    
    Parâmetros:
        variacao_pct: Variação percentual (negativa para split)
        
    Retorna:
        String com ratio estimado (ex: "2:1", "3:1") ou None
    """
    # Variação negativa indica split
    if variacao_pct >= 0:
        return None
    
    # Calcular ratio aproximado
    # Split 2:1 → -50%, Split 3:1 → -66.7%, Split 4:1 → -75%
    ratio = 1 / (1 + variacao_pct / 100)
    
    # Arredondar para ratios comuns
    if 1.8 <= ratio <= 2.2:
        return "2:1"
    elif 2.8 <= ratio <= 3.2:
        return "3:1"
    elif 3.8 <= ratio <= 4.2:
        return "4:1"
    elif 4.8 <= ratio <= 5.2:
        return "5:1"
    else:
        return f"~{ratio:.2f}:1"


def gerar_relatorio_auditoria(resultado: Dict, salvar_arquivo: Optional[str] = None) -> str:
    """
    Gera relatório textual da auditoria.
    
    Parâmetros:
        resultado: Dicionário retornado por auditar_dados()
        salvar_arquivo: Se fornecido, salva relatório neste arquivo
        
    Retorna:
        String com relatório formatado
    """
    relatorio = []
    relatorio.append("=" * 70)
    relatorio.append("RELATÓRIO DE AUDITORIA TÉCNICA DE DADOS")
    relatorio.append("=" * 70)
    relatorio.append(f"\nArquivo: {os.path.basename(resultado['arquivo'])}")
    relatorio.append(f"Status Geral: {resultado['status_geral']}")
    relatorio.append("\n" + "-" * 70)
    
    # Estrutura
    relatorio.append("\n1. ESTRUTURA DOS DADOS")
    relatorio.append("-" * 70)
    relatorio.append(f"  Shape: {resultado['estrutura']['shape']}")
    relatorio.append(f"  Colunas: {len(resultado['estrutura']['colunas'])}")
    relatorio.append(f"  Tipo de índice: {resultado['estrutura']['tipo_indice']}")
    relatorio.append(f"  É DatetimeIndex: {resultado['estrutura']['é_datetimeindex']}")
    
    # Período
    relatorio.append("\n2. PERÍODO DE COBERTURA")
    relatorio.append("-" * 70)
    relatorio.append(f"  Início: {resultado['periodo']['inicio']}")
    relatorio.append(f"  Fim: {resultado['periodo']['fim']}")
    relatorio.append(f"  Duração: {resultado['periodo']['duracao_dias']} dias "
                     f"({resultado['periodo']['duracao_anos']:.2f} anos)")
    relatorio.append(f"  Total de barras: {resultado['periodo']['total_barras']:,}")
    relatorio.append(f"  Barras por dia (média): {resultado['periodo']['barras_por_dia_medio']:.1f}")
    
    # Gaps
    relatorio.append("\n3. GAPS TEMPORAIS")
    relatorio.append("-" * 70)
    relatorio.append(f"  Total de gaps: {resultado['gaps']['total_gaps']}")
    if resultado['gaps']['total_gaps'] > 0:
        relatorio.append(f"  Maior gap: {resultado['gaps']['maior_gap_horas']:.1f} horas")
        relatorio.append(f"  Barras faltantes estimadas: {resultado['gaps']['total_barras_faltantes_estimadas']}")
        if resultado['gaps']['gaps_detalhes']:
            relatorio.append("\n  Primeiros gaps:")
            for gap in resultado['gaps']['gaps_detalhes'][:5]:
                relatorio.append(
                    f"    {gap['inicio']} -> {gap['fim']} "
                    f"({gap['duracao_horas']:.1f}h, ~{gap['barras_faltantes']} barras)"
                )
    else:
        relatorio.append("  [OK] Nenhum gap detectado")
    
    # Missing Values
    relatorio.append("\n4. MISSING VALUES")
    relatorio.append("-" * 70)
    if resultado['missing_values']['total_missing'] > 0:
        relatorio.append(f"  Total de missing: {resultado['missing_values']['total_missing']}")
        for col, n in resultado['missing_values']['por_coluna'].items():
            if n > 0:
                pct = resultado['missing_values']['percentual_por_coluna'][col]
                relatorio.append(f"    {col}: {n} ({pct}%)")
    else:
        relatorio.append("  [OK] Nenhum missing value")
    
    # Validação OHLC
    relatorio.append("\n5. VALIDAÇÃO OHLC")
    relatorio.append("-" * 70)
    if resultado['validacao_ohlc']['barras_invalidas'] > 0:
        relatorio.append(f"  [ERRO] {resultado['validacao_ohlc']['barras_invalidas']} barras inválidas")
        for problema in resultado['validacao_ohlc']['problemas']:
            relatorio.append(f"    - {problema}")
    else:
        relatorio.append("  [OK] Todas as barras têm lógica OHLC válida")
    
    # Horário de Pregão
    relatorio.append("\n6. HORÁRIO DE PREGÃO")
    relatorio.append("-" * 70)
    relatorio.append(f"  Barras dentro do pregão: {resultado['validacao_pregão']['barras_dentro_pregão']:,} "
                     f"({resultado['validacao_pregão']['percentual_dentro_pregão']:.1f}%)")
    relatorio.append(f"  Barras fora do pregão: {resultado['validacao_pregão']['barras_fora_pregão']:,} "
                     f"({resultado['validacao_pregão']['percentual_fora_pregão']:.1f}%)")
    
    # Anomalias
    relatorio.append("\n7. ANOMALIAS DE PREÇO")
    relatorio.append("-" * 70)
    relatorio.append(f"  Total de anomalias detectadas: {resultado['anomalias_preco']['total_anomalias']}")
    if resultado['anomalias_preco']['total_anomalias'] > 0:
        relatorio.append(f"  Threshold usado: {resultado['anomalias_preco']['threshold_usado']*100:.1f}%")
    
    # Possíveis Splits
    relatorio.append("\n8. POSSÍVEIS AJUSTES CORPORATIVOS")
    relatorio.append("-" * 70)
    if resultado['possiveis_splits']:
        relatorio.append(f"  Encontrados {len(resultado['possiveis_splits'])} possíveis eventos:")
        for split in resultado['possiveis_splits'][:5]:
            relatorio.append(
                f"    {split['timestamp']}: "
                f"R$ {split['preco_anterior']:.2f} -> R$ {split['preco_atual']:.2f} "
                f"({split['variacao_pct']:.1f}%)"
            )
            if split['possivel_split_ratio']:
                relatorio.append(f"      Possível split: {split['possivel_split_ratio']}")
    else:
        relatorio.append("  Nenhum evento suspeito detectado")
    
    # Estatísticas
    relatorio.append("\n9. ESTATÍSTICAS")
    relatorio.append("-" * 70)
    stats = resultado['estatisticas']
    relatorio.append(f"  Preço médio: R$ {stats['preco_medio']:.2f}")
    relatorio.append(f"  Preço mínimo: R$ {stats['preco_min']:.2f}")
    relatorio.append(f"  Preço máximo: R$ {stats['preco_max']:.2f}")
    relatorio.append(f"  Volatilidade média: {stats['volatilidade_media']*100:.2f}%")
    relatorio.append(f"  Volume médio: {stats['volume_medio']:,.0f}")
    relatorio.append(f"  Volume total: {stats['volume_total']:,.0f}")
    relatorio.append(f"  Barras com volume zero: {stats['barras_com_volume_zero']}")
    relatorio.append(f"  Timestamps duplicados: {stats['timestamps_duplicados']}")
    
    # Problemas e Warnings
    if resultado['problemas_criticos']:
        relatorio.append("\n10. PROBLEMAS CRÍTICOS")
        relatorio.append("-" * 70)
        for problema in resultado['problemas_criticos']:
            relatorio.append(f"  [ERRO] {problema}")
    
    if resultado['warnings']:
        relatorio.append("\n11. AVISOS")
        relatorio.append("-" * 70)
        for warning in resultado['warnings']:
            relatorio.append(f"  [!] {warning}")
    
    relatorio.append("\n" + "=" * 70)
    relatorio.append("FIM DO RELATÓRIO")
    relatorio.append("=" * 70)
    
    relatorio_texto = "\n".join(relatorio)
    
    if salvar_arquivo:
        with open(salvar_arquivo, 'w', encoding='utf-8') as f:
            f.write(relatorio_texto)
        print(f"\n[OK] Relatório salvo em: {salvar_arquivo}")
    
    return relatorio_texto


if __name__ == '__main__':
    """
    Teste básico do módulo.
    """
    import sys
    
    # Ajustar caminho para execução direta
    if __file__:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, script_dir)
    
    # Testar com arquivos disponíveis
    ativos = ['VALE3', 'PETR4', 'ITUB4']
    
    if len(sys.argv) > 1:
        arquivo_teste = sys.argv[1]
    else:
        # Tentar caminho relativo ao diretório do projeto
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        from ..config import obter_nome_arquivo_dados
        arquivo_teste = os.path.join(base_dir, 'data', 'raw', obter_nome_arquivo_dados('VALE3'))
    
    if os.path.exists(arquivo_teste):
        print("\n" + "=" * 70)
        print("EXECUTANDO AUDITORIA COMPLETA")
        print("=" * 70 + "\n")
        
        resultado = auditar_dados(arquivo_teste, verbose=True)
        
        print("\n" + "=" * 70)
        print("GERANDO RELATÓRIO")
        print("=" * 70 + "\n")
        
        relatorio = gerar_relatorio_auditoria(resultado)
        print(relatorio)
        
        # Salvar relatório
        nome_arquivo = os.path.basename(arquivo_teste).replace('.csv', '_auditoria.txt')
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        caminho_relatorio = os.path.join(base_dir, 'data', 'processed', nome_arquivo)
        os.makedirs(os.path.dirname(caminho_relatorio), exist_ok=True)
        gerar_relatorio_auditoria(resultado, salvar_arquivo=caminho_relatorio)
    else:
        print(f"[ERRO] Arquivo não encontrado: {arquivo_teste}")
        print("\nUso: python validate_data.py <caminho_arquivo>")
        print("\nOu execute sem argumentos para testar com VALE3")
