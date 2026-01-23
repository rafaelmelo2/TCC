import os
import zipfile
import requests
import pandas as pd
from datetime import datetime

# ---------------- CONFIGURAÇÕES ---------------- #
TICKERS_ALVO = ["PETR4", "VALE3", "ITUB4"]        # Lista de tickers desejados
ANOS = range(2020, 2026)                 # Intervalo de anos desejado
DIRETORIO_SAIDA = "dados_b3"             # Pasta para salvar os arquivos ZIP e CSV
ARQUIVO_FINAL = "cotacoes_filtradas.csv" # Nome do CSV de saída
# ----------------------------------------------- #

BASE_URL = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist"

def fetch_cotahist_year(year: int, out_dir: str) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, f"COTAHIST_A{year}.ZIP")
    url = f"{BASE_URL}/COTAHIST_A{year}.ZIP"

    print(f"▼ Baixando {year}...")
    resp = requests.get(url, timeout=120, verify=False)
    resp.raise_for_status()
    with open(zip_path, "wb") as fp:
        fp.write(resp.content)

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(zf.namelist()[0]) as raw:
            lines = raw.read().decode("latin-1").splitlines()

    registros = []
    for ln in lines[1:-1]:
        if ln[0:2] != "01":
            continue
        registros.append({
            "data":        datetime.strptime(ln[2:10], "%Y%m%d"),    # Data da negociação (formato YYYYMMDD)
            "ticker":      ln[12:24].strip(),                        # Código do ativo (ex: PETR4, VALE3)
            "cod_bdi":     ln[10:12],                                # Código BDI - tipo de negociação (02=ON, 12=PN, etc.)
            "preco_ab":    int(ln[56:69])  / 100,                    # Preço de abertura (em reais)
            "preco_max":   int(ln[69:82])  / 100,                    # Preço máximo do dia (em reais)
            "preco_min":   int(ln[82:95])  / 100,                    # Preço mínimo do dia (em reais)
            "preco_med":   int(ln[95:108]) / 100,                    # Preço médio ponderado (em reais)
            "preco_fech":  int(ln[108:121])/ 100,                    # Preço de fechamento (em reais)
            "qtd_neg":     int(ln[147:152]),                          # Quantidade de negócios realizados
            "qtde_tit":    int(ln[152:170]),                         # Quantidade de títulos negociados
            "vol_fin":     int(ln[170:188]) / 100,                   # Volume financeiro (em reais)
        })

    return pd.DataFrame(registros)

# ---------------- EXECUÇÃO PRINCIPAL ---------------- #
if __name__ == "__main__":
    # Baixar e empilhar os dados de todos os anos
    df_total = pd.concat([fetch_cotahist_year(ano, DIRETORIO_SAIDA) for ano in ANOS])

    # Filtrar apenas os tickers desejados
    df_filtrado = df_total[df_total["ticker"].isin(TICKERS_ALVO)]

    # --- salvar com o mesmo nome do(s) ativo(s) ---
    if len(TICKERS_ALVO) == 1:
        ARQUIVO_FINAL = f"{TICKERS_ALVO[0]}.csv"
        caminho_saida = os.path.join(DIRETORIO_SAIDA, ARQUIVO_FINAL)
        df_filtrado.to_csv(caminho_saida, index=False)
        print(f"\n✔ Dados salvos em: {caminho_saida}")
    else:
        for tk in TICKERS_ALVO:
            caminho_saida = os.path.join(DIRETORIO_SAIDA, f"{tk}.csv")
            df_filtrado[df_filtrado["ticker"] == tk].to_csv(caminho_saida, index=False)
            print(f"✔ {tk}: {caminho_saida} salvo.")

    print(df_filtrado.head())
