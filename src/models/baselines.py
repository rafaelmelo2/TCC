"""
Modelos baseline para comparação.

Conforme metodologia do TCC (Seção 4.3 - Modelos Baseline).
Implementa Naive, Drift e ARIMA como baselines de comparação.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from abc import ABC, abstractmethod

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from ..config import THRESHOLD_BANDA_MORTA
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import THRESHOLD_BANDA_MORTA


class BaseBaseline(ABC):
    """Classe base para todos os baselines."""
    
    def __init__(self, name: str):
        self.name = name
        self.fitted = False
    
    @abstractmethod
    def fit(self, train_data: pd.Series) -> None:
        """Treina o modelo."""
        pass
    
    @abstractmethod
    def predict(self, test_data: Optional[pd.Series] = None, steps: int = 1) -> np.ndarray:
        """Faz previsões."""
        pass


class NaiveBaseline(BaseBaseline):
    """
    Baseline Naive: assume que próximo movimento = último movimento.
    
    Conforme metodologia do TCC (Baseline 0).
    """
    
    def __init__(self):
        super().__init__('Naive')
        self.last_value = None
        self.last_direction = None
    
    def fit(self, train_data: pd.Series) -> None:
        """
        Treina o modelo (apenas armazena último valor).
        
        Parâmetros:
            train_data: Series com valores históricos (ex: retornos ou preços)
        """
        if len(train_data) == 0:
            raise ValueError("[ERRO] Dados de treino vazios")
        
        self.last_value = train_data.iloc[-1]
        self.last_direction = 1 if self.last_value > THRESHOLD_BANDA_MORTA else (
            -1 if self.last_value < -THRESHOLD_BANDA_MORTA else 0
        )
        self.fitted = True
    
    def predict(self, test_data: Optional[pd.Series] = None, steps: int = 1) -> np.ndarray:
        """
        Previsão: sempre retorna a direção do último movimento.
        
        Parâmetros:
            test_data: Não usado (mantido para compatibilidade)
            steps: Número de passos à frente
            
        Retorna:
            Array com previsões (mesma direção repetida)
        """
        if not self.fitted:
            raise ValueError("[ERRO] Modelo não foi treinado. Chame fit() primeiro.")
        
        return np.full(steps, self.last_direction)


class DriftBaseline(BaseBaseline):
    """
    Baseline Drift: assume tendência linear (drift).
    
    Calcula média dos retornos históricos e projeta linearmente.
    """
    
    def __init__(self):
        super().__init__('Drift')
        self.drift = None
        self.last_value = None
    
    def fit(self, train_data: pd.Series) -> None:
        """
        Treina o modelo (calcula drift médio).
        
        Parâmetros:
            train_data: Series com retornos históricos
        """
        if len(train_data) == 0:
            raise ValueError("[ERRO] Dados de treino vazios")
        
        self.drift = train_data.mean()
        self.last_value = train_data.iloc[-1]
        self.fitted = True
    
    def predict(self, test_data: Optional[pd.Series] = None, steps: int = 1) -> np.ndarray:
        """
        Previsão: projeta usando drift.
        
        Parâmetros:
            test_data: Não usado
            steps: Número de passos à frente
            
        Retorna:
            Array com previsões (direção baseada em drift)
        """
        if not self.fitted:
            raise ValueError("[ERRO] Modelo não foi treinado. Chame fit() primeiro.")
        
        predictions = []
        current_value = self.last_value
        
        for _ in range(steps):
            current_value += self.drift
            direction = 1 if current_value > THRESHOLD_BANDA_MORTA else (
                -1 if current_value < -THRESHOLD_BANDA_MORTA else 0
            )
            predictions.append(direction)
        
        return np.array(predictions)


class ARIMABaseline(BaseBaseline):
    """
    Baseline ARIMA: modelo estatístico Box-Jenkins.
    
    Conforme metodologia do TCC (Baseline 1).
    Otimiza ordem (p,d,q) por AIC.
    """
    
    def __init__(self, max_p: int = 3, max_d: int = 2, max_q: int = 3):
        super().__init__('ARIMA')
        if not STATSMODELS_AVAILABLE:
            raise ImportError("[ERRO] statsmodels não está instalado. Instale com: pip install statsmodels")
        
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.model = None
        self.best_order = None
        self.best_aic = None
    
    def fit(self, train_data: pd.Series) -> None:
        """
        Treina ARIMA com grid search para otimizar (p,d,q) por AIC.
        
        Parâmetros:
            train_data: Series com valores históricos
        """
        if len(train_data) == 0:
            raise ValueError("[ERRO] Dados de treino vazios")
        
        best_aic = np.inf
        best_model = None
        best_order = None
        
        # Grid search
        for p in range(self.max_p + 1):
            for d in range(self.max_d + 1):
                for q in range(self.max_q + 1):
                    try:
                        model = ARIMA(train_data, order=(p, d, q))
                        fitted = model.fit()
                        
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_model = fitted
                            best_order = (p, d, q)
                    except Exception:
                        continue
        
        if best_model is None:
            raise ValueError("[ERRO] Não foi possível ajustar nenhum modelo ARIMA")
        
        self.model = best_model
        self.best_order = best_order
        self.best_aic = best_aic
        self.fitted = True
    
    def predict(self, test_data: Optional[pd.Series] = None, steps: int = 1) -> np.ndarray:
        """
        Previsão usando modelo ARIMA ajustado.
        
        Parâmetros:
            test_data: Não usado (ARIMA usa apenas histórico interno)
            steps: Número de passos à frente
            
        Retorna:
            Array com previsões (direção baseada em forecast)
        """
        if not self.fitted:
            raise ValueError("[ERRO] Modelo não foi treinado. Chame fit() primeiro.")
        
        # Forecast
        forecast = self.model.forecast(steps=steps)
        
        # Converter para direção (assumindo que forecast é de retornos)
        # Se forecast > threshold: alta, < -threshold: baixa, else: neutro
        directions = np.where(
            forecast > THRESHOLD_BANDA_MORTA, 1,
            np.where(forecast < -THRESHOLD_BANDA_MORTA, -1, 0)
        )
        
        return directions
    
    def get_model_info(self) -> dict:
        """Retorna informações do modelo ajustado."""
        if not self.fitted:
            return {}
        
        return {
            'order': self.best_order,
            'aic': self.best_aic,
            'name': self.name
        }


def criar_baseline(tipo: str, **kwargs) -> BaseBaseline:
    """
    Factory function para criar baselines.
    
    Parâmetros:
        tipo: 'naive', 'drift' ou 'arima'
        **kwargs: Argumentos adicionais para o baseline
        
    Retorna:
        Instância do baseline solicitado
    """
    tipos = {
        'naive': NaiveBaseline,
        'drift': DriftBaseline,
        'arima': ARIMABaseline
    }
    
    tipo_lower = tipo.lower()
    if tipo_lower not in tipos:
        raise ValueError(f"[ERRO] Tipo de baseline inválido: {tipo}. Use: {list(tipos.keys())}")
    
    return tipos[tipo_lower](**kwargs)


if __name__ == '__main__':
    """
    Teste básico dos baselines.
    """
    import sys
    import os
    
    # Ajustar caminho
    if __file__:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, script_dir)
    
    from data_processing.load_data import carregar_dados
    from data_processing.feature_engineering import criar_features
    
    # Carregar e preparar dados
    arquivo = 'data/raw/VALE3_M15_20200101_20251231.csv'
    
    if os.path.exists(arquivo):
        print("=" * 70)
        print("TESTE DE BASELINES")
        print("=" * 70)
        
        # Carregar dados
        df = carregar_dados(arquivo, verbose=False)
        df_features = criar_features(df, verbose=False)
        
        # Dividir treino/teste simples (80/20)
        split_idx = int(len(df_features) * 0.8)
        train = df_features.iloc[:split_idx]
        test = df_features.iloc[split_idx:]
        
        print(f"\nTreino: {len(train)} barras")
        print(f"Teste: {len(test)} barras")
        
        # Testar Naive
        print("\n[1/3] Testando NaiveBaseline...")
        naive = NaiveBaseline()
        naive.fit(train['returns'])
        pred_naive = naive.predict(steps=len(test))
        print(f"[OK] Naive treinado. Previsões: {pred_naive[:10]}...")
        
        # Testar Drift
        print("\n[2/3] Testando DriftBaseline...")
        drift = DriftBaseline()
        drift.fit(train['returns'])
        pred_drift = drift.predict(steps=len(test))
        print(f"[OK] Drift treinado. Previsões: {pred_drift[:10]}...")
        
        # Testar ARIMA
        if STATSMODELS_AVAILABLE:
            print("\n[3/3] Testando ARIMABaseline...")
            arima = ARIMABaseline(max_p=2, max_d=1, max_q=2)  # Reduzido para teste rápido
            arima.fit(train['returns'].dropna())
            pred_arima = arima.predict(steps=min(100, len(test)))  # Limitar para não demorar
            print(f"[OK] ARIMA treinado. Ordem: {arima.best_order}, AIC: {arima.best_aic:.2f}")
            print(f"     Previsões: {pred_arima[:10]}...")
        else:
            print("\n[!] statsmodels não disponível. Pulando ARIMA.")
        
        print("\n" + "=" * 70)
        print("TESTE CONCLUÍDO")
        print("=" * 70)
    else:
        print(f"[ERRO] Arquivo não encontrado: {arquivo}")
