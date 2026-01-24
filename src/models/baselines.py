"""Modelos baseline para comparação."""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

from ..config import THRESHOLD_BANDA_MORTA


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
    def predict(self, steps: int = 1) -> np.ndarray:
        """Faz previsões."""
        pass


class NaiveBaseline(BaseBaseline):
    """Baseline Naive: assume que próximo movimento = último movimento."""
    
    def __init__(self):
        super().__init__('Naive')
        self.last_direction = None
    
    def fit(self, train_data: pd.Series) -> None:
        if len(train_data) == 0:
            raise ValueError("[ERRO] Dados de treino vazios")
        
        last_value = train_data.iloc[-1]
        self.last_direction = 1 if last_value > 0 else (-1 if last_value < 0 else 0)
        self.fitted = True
    
    def predict(self, steps: int = 1) -> np.ndarray:
        if not self.fitted:
            raise ValueError("[ERRO] Modelo não foi treinado")
        return np.full(steps, self.last_direction)


class DriftBaseline(BaseBaseline):
    """Baseline Drift: assume tendência linear (drift)."""
    
    def __init__(self):
        super().__init__('Drift')
        self.drift = None
        self.last_value = None
    
    def fit(self, train_data: pd.Series) -> None:
        if len(train_data) == 0:
            raise ValueError("[ERRO] Dados de treino vazios")
        
        self.drift = train_data.mean()
        self.last_value = train_data.iloc[-1]
        self.fitted = True
    
    def predict(self, steps: int = 1) -> np.ndarray:
        if not self.fitted:
            raise ValueError("[ERRO] Modelo não foi treinado")
        
        predictions = []
        current_value = self.last_value
        
        for _ in range(steps):
            current_value += self.drift
            direction = 1 if current_value > 0 else (-1 if current_value < 0 else 0)
            predictions.append(direction)
        
        return np.array(predictions)


class ARIMABaseline(BaseBaseline):
    """Baseline ARIMA: modelo estatístico Box-Jenkins."""
    
    def __init__(self, max_p: int = 3, max_d: int = 2, max_q: int = 3):
        super().__init__('ARIMA')
        if not STATSMODELS_AVAILABLE:
            raise ImportError("[ERRO] statsmodels não está instalado")
        
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.model = None
        self.best_order = None
    
    def fit(self, train_data: pd.Series) -> None:
        if len(train_data) == 0:
            raise ValueError("[ERRO] Dados de treino vazios")
        
        best_aic = np.inf
        best_model = None
        best_order = None
        
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
            raise ValueError("[ERRO] Não foi possível ajustar ARIMA")
        
        self.model = best_model
        self.best_order = best_order
        self.fitted = True
    
    def predict(self, steps: int = 1) -> np.ndarray:
        if not self.fitted:
            raise ValueError("[ERRO] Modelo não foi treinado")
        
        forecast = self.model.forecast(steps=steps)
        # Usa apenas o sinal do forecast (sem banda morta)
        return np.where(forecast > 0, 1, np.where(forecast < 0, -1, 0))
