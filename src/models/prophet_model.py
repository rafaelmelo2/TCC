"""Modelo baseline Prophet para comparação."""

import numpy as np
import pandas as pd
from typing import Optional

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

from .baselines import BaseBaseline


class ProphetBaseline(BaseBaseline):
    """
    Baseline Prophet: modelo de decomposição aditiva com sazonalidades.
    
    Prophet é adequado para séries temporais com padrões sazonais e tendências.
    Para dados intradiários, configura sazonalidade diária e semanal.
    
    Conforme metodologia do trabalho (Seção 3.2 - Baselines para Comparação).
    """
    
    def __init__(self, 
                 daily_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 yearly_seasonality: bool = False,
                 seasonality_mode: str = 'additive',
                 changepoint_prior_scale: float = 0.05):
        """
        Inicializa o modelo Prophet.
        
        Parâmetros:
            daily_seasonality: Se True, inclui sazonalidade diária (útil para dados intradiários)
            weekly_seasonality: Se True, inclui sazonalidade semanal
            yearly_seasonality: Se False, desabilita sazonalidade anual (não relevante para intradiário)
            seasonality_mode: 'additive' ou 'multiplicative'
            changepoint_prior_scale: Controle de flexibilidade da tendência (menor = mais rígido)
        """
        super().__init__('Prophet')
        
        if not PROPHET_AVAILABLE:
            raise ImportError("[ERRO] Prophet não está instalado. Instale com: uv pip install prophet")
        
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        
        self.model: Optional[Prophet] = None
        self.last_timestamp: Optional[pd.Timestamp] = None
        self.freq: Optional[str] = None
    
    def fit(self, train_data: pd.Series) -> None:
        """
        Treina o modelo Prophet.
        
        Parâmetros:
            train_data: Série temporal com índice datetime (pd.Series com DatetimeIndex)
        
        Exceções:
            ValueError: Se dados estão vazios ou sem índice datetime
        """
        if len(train_data) == 0:
            raise ValueError("[ERRO] Dados de treino vazios")
        
        if not isinstance(train_data.index, pd.DatetimeIndex):
            raise ValueError("[ERRO] train_data deve ter índice DatetimeIndex")
        
        # Preparar dados no formato Prophet (ds, y)
        df_prophet = pd.DataFrame({
            'ds': train_data.index,
            'y': train_data.values
        })
        
        # Criar e configurar modelo
        self.model = Prophet(
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        
        # Treinar modelo
        try:
            self.model.fit(df_prophet)
        except Exception as e:
            raise ValueError(f"[ERRO] Falha ao treinar Prophet: {e}")
        
        # Guardar informações para previsão
        self.last_timestamp = train_data.index[-1]
        
        # Inferir frequência dos dados
        if len(train_data) > 1:
            freq = pd.infer_freq(train_data.index)
            if freq is None:
                # Se não conseguir inferir, usar intervalo médio
                intervals = train_data.index.to_series().diff().dropna()
                median_interval = intervals.median()
                if pd.Timedelta(minutes=10) <= median_interval <= pd.Timedelta(minutes=20):
                    self.freq = '15min'
                elif pd.Timedelta(minutes=4) <= median_interval <= pd.Timedelta(minutes=6):
                    self.freq = '5min'
                elif pd.Timedelta(minutes=28) <= median_interval <= pd.Timedelta(minutes=32):
                    self.freq = '30min'
                else:
                    self.freq = '15min'  # Default
            else:
                self.freq = freq
        else:
            self.freq = '15min'  # Default
        
        self.fitted = True
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Faz previsões para os próximos steps períodos.
        
        Parâmetros:
            steps: Número de períodos à frente para prever
        
        Retorna:
            Array numpy com direções previstas (1, -1, ou 0)
            - 1: movimento positivo esperado
            - -1: movimento negativo esperado
            - 0: movimento neutro esperado
        
        Exceções:
            ValueError: Se modelo não foi treinado
        """
        if not self.fitted or self.model is None:
            raise ValueError("[ERRO] Modelo não foi treinado")
        
        # Criar datas futuras
        if self.freq:
            future_dates = pd.date_range(
                start=self.last_timestamp + pd.Timedelta(self.freq),
                periods=steps,
                freq=self.freq
            )
        else:
            # Fallback: usar intervalo de 15 minutos
            future_dates = pd.date_range(
                start=self.last_timestamp + pd.Timedelta(minutes=15),
                periods=steps,
                freq='15min'
            )
        
        df_future = pd.DataFrame({'ds': future_dates})
        
        # Fazer previsão
        try:
            forecast = self.model.predict(df_future)
            predictions = forecast['yhat'].values
        except Exception as e:
            raise ValueError(f"[ERRO] Falha ao prever com Prophet: {e}")
        
        # Converter valores previstos em direções (1, -1, 0)
        # Prophet prevê valores, então comparamos com zero
        directions = np.where(predictions > 0, 1, np.where(predictions < 0, -1, 0))
        
        return directions
