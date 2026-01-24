"""Validação walk-forward com embargo temporal."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any, Callable

from ..config import TAMANHO_TREINO_BARRAS, TAMANHO_TESTE_BARRAS, EMBARGO_BARRAS


@dataclass
class FoldInfo:
    """Informações sobre um fold de walk-forward."""
    fold_num: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    embargo_start: int
    embargo_end: int
    n_train: int
    n_test: int


class WalkForwardValidator:
    """Validador walk-forward com embargo temporal."""
    
    def __init__(self, train_size: int = TAMANHO_TREINO_BARRAS, test_size: int = TAMANHO_TESTE_BARRAS,
                 embargo: int = EMBARGO_BARRAS, min_train_size: int = None):
        self.train_size = train_size
        self.test_size = test_size
        self.embargo = embargo
        self.min_train_size = min_train_size if min_train_size is not None else train_size
        self.folds: List[FoldInfo] = []
        self.results: List[Dict[str, Any]] = []
    
    def _gerar_folds(self, n_samples: int) -> List[FoldInfo]:
        """Gera divisões walk-forward dos dados."""
        folds = []
        current_pos = 0
        fold_num = 0
        
        while current_pos < n_samples:
            train_start = current_pos
            train_end = min(train_start + self.train_size, n_samples)
            
            if train_end - train_start < self.min_train_size:
                break
            
            embargo_start = train_end
            embargo_end = min(embargo_start + self.embargo, n_samples)
            test_start = embargo_end
            test_end = min(test_start + self.test_size, n_samples)
            
            if test_end <= test_start:
                break
            
            fold = FoldInfo(
                fold_num=fold_num,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                embargo_start=embargo_start,
                embargo_end=embargo_end,
                n_train=train_end - train_start,
                n_test=test_end - test_start
            )
            
            folds.append(fold)
            current_pos = test_end
            fold_num += 1
        
        return folds
    
    def validate(self, model: Any, X: pd.DataFrame, y: pd.Series,
                 fit_func: Callable, predict_func: Callable,
                 metric_func: Callable = None, verbose: bool = True) -> Dict[str, Any]:
        """Executa validação walk-forward completa."""
        if len(X) != len(y):
            raise ValueError("[ERRO] X e y devem ter mesmo tamanho")
        
        n_samples = len(X)
        self.folds = self._gerar_folds(n_samples)
        
        if len(self.folds) == 0:
            raise ValueError("[ERRO] Não foi possível gerar nenhum fold")
        
        if verbose:
            print(f"[OK] {len(self.folds)} folds | Treino: {self.train_size} | Teste: {self.test_size} | Embargo: {self.embargo}")
        
        self.results = []
        
        for i, fold in enumerate(self.folds):
            if verbose:
                print(f"\n[Fold {i+1}/{len(self.folds)}] Treino:[{fold.train_start}:{fold.train_end}] "
                      f"Teste:[{fold.test_start}:{fold.test_end}]")
            
            X_train = X.iloc[fold.train_start:fold.train_end]
            y_train = y.iloc[fold.train_start:fold.train_end]
            X_test = X.iloc[fold.test_start:fold.test_end]
            y_test = y.iloc[fold.test_start:fold.test_end]
            
            try:
                model_fold = model.__class__(**model.__dict__) if hasattr(model, '__dict__') else model
            except:
                model_fold = model
            
            try:
                fit_func(model_fold, X_train, y_train)
            except Exception as e:
                if verbose:
                    print(f"[ERRO] Erro ao treinar fold {i+1}: {e}")
                continue
            
            try:
                y_pred = predict_func(model_fold, X_test)
                
                if len(y_pred) != len(y_test):
                    if len(y_pred) > len(y_test):
                        y_pred = y_pred[:len(y_test)]
                    else:
                        y_pred = np.pad(y_pred, (0, len(y_test) - len(y_pred)), mode='edge')
            except Exception as e:
                if verbose:
                    print(f"[ERRO] Erro ao prever fold {i+1}: {e}")
                continue
            
            fold_result = {
                'fold': i + 1,
                'train_start': fold.train_start,
                'train_end': fold.train_end,
                'test_start': fold.test_start,
                'test_end': fold.test_end,
                'n_train': fold.n_train,
                'n_test': fold.n_test,
                'y_true': y_test.values,
                'y_pred': y_pred
            }
            
            if metric_func is not None:
                try:
                    fold_result['metrics'] = metric_func(y_test.values, y_pred)
                except Exception:
                    pass
            
            self.results.append(fold_result)
        
        return self._agregar_resultados()
    
    def _agregar_resultados(self) -> Dict[str, Any]:
        """Agrega resultados de todos os folds."""
        if len(self.results) == 0:
            return {
                'n_folds': 0,
                'total_train_samples': 0,
                'total_test_samples': 0,
                'aggregated_metrics': {},
                'per_fold_metrics': [],
                'all_y_true': np.array([]),
                'all_y_pred': np.array([]),
                'folds': []
            }
        
        all_y_true = np.concatenate([r['y_true'] for r in self.results])
        all_y_pred = np.concatenate([r['y_pred'] for r in self.results])
        
        aggregated_metrics = {}
        if self.results and 'metrics' in self.results[0]:
            metric_names = list(self.results[0]['metrics'].keys())
            for metric_name in metric_names:
                values = [r['metrics'][metric_name] for r in self.results if 'metrics' in r]
                if values:
                    aggregated_metrics[f'{metric_name}_mean'] = np.mean(values)
                    aggregated_metrics[f'{metric_name}_std'] = np.std(values)
        
        return {
            'n_folds': len(self.results),
            'total_train_samples': sum(r['n_train'] for r in self.results),
            'total_test_samples': sum(r['n_test'] for r in self.results),
            'aggregated_metrics': aggregated_metrics,
            'per_fold_metrics': [r.get('metrics', {}) for r in self.results],
            'all_y_true': all_y_true,
            'all_y_pred': all_y_pred,
            'folds': self.results
        }
