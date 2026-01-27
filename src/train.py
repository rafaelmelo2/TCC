"""Script principal de treinamento de modelos de deep learning."""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

# Suprimir avisos do TensorFlow
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from sklearn.utils.class_weight import compute_class_weight

# Configurar GPU
def configurar_gpu(forcar_gpu: bool = True, memoria_crescimento: bool = True):
    """
    Configura TensorFlow para usar GPU.
    
    Parâmetros:
        forcar_gpu: Se True, força uso de GPU (erro se não disponível)
        memoria_crescimento: Se True, permite crescimento gradual de memória GPU
    
    Retorna:
        Lista de GPUs disponíveis
    """
    # Verificar se CUDA está disponível
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print("[INFO] nvidia-smi não encontrado. Verificando TensorFlow...")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("[INFO] nvidia-smi não encontrado. Verificando TensorFlow...")
    
    # Listar dispositivos físicos
    gpus = tf.config.list_physical_devices('GPU')
    
    if len(gpus) == 0:
        # Tentar verificar se há GPUs disponíveis de outra forma
        try:
            logical_gpus = tf.config.list_logical_devices('GPU')
            if len(logical_gpus) > 0:
                print(f"[INFO] {len(logical_gpus)} GPU(s) lógica(s) detectada(s)")
                for i, gpu in enumerate(logical_gpus):
                    print(f"     GPU {i}: {gpu.name}")
                return logical_gpus
        except:
            pass
        
        if forcar_gpu:
            print("[ERRO] GPU não encontrada pelo TensorFlow!")
            print("[INFO] Verificando instalação:")
            print(f"     TensorFlow version: {tf.__version__}")
            print(f"     CUDA built: {tf.test.is_built_with_cuda()}")
            print(f"     GPUs físicas: {tf.config.list_physical_devices('GPU')}")
            print("[INFO] Para usar CPU, execute com --no-gpu")
            raise RuntimeError("[ERRO] GPU não encontrada! Execute com --no-gpu para usar CPU.")
        else:
            print("[INFO] GPU não encontrada. Usando CPU.")
            return []
    
    print(f"[INFO] {len(gpus)} GPU(s) física(s) detectada(s):")
    for i, gpu in enumerate(gpus):
        print(f"     GPU {i}: {gpu.name}")
        try:
            # Obter detalhes da GPU
            gpu_details = tf.config.experimental.get_device_details(gpu)
            if gpu_details:
                print(f"          Detalhes: {gpu_details}")
        except:
            pass
        
        try:
            # Configurar crescimento de memória (evita alocar toda memória de uma vez)
            if memoria_crescimento:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"          Memória com crescimento habilitado")
            else:
                # Alocar toda memória disponível
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB por padrão
                )
        except RuntimeError as e:
            # Configuração deve ser feita antes de inicializar GPUs
            print(f"          Aviso: {e}")
    
    # Verificar dispositivos lógicos disponíveis
    logical_devices = tf.config.list_logical_devices()
    print(f"[INFO] Dispositivos disponíveis:")
    for device in logical_devices:
        print(f"     {device.name} ({device.device_type})")
    
    # Verificar se GPU será usada
    if len(gpus) > 0:
        print(f"[OK] TensorFlow configurado para usar GPU!")
        # Forçar uso de GPU
        try:
            with tf.device('/GPU:0'):
                # Teste simples para garantir que GPU funciona
                test_tensor = tf.constant([1.0, 2.0, 3.0])
                _ = test_tensor * 2
                print(f"[OK] Teste de GPU bem-sucedido!")
        except Exception as e:
            print(f"[AVISO] Erro ao testar GPU: {e}")
    
    return gpus

# Adicionar diretório pipeline ao path
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from src.data_processing.load_data import carregar_dados
from src.data_processing.feature_engineering import criar_features
from src.data_processing.prepare_sequences import preparar_dados_dl
from src.utils.validation import WalkForwardValidator
from src.utils.metrics import calcular_metricas_preditivas, calcular_acuracia_direcional
from src.utils.optuna_optimizer import otimizar_hiperparametros
from src.models.lstm_model import criar_modelo_lstm
from src.models.cnn_lstm_model import criar_modelo_cnn_lstm
from src.config import (
    TAMANHO_TREINO_BARRAS, TAMANHO_TESTE_BARRAS, EMBARGO_BARRAS,
    JANELA_TEMPORAL_STEPS, SEED
)

# Fixar seeds para reprodutibilidade
np.random.seed(SEED)
tf.random.set_seed(SEED)


def converter_target_para_binario(y: np.ndarray, remover_neutros: bool = True) -> tuple:
    """
    Converte target de (-1, 0, 1) para binário (0, 1).
    
    IMPORTANTE: Remove neutros do treinamento para evitar desbalanceamento.
    Neutros são removidos porque não representam movimento significativo
    e podem causar viés no modelo.
    
    Mapeamento:
    - 1 (alta) -> 1
    - -1 (baixa) -> 0
    - 0 (neutro) -> removido se remover_neutros=True
    
    Parâmetros:
        y: Array com targets (-1, 0, 1)
        remover_neutros: Se True, remove neutros (padrão: True)
    
    Retorna:
        Tupla (y_binary, mask) onde:
        - y_binary: Array com targets binários (0, 1)
        - mask: Array booleano indicando quais amostras foram mantidas
    """
    if remover_neutros:
        # Criar máscara: manter apenas alta (1) e baixa (-1), remover neutro (0)
        mask = y != 0
        y_filtered = y[mask]
    else:
        mask = np.ones(len(y), dtype=bool)
        y_filtered = y
    
    # Converter: 1 -> 1, -1 -> 0
    y_binary = np.where(y_filtered == 1, 1, 0)
    
    return y_binary, mask


def treinar_modelo_fold(model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                       epochs: int = 50, batch_size: int = 32,
                       verbose: int = 1, fold_num: Optional[int] = None,
                       ativo: Optional[str] = None, modelo_tipo: Optional[str] = None,
                       log_dir: Optional[Path] = None) -> keras.Model:
    """
    Treina modelo em um fold de walk-forward.
    
    IMPORTANTE: Remove neutros do treinamento para evitar desbalanceamento.
    
    Parâmetros:
        model: Modelo Keras
        X_train: Sequências de treino
        y_train: Targets de treino (-1, 0, 1)
        X_val: Sequências de validação (opcional)
        y_val: Targets de validação (opcional, -1, 0, 1)
        epochs: Número de épocas
        batch_size: Tamanho do batch
        verbose: Verbosidade (0=silencioso, 1=progresso)
    
    Retorna:
        Modelo treinado
    """
    # Converter targets para binário e remover neutros
    y_train_binary, mask_train = converter_target_para_binario(y_train, remover_neutros=True)
    X_train_filtered = X_train[mask_train]
    
    if verbose > 0:
        n_removidos = len(y_train) - len(y_train_binary)
        if n_removidos > 0:
            print(f"     Removidos {n_removidos} neutros do treino ({n_removidos/len(y_train)*100:.1f}%)")
    
    # Calcular class weights usando sklearn (mais robusto)
    # Isso previne que o modelo sempre preveja a mesma classe
    if len(np.unique(y_train_binary)) > 1:
        classes = np.unique(y_train_binary)
        weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=y_train_binary
        )
        class_weight = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
        if verbose > 0:
            print(f"     Class weights (sklearn balanced): {class_weight}")
    else:
        class_weight = None
        if verbose > 0:
            print(f"     [AVISO] Apenas uma classe presente no treino!")
    
    # Callbacks para treinamento otimizado (conforme TCC Seção 4.4)
    callbacks_list = [
        # Early stopping: para quando não há melhoria
        callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=10,
            restore_best_weights=True,
            verbose=1 if verbose > 0 else 0  # Mostrar quando para
        ),
        # Scheduler: reduz learning rate quando estagnado
        callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1 if verbose > 0 else 0  # Mostrar quando reduz LR
        )
    ]
    
    # Adicionar Cosine Annealing Scheduler (TCC Seção 4.4)
    # Melhora convergência e pode aumentar acurácia em 1-3%
    initial_lr = model.optimizer.learning_rate.numpy() if hasattr(model.optimizer.learning_rate, 'numpy') else float(model.optimizer.learning_rate)
    cosine_schedule = CosineDecayRestarts(
        initial_learning_rate=initial_lr,
        first_decay_steps=max(epochs // 2, 10),  # Primeira metade das épocas
        t_mul=2.0,  # Multiplicador de período
        m_mul=1.0,  # Multiplicador de learning rate mínimo
        alpha=1e-7  # Learning rate mínimo
    )
    callbacks_list.append(
        callbacks.LearningRateScheduler(
            lambda epoch: cosine_schedule(epoch).numpy(),
            verbose=0
        )
    )
    
    # Adicionar ModelCheckpoint e CSVLogger se fold_num especificado
    if fold_num is not None and ativo is not None and modelo_tipo is not None:
        models_dir = Path('models') / ativo / modelo_tipo
        models_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = models_dir / f'fold_{fold_num}_checkpoint.keras'
        callbacks_list.append(
            callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=1 if verbose > 0 else 0  # Mostrar quando salva
            )
        )
        
        # CSV Logger para salvar histórico de cada epoch
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            csv_log_path = log_dir / f'fold_{fold_num}_history.csv'
            callbacks_list.append(
                callbacks.CSVLogger(
                    str(csv_log_path),
                    separator=',',
                    append=False
                )
            )
            if verbose > 0:
                print(f"     [LOG] Salvando histórico em: {csv_log_path}")
    
    # Preparar validação (também removendo neutros)
    if X_val is not None:
        y_val_binary, mask_val = converter_target_para_binario(y_val, remover_neutros=True)
        X_val_filtered = X_val[mask_val]
        validation_data = (X_val_filtered, y_val_binary)
    else:
        validation_data = None
    
    model.fit(
        X_train_filtered, y_train_binary,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        class_weight=class_weight,  # Usar class weights para balancear
        verbose=verbose
    )
    
    return model


def prever_direcao(model: keras.Model, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Faz previsões de direção usando modelo treinado.
    
    Parâmetros:
        model: Modelo Keras treinado
        X: Sequências de entrada
        threshold: Limiar para classificação (0.5 = probabilidade)
    
    Retorna:
        Array com direções previstas (-1, 0, 1)
    """
    predictions = model.predict(X, verbose=0)
    predictions_binary = (predictions > threshold).astype(int).flatten()
    
    # Converter de binário (0, 1) para direção (-1, 1)
    # 1 -> 1 (alta), 0 -> -1 (baixa)
    directions = np.where(predictions_binary == 1, 1, -1)
    
    return directions


def main(ativo: str = "VALE3", modelo_tipo: str = "cnn_lstm",
         arquivo_dados: Optional[str] = None, epochs: int = 50,
         batch_size: int = 32, verbose: bool = True,
         usar_optuna: bool = False, n_trials_optuna: int = 20,
         usar_gpu: bool = True, folds_especificos: Optional[list] = None,
         usar_focal_loss: bool = False):
    """
    Função principal de treinamento.
    
    Parâmetros:
        ativo: Nome do ativo (ex: "VALE3")
        modelo_tipo: "lstm" ou "cnn_lstm"
        arquivo_dados: Caminho para arquivo de dados (opcional)
        epochs: Número de épocas de treinamento
        batch_size: Tamanho do batch (usado apenas se usar_optuna=False)
        verbose: Se True, imprime informações detalhadas
        usar_optuna: Se True, otimiza hiperparâmetros com Optuna em cada fold
        n_trials_optuna: Número de trials do Optuna (padrão: 20)
        usar_gpu: Se True, tenta usar GPU (padrão: True)
    """
    print("=" * 70)
    print("TREINAMENTO DE MODELO DE DEEP LEARNING")
    print("=" * 70)
    if usar_focal_loss:
        print("[INFO] 🔥 Usando FOCAL LOSS (gamma=5.0, alpha=0.5)")
        print("[INFO] Gamma alto (5.0) força modelo a focar em exemplos difíceis")
    
    # Configurar GPU se solicitado
    if usar_gpu:
        try:
            gpus = configurar_gpu(forcar_gpu=False, memoria_crescimento=True)  # Não forçar, apenas tentar
            if len(gpus) > 0:
                print(f"[OK] GPU configurada e pronta para uso!")
            else:
                print(f"[AVISO] GPU não detectada. Usando CPU.")
                print(f"[INFO] Para instalar TensorFlow com suporte GPU:")
                print(f"     1. Instale CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
                print(f"     2. Instale cuDNN: https://developer.nvidia.com/cudnn")
                print(f"     3. Reinstale TensorFlow: pip install tensorflow[and-cuda]")
        except Exception as e:
            print(f"[AVISO] Erro ao configurar GPU: {e}")
            print(f"[INFO] Continuando com CPU...")
            gpus = []
    else:
        print(f"[INFO] Usando CPU (GPU desabilitada)")
        gpus = []
    
    # Carregar dados
    if arquivo_dados is None:
        from src.config import obter_nome_arquivo_dados
        arquivo_dados = f'data/raw/{obter_nome_arquivo_dados(ativo)}'
    
    if not os.path.exists(arquivo_dados):
        print(f"[ERRO] Arquivo não encontrado: {arquivo_dados}")
        return
    
    print(f"\n[1/6] Carregando dados de {ativo}...")
    df = carregar_dados(arquivo_dados, verbose=False)
    print(f"[OK] Dados carregados: {df.shape}")
    
    # Criar features
    print(f"\n[2/6] Criando features...")
    df_features = criar_features(df, verbose=verbose)
    print(f"[OK] Features criadas: {df_features.shape}")
    
    # Criar validador walk-forward
    print(f"\n[3/6] Configurando walk-forward validation...")
    validator = WalkForwardValidator(
        train_size=TAMANHO_TREINO_BARRAS,
        test_size=TAMANHO_TESTE_BARRAS,
        embargo=EMBARGO_BARRAS
    )
    folds_originais = validator._gerar_folds(len(df_features))
    print(f"[OK] {len(folds_originais)} folds disponíveis")
    
    # Filtrar folds se especificado
    if folds_especificos is not None:
        folds_info = [folds_originais[i-1] for i in folds_especificos if 1 <= i <= len(folds_originais)]
        if not folds_info:
            print(f"[ERRO] Nenhum fold válido especificado. Folds disponíveis: 1-{len(folds_originais)}")
            return
        print(f"[INFO] Treinando apenas folds: {folds_especificos}")
    else:
        folds_info = folds_originais
    
    # Preparar dados para primeiro fold (para determinar dimensões)
    if len(folds_info) == 0:
        print("[ERRO] Não foi possível gerar folds")
        return
    
    first_fold = folds_info[0]
    X_train_sample, y_train_sample, _, _, _, feature_names = preparar_dados_dl(
        df_features,
        first_fold.train_start, first_fold.train_end,
        first_fold.test_start, first_fold.test_end,
        n_steps=JANELA_TEMPORAL_STEPS,
        verbose=False
    )
    
    n_steps, n_features = X_train_sample.shape[1], X_train_sample.shape[2]
    
    print(f"\n[4/6] Criando modelo {modelo_tipo.upper()}...")
    print(f"     Dimensões: n_steps={n_steps}, n_features={n_features}")
    
    # Criar modelo
    if modelo_tipo == "lstm":
        model_template = criar_modelo_lstm(n_steps, n_features)
    elif modelo_tipo == "cnn_lstm":
        model_template = criar_modelo_cnn_lstm(n_steps, n_features)
    else:
        raise ValueError(f"[ERRO] Tipo de modelo inválido: {modelo_tipo}")
    
    print(f"[OK] Modelo criado: {model_template.count_params()} parâmetros")
    
    # Treinar com walk-forward
    print(f"\n[5/6] Treinando com walk-forward validation...")
    results = []
    
    for i, fold in enumerate(folds_info):
        # Encontrar índice original do fold
        fold_original_idx = folds_originais.index(fold)
        fold_num = fold_original_idx + 1
        print(f"\n[Fold {fold_num}/{len(folds_originais)}] Treino:[{fold.train_start}:{fold.train_end}] "
              f"Teste:[{fold.test_start}:{fold.test_end}]")
        
        # Preparar dados do fold
        X_train, y_train, X_test, y_test, scaler, _ = preparar_dados_dl(
            df_features,
            fold.train_start, fold.train_end,
            fold.test_start, fold.test_end,
            n_steps=JANELA_TEMPORAL_STEPS,
            verbose=False
        )
        
        # Dividir treino em treino e validação interna (para Optuna)
        if usar_optuna:
            # Usar 80% para treino, 20% para validação interna
            split_idx = int(len(X_train) * 0.8)
            X_train_opt, X_val_opt = X_train[:split_idx], X_train[split_idx:]
            y_train_opt, y_val_opt = y_train[:split_idx], y_train[split_idx:]
            
            if verbose:
                print(f"     Dividindo treino: {len(X_train_opt)} treino, {len(X_val_opt)} validação interna")
            
            # Otimizar hiperparâmetros
            melhores_hiperparams, study, model = otimizar_hiperparametros(
                X_train_opt, y_train_opt,
                X_val_opt, y_val_opt,
                modelo_tipo=modelo_tipo,
                n_steps=n_steps,
                n_features=n_features,
                n_trials=n_trials_optuna,
                epochs=epochs,
                verbose=verbose,
                use_focal_loss=usar_focal_loss
            )
            
            # Criar modelo com hiperparâmetros otimizados
            if modelo_tipo == "lstm":
                model = criar_modelo_lstm(
                    n_steps=n_steps,
                    n_features=n_features,
                    lstm_units=melhores_hiperparams['lstm_units'],
                    dropout=melhores_hiperparams['dropout'],
                    learning_rate=melhores_hiperparams['learning_rate'],
                    use_focal_loss=usar_focal_loss
                )
                batch_size_otimizado = melhores_hiperparams['batch_size']
            else:  # cnn_lstm
                model = criar_modelo_cnn_lstm(
                    n_steps=n_steps,
                    n_features=n_features,
                    conv_filters=melhores_hiperparams['conv_filters'],
                    conv_kernel_size=melhores_hiperparams['conv_kernel_size'],
                    pool_size=2,  # Fixo
                    lstm_units=melhores_hiperparams['lstm_units'],
                    dropout=melhores_hiperparams['dropout'],
                    learning_rate=melhores_hiperparams['learning_rate'],
                    use_focal_loss=usar_focal_loss
                )
                batch_size_otimizado = melhores_hiperparams['batch_size']
            
            # Modelo já foi treinado durante otimização Optuna
            # Salvar modelo otimizado
            models_dir = Path('models') / ativo / modelo_tipo
            models_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = models_dir / f'fold_{fold_num}_checkpoint.keras'
            model.save(str(checkpoint_path))
            if verbose:
                print(f"     [OK] Modelo otimizado salvo: {checkpoint_path}")
        else:
            # Criar modelo com hiperparâmetros padrão
            if modelo_tipo == "lstm":
                model = criar_modelo_lstm(n_steps, n_features, use_focal_loss=usar_focal_loss)
            else:
                model = criar_modelo_cnn_lstm(n_steps, n_features, use_focal_loss=usar_focal_loss)
            
            # Treinar
            log_dir = Path('logs') / 'training_history' / ativo / modelo_tipo
            model = treinar_modelo_fold(
                model, X_train, y_train,
                epochs=epochs, batch_size=batch_size,
                verbose=1 if verbose else 0,
                fold_num=fold_num, ativo=ativo, modelo_tipo=modelo_tipo,
                log_dir=log_dir
            )
        
        # Prever
        y_pred = prever_direcao(model, X_test)
        
        # Calcular métricas
        metricas = calcular_metricas_preditivas(y_test, y_pred)
        metricas['accuracy_direcional'] = calcular_acuracia_direcional(y_test, y_pred)
        
        results.append({
            'fold': fold_num,
            'train_start': fold.train_start,
            'train_end': fold.train_end,
            'test_start': fold.test_start,
            'test_end': fold.test_end,
            'n_train': fold.n_train,
            'n_test': fold.n_test,
            'metricas': metricas
        })
        
        print(f"     Acurácia Direcional: {metricas['accuracy_direcional']:.4f}")
    
    # Agregar resultados
    print(f"\n[6/6] Consolidando resultados...")
    df_results = pd.DataFrame([{
        'fold': r['fold'],
        'accuracy_direcional': r['metricas']['accuracy_direcional'],
        'accuracy': r['metricas'].get('accuracy', np.nan),
        'f1_score': r['metricas'].get('f1_score', np.nan),
        'mcc': r['metricas'].get('mcc', np.nan),
    } for r in results])
    
    metricas_medias = df_results[['accuracy_direcional', 'accuracy', 'f1_score', 'mcc']].mean()
    
    print(f"[OK] Resultados consolidados!")
    print(f"\n{'='*70}")
    print("RESULTADOS FINAIS")
    print("="*70)
    print(f"Acurácia Direcional Média: {metricas_medias['accuracy_direcional']:.4f}")
    print(f"Acurácia Média: {metricas_medias['accuracy']:.4f}")
    print(f"F1-Score Médio: {metricas_medias['f1_score']:.4f}")
    print(f"MCC Médio: {metricas_medias['mcc']:.4f}")
    
    # Salvar resultados
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    arquivo_resultados = output_dir / f'{ativo}_{modelo_tipo}_walkforward.csv'
    df_results.to_csv(arquivo_resultados, index=False)
    print(f"\n[OK] Resultados salvos em: {arquivo_resultados}")
    
    print("\n" + "="*70)
    print("TREINAMENTO CONCLUÍDO")
    print("="*70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Treinar modelo de deep learning')
    parser.add_argument('--ativo', type=str, default='VALE3', help='Nome do ativo')
    parser.add_argument('--modelo', type=str, default='cnn_lstm', choices=['lstm', 'cnn_lstm'],
                       help='Tipo de modelo: lstm ou cnn_lstm')
    parser.add_argument('--arquivo', type=str, default=None, help='Caminho para arquivo de dados')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas (com early stopping)')
    parser.add_argument('--batch-size', type=int, default=32, help='Tamanho do batch (ignorado se --optuna)')
    parser.add_argument('--optuna', action='store_true', help='Usar Optuna para otimizar hiperparâmetros')
    parser.add_argument('--n-trials', type=int, default=20, help='Número de trials do Optuna')
    parser.add_argument('--gpu', action='store_true', default=True, help='Usar GPU (padrão: True)')
    parser.add_argument('--no-gpu', dest='gpu', action='store_false', help='Forçar uso de CPU')
    parser.add_argument('--folds', type=str, default=None, 
                       help='Folds específicos para treinar (ex: "4,5" ou "1-3")')
    parser.add_argument('--focal-loss', action='store_true', 
                       help='Usar Focal Loss ao invés de Binary Crossentropy')
    
    args = parser.parse_args()
    
    # Processar parâmetro --folds
    folds_especificos = None
    if args.folds:
        try:
            if '-' in args.folds:
                # Range: "1-3" -> [1, 2, 3]
                start, end = map(int, args.folds.split('-'))
                folds_especificos = list(range(start, end + 1))
            else:
                # Lista: "4,5" -> [4, 5]
                folds_especificos = [int(x.strip()) for x in args.folds.split(',')]
        except ValueError:
            print(f"[ERRO] Formato inválido para --folds: {args.folds}")
            print("[INFO] Use formato: --folds 4,5 ou --folds 1-3")
            sys.exit(1)
    
    main(
        ativo=args.ativo,
        modelo_tipo=args.modelo,
        arquivo_dados=args.arquivo,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True,
        usar_optuna=args.optuna,
        n_trials_optuna=args.n_trials,
        usar_gpu=args.gpu,
        folds_especificos=folds_especificos,
        usar_focal_loss=args.focal_loss
    )
