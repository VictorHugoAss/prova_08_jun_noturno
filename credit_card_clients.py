import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)


# CONFIGURAÇÕES DO DATASET

COLUNA_ALVO = 'default payment next month'

# Colunas numéricas contínuas
COLUNAS_NUM = [
    'LIMIT_BAL', 'AGE',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
    'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1',  'PAY_AMT2',  'PAY_AMT3',
    'PAY_AMT4',  'PAY_AMT5',  'PAY_AMT6',
]

# Colunas categóricas
COLUNAS_CAT = ['SEX', 'EDUCATION', 'MARRIAGE']

# Normalizados junto com os outros numéricos
COLUNAS_PAY = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']


# CLASSE PRINCIPAL

class PipelineDefault:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.normalizador = MinMaxScaler()
        self.modelos: dict = {}
        self.resultados: dict = {}
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.colunas_dummy = []
        self.colunas_modelo = []

    # ETAPA 1 — Carregar dados
    
    def carregar(self, caminho: str = 'default_of_credit_card_clients.csv') -> pd.DataFrame:
        df = pd.read_csv(caminho, sep=';')
        df = df.drop(columns=['ID'])

        print(f"[Dados] Shape: {df.shape}")
        print(f"[Dados] Inadimplentes (1): {df[COLUNA_ALVO].sum()} "
              f"({100 * df[COLUNA_ALVO].mean():.1f}%)")
        print(f"[Dados] Adimplentes   (0): {(df[COLUNA_ALVO] == 0).sum()} "
              f"({100 * (1 - df[COLUNA_ALVO].mean()):.1f}%)")
        return df

    # ETAPA 2 — Pré-processamento

    def preprocessar(self, df: pd.DataFrame) -> pd.DataFrame:

        # Separar features e alvo
        X = df.drop(columns=[COLUNA_ALVO])
        y = df[COLUNA_ALVO]

        # One Hot nas categóricas
        dummies = pd.get_dummies(X[COLUNAS_CAT],
                                 prefix=COLUNAS_CAT,
                                 prefix_sep='_', dtype=int)
        self.colunas_dummy = list(dummies.columns)

        # Montando X completo com numéricos + PAY + dummies
        colunas_numericas = COLUNAS_NUM + COLUNAS_PAY
        X_proc = pd.concat([X[colunas_numericas], dummies], axis=1)
        self.colunas_modelo = list(X_proc.columns)

        print(f"\n[Preprocessamento] Features após encoding: {X_proc.shape[1]}")
        return X_proc, y

    def dividir_e_normalizar(self, X: pd.DataFrame, y: pd.Series) -> None:

        self.X_train_df, self.X_test_df, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y 
        )

        # Fit para o treino
        self.X_train = self.normalizador.fit_transform(self.X_train_df)
        self.X_test  = self.normalizador.transform(self.X_test_df)  # só transform!

        print(f"[Divisão] Treino: {len(self.X_train)} | Teste: {len(self.X_test)}")
        print(f"[Divisão] % inadimplentes treino: {100 * self.y_train.mean():.1f}%")

    # ETAPA 3 — Treinamento

    def treinar_modelos(self) -> None:

        configs = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                class_weight='balanced',  # compensa desbalanceamento
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=self.random_state
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,   # habilita predict_proba
                class_weight='balanced',
                random_state=self.random_state
            ),
        }

        for nome, modelo in configs.items():
            print(f"\n[Treino] Treinando {nome}...")
            modelo.fit(self.X_train, self.y_train)
            self.modelos[nome] = modelo
            print(f"[Treino] {nome} concluído.")



    def avaliar(self) -> pd.DataFrame:

        kfold = StratifiedKFold(n_splits=5, shuffle=True,
                                random_state=self.random_state)
        linhas = []

        for nome, modelo in self.modelos.items():
            y_pred  = modelo.predict(self.X_test)
            y_proba = modelo.predict_proba(self.X_test)[:, 1]  # prob. da classe 1

            # Cross-validation com AUC-ROC
            cv_auc = cross_val_score(modelo, self.X_train, self.y_train,
                                     cv=kfold, scoring='roc_auc', n_jobs=-1)

            linhas.append({
                'Modelo':       nome,
                'Acurácia':     round(accuracy_score(self.y_test, y_pred), 4),
                'F1 (default)': round(f1_score(self.y_test, y_pred, pos_label=1), 4),
                'AUC-ROC':      round(roc_auc_score(self.y_test, y_proba), 4),
                'Recall (def.)':round(f1_score(self.y_test, y_pred,
                                               average=None)[1], 4),
                'CV AUC Média': round(cv_auc.mean(), 4),
                'CV Desvio':    round(cv_auc.std(), 4),
            })
            self.resultados[nome] = (y_pred, y_proba)

        df_res = pd.DataFrame(linhas).set_index('Modelo')
        print("\n" + "=" * 70)
        print("COMPARAÇÃO DE MODELOS")
        print("=" * 70)
        print(df_res.to_string())
        return df_res

    def selecionar_melhor(self, df_avaliacao: pd.DataFrame) -> str:

        df = df_avaliacao.copy()
        df['Score_Final'] = df['CV AUC Média'] - 0.5 * df['CV Desvio']
        melhor = df['Score_Final'].idxmax()

        print(f"\n{'=' * 70}")
        print(f"modelo recomendado: {melhor}")
        print(f"{'=' * 70}")
        print(f"\nJustificativa:")
        print(f"AUC-ROC CV:{df.loc[melhor, 'CV AUC Média']:.4f}")
        print(f"Desvio CV:{df.loc[melhor, 'CV Desvio']:.4f}")
        print(f"Score Final:{df.loc[melhor, 'Score_Final']:.4f}")
        print(f"\nO AUC-ROC mede a capacidade do modelo de separar inadimplentes")
        print(f"de adimplentes, independente do threshold. É a métrica padrão")
        print(f"para modelos de crédito/risco em produção.")
        return melhor

    def matriz_confusao(self, nome_modelo: str) -> None:

        if nome_modelo not in self.resultados:
            raise ValueError(f"Rode avaliar() primeiro.")
        y_pred, _ = self.resultados[nome_modelo]
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\n[Matriz de Confusão — {nome_modelo}]")
        print(f"{'':25} Previsto 0  Previsto 1")
        print(f"{'Real 0 (Adimplente)':25} {cm[0,0]:10}  {cm[0,1]:10}")
        print(f"{'Real 1 (Inadimplente)':25} {cm[1,0]:10}  {cm[1,1]:10}")
        print(f"\nFalsos Negativos (FN): {cm[1,0]} inadimplentes não detectados")
        print(f"Falsos Positivos (FP): {cm[0,1]} adimplentes incorretamente bloqueados")

    # ETAPA 5 — Inferência com Score de Risco

    def inferir_cliente(self, dados_cliente: dict,
                         nome_modelo: str) -> dict:

        modelo = self.modelos.get(nome_modelo)
        if modelo is None:
            raise ValueError(f"Modelo '{nome_modelo}' não encontrado.")

        # Dados numéricos (numéricos + PAY)
        colunas_numericas = COLUNAS_NUM + COLUNAS_PAY
        df_num = pd.DataFrame(
            [[dados_cliente[c] for c in colunas_numericas]],
            columns=colunas_numericas
        )

        # One Hot nas categóricas + alinhamento com colunas do treino
        df_cat = pd.DataFrame(
            [[dados_cliente[c] for c in COLUNAS_CAT]],
            columns=COLUNAS_CAT
        )
        dummies = pd.get_dummies(df_cat, prefix=COLUNAS_CAT,
                                 prefix_sep='_', dtype=int)
        dummies = dummies.reindex(columns=self.colunas_dummy, fill_value=0)

        # instância completa e garantir mesma ordem das colunas
        df_completo = pd.concat([df_num, dummies], axis=1)
        df_completo = df_completo.reindex(columns=self.colunas_modelo, fill_value=0)

        # Normalizar com o scaler do treino
        df_norm = self.normalizador.transform(df_completo)

        # Predição e probabilidades
        classe = modelo.predict(df_norm)[0]

        probas = modelo.predict_proba(df_norm)[0]
        score_risco = round(float(probas[1]), 4)
        prob_adimplente = round(float(probas[0]), 4)

        # Categorizar o nível de risco
        if score_risco < 0.20:
            nivel = "BAIXO"
        elif score_risco < 0.40:
            nivel = "MÉDIO"
        elif score_risco < 0.65:
            nivel = "ALTO"
        else:
            nivel = "CRÍTICO"

        resultado = {
            'classificacao':'RISCO DE DEFAULT' if classe == 1 else 'PAGADOR REGULAR',
            'score_risco': score_risco,
            'prob_adimplente': prob_adimplente,
            'nivel_risco':nivel,
            'distribuicao_proba': {
                'Classe 0 (Adimplente)':prob_adimplente,
                'Classe 1 (Inadimplente)': score_risco
            }
        }
        return resultado

    # PERSISTÊNCIA

    def salvar(self, prefixo: str = 'credit_default'):
        for nome, modelo in self.modelos.items():
            fname = f"{prefixo}_{nome.replace(' ', '_').lower()}.pkl"
            pickle.dump(modelo, open(fname, 'wb'))
        pickle.dump(self.normalizador, open(f'{prefixo}_normalizador.pkl', 'wb'))
        pickle.dump({
            'colunas_dummy':  self.colunas_dummy,
            'colunas_modelo': self.colunas_modelo
        }, open(f'{prefixo}_estrutura.pkl', 'wb'))
        print(f"\n[Salvo] Modelos e estrutura salvos com prefixo '{prefixo}'")


# MAIN

if __name__ == '__main__':
    print("=" * 70)
    print("Sistema de detecção de inadimplência")
    print("=" * 70)

    pipeline = PipelineDefault(test_size=0.2, random_state=42)

    # 1. Carregar
    df = pipeline.carregar('dados/default_of_credit_card_clients.csv')

    # 2. Pré-processar
    X, y = pipeline.preprocessar(df)

    # 3. Dividir e normalizar
    pipeline.dividir_e_normalizar(X, y)

    # 4. Treinar todos os modelos
    pipeline.treinar_modelos()

    # 5. Avaliar e comparar
    df_aval = pipeline.avaliar()

    # 6. Mostrar matrizes de confusão
    for nome in pipeline.modelos:
        pipeline.matriz_confusao(nome)

    # 7. Selecionar melhor modelo
    melhor = pipeline.selecionar_melhor(df_aval)

    # 8. Relatório detalhado do melhor
    y_pred, _ = pipeline.resultados[melhor]
    print(f"\n[Relatório Completo — {melhor}]")
    print(classification_report(pipeline.y_test, y_pred,
                                  target_names=['Adimplente', 'Inadimplente']))

    # 9. Inferência
    print("\n" + "=" * 70)
    print("INFERÊNCIA — Análise de Novo Cliente")
    print("=" * 70)

    # Exemplo
    cliente_exemplo = {
        # Perfil financeiro
        'LIMIT_BAL': 50000,
        'AGE': 30,
        'SEX': 'M',
        'EDUCATION': 'High School',
        'MARRIAGE': 'Married',
        'PAY_0': 2,   
        'PAY_2': 2,    
        'PAY_3': 1,
        'PAY_4': 0,
        'PAY_5': 0,
        'PAY_6': 0,
        'BILL_AMT1': 45000,
        'BILL_AMT2': 44000,
        'BILL_AMT3': 43000,
        'BILL_AMT4': 30000,
        'BILL_AMT5': 28000,
        'BILL_AMT6': 25000,
        'PAY_AMT1': 0,
        'PAY_AMT2': 1000,
        'PAY_AMT3': 1000,
        'PAY_AMT4': 1000,
        'PAY_AMT5': 1000,
        'PAY_AMT6': 1000,
    }

    resultado = pipeline.inferir_cliente(cliente_exemplo, melhor)

    print(f"\nModelo usado: {melhor}")
    print(f"Classificação: {resultado['classificacao']}")
    print(f"Nível de risco: {resultado['nivel_risco']}")
    print(f"\nDistribuição de Probabilidade")
    for classe, prob in resultado['distribuicao_proba'].items():
        barra = '█' * int(prob * 40)
        print(f"  {classe:35}: {prob:.4f}  {barra}")
    print(f"\nScore de risco (inadimplência): {resultado['score_risco']:.4f}")
    print(f"{resultado['score_risco'] * 100:.1f}% de chance de dar default")

    # Salvar modelos
    pipeline.salvar()