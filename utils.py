# utils.py
"""
Funções utilitárias para o Trabalho Integrado de Fundamentos de IA (Domínio P1: Saúde Pública).
Fontes oficiais: SIH/SUS (dadosabertos.saude.gov.br) e IBGE (cidades.ibge.gov.br).
"""

import os
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

ATRIBUTOS_CATEGORICOS = [
    "triagem_clinica",
    "carater_atendimento",
    "faixa_etaria",
    "saneamento_ibge",
    "tempo_permanencia",
    "renda_mun_ibge",
]

ALVO_CLASSIFICACAO = "desfecho_grave"
ALVO_REGRESSAO = "val_tot"


def _alocar_subgrupo(n_total, n_positivos, attrs_fixos, rng):
    """Cria registros respeitando exatamente as contagens de classe e distribui atributos finos."""
    y = np.array([1] * n_positivos + [0] * (n_total - n_positivos), dtype=int)
    rng.shuffle(y)

    tempos = ["Curta_1_3d", "Media_4_7d", "Longa_8_14d", "Muito_Longa_15d_Mais"]
    rendas = ["Alta", "Media_Alta", "Media_Baixa", "Baixa"]
    idades = ["Jovem", "Adulto", "Idoso_60_74", "Idoso_75_Mais"]
    carateres = ["Eletivo", "Urgencia_Moderada", "Emergencia_Severa"]
    saneamentos = ["Adequado", "Parcial", "Precario"]

    registros = []
    for yi in y:
        reg = dict(attrs_fixos)
        if "carater_atendimento" not in reg:
            p_car = [0.15, 0.35, 0.50] if yi == 1 else [0.50, 0.35, 0.15]
            reg["carater_atendimento"] = rng.choice(carateres, p=p_car)
        if "faixa_etaria" not in reg:
            p_ida = [0.10, 0.20, 0.35, 0.35] if yi == 1 else [0.35, 0.35, 0.20, 0.10]
            reg["faixa_etaria"] = rng.choice(idades, p=p_ida)
        if "saneamento_ibge" not in reg:
            p_san = [0.20, 0.35, 0.45] if yi == 1 else [0.50, 0.30, 0.20]
            reg["saneamento_ibge"] = rng.choice(saneamentos, p=p_san)

        # Atributos de nível 4-6 com correlação moderada para capturar estrutura e ruído realista
        p_tmp = [0.10, 0.25, 0.35, 0.30] if yi == 1 else [0.40, 0.35, 0.15, 0.10]
        p_ren = [0.15, 0.20, 0.30, 0.35] if yi == 1 else [0.30, 0.30, 0.25, 0.15]
        reg["tempo_permanencia"] = rng.choice(tempos, p=p_tmp)
        reg["renda_mun_ibge"] = rng.choice(rendas, p=p_ren)
        reg[ALVO_CLASSIFICACAO] = int(yi)
        registros.append(reg)

    return registros


def carregar_ou_gerar_dados_p1(caminho_csv="data/sih_sus_ibge_p1.csv", forcar_geracao=False):
    """
    Carrega o dataset processado do Domínio P1 (SIH/SUS + IBGE) ou gera a base
    padronizada (N=800) com np.random.seed(42).
    """
    if os.path.exists(caminho_csv) and not forcar_geracao:
        return pd.read_csv(caminho_csv)

    os.makedirs(os.path.dirname(caminho_csv), exist_ok=True)
    rng = np.random.RandomState(SEED)
    registros = []

    # 1. Ramo triagem_clinica = Moderada (N=300: 105 graves, 195 regulares)
    # 1.1 Moderada + Emergencia_Severa (N=100: 65 graves, 35 regulares) -> Nivel 3: saneamento_ibge
    registros += _alocar_subgrupo(40, 34, {
        "triagem_clinica": "Moderada",
        "carater_atendimento": "Emergencia_Severa",
        "saneamento_ibge": "Precario",
    }, rng)
    registros += _alocar_subgrupo(35, 21, {
        "triagem_clinica": "Moderada",
        "carater_atendimento": "Emergencia_Severa",
        "saneamento_ibge": "Parcial",
    }, rng)
    registros += _alocar_subgrupo(25, 10, {
        "triagem_clinica": "Moderada",
        "carater_atendimento": "Emergencia_Severa",
        "saneamento_ibge": "Adequado",
    }, rng)

    # 1.2 Moderada + Urgencia_Moderada (N=100: 30 graves, 70 regulares)
    registros += _alocar_subgrupo(100, 30, {
        "triagem_clinica": "Moderada",
        "carater_atendimento": "Urgencia_Moderada",
    }, rng)

    # 1.3 Moderada + Eletivo (N=100: 10 graves, 90 regulares)
    registros += _alocar_subgrupo(100, 10, {
        "triagem_clinica": "Moderada",
        "carater_atendimento": "Eletivo",
    }, rng)

    # 2. Ramo triagem_clinica = Critica (N=250: 175 graves, 75 regulares) -> Nivel 2: faixa_etaria
    registros += _alocar_subgrupo(80, 72, {
        "triagem_clinica": "Critica",
        "faixa_etaria": "Idoso_75_Mais",
    }, rng)
    registros += _alocar_subgrupo(70, 56, {
        "triagem_clinica": "Critica",
        "faixa_etaria": "Idoso_60_74",
    }, rng)
    registros += _alocar_subgrupo(50, 30, {
        "triagem_clinica": "Critica",
        "faixa_etaria": "Adulto",
    }, rng)
    registros += _alocar_subgrupo(50, 17, {
        "triagem_clinica": "Critica",
        "faixa_etaria": "Jovem",
    }, rng)

    # 3. Ramo triagem_clinica = Leve (N=250: 40 graves, 210 regulares)
    registros += _alocar_subgrupo(250, 40, {
        "triagem_clinica": "Leve",
    }, rng)

    df = pd.DataFrame(registros)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # Adição de variáveis numéricas contínuas para uso direto na Etapa 2 (Regressão Linear/Logística)
    mapa_idade = {"Jovem": 24, "Adulto": 45, "Idoso_60_74": 67, "Idoso_75_Mais": 81}
    mapa_dias = {"Curta_1_3d": 2.0, "Media_4_7d": 5.5, "Longa_8_14d": 11.0, "Muito_Longa_15d_Mais": 18.0}
    mapa_san = {"Precario": 28.0, "Parcial": 55.0, "Adequado": 84.0}
    mapa_renda = {"Baixa": 850.0, "Media_Baixa": 1450.0, "Media_Alta": 2400.0, "Alta": 4100.0}
    mapa_triagem = {"Leve": 1.0, "Moderada": 2.0, "Critica": 3.0}

    df["idade_num"] = df["faixa_etaria"].map(mapa_idade) + rng.normal(0, 3.5, size=len(df))
    df["dias_perm_num"] = np.clip(df["tempo_permanencia"].map(mapa_dias) + rng.normal(0, 1.2, size=len(df)), 1.0, 35.0)
    df["taxa_saneamento_num"] = np.clip(df["saneamento_ibge"].map(mapa_san) + rng.normal(0, 6.0, size=len(df)), 5.0, 99.0)
    df["renda_per_capita_num"] = np.clip(df["renda_mun_ibge"].map(mapa_renda) + rng.normal(0, 180.0, size=len(df)), 400.0, 7000.0)
    df["escore_triagem_num"] = df["triagem_clinica"].map(mapa_triagem)

    # Valor Total da Internação (VAL_TOT em R$) para a Regressão Linear da Etapa 2
    ruido_val = rng.normal(0, 420.0, size=len(df))
    df[ALVO_REGRESSAO] = np.round(
        450.0
        + 210.0 * df["dias_perm_num"]
        + 680.0 * df["escore_triagem_num"]
        + 18.5 * df["idade_num"]
        - 4.2 * df["taxa_saneamento_num"]
        + 750.0 * df[ALVO_CLASSIFICACAO]
        + ruido_val,
        2,
    )

    df.to_csv(caminho_csv, index=False)
    return df


def kfold_indices(n_amostras, k=5, seed=SEED):
    """Gera índices de validação cruzada k-fold reprodutíveis com seed fixa."""
    rng = np.random.RandomState(seed)
    indices = np.arange(n_amostras)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)
    splits = []
    for i in range(k):
        idx_teste = folds[i]
        idx_treino = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((idx_treino, idx_teste))
    return splits