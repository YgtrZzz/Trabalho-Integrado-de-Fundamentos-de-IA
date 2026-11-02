# fuzzy.py
"""
Módulo de Lógica Fuzzy e Inferência Nebulosa Mamdani — Etapa 6 (Integração Final).
Domínio P1: Saúde Pública — Regulação de Leitos e Priorização de UTI (SIH/SUS + IBGE).

Implementa:
1. `executar_exemplo_listing4(valor_risco=0.75)`: reprodução literal do Listing 4 (Seção 9.3 do PDF);
2. `criar_sistema_mamdani_integrado()`: sistema Mamdani completo com 3 variáveis linguísticas
   triangulares (`risco`, `gravidade_em`, `urgencia`) e 5 regras Se-Então interpretáveis
   usando conectivos E (`&`) e OU (`|`);
3. `avaliar_pacientes_fuzzy_vs_hard(df)`: alimentação dos N=800 pacientes com as saídas
   probabilísticas das Etapas 5 (`prob_bayes_etapa5`) e 4 (`prob_em_etapa4`), comparando
   a decisão contínua suave Mamdani contra o threshold rígido booleano (0,5).
"""

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from utils import SEED

np.random.seed(SEED)


# ==============================================================================
# 1. REPRODUÇÃO LITERAL DO LISTING 4 (SEÇÃO 9.3 DO PDF)
# ==============================================================================
def executar_exemplo_listing4(valor_risco: float = 0.75) -> dict:
    """
    Reproduz literalmente o Listing 4 da Seção 9.3 do PDF usando `scikit-fuzzy`:
    - Antecedente `risco` em [0, 1] com termos 'baixo' [0, 0, 0.4] e 'alto' [0.6, 1, 1]
    - Consequente `urgencia` em [0, 100] com termos 'baixa' [0, 0, 50] e 'alta' [50, 100, 100]
    - Regras r1 (alto -> alta) e r2 (baixo -> baixa)
    - Defuzzificação por centroide para `risco = 0.75` (~75-79).
    """
    risco = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "risco")
    urgencia = ctrl.Consequent(np.arange(0, 101, 1), "urgencia")

    risco["baixo"] = fuzz.trimf(risco.universe, [0, 0, 0.4])
    risco["alto"] = fuzz.trimf(risco.universe, [0.6, 1, 1])
    urgencia["baixa"] = fuzz.trimf(urgencia.universe, [0, 0, 50])
    urgencia["alta"] = fuzz.trimf(urgencia.universe, [50, 100, 100])

    r1 = ctrl.Rule(risco["alto"], urgencia["alta"])
    r2 = ctrl.Rule(risco["baixo"], urgencia["baixa"])

    sistema = ctrl.ControlSystem([r1, r2])
    sim = ctrl.ControlSystemSimulation(sistema)
    sim.input["risco"] = float(valor_risco)
    sim.compute()
    urgencia_defuzz = float(sim.output["urgencia"])

    # Graus de pertinência no ponto avaliado
    pert_baixo = float(fuzz.interp_membership(risco.universe, risco["baixo"].mf, valor_risco))
    pert_alto = float(fuzz.interp_membership(risco.universe, risco["alto"].mf, valor_risco))

    return {
        "valor_risco": float(valor_risco),
        "pertinencia_baixo": pert_baixo,
        "pertinencia_alto": pert_alto,
        "urgencia_defuzzificada": urgencia_defuzz,
    }


# ==============================================================================
# 2. SISTEMA DE INFERÊNCIA FUZZY MAMDANI INTEGRADO (TAREFAS 9.2.1 A 9.2.3)
# ==============================================================================
def criar_sistema_mamdani_integrado():
    """
    Constrói o Sistema de Inferência Fuzzy Mamdani da Etapa 6 integrando:
    - Antecedente 1 (`risco`): probabilidade a posteriori da Rede Bayesiana (Etapa 5);
    - Antecedente 2 (`gravidade_em`): probabilidade do Naive Bayes + EM com variável oculta Z (Etapa 4);
    - Consequente (`urgencia`): escore contínuo [0, 100] de prioridade de regulação de leito/UTI no SUS.
    """
    risco = ctrl.Antecedent(np.arange(0.0, 1.01, 0.01), "risco")
    gravidade_em = ctrl.Antecedent(np.arange(0.0, 1.01, 0.01), "gravidade_em")
    urgencia = ctrl.Consequent(np.arange(0, 101, 1), "urgencia")

    # Funções de pertinência triangulares (fuzz.trimf) — Antecedente 1: risco (Etapa 5)
    risco["baixo"] = fuzz.trimf(risco.universe, [0.0, 0.0, 0.45])
    risco["moderado"] = fuzz.trimf(risco.universe, [0.20, 0.50, 0.80])
    risco["alto"] = fuzz.trimf(risco.universe, [0.55, 1.0, 1.0])

    # Funções de pertinência triangulares (fuzz.trimf) — Antecedente 2: gravidade_em (Etapa 4)
    gravidade_em["baixa"] = fuzz.trimf(gravidade_em.universe, [0.0, 0.0, 0.45])
    gravidade_em["intermediaria"] = fuzz.trimf(gravidade_em.universe, [0.20, 0.50, 0.80])
    gravidade_em["severa"] = fuzz.trimf(gravidade_em.universe, [0.55, 1.0, 1.0])

    # Funções de pertinência triangulares (fuzz.trimf) — Consequente: urgencia [0, 100]
    urgencia["baixa"] = fuzz.trimf(urgencia.universe, [0, 0, 45])
    urgencia["media"] = fuzz.trimf(urgencia.universe, [20, 50, 80])
    urgencia["alta"] = fuzz.trimf(urgencia.universe, [55, 100, 100])

    # Base de 5 Regras Se-Então com conectivos E (&) e OU (|)
    r1 = ctrl.Rule(
        risco["alto"] | gravidade_em["severa"],
        urgencia["alta"],
        label="R1_Alerta_Maximo_UTI",
    )
    r2 = ctrl.Rule(
        risco["baixo"] & gravidade_em["baixa"],
        urgencia["baixa"],
        label="R2_Estabilidade_Enfermaria",
    )
    r3 = ctrl.Rule(
        risco["moderado"] & gravidade_em["intermediaria"],
        urgencia["media"],
        label="R3_Monitoramento_Intermediario",
    )
    r4 = ctrl.Rule(
        (risco["moderado"] & gravidade_em["baixa"])
        | (risco["baixo"] & gravidade_em["intermediaria"]),
        urgencia["media"],
        label="R4_Discordancia_Moderada_Observacao",
    )
    r5 = ctrl.Rule(
        (risco["alto"] & gravidade_em["intermediaria"])
        | (risco["moderado"] & gravidade_em["severa"]),
        urgencia["alta"],
        label="R5_Sinergia_Agravamento_UTI",
    )

    regras_lista = [r1, r2, r3, r4, r5]
    sistema_ctrl = ctrl.ControlSystem(regras_lista)

    descricoes_regras = [
        {
            "Regra": "Regra 1 (OU)",
            "Expressão Lógica Fuzzy": "SE risco['alto'] OU gravidade_em['severa'] ENTÃO urgencia['alta']",
            "Operador": "OU (max)",
            "Interpretação Clínica (Regulação SUS)": (
                "Basta que a Rede Bayesiana causal (Etapa 5) OU o subperfil latente EM (Etapa 4) "
                "aponte risco elevado para acionar prioridade máxima de vaga em UTI."
            ),
        },
        {
            "Regra": "Regra 2 (E)",
            "Expressão Lógica Fuzzy": "SE risco['baixo'] E gravidade_em['baixa'] ENTÃO urgencia['baixa']",
            "Operador": "E (min)",
            "Interpretação Clínica (Regulação SUS)": (
                "Exige concordância simultânea de baixo risco causal e baixa gravidade latente "
                "para classificar o paciente em baixa urgência (leito clínico regular)."
            ),
        },
        {
            "Regra": "Regra 3 (E)",
            "Expressão Lógica Fuzzy": "SE risco['moderado'] E gravidade_em['intermediaria'] ENTÃO urgencia['media']",
            "Operador": "E (min)",
            "Interpretação Clínica (Regulação SUS)": (
                "Pacientes na zona intermediária em ambos os modelos probabilísticos recebem "
                "prioridade média (unidade semi-intensiva / observação monitorada)."
            ),
        },
        {
            "Regra": "Regra 4 (E/OU)",
            "Expressão Lógica Fuzzy": "SE (risco['moderado'] E gravidade_em['baixa']) OU (risco['baixo'] E gravidade_em['intermediaria']) ENTÃO urgencia['media']",
            "Operador": "E + OU",
            "Interpretação Clínica (Regulação SUS)": (
                "Quando um modelo indica alerta moderado na fronteira e o outro indica risco baixo, "
                "evita-se a alta precoce mantendo o paciente em observação intermediária."
            ),
        },
        {
            "Regra": "Regra 5 (E/OU)",
            "Expressão Lógica Fuzzy": "SE (risco['alto'] E gravidade_em['intermediaria']) OU (risco['moderado'] E gravidade_em['severa']) ENTÃO urgencia['alta']",
            "Operador": "E + OU",
            "Interpretação Clínica (Regulação SUS)": (
                "Reforça a escalada para alta urgência quando há combinação de risco alto com "
                "gravidade intermediária ou risco moderado com subperfil severo."
            ),
        },
    ]

    variaveis = {
        "risco": risco,
        "gravidade_em": gravidade_em,
        "urgencia": urgencia,
    }
    return sistema_ctrl, variaveis, pd.DataFrame(descricoes_regras)


def computar_urgencia_mamdani(
    sistema_ctrl: ctrl.ControlSystem,
    val_risco: float,
    val_gravidade_em: float,
) -> float:
    """Executa uma simulação Mamdani para um par de entradas (risco, gravidade_em)."""
    sim = ctrl.ControlSystemSimulation(sistema_ctrl)
    sim.input["risco"] = float(np.clip(val_risco, 0.0, 1.0))
    sim.input["gravidade_em"] = float(np.clip(val_gravidade_em, 0.0, 1.0))
    sim.compute()
    return float(sim.output["urgencia"])


def classificar_faixa_regulacao(urgencia_val: float) -> str:
    """Mapeia o escore contínuo de urgência [0, 100] em 3 níveis clínicos interpretáveis."""
    if urgencia_val < 40.0:
        return "Baixa Prioridade / Enfermaria"
    if urgencia_val < 62.0:
        return "Média Prioridade / Observação Intensiva"
    return "Alta Prioridade / Vaga UTI Imediata"


# ==============================================================================
# 3. AVALIAÇÃO DOS PACIENTES: FUZZY MAMDANI vs. HARD THRESHOLD (0,5)
# ==============================================================================
def avaliar_pacientes_fuzzy_vs_hard(df: pd.DataFrame, seed: int = SEED) -> dict:
    """
    Alimenta o sistema Fuzzy Mamdani com `prob_bayes_etapa5` e `prob_em_etapa4`
    para todos os N=800 pacientes e compara quantitativamente a decisão suave
    contra o threshold rígido (0.5).
    """
    sistema_ctrl, _, _ = criar_sistema_mamdani_integrado()

    riscos_etapa5 = df["prob_bayes_etapa5"].values.astype(float)
    gravs_etapa4 = df["prob_em_etapa4"].values.astype(float)
    y_real = df["desfecho_grave"].values.astype(int)

    urgencias_fuzzy = np.zeros(len(df), dtype=float)
    faixas_clinicas = []
    for i in range(len(df)):
        u = computar_urgencia_mamdani(sistema_ctrl, riscos_etapa5[i], gravs_etapa4[i])
        urgencias_fuzzy[i] = u
        faixas_clinicas.append(classificar_faixa_regulacao(u))

    # Decisão Hard Threshold (0.5) sobre a probabilidade Bayesiana da Etapa 5 -> {0, 100}
    decisao_hard_bin = (riscos_etapa5 >= 0.50).astype(int)
    decisao_hard_100 = 100.0 * decisao_hard_bin

    # Decisão Fuzzy calibrada para triagem segura (urgencia >= 50.0)
    decisao_fuzzy_bin = (urgencias_fuzzy >= 50.0).astype(int)

    # Métricas globais (N = 800)
    acc_hard_global = 100.0 * float(np.mean(decisao_hard_bin == y_real))
    acc_fuzzy_global = 100.0 * float(np.mean(decisao_fuzzy_bin == y_real))

    # Erro quadrático médio de severidade (normalizando urgência para [0, 1] vs. y_real)
    brier_hard_global = float(np.mean(((decisao_hard_100 / 100.0) - y_real) ** 2))
    brier_fuzzy_global = float(np.mean(((urgencias_fuzzy / 100.0) - y_real) ** 2))

    # Região de fronteira ambígua (0.35 <= prob_bayes_etapa5 <= 0.65)
    mascara_fronteira = (riscos_etapa5 >= 0.35) & (riscos_etapa5 <= 0.65)
    n_fronteira = int(np.sum(mascara_fronteira))
    y_front = y_real[mascara_fronteira]
    hard_front = decisao_hard_bin[mascara_fronteira]
    fuzzy_front = decisao_fuzzy_bin[mascara_fronteira]
    urg_front = urgencias_fuzzy[mascara_fronteira]

    acc_hard_front = 100.0 * float(np.mean(hard_front == y_front))
    acc_fuzzy_front = 100.0 * float(np.mean(fuzzy_front == y_front))

    # Sensibilidade (Recall para casos graves D=1) na zona de fronteira
    graves_front = y_front == 1
    sens_hard_front = 100.0 * float(np.mean(hard_front[graves_front] == 1)) if np.any(graves_front) else 0.0
    sens_fuzzy_front = 100.0 * float(np.mean(fuzzy_front[graves_front] == 1)) if np.any(graves_front) else 0.0

    brier_hard_front = float(np.mean((hard_front.astype(float) - y_front) ** 2))
    brier_fuzzy_front = float(np.mean(((urg_front / 100.0) - y_front) ** 2))

    # Estabilidade sob ruído de aferição clínica (sigma = 0.05) na zona de fronteira
    rng = np.random.RandomState(seed)
    ruido_r = rng.normal(0.0, 0.05, size=n_fronteira)
    ruido_g = rng.normal(0.0, 0.05, size=n_fronteira)
    riscos_pert = np.clip(riscos_etapa5[mascara_fronteira] + ruido_r, 0.0, 1.0)
    gravs_pert = np.clip(gravs_etapa4[mascara_fronteira] + ruido_g, 0.0, 1.0)

    hard_pert_100 = 100.0 * (riscos_pert >= 0.50).astype(float)
    urg_pert = np.array(
        [
            computar_urgencia_mamdani(sistema_ctrl, riscos_pert[j], gravs_pert[j])
            for j in range(n_fronteira)
        ],
        dtype=float,
    )

    # Salto médio absoluto na escala [0, 100] provocado por pequeno ruído de 5%
    salto_medio_hard = float(np.mean(np.abs(hard_pert_100 - decisao_hard_100[mascara_fronteira])))
    salto_medio_fuzzy = float(np.mean(np.abs(urg_pert - urg_front)))
    taxa_inversao_abrupta_hard = 100.0 * float(
        np.mean((riscos_pert >= 0.50) != (riscos_etapa5[mascara_fronteira] >= 0.50))
    )

    df_out = df.copy()
    df_out["urgencia_fuzzy"] = np.round(urgencias_fuzzy, 4)
    df_out["faixa_regulacao_fuzzy"] = faixas_clinicas
    df_out["decisao_hard_05"] = decisao_hard_100.astype(int)

    return {
        "df_resultado": df_out,
        "metricas": {
            "acc_hard_global": acc_hard_global,
            "acc_fuzzy_global": acc_fuzzy_global,
            "brier_hard_global": brier_hard_global,
            "brier_fuzzy_global": brier_fuzzy_global,
            "n_fronteira": n_fronteira,
            "acc_hard_fronteira": acc_hard_front,
            "acc_fuzzy_fronteira": acc_fuzzy_front,
            "sens_hard_fronteira": sens_hard_front,
            "sens_fuzzy_fronteira": sens_fuzzy_front,
            "brier_hard_fronteira": brier_hard_front,
            "brier_fuzzy_fronteira": brier_fuzzy_front,
            "salto_medio_hard_ruido": salto_medio_hard,
            "salto_medio_fuzzy_ruido": salto_medio_fuzzy,
            "taxa_inversao_abrupta_hard": taxa_inversao_abrupta_hard,
        },
    }
