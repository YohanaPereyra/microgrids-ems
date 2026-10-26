# -*- coding: utf-8 -*-
"""MOC - Primera UCF empirica con errores historicos del codigo 08.

La distribucion se construye SOLO con los dias de calibracion. Cada dia
aporta un vector completo de 24 errores y peso 1/N. Para un pronostico:
    W[k,t] = max(0, pronostico[t] + error_calibracion[k,t]).
Se impone W=0 en intervalos cuya altura solar PVGIS H_sun es <=0.
H_sun es geometria conocida por fecha y lugar; no es meteorologia futura.
Se traslada la distribucion de errores en kW por hora local: es un modelo
inicial, sin condicionamiento por estacion, nubosidad o nivel pronosticado.
No se supone que 5 kWp sea un limite rigido de potencia AC del inversor.

Se minimiza la UCF AISLADA de las 24 horas del primer dia de prueba.
Es un LP continuo, no un EMS ni un MILP nuevo. El limite de programacion
0 <= s[t] <= pronostico[t] mantiene la convencion del archivo 06.
cf=100 y ce=10 u.m./kWh siguen siendo supuestos economicos: se estima la
distribucion de errores, no esos precios. La referencia de prueba se usa
solo despues de elegir s, para ilustrar la penalizacion de ese unico dia.
La mejora esperada de una UCF aislada no demuestra ahorro de red.

Referencias de implementacion:
https://pyomo.readthedocs.io/en/stable/getting_started/pyomo_overview/simple_examples.html
https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/pvgis-5-tools/hourly-radiation_en
"""

import hashlib
import json
from datetime import timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo


COSTO_FALTANTE = 100.0
COSTO_EXCEDENTE = 10.0
PASO_H = 1.0
MOSTRAR_GRAFICO = True
CARPETA = Path(__file__).resolve().parent
CARPETA_ENTRADA = CARPETA / "resultados_08"
CARPETA_DATOS = CARPETA / "datos_08"
CARPETA_RESULTADOS = CARPETA / "resultados_09"
TZ_LOCAL = timezone(timedelta(hours=-3))


def leer_pronosticos(ruta, periodo):
    """Revisar dias completos, signo del error y disponibilidad temporal."""
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta}. Ejecuta primero el codigo 08 en la carpeta MOC.")
    tabla = pd.read_csv(ruta, encoding="utf-8-sig")
    requeridas = {"inicio_local", "centro_utc", "origen_pronostico_local",
                  "emision_pronostico_local", "fecha", "hora", "periodo",
                  "solar_referencia_kw", "solar_pronostico_kw", "error_kw"}
    if not requeridas.issubset(tabla.columns) or tabla.empty:
        raise ValueError(f"Formato incompleto en {ruta.name}; vuelve a ejecutar el 08.")
    for columna in ("inicio_local", "origen_pronostico_local", "emision_pronostico_local"):
        tabla[columna] = pd.to_datetime(tabla[columna], utc=True).dt.tz_convert(TZ_LOCAL)
    tabla["centro_utc"] = pd.to_datetime(tabla["centro_utc"], utc=True)
    tabla = tabla.sort_values("inicio_local").reset_index(drop=True)
    valores = tabla[["hora", "solar_referencia_kw", "solar_pronostico_kw", "error_kw"]].to_numpy(dtype=float)
    if (not np.isfinite(valores).all() or (tabla["periodo"] != periodo).any()
            or tabla["inicio_local"].duplicated().any()
            or (tabla[["solar_referencia_kw", "solar_pronostico_kw"]] < 0).any().any()):
        raise ValueError(f"Hay filas invalidas en {ruta.name}.")
    if (not (tabla["hora"] == tabla["inicio_local"].dt.hour).all()
            or not (tabla["fecha"] == tabla["inicio_local"].dt.strftime("%Y-%m-%d")).all()
            or not (tabla["inicio_local"].dt.minute == 0).all()):
        raise ValueError(f"Revisa las horas locales de {ruta.name}.")
    for _, dia in tabla.groupby("fecha", sort=True):
        if len(dia) != 24 or not np.array_equal(dia["hora"].to_numpy(), np.arange(24)):
            raise ValueError("Cada fecha debe tener exactamente los intervalos 0 a 23.")
    origen = tabla["origen_pronostico_local"]
    emision = tabla["emision_pronostico_local"]
    if (not (origen == tabla["inicio_local"] - pd.Timedelta(days=1)).all()
            or not (emision == tabla["inicio_local"].dt.normalize()).all()
            or not (origen + pd.Timedelta(hours=1) <= emision).all()
            or not (tabla["centro_utc"] == tabla["inicio_local"].dt.tz_convert("UTC") + pd.Timedelta(minutes=30)).all()
            or not np.allclose(tabla["error_kw"], tabla["solar_referencia_kw"] - tabla["solar_pronostico_kw"], atol=1e-9, rtol=0)):
        raise ValueError("Los errores o los tiempos no coinciden con la convencion del 08.")
    return tabla


def calibrar_errores(calibracion):
    """Esta funcion recibe exclusivamente el periodo de calibracion."""
    if (calibracion["periodo"] != "calibracion").any():
        raise ValueError("El banco de errores solo puede usar calibracion.")
    banco = calibracion.pivot(index="fecha", columns="hora", values="error_kw").sort_index()
    banco = banco.reindex(columns=np.arange(24))
    if len(banco) < 2 or not np.isfinite(banco.to_numpy()).all():
        raise ValueError("Se necesitan al menos dos dias completos para calibrar.")
    return banco


def leer_geometria_ejemplo(ejemplo, metadatos):
    """Del JSON original se lee SOLO time y H_sun para el dia de ejemplo."""
    nombre = Path(metadatos["archivo_respuesta"])
    if nombre.name != str(nombre):
        raise ValueError("Nombre de respuesta PVGIS no valido.")
    ruta = CARPETA_DATOS / nombre
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta}. Conserva la carpeta datos_08 del codigo 08.")
    bruto = ruta.read_bytes()
    if hashlib.sha256(bruto).hexdigest() != metadatos["sha256_respuesta"]:
        raise ValueError("El JSON de PVGIS no coincide con la huella registrada por el 08.")
    respuesta = json.loads(bruto)
    geometria = pd.DataFrame([{k: fila[k] for k in ("time", "H_sun")}
                             for fila in respuesta["outputs"]["hourly"]])
    geometria.index = pd.to_datetime(geometria["time"], format="%Y%m%d:%H%M", utc=True)
    if not geometria.index.is_unique:
        raise ValueError("Hay fechas UTC duplicadas en la geometria PVGIS.")
    altura = geometria["H_sun"].reindex(pd.DatetimeIndex(ejemplo["centro_utc"])).to_numpy(dtype=float)
    if not np.isfinite(altura).all():
        raise ValueError("No se pudo localizar la geometria de las 24 horas del ejemplo.")
    return altura


def construir_escenarios(pronostico, errores, altura_solar):
    pronostico = np.asarray(pronostico, dtype=float)
    errores = np.asarray(errores, dtype=float)
    altura_solar = np.asarray(altura_solar, dtype=float)
    if (pronostico.shape != (24,) or altura_solar.shape != (24,)
            or errores.ndim != 2 or errores.shape[1] != 24 or len(errores) < 2
            or not np.isfinite(pronostico).all() or np.any(pronostico < 0)
            or not np.isfinite(errores).all() or not np.isfinite(altura_solar).all()):
        raise ValueError("Revisa pronostico, errores y altura solar.")
    solares = np.maximum(pronostico[None, :] + errores, 0.0)
    solares[:, altura_solar <= 0] = 0.0
    return solares


def evaluar_ucf(programada, escenarios, cf=COSTO_FALTANTE, ce=COSTO_EXCEDENTE):
    """Promedio empirico por hora; todas las filas tienen peso 1/N."""
    programada = np.asarray(programada, dtype=float)
    escenarios = np.asarray(escenarios, dtype=float)
    if (escenarios.ndim != 2 or programada.shape != (escenarios.shape[1],)
            or len(escenarios) == 0 or not np.isfinite(escenarios).all()
            or np.any(escenarios < 0) or not np.isfinite(programada).all()
            or np.any(programada < 0) or not np.isfinite([cf, ce, PASO_H]).all()
            or min(cf, ce, PASO_H) <= 0):
        raise ValueError("Revisa potencias, penalizaciones e intervalo.")
    faltante = np.maximum(programada[None, :] - escenarios, 0.0)
    excedente = np.maximum(escenarios - programada[None, :], 0.0)
    return (cf * faltante + ce * excedente).mean(axis=0) * PASO_H


def resolver_ucf_aislada(pronostico, escenarios):
    """LP del 05 extendido a 24 horas con una distribucion empirica."""
    evaluar_ucf(pronostico, escenarios)  # Validar antes de construir el modelo.
    n = len(escenarios)
    modelo = pyo.ConcreteModel()
    modelo.horas = pyo.RangeSet(0, 23)
    modelo.escenarios = pyo.RangeSet(0, n - 1)
    modelo.programada = pyo.Var(modelo.horas, domain=pyo.NonNegativeReals,
                               bounds=lambda m, t: (0.0, float(pronostico[t])))
    modelo.faltante = pyo.Var(modelo.escenarios, modelo.horas, domain=pyo.NonNegativeReals)
    modelo.excedente = pyo.Var(modelo.escenarios, modelo.horas, domain=pyo.NonNegativeReals)
    modelo.cota_faltante = pyo.Constraint(modelo.escenarios, modelo.horas,
        rule=lambda m, k, t: m.faltante[k, t] >= m.programada[t] - float(escenarios[k, t]))
    modelo.cota_excedente = pyo.Constraint(modelo.escenarios, modelo.horas,
        rule=lambda m, k, t: m.excedente[k, t] >= float(escenarios[k, t]) - m.programada[t])
    modelo.objetivo = pyo.Objective(expr=PASO_H / n * sum(
        COSTO_FALTANTE * modelo.faltante[k, t] + COSTO_EXCEDENTE * modelo.excedente[k, t]
        for k in modelo.escenarios for t in modelo.horas), sense=pyo.minimize)
    solver = pyo.SolverFactory("appsi_highs")
    if not solver.available(exception_flag=False):
        raise RuntimeError("HiGHS no esta disponible. Selecciona el interprete de .venv que usaste en el 06.")
    solucion = solver.solve(modelo)
    if not pyo.check_optimal_termination(solucion):
        raise RuntimeError(f"El LP no termino con solucion optima: {solucion.solver.termination_condition}")
    programada = np.array([pyo.value(modelo.programada[t]) for t in modelo.horas])
    if np.any(programada < -1e-7) or np.any(programada > pronostico + 1e-7):
        raise RuntimeError("La programacion no respeta sus limites.")
    programada = np.clip(programada, 0, pronostico)
    costo = float(evaluar_ucf(programada, escenarios).sum())
    if not np.isclose(costo, pyo.value(modelo.objetivo), atol=1e-6, rtol=0):
        raise RuntimeError("El objetivo del LP no coincide con la UCF recalculada.")
    if costo > float(evaluar_ucf(pronostico, escenarios).sum()) + 1e-6:
        raise RuntimeError("La UCF optima no puede superar a la referencia factible.")
    return programada


def crear_curva(pronostico, escenarios, programada):
    hora = int(np.argmax(pronostico))  # Se elige sin mirar la referencia de prueba.
    potencias = np.unique(np.r_[np.linspace(0, pronostico[hora], 401), programada[hora], pronostico[hora]])
    disponibles = escenarios[:, hora]
    costos = PASO_H * (COSTO_FALTANTE * np.maximum(potencias[:, None] - disponibles, 0)
                      + COSTO_EXCEDENTE * np.maximum(disponibles - potencias[:, None], 0)).mean(axis=1)
    return hora, pd.DataFrame({"programacion_kw": potencias, "ucf_um": costos})


def graficar(tabla, escenarios, curva, hora_curva, comparacion, carpeta):
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), layout="constrained")
    fig.canvas.manager.set_window_title("MOC - Primera UCF historica")
    fecha = tabla["fecha"].iloc[0]
    fig.suptitle(f"UCF empirica aislada - ejemplo {fecha}\n"
                 f"{len(escenarios)} vectores de error de calibracion; referencia FV estimada", fontsize=13)
    bordes = np.arange(25)
    q10, q90 = np.quantile(escenarios, [0.10, 0.90], axis=0, method="inverted_cdf")
    axes[0].fill_between(bordes, np.r_[q10, q10[-1]], np.r_[q90, q90[-1]],
                         step="post", color="#aebdce", alpha=0.35, label="P10-P90 por hora de los escenarios")
    for col, label, color, ls in (
        ("solar_pronostico_kw", "Pronostico de persistencia", "#dc8b15", "--"),
        ("solar_programada_ucf_kw", "Programacion que minimiza solo UCF", "#18856d", "-"),
        ("solar_referencia_kw", "Referencia del dia, solo para evaluar", "#2455a4", ":"),
    ):
        axes[0].stairs(tabla[col], bordes, label=label, color=color, linestyle=ls, linewidth=1.8)
    axes[0].set(title="1. Pronostico, escenarios y programacion solar del ejemplo",
                xlabel="Hora local UTC-3", ylabel="Potencia (kW)", xlim=(0, 24))
    axes[0].legend(fontsize=8.5, ncol=2)
    axes[1].plot(curva["programacion_kw"], curva["ucf_um"], color="#2455a4", linewidth=2)
    for col, label, color in (("solar_pronostico_kw", "Programar el pronostico", "#dc8b15"),
                              ("solar_programada_ucf_kw", "Minimo de la UCF aislada", "#18856d")):
        s = float(tabla[col].iloc[hora_curva])
        costo = float(evaluar_ucf(np.array([s]), escenarios[:, [hora_curva]])[0])
        axes[1].scatter(s, costo, s=45, color=color, label=f"{label}: {s:.3f} kW; {costo:.2f} u.m.", zorder=3)
    axes[1].set(title=f"2. UCF del intervalo {hora_curva:02}:00-{hora_curva + 1:02}:00",
                xlabel="Potencia que se decide programar (kW)", ylabel="UCF del intervalo (u.m.)")
    axes[1].legend(fontsize=8.5)
    x = np.arange(2)
    bars = axes[2].bar(x, comparacion["ucf_esperada_ejemplo_um"], color=["#dc8b15", "#18856d"], width=0.5)
    axes[2].bar_label(bars, fmt="%.2f", padding=4)
    axes[2].set_xticks(x, ["Programar el pronostico", "Minimizar solo UCF"])
    axes[2].set(title="3. UCF diaria estimada con los escenarios de calibracion",
                ylabel="Penalizacion esperada (u.m.)")
    axes[2].set_ylim(0, max(1, comparacion["ucf_esperada_ejemplo_um"].max() * 1.2))
    for ax in axes:
        ax.grid(alpha=0.2); ax.set_axisbelow(True)
    fig.savefig(carpeta / "calibracion_ucf_historica.png", dpi=160)
    return fig


def main():
    ruta_meta = CARPETA_ENTRADA / "metadatos_datos_historicos.json"
    if not ruta_meta.exists():
        raise FileNotFoundError("Falta resultados_08/metadatos_datos_historicos.json. Ejecuta primero el 08.")
    metadatos = json.loads(ruta_meta.read_text(encoding="utf-8"))
    if metadatos["zona_local"] != "UTC-3" or metadatos["signo_error"] != "referencia menos pronostico":
        raise ValueError("Este archivo requiere el horario y signo de error del codigo 08.")
    ruta_cal = CARPETA_ENTRADA / "pronosticos_calibracion.csv"
    calibracion = leer_pronosticos(ruta_cal, "calibracion")
    banco = calibrar_errores(calibracion)
    # El banco queda fijado ANTES de leer la prueba.
    prueba = leer_pronosticos(CARPETA_ENTRADA / "pronosticos_prueba.csv", "prueba")
    if (calibracion["inicio_local"].max() >= prueba["inicio_local"].min()
            or len(banco) != metadatos["dias_calibracion"]
            or prueba["fecha"].nunique() != metadatos["dias_prueba"]
            or prueba["fecha"].iloc[0] != metadatos["primera_fecha_prueba"]):
        raise ValueError("La particion de calibracion y prueba no coincide con el 08.")
    ejemplo = prueba.loc[prueba["fecha"] == prueba["fecha"].iloc[0]].copy().reset_index(drop=True)
    pronostico = ejemplo["solar_pronostico_kw"].to_numpy()
    altura = leer_geometria_ejemplo(ejemplo, metadatos)
    escenarios = construir_escenarios(pronostico, banco.to_numpy(), altura)
    programada = resolver_ucf_aislada(pronostico, escenarios)
    hora_curva, curva = crear_curva(pronostico, escenarios, programada)
    ejemplo["altura_solar_grados"] = altura
    ejemplo["solar_programada_ucf_kw"] = programada
    ejemplo["ucf_pronostico_um"] = evaluar_ucf(pronostico, escenarios)
    ejemplo["ucf_programacion_um"] = evaluar_ucf(programada, escenarios)
    # La referencia del dia aparece unicamente DESPUES de resolver el LP.
    referencia = ejemplo["solar_referencia_kw"].to_numpy()
    filas = []
    for nombre, s in (("programar_pronostico", pronostico), ("minimizar_ucf_aislada", programada)):
        realizada = PASO_H * (COSTO_FALTANTE * np.maximum(s - referencia, 0)
                             + COSTO_EXCEDENTE * np.maximum(referencia - s, 0)).sum()
        filas.append({"alternativa": nombre, "solar_programada_kwh": float(s.sum() * PASO_H),
                      "ucf_esperada_ejemplo_um": float(evaluar_ucf(s, escenarios).sum()),
                      "penalizacion_realizada_ejemplo_um": float(realizada)})
    comparacion = pd.DataFrame(filas)

    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    banco.to_csv(CARPETA_RESULTADOS / "errores_calibracion_por_dia.csv", encoding="utf-8-sig")
    e = banco.to_numpy()
    q05, mediana, q95 = np.quantile(e, [0.05, 0.5, 0.95], axis=0, method="inverted_cdf")
    estadisticas = pd.DataFrame({"hora": np.arange(24), "dias_calibracion": len(banco),
        "sesgo_kw": e.mean(axis=0), "mae_kw": np.abs(e).mean(axis=0),
        "error_p05_kw": q05, "error_mediana_kw": mediana, "error_p95_kw": q95})
    estadisticas.to_csv(CARPETA_RESULTADOS / "estadisticas_error_por_hora.csv", index=False, encoding="utf-8-sig")
    escenarios_csv = pd.DataFrame({"escenario": np.repeat(np.arange(len(banco)), 24),
        "fecha_error_calibracion": np.repeat(banco.index.to_numpy(), 24),
        "probabilidad": 1.0 / len(banco), "hora": np.tile(np.arange(24), len(banco)),
        "error_calibracion_kw": e.ravel(), "solar_disponible_kw": escenarios.ravel()})
    escenarios_csv.to_csv(CARPETA_RESULTADOS / "escenarios_ejemplo.csv", index=False, encoding="utf-8-sig")
    ejemplo.to_csv(CARPETA_RESULTADOS / "programacion_ejemplo.csv", index=False, encoding="utf-8-sig")
    curva.to_csv(CARPETA_RESULTADOS / "curva_ucf_hora_ejemplo.csv", index=False, encoding="utf-8-sig")
    comparacion.to_csv(CARPETA_RESULTADOS / "comparacion_ejemplo.csv", index=False, encoding="utf-8-sig")
    configuracion = {"metodo": "Distribucion empirica de vectores de error diarios por hora local",
        "dias_calibracion": len(banco), "primera_fecha_calibracion": str(banco.index[0]),
        "ultima_fecha_calibracion": str(banco.index[-1]), "probabilidad_por_vector": 1.0 / len(banco),
        "fecha_ejemplo": ejemplo["fecha"].iloc[0], "hora_curva": hora_curva,
        "signo_error": "referencia menos pronostico", "formula_escenarios": "max(0, pronostico + error); cero si H_sun <= 0",
        "coeficientes_economicos_calibrados": False, "costo_faltante_um_kwh": COSTO_FALTANTE,
        "costo_excedente_um_kwh": COSTO_EXCEDENTE, "paso_h": PASO_H,
        "cota_programacion": "0 <= s[t] <= pronostico[t]", "tipo_modelo": "LP de UCF aislada",
        "referencia_fv": metadatos["referencia"], "sha256_calibracion_csv": hashlib.sha256(ruta_cal.read_bytes()).hexdigest(),
        "sha256_respuesta_pvgis": metadatos["sha256_respuesta"], "numpy": np.__version__, "pandas": pd.__version__,
        "limitaciones": ["Errores aditivos agrupados por hora sin condicionamiento estacional",
                         "La no negatividad y la mascara solar modifican los errores trasladados",
                         "No modela limite AC del inversor ni operacion de la microrred",
                         "Ejemplo de un dia de prueba; no es validacion del EMS ni ahorro de factura"]}
    (CARPETA_RESULTADOS / "configuracion_ucf.json").write_text(json.dumps(configuracion, ensure_ascii=False, indent=2), encoding="utf-8")
    resumen = (
        "MOC - Primera UCF empirica con errores historicos del codigo 08.\n"
        f"Calibracion: {len(banco)} dias, desde {banco.index[0]} hasta {banco.index[-1]}.\n"
        f"Escenarios del ejemplo: {len(banco)} vectores de 24 horas con igual peso.\n"
        f"Dia ilustrativo: {ejemplo['fecha'].iloc[0]}; primer pronostico del periodo de prueba.\n"
        f"Coeficientes asumidos: faltante {COSTO_FALTANTE:g}; excedente {COSTO_EXCEDENTE:g} u.m./kWh.\n"
        "Estado: solucion optima de un LP continuo de UCF AISLADA de 24 horas.\n\n"
        + comparacion.to_string(index=False, float_format=lambda x: f"{x:.2f}") + "\n\n"
        "UCF esperada: media sobre los escenarios construidos con calibracion.\n"
        "Penalizacion realizada: usa la referencia PVGIS del UNICO dia ilustrativo.\n"
        "La mejora esperada no garantiza mejorar ese dia concreto.\n"
        "La referencia FV es estimada, no una medicion. Los coeficientes economicos no se estiman aqui.\n"
        "No se ejecuta el EMS, no hay plan de bateria y no se calcula gasto de red.\n"
        "La siguiente etapa integra estos escenarios y la UCF al MILP del EMS.\n")
    print(resumen)
    (CARPETA_RESULTADOS / "resumen_calibracion_ucf.txt").write_text(resumen, encoding="utf-8")
    fig = graficar(ejemplo, escenarios, curva, hora_curva, comparacion, CARPETA_RESULTADOS)
    print(f"Resultados guardados en: {CARPETA_RESULTADOS}")
    if MOSTRAR_GRAFICO:
        print("Cerra el grafico para finalizar la ejecucion.")
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
