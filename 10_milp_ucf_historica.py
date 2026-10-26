# -*- coding: utf-8 -*-
"""MOC - EMS MILP con escenarios solares historicos del codigo 09.

Se comparan dos planes del PRIMER dia de prueba, con identica factibilidad:
    sin UCF: min C_programado
    con UCF: min C_programado + UCF_esperada
En ambos, 0 <= solar_programada <= pronostico. El margen no programado
NO es un vertido fisico. Se mantiene la convencion de programacion del 06.
La UCF usa 254 vectores de errores de calibracion; cf=100, ce=10 siguen
siendo supuestos. La referencia PVGIS es estimada, no una medicion.

Cada plan tiene UNA trayectoria de bateria comun a todos sus escenarios.
La red compensa cada disponibilidad FV usando toda la solar del escenario,
sin vertido ni adaptacion de bateria. Se exige |red| <= limite en todos
los escenarios, tanto para el plan sin UCF como para el plan con UCF.
La referencia del dia NO entra al optimizador: solo evalua despues.

C_programado + penalizacion es un criterio de planificacion, no factura.
El gasto neto de red simulado se calcula aparte; nunca se suma otra vez
con la penalizacion. Un dia no constituye una validacion del EMS.

Interfaz del solver: https://pyomo.readthedocs.io/en/stable/api/
pyomo.contrib.appsi.solvers.highs.Highs.html
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo


CARPETA = Path(__file__).resolve().parent
CARPETA_RESULTADOS = CARPETA / "resultados_10"
MOSTRAR_GRAFICOS = True
TOL = 1e-6


def cargar_modulo(nombre, archivo):
    ruta = CARPETA / archivo
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {archivo}. Guardalo junto al codigo 10 en MOC.")
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def huella(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def preparar_caso(base, historico):
    """Calibrar sin prueba y preparar el primer pronostico de prueba."""
    entrada = CARPETA / "resultados_08"
    ruta_meta = entrada / "metadatos_datos_historicos.json"
    if not ruta_meta.exists():
        raise FileNotFoundError("Falta resultados_08. Ejecuta primero el codigo 08.")
    meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
    if meta["zona_local"] != "UTC-3" or meta["signo_error"] != "referencia menos pronostico":
        raise ValueError("Se requieren el horario y el signo de error del codigo 08.")
    calibracion = historico.leer_pronosticos(entrada / "pronosticos_calibracion.csv", "calibracion")
    banco = historico.calibrar_errores(calibracion)
    # El banco ya esta fijado al acceder al periodo de prueba.
    prueba = historico.leer_pronosticos(entrada / "pronosticos_prueba.csv", "prueba")
    if (calibracion["inicio_local"].max() >= prueba["inicio_local"].min()
            or len(banco) != meta["dias_calibracion"]
            or prueba["fecha"].nunique() != meta["dias_prueba"]
            or prueba["fecha"].iloc[0] != meta["primera_fecha_prueba"]):
        raise ValueError("La particion temporal no coincide con el registro del 08.")
    ejemplo = prueba.loc[prueba["fecha"] == prueba["fecha"].iloc[0]].copy().reset_index(drop=True)
    altura = historico.leer_geometria_ejemplo(ejemplo, meta)
    pronostico = ejemplo["solar_pronostico_kw"].to_numpy()
    escenarios = historico.construir_escenarios(pronostico, banco.to_numpy(), altura)
    datos = base.DATOS.copy().reset_index(drop=True)
    if len(datos) != 24 or not np.array_equal(datos["hora"], np.arange(24)):
        raise ValueError("El archivo 02 debe contener los 24 intervalos de 0 a 23.")
    datos["solar_disponible_kw"] = pronostico
    p = dict(base.PARAMETROS)
    if not np.isclose(p["paso_h"], historico.PASO_H) or not np.isclose(p["paso_h"], 1):
        raise ValueError("Los archivos 02 y 09 deben utilizar intervalos de una hora.")
    ejemplo["altura_solar_grados"] = altura
    return datos, p, escenarios, banco, ejemplo, meta


def resolver_ems(base, historico, datos, p, escenarios, incluir_ucf):
    """Resolver sin recibir la referencia futura del dia."""
    pronostico = datos["solar_disponible_kw"].to_numpy()
    historico.evaluar_ucf(pronostico, escenarios)
    cf, ce = historico.COSTO_FALTANTE, historico.COSTO_EXCEDENTE
    n = len(escenarios)
    m = base.crear_modelo(datos, p)
    m.escenarios = pyo.RangeSet(0, n - 1)
    m.red_escenario = pyo.Expression(m.escenarios, m.horas,
        rule=lambda m, k, t: float(datos.loc[t, "demanda_kw"]) + m.carga[t]
        - m.descarga[t] - float(escenarios[k, t]))
    m.factibilidad_escenarios = pyo.Constraint(m.escenarios, m.horas,
        rule=lambda m, k, t: pyo.inequality(-p["limite_red_kw"],
                                          m.red_escenario[k, t], p["limite_red_kw"]))
    # Misma factibilidad en ambos planes. Solo cambia el objetivo.
    m.costo_programado = pyo.Expression(expr=m.costo.expr)
    m.costo.deactivate()
    if incluir_ucf:
        m.faltante = pyo.Var(m.escenarios, m.horas, domain=pyo.NonNegativeReals)
        m.excedente = pyo.Var(m.escenarios, m.horas, domain=pyo.NonNegativeReals)
        m.cota_faltante = pyo.Constraint(m.escenarios, m.horas,
            rule=lambda m, k, t: m.faltante[k, t] >= m.pv[t] - float(escenarios[k, t]))
        m.cota_excedente = pyo.Constraint(m.escenarios, m.horas,
            rule=lambda m, k, t: m.excedente[k, t] >= float(escenarios[k, t]) - m.pv[t])
        m.ucf = pyo.Expression(expr=p["paso_h"] / n * sum(
            cf * m.faltante[k, t] + ce * m.excedente[k, t]
            for k in m.escenarios for t in m.horas))
        objetivo = m.costo_programado + m.ucf
    else:
        objetivo = m.costo_programado
    m.objetivo_10 = pyo.Objective(expr=objetivo, sense=pyo.minimize)
    solver = pyo.SolverFactory("appsi_highs")
    if not solver.available(exception_flag=False):
        raise RuntimeError("HiGHS no esta disponible. Selecciona el interprete de .venv.")
    solver.options["mip_rel_gap"] = 1e-8
    resultado = solver.solve(m, load_solutions=False)
    if not pyo.check_optimal_termination(resultado):
        raise RuntimeError(f"El MILP no termino con solucion optima: {resultado.solver.termination_condition}")
    solver.load_vars()
    plan = datos.copy()
    for columna, variable in (("compra_kw", m.compra), ("venta_kw", m.venta),
            ("solar_usada_kw", m.pv), ("carga_kw", m.carga), ("descarga_kw", m.descarga)):
        plan[columna] = [pyo.value(variable[t]) for t in m.horas]
    plan["energia_inicial_kwh"] = [pyo.value(m.energia[t]) for t in m.horas]
    plan["energia_final_kwh"] = [pyo.value(m.energia[t + 1]) for t in m.horas]
    plan["soc_final_pct"] = 100 * plan["energia_final_kwh"] / p["capacidad_kwh"]
    plan["solar_no_usada_kw"] = pronostico - plan["solar_usada_kw"]
    plan["costo_intervalo_um"] = p["paso_h"] * (
        plan["precio_compra_um_kwh"] * plan["compra_kw"]
        - plan["precio_venta_um_kwh"] * plan["venta_kw"])
    errores = base.validar_resultados(plan, p)
    plan = plan.rename(columns={"solar_disponible_kw": "solar_pronostico_kw",
        "solar_usada_kw": "solar_programada_kw", "solar_no_usada_kw": "margen_no_programado_kw",
        "compra_kw": "compra_programada_kw", "venta_kw": "venta_programada_kw",
        "costo_intervalo_um": "costo_programado_intervalo_um"})
    plan["ucf_esperada_intervalo_um"] = historico.evaluar_ucf(
        plan["solar_programada_kw"].to_numpy(), escenarios, cf, ce)
    c = float(plan["costo_programado_intervalo_um"].sum())
    u = float(plan["ucf_esperada_intervalo_um"].sum())
    if not np.isclose(pyo.value(m.objetivo_10), c + (u if incluir_ucf else 0), atol=TOL, rtol=0):
        raise RuntimeError("El objetivo del MILP no coincide con el costo recalculado.")
    red = (plan["demanda_kw"].to_numpy() + plan["carga_kw"].to_numpy()
           - plan["descarga_kw"].to_numpy() - escenarios)
    if np.max(np.abs(red)) > p["limite_red_kw"] + TOL:
        raise RuntimeError("El plan viola el limite de red en algun escenario.")
    verificacion = {**errores, "max_abs_red_escenarios_kw": float(np.max(np.abs(red))),
                   "terminacion_solver": str(resultado.solver.termination_condition)}
    return plan, verificacion


def evaluar_referencia(plan, referencia, p, cf, ce):
    """Simular red con bateria fija; marcar limites sin recortar potencias."""
    referencia = np.asarray(referencia, dtype=float)
    if referencia.shape != (len(plan),) or not np.isfinite(referencia).all() or np.any(referencia < 0):
        raise ValueError("Referencia solar no valida.")
    tabla = plan.copy()
    tabla["solar_referencia_kw"] = referencia
    red = (tabla["demanda_kw"].to_numpy() + tabla["carga_kw"].to_numpy()
           - tabla["descarga_kw"].to_numpy() - referencia)
    tabla["red_simulada_kw"] = red  # positivo: compra; negativo: venta.
    tabla["red_factible_intervalo"] = np.abs(red) <= p["limite_red_kw"] + TOL
    tabla["compra_simulada_kw"] = np.maximum(red, 0)
    tabla["venta_simulada_kw"] = np.maximum(-red, 0)
    costo_red = p["paso_h"] * (
        tabla["precio_compra_um_kwh"] * tabla["compra_simulada_kw"]
        - tabla["precio_venta_um_kwh"] * tabla["venta_simulada_kw"])
    tabla["gasto_red_simulado_intervalo_um"] = costo_red.where(tabla["red_factible_intervalo"])
    s = tabla["solar_programada_kw"].to_numpy()
    tabla["penalizacion_realizada_intervalo_um"] = p["paso_h"] * (
        cf * np.maximum(s - referencia, 0) + ce * np.maximum(referencia - s, 0))
    c = float(tabla["costo_programado_intervalo_um"].sum())
    u = float(tabla["ucf_esperada_intervalo_um"].sum())
    realizada = float(tabla["penalizacion_realizada_intervalo_um"].sum())
    factible = bool(tabla["red_factible_intervalo"].all())
    indicadores = {"costo_programado_um": c, "ucf_esperada_um": u,
        "criterio_esperado_um": c + u, "penalizacion_realizada_um": realizada,
        "criterio_realizado_um": c + realizada if factible else np.nan,
        "gasto_red_simulado_um": float(costo_red.sum()) if factible else np.nan,
        "red_factible_en_referencia": factible,
        "solar_programada_kwh": float(s.sum() * p["paso_h"]),
        "compra_simulada_kwh": float(tabla["compra_simulada_kw"].sum() * p["paso_h"]),
        "venta_simulada_kwh": float(tabla["venta_simulada_kw"].sum() * p["paso_h"])}
    return tabla, indicadores


def graficar(planes, escenarios, p, fecha, comparacion, carpeta):
    colores = {"sin_ucf": "#dc8b15", "con_ucf": "#18856d"}
    etiquetas = {"sin_ucf": "Sin UCF en objetivo", "con_ucf": "Con UCF en objetivo"}
    bordes = np.arange(25)
    primero = planes["sin_ucf"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), layout="constrained")
    fig.canvas.manager.set_window_title("MOC 10 - Planes del EMS con UCF historica")
    fig.suptitle(f"EMS MILP del {fecha}: {len(escenarios)} escenarios historicos\n"
                 "Demanda y tarifas ficticias; referencia FV estimada", fontsize=12)
    q10, q90 = np.quantile(escenarios, [0.1, 0.9], axis=0, method="inverted_cdf")
    axes[0].fill_between(bordes, np.r_[q10, q10[-1]], np.r_[q90, q90[-1]],
                         step="post", alpha=.25, color="#aebdce", label="P10-P90 por hora de escenarios")
    axes[0].stairs(primero["solar_pronostico_kw"], bordes, color="#555555", linestyle="--", label="Pronostico")
    axes[0].stairs(primero["solar_referencia_kw"], bordes, color="#2455a4", linestyle=":", label="Referencia para evaluar")
    for nombre, tabla in planes.items():
        axes[0].stairs(tabla["solar_programada_kw"], bordes, color=colores[nombre], label=etiquetas[nombre])
        energia = np.r_[tabla["energia_inicial_kwh"].iloc[0], tabla["energia_final_kwh"]]
        axes[1].plot(bordes, energia, color=colores[nombre], label=etiquetas[nombre], marker=".", markersize=3)
        axes[2].stairs(tabla["red_simulada_kw"], bordes, color=colores[nombre], label=etiquetas[nombre])
    axes[0].set(title="Programacion solar y escenarios", ylabel="Potencia solar (kW)")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].axhline(p["soc_min"] * p["capacidad_kwh"], color="gray", linestyle=":")
    axes[1].axhline(p["soc_max"] * p["capacidad_kwh"], color="gray", linestyle=":")
    axes[1].set(title="Energia de la bateria en cada instante", ylabel="Energia (kWh)")
    axes[1].legend(fontsize=8)
    axes[2].axhline(0, color="black", linewidth=.6)
    axes[2].axhline(p["limite_red_kw"], color="gray", linestyle=":", label="Limites de red")
    axes[2].axhline(-p["limite_red_kw"], color="gray", linestyle=":")
    axes[2].set(title="Red simulada con referencia FV y bateria fija; positivo compra, negativo venta",
                ylabel="Potencia de red (kW)", xlabel="Hora local UTC-3")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlim(0, 24); ax.set_xticks(np.arange(0, 25, 2)); ax.grid(alpha=.2); ax.set_axisbelow(True)
    fig.savefig(carpeta / "planes_ems_ucf_historica.png", dpi=160)

    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    fig2.canvas.manager.set_window_title("MOC 10 - Comparacion del criterio esperado")
    fig2.suptitle("Criterio de planificacion; no es una factura de red", fontsize=12)
    for ax, columna, titulo in zip(axes2,
            ["costo_programado_um", "ucf_esperada_um", "criterio_esperado_um"],
            ["Costo programado", "UCF esperada", "Costo programado + UCF"]):
        bars = ax.bar([0, 1], comparacion[columna], color=list(colores.values()), width=.55)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set_xticks([0, 1], ["Sin UCF\nen objetivo", "Con UCF\nen objetivo"])
        ax.set(title=titulo, ylabel="Unidades monetarias ficticias")
        ax.margins(y=.2); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig2.savefig(carpeta / "criterio_ems_ucf_historica.png", dpi=160)
    return fig, fig2


def main():
    base = cargar_modulo("moc02_para_10", "02_microrred_24_horas.py")
    historico = cargar_modulo("moc09_para_10", "09_calibracion_ucf_historica.py")
    datos, p, escenarios, banco, ejemplo, meta = preparar_caso(base, historico)
    fecha = str(ejemplo["fecha"].iloc[0])
    print(f"MOC - Codigo 10: EMS MILP con UCF historica. Dia {fecha}.")
    print(f"Calibracion: {len(banco)} dias; errores hasta {banco.index[-1]}.")
    planes, verificaciones = {}, {}
    for nombre, incluir in (("sin_ucf", False), ("con_ucf", True)):
        print(f"Resolviendo plan {nombre}...")
        planes[nombre], verificaciones[nombre] = resolver_ems(base, historico, datos, p, escenarios, incluir)
    # La referencia solar no fue recibida por resolver_ems en ninguno de los planes.
    filas = []
    for nombre in planes:
        planes[nombre], indicadores = evaluar_referencia(planes[nombre], ejemplo["solar_referencia_kw"].to_numpy(),
            p, historico.COSTO_FALTANTE, historico.COSTO_EXCEDENTE)
        planes[nombre].insert(0, "fecha", fecha)
        filas.append({"plan": nombre, **indicadores})
    comparacion = pd.DataFrame(filas)
    if comparacion.loc[1, "criterio_esperado_um"] > comparacion.loc[0, "criterio_esperado_um"] + TOL:
        raise RuntimeError("El optimo con UCF supera un plan factible bajo el mismo criterio.")
    if comparacion.loc[0, "costo_programado_um"] > comparacion.loc[1, "costo_programado_um"] + TOL:
        raise RuntimeError("El optimo sin UCF no minimiza el costo programado.")
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    for nombre, tabla in planes.items():
        tabla.to_csv(CARPETA_RESULTADOS / f"plan_{nombre}.csv", index=False, encoding="utf-8-sig")
    comparacion.to_csv(CARPETA_RESULTADOS / "comparacion_ems_historica.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"escenario": np.repeat(np.arange(len(banco)), 24),
        "fecha_error_calibracion": np.repeat(banco.index.to_numpy(), 24),
        "probabilidad": 1 / len(banco), "hora": np.tile(np.arange(24), len(banco)),
        "solar_disponible_kw": escenarios.ravel()}).to_csv(
            CARPETA_RESULTADOS / "escenarios_ejemplo.csv", index=False, encoding="utf-8-sig")
    configuracion = {"fecha_ejemplo": fecha, "tipo_modelo": "MILP EMS con bateria comun por plan",
        "dias_calibracion": len(banco), "primera_fecha_calibracion": str(banco.index[0]),
        "ultima_fecha_calibracion": str(banco.index[-1]), "dias_prueba_reservados": meta["dias_prueba"],
        "costo_faltante_um_kwh": historico.COSTO_FALTANTE, "costo_excedente_um_kwh": historico.COSTO_EXCEDENTE,
        "parametros_ems": p, "demanda_y_precios": "Perfiles ficticios del codigo 02",
        "referencia_fv": meta["referencia"], "metodo_escenarios": "Funciones del 09; banco fijado solo con calibracion",
        "factibilidad": "Todos los escenarios en ambos planes; bateria fija, red compensa, sin vertido",
        "formulas_criterio": {"esperado": "costo programado + UCF esperada",
                             "realizado": "costo programado + penalizacion con referencia",
                             "red": "compras simuladas menos ingresos por ventas; calculo separado"},
        "sha256_archivos": {nombre: huella(CARPETA / nombre) for nombre in (
            "02_microrred_24_horas.py", "09_calibracion_ucf_historica.py", "10_milp_ucf_historica.py",
            "resultados_08/pronosticos_calibracion.csv", "resultados_08/pronosticos_prueba.csv",
            "resultados_08/metadatos_datos_historicos.json")},
        "sha256_respuesta_pvgis": meta["sha256_respuesta"], "verificacion_planes": verificaciones,
        "versiones": {"numpy": np.__version__, "pandas": pd.__version__,
                      "pyomo": __import__("importlib.metadata", fromlist=["version"]).version("pyomo")},
        "limitaciones": ["Un solo dia; no validacion de los 109 dias de prueba",
            "Errores aditivos por hora sin condicionamiento estacional; recorte a cero y mascara solar del 09",
            "Demanda, tarifas y penalizaciones ficticias; FV estimada, no medida",
            "Bateria sin recarga adaptativa ni degradacion; no se modela limite AC del inversor"]}
    (CARPETA_RESULTADOS / "configuracion_ems_historica.json").write_text(
        json.dumps(configuracion, ensure_ascii=False, indent=2), encoding="utf-8")
    resumen = (f"MOC - EMS MILP con escenarios historicos. Dia {fecha}.\n"
        f"Calibracion: {len(banco)} dias; {banco.index[0]} a {banco.index[-1]}.\n"
        "Ambos planes: solucion optima; misma factibilidad para todos los escenarios.\n"
        "Demanda y tarifas del 02: ficticias. Referencia FV PVGIS: estimada, no medida.\n\n"
        "PLANIFICACION - unidades monetarias ficticias\n"
        + comparacion[["plan", "costo_programado_um", "ucf_esperada_um", "criterio_esperado_um"]].to_string(index=False, float_format=lambda v: f"{v:.2f}")
        + "\n\nEVALUACION DEL UNICO DIA - unidades monetarias ficticias\n"
        + comparacion[["plan", "penalizacion_realizada_um", "criterio_realizado_um", "gasto_red_simulado_um", "red_factible_en_referencia"]].to_string(index=False, float_format=lambda v: f"{v:.2f}")
        + "\n\nCriterio esperado = costo programado + UCF esperada.\n"
        "Criterio realizado = costo programado + penalizacion del dia. No es factura.\n"
        "Gasto de red simulado: calculo separado con toda la FV de referencia y bateria fija.\n"
        "No sumar el gasto de red nuevamente con los criterios anteriores.\n"
        "Si la referencia viola la red, se marca False y no se informa costo valido (NaN).\n"
        "La mejora del criterio esperado no garantiza mejorar el dia ni el gasto de red.\n"
        "Proxima etapa: evaluacion diaria durante los 109 dias de prueba.\n")
    (CARPETA_RESULTADOS / "resumen_ems_historica.txt").write_text(resumen, encoding="utf-8")
    print("\n" + resumen)
    figuras = graficar(planes, escenarios, p, fecha, comparacion, CARPETA_RESULTADOS)
    print(f"Resultados guardados en: {CARPETA_RESULTADOS}")
    if MOSTRAR_GRAFICOS:
        print("Cerra las dos ventanas de graficos para finalizar.")
        plt.show()
    else:
        for fig in figuras:
            plt.close(fig)


if __name__ == "__main__":
    main()
