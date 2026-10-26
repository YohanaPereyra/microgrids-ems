# -*- coding: utf-8 -*-
"""MOC - Codigo 12: sensibilidad exploratoria del EMS con UCF historica.

Siete casos, un factor por vez: cf=50/100/200, ce=5/10/20 y bateria
5/10/15 kWh. El caso central es comun. Los valores son SUPUESTOS, no
tarifas estimadas ni recomendaciones de dimensionamiento. La potencia de
bateria, eficiencias y fracciones de SOC se mantienen como en el 02.

Se reutiliza SIN CAMBIAR la formulacion MILP del 10 y el banco fijo del 11.
Cada caso compara sin/con UCF con la misma factibilidad y coeficientes.
El plan sin UCF no depende de cf/ce: se resuelve una vez por capacidad y
dia y se reevalua con los coeficientes de cada caso. No se reutiliza entre
capacidades ni fechas. Son 10 MILP por dia, 1090 con los datos originales.

Los 109 dias del 11 YA fueron observados: este barrido es exploratorio.
No selecciona un 'mejor' caso ni constituye una nueva validacion ciega.
Comparar criterios absolutos con penalizaciones distintas cambia la regla
economica. Se muestran diferencias sin/con UCF DENTRO de cada caso.
El gasto de red se calcula aparte con bateria fija y toda la FV de
referencia. No se suma otra vez al criterio con penalizacion.

Dias independientes: SOC inicial/final se reinicia por fecha y capacidad.
FV estimada; demanda/precios ficticios; sin inversion ni degradacion.
No calcula capacidad optima, rentabilidad ni operacion continua de bateria.
"""

import gc
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CARPETA = Path(__file__).resolve().parent
SALIDA = CARPETA / "resultados_12"
MOSTRAR_GRAFICOS = True
TOL = 1e-6
# None ejecuta TODOS los dias. Para una prueba rapida local: MAX_DIAS = 3.
# Un resultado parcial queda identificado y no se informa como ensayo completo.
MAX_DIAS = None


def cargar_11():
    ruta = CARPETA / "11_validacion_historica_ems.py"
    if not ruta.exists():
        raise FileNotFoundError("Falta 11_validacion_historica_ems.py. Guardalo junto al codigo 12.")
    spec = importlib.util.spec_from_file_location("moc11_para_12", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def definir_casos(p, historico):
    """Barrido un factor por vez alrededor de la configuracion original."""
    capacidad = float(p["capacidad_kwh"])
    cf, ce = float(historico.COSTO_FALTANTE), float(historico.COSTO_EXCEDENTE)
    filas = [{"caso": "base", "factor": "base", "multiplicador": 1.0,
              "capacidad_kwh": capacidad, "cf_um_kwh": cf, "ce_um_kwh": ce}]
    for factor, valores in (("faltante", (.5, 2.0)), ("excedente", (.5, 2.0)),
                            ("capacidad", (.5, 1.5))):
        for k in valores:
            fila = dict(filas[0], caso=f"{factor}_{str(k).replace('.', 'p')}",
                        factor=factor, multiplicador=k)
            columna = {"faltante": "cf_um_kwh", "excedente": "ce_um_kwh",
                       "capacidad": "capacidad_kwh"}[factor]
            fila[columna] *= k
            filas.append(fila)
    casos = pd.DataFrame(filas)
    if (not np.isfinite(casos[["capacidad_kwh", "cf_um_kwh", "ce_um_kwh"]]).all().all()
            or (casos[["capacidad_kwh", "cf_um_kwh", "ce_um_kwh"]] <= 0).any().any()):
        raise ValueError("Capacidades y penalizaciones deben ser positivas y finitas.")
    return casos


def coeficientes_caso(historico, caso):
    # Funciones compartidas; constantes locales. No modifica el modulo 09.
    return SimpleNamespace(COSTO_FALTANTE=float(caso["cf_um_kwh"]),
        COSTO_EXCEDENTE=float(caso["ce_um_kwh"]), evaluar_ucf=historico.evaluar_ucf)


def resolver_o_registrar(m10, base, h, datos, p, escenarios, incluir_ucf):
    try:
        return m10.resolver_ems(base, h, datos, p, escenarios, incluir_ucf)
    except RuntimeError as error:
        mensaje = str(error)
        if mensaje.startswith("El MILP no termino con solucion optima:") and "infeasible" in mensaje.lower():
            return None, {"terminacion_solver": "infeasible", "detalle": mensaje}
        raise


def evaluar_caso(m10, historico, planes, estados, referencia, p, caso, fecha, metricas):
    """La referencia entra DESPUES de obtener todos los planes del dia."""
    cf, ce = float(caso["cf_um_kwh"]), float(caso["ce_um_kwh"])
    filas, horarios, verificaciones = [], [], []
    for nombre in ("sin_ucf", "con_ucf"):
        plan = planes[nombre]
        fila = {"caso": caso["caso"], "fecha": fecha, "plan": nombre,
            "optimo": plan is not None, "estado_solver": estados[nombre]["terminacion_solver"],
            "red_factible_en_referencia": False, **{k: np.nan for k in metricas}}
        verificaciones.append({"caso": caso["caso"], "fecha": fecha,
                               "plan": nombre, **estados[nombre]})
        if plan is not None:
            plan = plan.copy()
            # El plan sin UCF se reutiliza; su UCF SI se recalcula para este caso.
            plan["ucf_esperada_intervalo_um"] = historico.evaluar_ucf(
                plan["solar_programada_kw"].to_numpy(), planes["escenarios"], cf, ce)
            tabla, valores = m10.evaluar_referencia(plan, referencia, p, cf, ce)
            fila.update(valores)
            fila["carga_bateria_kwh"] = float(tabla["carga_kw"].sum() * p["paso_h"])
            fila["descarga_bateria_kwh"] = float(tabla["descarga_kw"].sum() * p["paso_h"])
            for posicion, (columna, valor) in enumerate((("caso", caso["caso"]),
                                                        ("fecha", fecha), ("plan", nombre))):
                tabla.insert(posicion, columna, valor)
            horarios.append(tabla)
        filas.append(fila)
    if all(f["optimo"] for f in filas):
        if (filas[0]["costo_programado_um"] > filas[1]["costo_programado_um"] + TOL
                or filas[1]["criterio_esperado_um"] > filas[0]["criterio_esperado_um"] + TOL):
            raise RuntimeError(f"Fallo de los objetivos optimos en {fecha}, {caso['caso']}.")
    return filas, horarios, verificaciones


def resumir_barrido(m11, diarios, casos):
    """Comparaciones por caso y resumen adicional sobre la interseccion global."""
    ids = list(casos["caso"])
    if set(diarios["caso"]) != set(ids):
        raise ValueError("Faltan casos en los resultados diarios.")
    pares_todos, stats_pares, frecuencias_pares, factibilidades = [], [], [], []
    for caso in ids:
        d = diarios.loc[diarios["caso"] == caso].drop(columns="caso")
        pares, stats, frec, fact = m11.resumir(d)
        for tabla in (pares, stats, frec, fact):
            tabla.insert(0, "caso", caso)
        pares_todos.append(pares);stats_pares.append(stats)
        frecuencias_pares.append(frec);factibilidades.append(fact)
    pares = pd.concat(pares_todos, ignore_index=True)
    if pares.groupby("fecha")["caso"].nunique().ne(len(ids)).any():
        raise ValueError("Cada caso debe intentar las mismas fechas.")
    fechas_comunes = set(pares.groupby("fecha")["dia_comun_valido"].all().loc[lambda s: s].index)
    pares["dia_valido_todos_los_casos"] = pares["fecha"].isin(fechas_comunes)
    stats_globales, frecuencias_globales, comparacion = [], [], []
    for caso in ids:
        d = diarios.loc[diarios["caso"] == caso].drop(columns="caso").copy()
        # Se conservan las fechas en el calculo, pero solo la interseccion global
        # aporta costos. La factibilidad verdadera esta en factibilidad.csv.
        d["red_factible_en_referencia"] &= d["fecha"].isin(fechas_comunes)
        _, stats, frec, _ = m11.resumir(d)
        for tabla in (stats, frec):tabla.insert(0, "caso", caso)
        stats_globales.append(stats);frecuencias_globales.append(frec)
        fila = casos.loc[casos["caso"] == caso].iloc[0].to_dict()
        fila["dias_comunes_todos_los_casos"] = len(fechas_comunes)
        fila["dias_pareados_del_caso"] = int(pares.loc[pares["caso"] == caso, "dia_comun_valido"].sum())
        for metrica, corto in (("criterio_realizado_um", "criterio"),
                               ("gasto_red_simulado_um", "red")):
            r = stats.loc[stats["metrica"] == metrica].set_index("plan")
            for plan in m11.PLANES:
                for medida in ("media", "p95"):
                    fila[f"{corto}_{medida}_{plan}_um"] = float(r.loc[plan, medida])
            a, b = float(r.loc["sin_ucf", "media"]), float(r.loc["con_ucf", "media"])
            fila[f"delta_{corto}_media_um"] = b - a
            fila[f"reduccion_{corto}_media_pct"] = 100 * (a - b) / abs(a) if np.isfinite(a) and abs(a) > TOL else np.nan
            f = frec.loc[frec["metrica"] == metrica].iloc[0]
            for col in ("dias_menor_con_ucf", "dias_mayor_con_ucf", "dias_iguales"):
                fila[f"{corto}_{col}"] = int(f[col])
        comparacion.append(fila)
    return {"comparacion_por_dia": pares, "metricas_dias_pareados": pd.concat(stats_pares, ignore_index=True),
        "frecuencias_dias_pareados": pd.concat(frecuencias_pares, ignore_index=True),
        "factibilidad": pd.concat(factibilidades, ignore_index=True),
        "metricas_dias_comunes": pd.concat(stats_globales, ignore_index=True),
        "frecuencias_dias_comunes": pd.concat(frecuencias_globales, ignore_index=True),
        "comparacion_sensibilidad": pd.DataFrame(comparacion)}


def crear_resumen(tablas, fechas, total_disponible):
    t = tablas["comparacion_sensibilidad"]
    n = int(t["dias_comunes_todos_los_casos"].iloc[0])
    cols = ["caso", "capacidad_kwh", "cf_um_kwh", "ce_um_kwh",
            "delta_criterio_media_um", "reduccion_criterio_media_pct",
            "delta_red_media_um", "reduccion_red_media_pct"]
    texto = (f"MOC - Codigo 12: sensibilidad EXPLORATORIA; {len(fechas)}/{total_disponible} dias.\n"
        f"Periodo ejecutado: {fechas[0]} a {fechas[-1]}; siete casos, un factor por vez.\n"
        "Mismo banco y escenarios diarios; misma factibilidad sin/con UCF dentro de cada caso.\n"
        "Cada dia y capacidad reinicia el SOC inicial y exige el SOC final del 02.\n\n"
        "FACTIBILIDAD - todos los intentos, sin excluir fallos\n"
        + tablas["factibilidad"].to_string(index=False)
        + f"\n\nDias comunes a TODOS los casos y ambos planes: {n}/{len(fechas)}.\n"
        "DIFERENCIAS CON UCF MENOS SIN UCF, dentro de cada caso\n"
        "Delta negativo y reduccion porcentual positiva indican menor costo con UCF.\n"
        + t[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}")
        + "\n\nCAPACIDAD - gasto neto de red, mismos dias comunes\n"
        + t.loc[t["factor"].isin(["base", "capacidad"]), ["caso", "capacidad_kwh",
            "red_media_sin_ucf_um", "red_media_con_ucf_um", "red_p95_sin_ucf_um", "red_p95_con_ucf_um"]]
            .sort_values("capacidad_kwh").to_string(index=False, float_format=lambda v: f"{v:.2f}")
        + "\n\nCriterio realizado = costo programado + penalizacion del dia; no es factura.\n"
        "Gasto de red: compras simuladas menos ventas; indicador separado, no se suman.\n"
        "Al cambiar cf/ce cambia la regla del criterio: no ordenar casos por su criterio absoluto.\n"
        "Los porcentajes comparan sin/con UCF bajo la MISMA regla dentro de cada caso.\n"
        "El gasto de red mantiene los mismos precios entre casos; no incluye inversion ni degradacion.\n"
        "Mayor capacidad mantiene fija la potencia en kW y las fracciones de SOC; no es dimensionamiento optimo.\n"
        "P95: inversa de la distribucion empirica; no garantia futura ni prueba de significancia.\n"
        "Tambien se guardan resumenes sobre los dias pareados de cada caso y todos los fallos.\n"
        "Los dias del 11 ya fueron vistos: no seleccionar parametros y declarar validacion nueva con estos mismos dias.\n"
        "FV estimada; demanda, precios y coeficientes ficticios; un lugar y tramo de un anio.\n")
    if len(fechas) != total_disponible:
        texto += "EJECUCION PARCIAL: no representa los 109 dias completos del ensayo original.\n"
    if not n:
        texto += "SIN DIAS COMUNES GLOBALES: no se interpreta una comparacion economica entre todos los casos.\n"
    return texto


def graficar(comparacion, carpeta):
    if not comparacion["dias_comunes_todos_los_casos"].iloc[0]:
        return []
    n = int(comparacion["dias_comunes_todos_los_casos"].iloc[0])
    color_criterio, color_red = "#2455a4", "#18856d"
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    fig.canvas.manager.set_window_title("MOC 12 - Sensibilidad a las penalizaciones")
    fig.suptitle(f"Sensibilidad exploratoria: diferencias con UCF frente a sin UCF\n"
                 f"{n} dias comunes; cada punto usa su propia regla de penalizacion", fontsize=12)
    for fila, factor, col, nombre in ((0, "faltante", "cf_um_kwh", "Penalizacion de faltante cf"),
                                      (1, "excedente", "ce_um_kwh", "Penalizacion de excedente ce")):
        d = comparacion.loc[comparacion["factor"].isin(["base", factor])].sort_values(col)
        for ax, metrica, color, titulo, ylabel in (
            (axes[fila, 0], "reduccion_criterio_media_pct", color_criterio,
             "Reduccion media del criterio realizado", "Reduccion (%)"),
            (axes[fila, 1], "delta_red_media_um", color_red,
             "Cambio medio del gasto neto de red", "Con UCF menos sin UCF (u.m./dia)")):
            ax.plot(d[col], d[metrica], "o-", color=color, linewidth=1.8)
            b = d.loc[d["caso"] == "base"].iloc[0]
            ax.scatter([b[col]], [b[metrica]], s=90, facecolors="none", edgecolors="black", label="Caso base", zorder=3)
            ax.axhline(0, color="gray", linewidth=.8)
            ax.set(title=titulo, xlabel=f"{nombre} (u.m./kWh)", ylabel=ylabel)
            ax.set_xticks(d[col]);ax.grid(alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=8)
    fig.savefig(carpeta / "sensibilidad_penalizaciones.png", dpi=160)

    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    fig2.canvas.manager.set_window_title("MOC 12 - Sensibilidad a capacidad de bateria")
    fig2.suptitle(f"Capacidad de bateria: mismo cf/ce y precios, {n} dias comunes\n"
                  "Potencia fija; no incluye costo de inversion ni degradacion", fontsize=12)
    d = comparacion.loc[comparacion["factor"].isin(["base", "capacidad"])].sort_values("capacidad_kwh")
    for ax, medida, titulo in zip(axes2, ["media", "p95"], ["Media del gasto neto de red", "P95 del gasto neto de red"]):
        for plan, color, label in (("sin_ucf", "#dc8b15", "Sin UCF en objetivo"),
                                   ("con_ucf", "#18856d", "Con UCF en objetivo")):
            ax.plot(d["capacidad_kwh"], d[f"red_{medida}_{plan}_um"], "o-", color=color, label=label)
        ax.set(title=titulo, xlabel="Capacidad nominal (kWh)", ylabel="Gasto simulado (u.m./dia)")
        ax.set_xticks(d["capacidad_kwh"]);ax.grid(alpha=.2);ax.set_axisbelow(True)
    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    fig2.savefig(carpeta / "sensibilidad_capacidad_bateria.png", dpi=160)
    return [fig, fig2]


def main():
    inicio = perf_counter()
    m11 = cargar_11()
    m10 = m11.cargar_10()
    base = m10.cargar_modulo("moc02_para_12", "02_microrred_24_horas.py")
    historico = m10.cargar_modulo("moc09_para_12", "09_calibracion_ucf_historica.py")
    plantilla, p_base, banco, prueba, meta = m11.preparar_periodo(base, historico)
    casos = definir_casos(p_base, historico)
    disponibles = sorted(prueba["fecha"].unique())
    if MAX_DIAS is not None and (isinstance(MAX_DIAS, bool) or not isinstance(MAX_DIAS, int) or MAX_DIAS < 1):
        raise ValueError("MAX_DIAS debe ser None o un entero positivo.")
    fechas = disponibles if MAX_DIAS is None else disponibles[:MAX_DIAS]
    capacidades = sorted(casos["capacidad_kwh"].unique())
    milp_por_dia = len(capacidades) + len(casos)
    SALIDA.mkdir(parents=True, exist_ok=True)
    casos.to_csv(SALIDA / "casos_sensibilidad.csv", index=False, encoding="utf-8-sig")
    banco.to_csv(SALIDA / "errores_calibracion_utilizados.csv", encoding="utf-8-sig")
    archivos = ("02_microrred_24_horas.py", "09_calibracion_ucf_historica.py", "10_milp_ucf_historica.py",
        "11_validacion_historica_ems.py", "12_sensibilidad_ems_ucf.py", "resultados_08/pronosticos_calibracion.csv",
        "resultados_08/pronosticos_prueba.csv", "resultados_08/metadatos_datos_historicos.json")
    config = {"tipo_estudio": "Sensibilidad exploratoria de un factor por vez; sin seleccion automatica",
        "dias_del_11_ya_observados": True, "dias_disponibles": len(disponibles), "dias_ejecutar": len(fechas),
        "ensayo_parcial": len(fechas) != len(disponibles), "inicio": fechas[0], "fin": fechas[-1],
        "dias_calibracion": len(banco), "inicio_calibracion": str(banco.index[0]), "fin_calibracion": str(banco.index[-1]),
        "escenarios_por_dia": len(banco), "banco_fijo_solo_calibracion": True,
        "parametros_base": p_base, "casos": casos.to_dict("records"),
        "potencia_bateria_fija_kw": [p_base["potencia_carga_kw"], p_base["potencia_descarga_kw"]],
        "soc": "Mismas fracciones de SOC; kWh inicial/final proporcionales a capacidad. Dias independientes.",
        "cache_sin_ucf": "Solo por fecha y capacidad; se recalculan UCF/penalizacion de cada caso",
        "milp_por_dia": milp_por_dia, "milp_totales_previstos": len(fechas) * milp_por_dia,
        "resumen_global": "Interseccion de dias optimos y red factible para ambos planes de TODOS los casos",
        "resumen_pareado": "Tambien se informan dias comunes a los dos planes de cada caso",
        "delta": "con UCF menos sin UCF", "metodo_p95": "inverted_cdf",
        "pronostico": meta["pronostico"], "disponibilidad_datos": meta["disponibilidad_datos"],
        "referencia_fv": meta["referencia"], "sha256_pvgis": meta["sha256_respuesta"],
        "sha256_archivos": {n: m10.huella(CARPETA / n) for n in archivos},
        "versiones": {n: version(n) for n in ("numpy", "pandas", "pyomo", "highspy", "matplotlib")},
        "limitaciones": ["No cambia demanda, precios, potencia, eficiencia ni banco historico",
            "No estudia interacciones entre factores ni identifica un optimo",
            "Criterios absolutos no comparables bajo distintas penalizaciones",
            "Sin inversion, degradacion, potencia reactiva, vertido ni limite AC del inversor",
            "Posibles soluciones multiples con igual objetivo; no es prueba de significancia",
            "Una ubicacion, un anio, FV estimada y precios/demanda ficticios"]}
    (SALIDA / "configuracion_sensibilidad.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    def guardar_estado(dias, finalizado):
        estado = {"dias_objetivo": len(fechas), "dias_completados": dias,
            "barrido_completo": finalizado, "ensayo_parcial": len(fechas) != len(disponibles),
            "milp_resueltos_o_intentados": dias * milp_por_dia, "segundos": perf_counter() - inicio}
        (SALIDA / "estado_ejecucion.json").write_text(json.dumps(estado, indent=2), encoding="utf-8")
    guardar_estado(0, False)
    print(f"MOC - Codigo 12: {len(casos)} casos; {len(fechas)} dias; {milp_por_dia} MILP por dia.", flush=True)
    print(f"Total previsto: {len(fechas) * milp_por_dia} MILP. Puede demorar varias decenas de minutos.", flush=True)
    print("Sensibilidad EXPLORATORIA sobre dias ya observados. No se selecciona un mejor caso.", flush=True)
    print(casos.to_string(index=False), flush=True)
    filas, verificaciones = [], []
    ruta_horarios = SALIDA / "planes_horarios.csv"
    # Solo se reinicia la salida horaria de ESTE programa; no se tocan datos de entrada.
    ruta_horarios.write_text("", encoding="utf-8-sig")
    for i, fecha in enumerate(fechas, start=1):
        dia = prueba.loc[prueba["fecha"] == fecha].copy().reset_index(drop=True)
        altura = historico.leer_geometria_ejemplo(dia, meta)
        datos = plantilla.copy()
        datos["solar_disponible_kw"] = dia["solar_pronostico_kw"].to_numpy()
        escenarios = historico.construir_escenarios(datos["solar_disponible_kw"].to_numpy(), banco.to_numpy(), altura)
        sin_ucf, con_ucf, estados_sin, estados_con = {}, {}, {}, {}
        for capacidad in capacidades:
            p = dict(p_base, capacidad_kwh=float(capacidad))
            sin_ucf[capacidad], estados_sin[capacidad] = resolver_o_registrar(
                m10, base, historico, datos, p, escenarios, False)
        for caso in casos.to_dict("records"):
            p = dict(p_base, capacidad_kwh=float(caso["capacidad_kwh"]))
            h = coeficientes_caso(historico, caso)
            con_ucf[caso["caso"]], estados_con[caso["caso"]] = resolver_o_registrar(
                m10, base, h, datos, p, escenarios, True)
        # Ningun optimizador anterior recibio solar_referencia_kw del dia.
        horarios_dia = []
        for caso in casos.to_dict("records"):
            cap, cid = caso["capacidad_kwh"], caso["caso"]
            p = dict(p_base, capacidad_kwh=float(cap))
            planes = {"sin_ucf": sin_ucf[cap], "con_ucf": con_ucf[cid], "escenarios": escenarios}
            estados = {"sin_ucf": estados_sin[cap], "con_ucf": estados_con[cid]}
            f, h, v = evaluar_caso(m10, historico, planes, estados, dia["solar_referencia_kw"].to_numpy(), p, caso, fecha, m11.METRICAS)
            filas.extend(f);horarios_dia.extend(h);verificaciones.extend(v)
        pd.DataFrame(filas).to_csv(SALIDA / "resultados_diarios.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(verificaciones).to_csv(SALIDA / "verificaciones_planes.csv", index=False, encoding="utf-8-sig")
        if horarios_dia:
            pd.concat(horarios_dia, ignore_index=True).to_csv(ruta_horarios,
                mode="a", header=ruta_horarios.stat().st_size <= 3, index=False, encoding="utf-8")
        guardar_estado(i, False)
        if i == 1 or i % 5 == 0 or i == len(fechas):
            print(f"Dias completados: {i}/{len(fechas)}; ultimo {fecha}; {perf_counter() - inicio:.1f} s.", flush=True)
            gc.collect()
    tablas = resumir_barrido(m11, pd.DataFrame(filas), casos)
    for nombre, tabla in tablas.items():
        tabla.to_csv(SALIDA / f"{nombre}.csv", index=False, encoding="utf-8-sig")
    texto = crear_resumen(tablas, fechas, len(disponibles))
    (SALIDA / "resumen_sensibilidad.txt").write_text(texto, encoding="utf-8")
    figuras = graficar(tablas["comparacion_sensibilidad"], SALIDA)
    guardar_estado(len(fechas), True)
    print("\n" + texto)
    print(f"Resultados guardados en: {SALIDA}")
    if MOSTRAR_GRAFICOS and figuras:
        print("Cerra las dos ventanas de graficos para finalizar.")
        plt.show()
    else:
        for fig in figuras:plt.close(fig)


if __name__ == "__main__":
    main()
