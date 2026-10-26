# -*- coding: utf-8 -*-
"""Primer código de MILP.

Caso de ejemplo: tres intervalos de una hora, PV, bateria y compra a red.
Los datos y precios son ficticios. Se supone eficiencia de bateria del
100 %, SOC entre 0 y 100 %, sin venta a red ni incertidumbre. Estos
supuestos se ampliaran en el modelo de investigacion.

Las potencias se expresan en kW, la energia en kWh y los precios en
unidades monetarias por kWh. La energia inicial y final es cero.
"""

import pyomo.environ as pyo


def crear_modelo():
    # 1. DATOS: estas cantidades ya son conocidas por el EMS.
    demanda = [1.0, 1.0, 2.0]
    solar = [0.0, 2.0, 0.0]
    precio = [60.0, 20.0, 50.0]    
    paso_h = 1.0
    capacidad_kwh = 2.0
    potencia_max_kw = 2.0

    m = pyo.ConcreteModel()
    m.horas = pyo.RangeSet(0, 2)
    m.instantes = pyo.RangeSet(0, 3)

    # 2. VARIABLES: el solver elegira sus valores.
    m.red = pyo.Var(m.horas, domain=pyo.NonNegativeReals)
    m.pv = pyo.Var(m.horas, bounds=lambda m, t: (0, solar[t]))
    m.carga = pyo.Var(m.horas, bounds=(0, potencia_max_kw))
    m.descarga = pyo.Var(m.horas, bounds=(0, potencia_max_kw))
    m.energia = pyo.Var(m.instantes, bounds=(0, capacidad_kwh))
    m.modo_carga = pyo.Var(m.horas, domain=pyo.Binary)

    # 3. RESTRICCIONES: suministro, almacenamiento y modo de bateria.
    m.balance = pyo.Constraint(
        m.horas,
        rule=lambda m, t: m.red[t] + m.pv[t] + m.descarga[t]
        == demanda[t] + m.carga[t],
    )
    m.evolucion = pyo.Constraint(
        m.horas,
        rule=lambda m, t: m.energia[t + 1]
        == m.energia[t] + (m.carga[t] - m.descarga[t]) * paso_h,
    )
    m.limite_carga = pyo.Constraint(
        m.horas,
        rule=lambda m, t: m.carga[t] <= potencia_max_kw * m.modo_carga[t],
    )
    m.limite_descarga = pyo.Constraint(
        m.horas,
        rule=lambda m, t: m.descarga[t]
        <= potencia_max_kw * (1 - m.modo_carga[t]),
    )
    m.energia[0].fix(0)
    m.energia[3].fix(0)

    # 4. OBJETIVO: minimizar el costo total de comprar energia.
    m.costo = pyo.Objective(
        expr=sum(precio[t] * m.red[t] * paso_h for t in m.horas),
        sense=pyo.minimize,
    )
    return m, demanda, solar, precio, paso_h


def main():
    m, demanda, solar, precio, paso_h = crear_modelo()

    # 5. RESOLUCION: HiGHS calcula el despacho optimo.
    solver = pyo.SolverFactory("appsi_highs")
    resultado = solver.solve(m)
    if not pyo.check_optimal_termination(resultado):
        raise RuntimeError("El solver no termino con una solucion optima.")

    print("Estado: solucion optima")
    print("Potencias en kW; energia al final de cada intervalo en kWh.")
    print("Hora  Demanda  PV usada  Red  Carga  Descarga  Energia final")
    for t in m.horas:
        print(
            f"{t + 1:4d}  {demanda[t]:7.2f}  {pyo.value(m.pv[t]):8.2f}"
            f"  {pyo.value(m.red[t]):3.2f}  {pyo.value(m.carga[t]):5.2f}"
            f"  {pyo.value(m.descarga[t]):8.2f}"
            f"  {pyo.value(m.energia[t + 1]):13.2f}"
        )

    costo_sin_bateria = sum(
        precio[t] * max(demanda[t] - solar[t], 0) * paso_h for t in m.horas
    )
    print(f"\nCosto con bateria: {pyo.value(m.costo):.2f} unidades monetarias")
    print(f"Costo sin bateria: {costo_sin_bateria:.2f} unidades monetarias")
    print("\nEstos resultados corresponden solo al caso ficticio del ejercicio.")


if __name__ == "__main__":
    main()
