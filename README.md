# Sistema de Gestión de Energía para Microrredes (EMS)

**Investigación en desarrollo | Python · Optimización MILP · Incertidumbre (UCF)**

## Descripción

Este repositorio reúne modelos y experimentos desarrollados en Python para estudiar la **gestión óptima de energía en microrredes** (*Energy Management Systems*, EMS).

El trabajo explora la programación del intercambio de energía con la red eléctrica, el uso de almacenamiento y el efecto de la incertidumbre en las decisiones de operación. Se estudia la **programación lineal entera mixta (MILP)** y la incorporación de **funciones de costo de incertidumbre (UCF, *Uncertainty Cost Functions*)** para evaluar decisiones bajo distintos escenarios.

El código tiene carácter **experimental y académico**: los modelos, supuestos y pruebas se encuentran en evolución.

## Objetivos

- Desarrollar y ampliar modelos de optimización para la operación de una microrred.
- Analizar el despacho de energía y el comportamiento de sistemas de almacenamiento.
- Estudiar el efecto de errores de pronóstico y distintos escenarios de operación.
- Incorporar penalizaciones esperadas por incertidumbre mediante UCF.
- Comparar alternativas de planificación mediante métricas de costo y resultados de simulación.
- Evaluar la sensibilidad y la capacidad de generalización de los modelos.

## Estructura del repositorio

| Archivo | Contenido |
| --- | --- |
| `01_microrred_3_horas.py` | Primer modelo de microrred con un horizonte de simulación de 3 horas. |
| `02_microrred_24_horas.py` | Ampliación del modelo a un horizonte de 24 horas. |
| `02_microrred_24_horas_extra.py` | Variante adicional del modelo de 24 horas; sus diferencias específicas están pendientes de documentar. |
| `03_analisis_sensibilidad.py` | Experimentos para analizar cómo cambian los resultados al modificar parámetros. |
| `04_error_pronostico.py` | Experimentos relacionados con los errores de pronóstico. |
| `05_primera_ucf.py` | Primeras pruebas de incorporación de funciones de costo de incertidumbre. |
| `06_milp_con_ucf.py` | Desarrollo del modelo de optimización MILP con UCF. |
| `07_validacion_nuevos_dias.py` | Pruebas de validación con nuevos días o conjuntos de datos. |

Las carpetas locales `resultados_02`, `resultados_03`, etc. pueden contener salidas de simulaciones, pero **no forman parte de la versión publicada inicialmente en GitHub**.

## Enfoque metodológico

Se parte de un modelo de programación de la operación de la microrred y se incorporan progresivamente herramientas para estudiar su comportamiento ante condiciones variables.

Una parte central del análisis consiste en comparar estrategias que optimizan el costo programado con otras que también consideran una penalización esperada por incertidumbre. Como criterio de evaluación puede utilizarse:

**Criterio total = costo programado + penalización esperada por incertidumbre.**

Esta suma es un **criterio de planificación del modelo**, no una factura eléctrica medida. Los valores económicos de los ejemplos son unidades monetarias de simulación, salvo que se indique expresamente lo contrario.

## Requisitos y ejecución

- Python instalado.
- Un entorno virtual de Python (recomendado).
- Las bibliotecas y el solucionador de optimización que utilice cada script. En los modelos MILP se trabaja con **HiGHS**.

Para clonar el repositorio:

```bash
git clone https://github.com/YohanaPereyra/microgrids-ems.git
cd microgrids-ems
```

Para crear y activar un entorno virtual en **Windows PowerShell**:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

A modo de punto de partida, pueden instalarse las siguientes dependencias habituales del proyecto:

```powershell
python -m pip install pyomo highspy numpy pandas matplotlib scipy
```

> **Nota:** esta lista es orientativa y debe verificarse contra las importaciones reales de todos los scripts antes de considerarla definitiva. Todavía no se incluye un archivo `requirements.txt` validado.

Para ejecutar, por ejemplo, el primer modelo:

```powershell
python 01_microrred_3_horas.py
```

La ejecución de otros scripts puede depender de sus parámetros, datos de entrada y rutas locales. Consultar cada archivo antes de utilizarlo.

## Estado y próximos pasos

El repositorio contiene una secuencia de experimentos que avanza desde modelos básicos de microrredes hasta análisis con incertidumbre y validación. Entre los próximos pasos se encuentran:

- Documentar los supuestos, variables y restricciones de cada modelo.
- Identificar las diferencias entre ambas versiones del modelo de 24 horas.
- Estandarizar los datos de entrada y los resultados.
- Validar y fijar las dependencias en `requirements.txt`.
- Ampliar las comparaciones entre enfoques de optimización.

## Autora

**Yohana Pereyra**  
Proyecto de investigación en gestión energética de microrredes.
