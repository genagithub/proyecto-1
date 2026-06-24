### 📊 Predicción de Rotación de Personal y Fuga de Talento en RRHH

#### 🎯 El Problema de Negocio
La pérdida de empleados genera altos costos de contratación y frena la productividad. La empresa necesita identificar qué sectores y perfiles tienen mayor riesgo de abandonar la organización para activar planes de retención antes de que renuncien.

---

#### 💡 Hallazgos Clave de la Investigación (¿Por qué se va el talento?)
A pesar de lo que se suele pensar, el salario no es el motivo de la renuncia en esta organización. Las medias de pago y experiencia son idénticas entre los empleados que se quedan y los que se van. Los verdaderos puntos críticos son:
- **Fuga de Talento Especializado:** El 48% de los empleados con Maestría (Master) renuncia. Esto duplica la baja de los perfiles con Doctorado (PHD, 25%). Falta una ruta de crecimiento para ellos.
- **Alerta de Género:** El riesgo de salida en mujeres (47%) casi duplica al de los hombres (26%). Existe un problema urgente en el clima laboral o el balance vida-trabajo.
- **Foco Geográfico:** Las alertas rojas de deserción se concentran críticamente en las sedes de Pune y New Delhi.
- **Anomalía Histórica:** Los contratados en el año 2018 presentan un 98% de deserción. Esto señala un fallo grave en la selección o inducción de ese grupo específico.

---

#### 🛠️ Enfoque Técnico y Modelado
Para resolver esto, se construyó un modelo predictivo basado en un Árbol de Decisión (CART). Este algoritmo permite mapear los nodos y reglas exactas que determinan la salida de un empleado.

#### 📈 Rendimiento del Modelo
El sistema cuenta con un enfoque de alta confianza para evitar falsas alarmas:
- **Precisión(Precision):** 85% ── Cuando el modelo alerta sobre un empleado, la probabilidad de acierto es casi total.
- **Exactitud(Accuracy):** 83% ── El rendimiento general del clasificador es sólido y equilibrado.
- **Balance (F1-Score):** 75% ── Valida que el modelo es confiable para la toma de decisiones.

⚠️ **Nota Técnica sobre el Sesgo:** El modelo tiene un Recall de 67%. Esto ocurre porque variables numéricas como la Edad o el Pago son idénticas en ambos grupos ("invisibles" para el algoritmo). Por ende, el modelo se apoya en patrones categóricos y es conservador: prioriza la precisión sobre la cobertura total.

---

#### 🚀 Recomendaciones Estratégicas
1. **Frenar aumentos lineales:** El dinero no retiene a este equipo.
2. **Plan de carrera "Master":** Crear incentivos de crecimiento post-maestría.
3. **Auditoría de Género:** Revisar las políticas de equidad y cultura interna para retener al talento femenino.
