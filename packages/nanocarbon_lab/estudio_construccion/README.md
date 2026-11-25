# Construcción de las superestructuras

Informe completo: `informe_construccion.html`. Scripts, en orden:

- `auditoria.py <carpeta>`: las trece estructuras entregadas a carbonforge
  (`validacion_rendijas/{cristalinas,superestructuras,jaulas}`) auditadas con una
  sola regla de enlaces y anillos → `auditoria.json`.
- `prueba_rutas.py [fcc]`: intenta la ruta cristalina (nodos de rodilla pegados)
  en jaulas platónicas y en el nodo fcc de 12 brazos → `prueba_rutas.txt`.
- `barrido_perfectas.py`: formas exactas de Schwarz P y del súper-cuadrado,
  relajadas y juzgadas con la ventana sp² → `barrido_perfectas.jsonl`.
- `figuras_construccion.py <carpeta>`: guarda las cinco cristalinas limpias en
  `perfectas/` y dibuja todas con los defectos coloreados → `figuras/`.
- `escala_D.py`: celdas D de k = 18 con 5, 7 y 9 filas para el escalado del gap
  (lo calcula `tbkit/estudios/superredes/escala_D.py`).

Resultado: limpias D, giroide y súper-grafeno; P y súper-cuadrado limpios solo
con brazos cortos (k20 n7, 488 átomos; k12 n5, 180 átomos). Super-fcc y jaulas no
admiten la ruta cristalina.
