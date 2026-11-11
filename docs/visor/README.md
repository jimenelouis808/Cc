# Visor 3D de las celdas

`celdas_3d.html` carga los ensambles de `modules/assemblies.py` (GLB en base64 + manifiesto) y
muestra cada celda con sus electrodos: piezas por separado, cuerpos transparentes, plano de corte.

```bash
python modules/assemblies.py              # escribe out/ensamble_<celda>.{step,glb,glb.txt,json}
cp docs/visor/celdas_3d.html out/
cd out && python -m http.server 8000      # abrir http://localhost:8000/celdas_3d.html
```

El archivo es el cuerpo de la página (sin `<html>` ni `<head>`): los navegadores lo abren igual.
Necesita internet para three.js (cdnjs / jsDelivr).

Las figuras 2D (`docs/figuras/`) salen de `python docs/figuras.py --vistas3d <carpeta>`, donde la
carpeta tiene las capturas `stage_<celda>.png` del visor; sin `--vistas3d` se omite el panel 3D.
