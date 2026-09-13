"""Fases no carbonosas: por qué el escaneo va encendido por defecto.

    python -m ramancarbon.examples.ex11_fases
"""

from __future__ import annotations

from ramancarbon import analyse
from ramancarbon.examples.demo_data import make_demo


def main() -> None:
    print("Una muestra de nanotubos decorada con FeSe pone modos que NO son")
    print("carbono en 181, 196, 237 y 254 cm⁻¹. Los cuatro caen dentro de la")
    print("ventana del modo de respiración radial.\n")

    spectrum = make_demo("MWCNT_FeSe", laser_nm=532.0, seed=3)

    without = analyse(spectrum, auto_preprocess=True, check_phases=False)
    print("SIN el escaneo de fases:")
    print(f"  diámetros deducidos : "
          f"{[f'{d.diameter_nm:.2f} nm' for d in without.rbm.diameters]}")
    print(f"  clasificación       : {without.classification.label} "
          f"({without.classification.confidence})")
    print("  Los cuatro son creíbles. Los cuatro son falsos: no hay ni un\n"
          "  nanotubo de pared única en esta muestra.\n")

    with_scan = analyse(spectrum, auto_preprocess=True)
    print("CON el escaneo (lo que hace el programa por defecto):")
    print(f"  diámetros deducidos : "
          f"{[f'{d.diameter_nm:.2f} nm' for d in with_scan.rbm.diameters] or 'ninguno'}")
    print(f"  clasificación       : {with_scan.classification.label} "
          f"({with_scan.classification.confidence})\n")

    print(with_scan.phases.summary())
    print()
    print("Fíjate en la última sección: Raman PROPONE la fase, y dice qué")
    print("reflexión de difracción la demostraría. Esa es la pestaña de DRX.")

    print("\n" + "─" * 68)
    print("El otro caso: la anchura, no la posición, separa polimorfos.\n")
    amorphous = analyse(make_demo("MWCNT_Se", laser_nm=532.0, seed=4),
                        auto_preprocess=True)
    for identification in amorphous.phases.identifications:
        if identification.corroborated:
            print(f"  {identification}")
    print("\n  El selenio amorfo y el monoclínico están los dos hacia 250 cm⁻¹.")
    print("  Lo único que los distingue es el ancho de banda — y solo un")
    print("  límite INFERIOR de anchura sirve como prueba, porque «más")
    print("  estrecha de 15 cm⁻¹» lo cumple cualquier RBM.")


if __name__ == "__main__":
    main()
