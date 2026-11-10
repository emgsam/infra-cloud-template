"""Ciclo de vida del modelo de ventas perdidas (Trabajo Final, Curso 2).

Se separa en tres módulos para que la API pueda reusar lo mismo que el trainer:
  datos.py   -> carga del extracto y armado de la tabla de compras cliente-mes
  modelo.py  -> entrenamiento, líneas base, evaluación y predicción
  almacen.py -> lectura y escritura en una carpeta local o en gs://bucket/...
"""
