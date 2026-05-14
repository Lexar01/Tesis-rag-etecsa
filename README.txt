Pasos para instalar el Proyecto:

1- Usar el backup "Tesis.nb3" para crear la base de datos en postgres, no es mas q crear una base de datos nueva, pegar el backup, ejecutarlo y entonces la BD tendra las tablas y atributos respectivos necesarios para el Proyecto

2-Ir con vpn a la pagina " " y descargar ollama para windows, para el Proyecto siempre debe estar abierto igual q para el siguiente paso

3-Luego de instalar Ollama, en la misma pagina

4- Instalar primero la dependecia " " que es necesaria para instalar el resto de las dependencias

5- Luego deberan instalar las dependencias de requirements.txt

6- En los scripts "Tools.py" y "AgenteAI.py" deben modificar todos los parametros de los cursores q accederan a la BD con los datos de la BD en la q usaran el backup usuario, contraseña, nombre de la BD, etc (basta q en ambos scripts abran el buscador y escriban "psycopg2.connect" y modificar dentro de los parentesis, deberian ser unos 5, 1 dentro de "Tools.py" y 4 dentro "AgenteAI.py")