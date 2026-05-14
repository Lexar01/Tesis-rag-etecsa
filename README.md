PASOS PARA INSTALAR EL PROYECTO:

1- Usar el backup "Tesis.nb3" para crear la base de datos en postgres, no es mas que crear una base de datos nueva, pegar el backup, ejecutarlo y entonces la BD tendra las tablas y atributos respectivos necesarios para el Proyecto.

2-Ir con vpn a la pagina "Ollama.com" y descargar ollama para windows, para el Proyecto siempre debe estar ejecutandose, igual que para el siguiente paso.

3-Luego de instalar Ollama, abrir una consola cmd (yo instale el modelo "llama3.2" en la pc pq lo quiero usar en otros proyectos, pero se puede instalar Tambien en un entorno virtual) y escriben si quieren Ollama para que sepan que lo tiene instalado deberian salirles un menu con descripciones, pero mas importante en la consola deben escribrir "ollama run llama3.2" y entonces se instalara "llama3.2", que fue el modelo generative usado en el Proyecto y usara los embeddings y la informacion correspondiente para generar respuestas.

NOTA: Es el LLM q yo use para que no se compliquen buscando otro, pero si asi lo desean pueden hacerlo y solo tiene q ir a "AgenteAI.py" y cambiar el nombre del modelo que esta por el que ustedes deseen en la variable "model" (buscar con el buscador), ahi se importa el tipo de LLM y le pasas por parametron el nombre de tu modelo.

4- Primero ejecutar en consola de visual code "pip install --upgrade pip setuptools wheel" que es necesaria para instalar las dependencias.

5- Luego deberan instalar las dependencias de requirements.txt.

NOTA: Un entorno virual de python donde instalarlas seria muy util

6- En los scripts "Tools.py" y "AgenteAI.py" deben modificar todos los parametros de los cursores que accederan a la BD con los datos de la BD en la que usaran el backup usuario, contraseña, nombre de la BD, etc (basta que en ambos scripts abran el buscador y escriban "psycopg2.connect" y modificar dentro de los parentesis, deberian ser unos 5, 1 dentro de "Tools.py" y 4 dentro "AgenteAI.py").

7- En la carpeta Storage que es donde se almacenan el contenido con que interactua el Proyecto esta:

Archivos json de los codigos: Aqui se guardan los archivos json que generen o que necesite el Proyecto.

Dataset para convertir en vectores(embeddings): Aqui se pegan los archivos que se quieran almacenar el BD para uso permanente del Proyecto, su contenido se convierte en embedding y se inserta en la bd para su posterior uso ( luego se puede borrar de la carpeta pues se queda almacenado en la BD).

Dataset para entrenar embeddings: Aqui se almacena el corpus que sera usado para entrenar el modelo generador de embeddings, el modelo a entrenar solo acepta pdf por temas de cautela, ademas una vez se logra entrenar no se necesita volverlo hacer a menos de ser necesario (correcciones o mejoras).

Modelos Generados: Aqui se almacenan los modelos generados por el entrenamiento, en este caso solo el del embedding, aunque se almacena en caso de necesidad y uso, pues el proceso importante es toma los pesos del modelo y procesarlos en un archivo que es almacenado en archivos jso de los codigos.


"""NOTA IMPORTANTE"""
Si desean ejecutar el script "Modelo notebook.ipynb" por favor tener cuidado con las variables y el tamaño del corpus, pues todo fue cuidadosamente elegido para entrenar el modelo, un simple cambio es capaz de provocar que las similitudes de cosenos se alinen y tengan un valor casi identico y eso seria fatal, por seguridad hacer una copia de seguridad del archivo "embeddings_final.npy" almacenado en "archivos json de los codigos"

Ejemplo mas simple: 

Con determinados valores de variables, y un corpus de 10k tokens se obtuvieron resultados satisfactorios

Con los mismo valores de variables pero con un corpus de 10M, que es lo que deberia usarse para entrenar un embedding bastante satisfactorio, se alinearon las variables en muchas ocasiones e incluso los resultados no alineados no solian ser satisfactorios, fue necesario aumentar parejas de contextos, parejas negativas, disminuir frecuencia de entrenamiento, filtrado de palabras de pausa(preposiciones, conjuciones, entre otras, por el ruido que provocaban), etc

Recomendacion: Cuidado al modificar las variables del codigo, guardar el archivo de embedding que mencione antes que ese es el principal que usa el Proyecto para responder y el Proyecto ahora mismo esta adaptado a un corpus de 10M de puro lenguaje tecnico de puras asignaturas de Universidad ( es solo copiar pdfs a la carpeta "Dataset para entrenar embeddings" y el solo los transforma para su uso en el entrenamiento). Asi que por favor haganlo teniendo en cuenta los riesgos comentados y el tiempo necesario (que no es poco y puede extenderse dependiendo de la configuracion y tipo de corpus), o de lo contrario no usar el script.


COMO INICIAR EL PROYECTO:

Desde el visual code:

1- Abrir 2 consolas

2- En ambas escribir: cd NombreDisco:\...\Tesis completada\src, basicamente direccionar a donde se encuentran los scripts

3- Luego en 1 de las 2 escribir: uvicorn app:app --reload y python -m http.server 5500 en la otra

4- Abrir en navegador http://127.0.0.1:5500/index.html


OPCIONES:
El proyecto es capaz de:

1-En dependencia de la pregunta generar una respuesta usando de la BD los embeddings almacenado y el contenido de los pdf, busca similitud y la mas parecida la responde, pero añadi como restriccion que si no hay suficiente informacion o la informacion tiene cierto nivel de error, ya sea incoherencia, redaccion, o alguna otro motivo, no responda y envie una respuesta adecuada a la situacion.

2- Realiza una auditoria que la almacena de forma local y en la BD en dependencia de lo respondido, brindado infromacion del suceso, puede ser de un error, puede ser de una respuesta adecuada.

3- Tiene un historial de preguntas realizadas, de forma que puedas saber que preguntas se hicieron y si alguna te es util y ver la respuesta del momento en que se hizo desde el historial, tambien puedes clickearla y esta se convierte en pregunta que puedes usar para que obtener una respuesta similar.

4- Permite hacer feedback de la respuesta, para saber si fue util o no y lo almacena en la ultima auditoria realizada

5- Permite añadir temporalmente texto o archivos, a diferencia del proceso estandar por carpeta que da permanencia, estos son temporales y son eliminados si recargas la pagina o cada vez que inicies o reinicies el proyecto.

NOTA: El proyecto para almacenar informacion si acepta doc y txt ademas de pdf, aunque tiene un proceso de filtrado para que no se acepten erroneos o innecesarios.

ejemplo: Un archivo con todas las dimensiones del embedding en 0 no tiene ninguna utilidad.

