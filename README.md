# Amazonia.IA - Traductor de Lenguas Amazónicas

> **Preservando la riqueza lingüística de la Amazonía a través de la tecnología.**

##  Descripción

**Amazonia.IA** es una aplicación de código abierto diseñada para traducir entre español y lenguas indígenas de la región amazónica (Guahibo).

Este proyecto busca reducir las barreras de comunicación y apoyar la revitalización de lenguas en peligro de extinción mediante el uso de modelos de aprendizaje automático y colaboración comunitaria.

###  Características Principales

* **Reconocimiento de Voz (ASR):** Capacidad de traducir a partir de audio.
* **Modo Offline:** Funcionalidad básica sin conexión a internet para zonas remotas.
* **Colaboración Abierta:** Sistema para que hablantes nativos sugieran correcciones.

## Tecnologías Utilizadas

El proyecto está construido con una arquitectura moderna:

| Componente | Tecnología | Descripción |
| :--- | :--- | :--- |
| **Frontend** | |
| **Backend** | |
| **Base de Datos** |  |
| **IA / ML** |  |

## Instalación y Configuración

Sigue estos pasos para ejecutar el proyecto en tu entorno local.

### Pasos

1.  **Clonar el repositorio**
    ```bash
    git clone [https://github.com/jimenaazocar05/Amazonia.IA.git](https://github.com/jimenaazocar05/Amazonia.IA)
    cd nombre-del-proyecto
    ```

2.  **Configurar Variables de Entorno**
    Crea un archivo `.env` en la raíz del proyecto y añade las siguientes variables:
    ```env
    HF_TOKEN=tu_token_de_huggingface_aqui
    HF_REPO_ID=usuario/nombre-del-repo
    ```

3.  **Instalar dependencias**
    Ejecuta el siguiente comando para instalar las librerías necesarias (Backend y UI):
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configurar Modelo Guahibo**
    Para que el reconocimiento de voz funcione, necesitas el modelo entrenado:
    1.  Descarga el archivo del modelo (por ejemplo `modelo_guahibo_gpu.zip`) del siguiente link: https://drive.google.com/file/d/1LNPWc2YbqIFK6V4We28w7FhD5LsCJVS3/view?usp=sharing
    2.  Descomprime el archivo en la raíz del repositorio.
    3.  Asegúrate de que la carpeta resultante se llame **`modelo_guahibo_gpu`**.
    4.  Verifica que dentro existan archivos como `model.safetensors` y `config.json`.

5.  **Ejecutar la aplicación**
    Inicia el servidor local de Gradio:
    ```bash
    python app.py
    ```
    La aplicación se abrirá automáticamente en tu navegador (o visita `http://127.0.0.1:7860`).
    

## Licencia

Este proyecto está bajo la Licencia MIT - mira el archivo [LICENSE.md](LICENSE.md) para más detalles.
