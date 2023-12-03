# Instalações
    python -m venv mlflow-env
    .\mlflow-env\Scripts\activate
    pip install mlflow
    pip install scikit-learn pandas numpy
    pip install onnx onnxruntime skl2onnx

# O codig presente em guide.py

1. Treina um Modelo de Regressão Logística com o dataset Iris.
2. Converte o Modelo Treinado para ONNX e o salva como um arquivo.
3. Registra o Modelo ONNX no MLflow.
4. Exporta em txt o arquivo.