import datetime
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import mlflow.onnx

class MLExperiment:
    def __init__(self, registered_name):
        self.registered_name = registered_name
        self.experiment_id = self.create_experiment()

    @staticmethod
    def log_timestamp():
        now = datetime.datetime.now()
        return now.strftime("%Y%m%d%H%M%S")

    def create_experiment(self):
        timestamp = MLExperiment.log_timestamp()
        return mlflow.create_experiment(timestamp)

    def train_model(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={{cookiecutter.test_size}}, random_state={{cookiecutter.random_state}})
        model = LogisticRegression(max_iter={{cookiecutter.max_iter}})
        model.fit(X_train, y_train)
        return model, X_test, y_test

    def convert_and_log_model(self, model, X_test, y_test):
        initial_type = [('float_input', FloatTensorType([None, X_test.shape[1]]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        with open("{{cookiecutter.model_filename}}.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())

        with mlflow.start_run(experiment_id=self.experiment_id):
            mlflow.autolog()
            accuracy = model.score(X_test, y_test)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.onnx.log_model(onnx_model, "model", registered_model_name=self.registered_name)

class MLFlowDataExporter:
    @staticmethod
    def export_mlflow_results_to_txt(experiment_id, output_file):
        runs = mlflow.search_runs(experiment_id)

        if runs.empty:
            print("Nenhuma execução encontrada para o experimento.")
            return

        last_run = runs.iloc[0]
        run_id = last_run['run_id']

        with open(output_file, 'w') as file:
            file.write(f"Resultados do Modelo - Run ID: {run_id}\n")
            
            file.write("Metricas:\n")
            for col in runs.columns:
                if col.startswith('metrics.'):
                    metric_name = col.split('.', 1)[1]
                    metric_value = last_run[col]
                    file.write(f"{metric_name}: {metric_value}\n")

            file.write("\nParametros:\n")
            for col in runs.columns:
                if col.startswith('params.'):
                    param_name = col.split('.', 1)[1]
                    param_value = last_run[col]
                    file.write(f"{param_name}: {param_value}\n")

        print(f"Arquivo '{output_file}' criado com sucesso.")

# Utilização das classes
experiment = MLExperiment("{{cookiecutter.registered_name}}")
model, X_test, y_test = experiment.train_model()
experiment.convert_and_log_model(model, X_test, y_test)

output_file = '{{cookiecutter.results_filename}}.txt'
MLFlowDataExporter.export_mlflow_results_to_txt(experiment.experiment_id, output_file)
