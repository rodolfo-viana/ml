import numpy as np
from typing import Union, List, Optional


class Perceptron:
    """
    Implementação básica do perceptron com número arbitrário de entradas, dado:

    f(x) = 1 se w·x + b ≥ 0
           0 caso contrário

    onde:
    - w é o vetor de pesos
    - x é o vetor de entrada
    - b é o viés (bias)
    - · representa o produto escalar
    """

    def __init__(
        self,
        num_inputs: int = 2,
        learning_rate: float = 0.1,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Parâmetros:
            num_inputs: Número de features de entrada.
            learning_rate: Taxa de aprendizado para atualização dos pesos.
                           Determina o tamanho do passo durante o gradiente descendente.
                           Fórmula: w_atual = w_antigo + learning_rate * error * x
            random_seed: Opcional, garante reprodutibilidade quando definido.
        """
        self.num_inputs: int = num_inputs
        self.learning_rate: float = learning_rate
        if random_seed is not None:
            np.random.seed(random_seed)
        self.weights: np.ndarray = np.random.randn(num_inputs) * 0.01
        self.bias: float = 0.0

    @staticmethod
    def step_activation(z: float) -> int:
        """
        Função degrau que retorna 1 se z >= 0, caso contrário 0.

        Esta é a função de ativação clássica, definida como:

        φ(z) = 1 se z ≥ 0
               0 se z < 0

        Parâmetros:
            z: Valor de entrada (geralmente o produto escalar w·x + b)
        """
        return 1 if z >= 0 else 0

    def predict(self, X: Union[List[float], np.ndarray]) -> Union[int, np.ndarray]:
        """
        Prediz a saída para uma única amostra ou um lote de amostras.

        O processo de previsão segue estas etapas:
        1. Calcula z = w·x + b para cada amostra
        2. Aplica a função degrau: y_pred = φ(z)

        Parâmetros:
            X: Se for um array 1D de formato (num_inputs,), retorna uma única previsão.
               Se for um array 2D de formato (num_samples, num_inputs), retorna um array de previsões.
        """
        X = np.array(X)
        single_sample = X.ndim == 1
        if single_sample:
            X = X.reshape(1, -1)

        z = X.dot(self.weights) + self.bias
        y_pred = np.array([self.step_activation(val) for val in z])

        return y_pred if not single_sample else y_pred[0]

    def fit(
        self,
        X: Union[List[List[float]], np.ndarray],
        y: Union[List[int], np.ndarray],
        epochs: int = 10,
    ) -> List[int]:
        """
        Treina o modelo conforme a regra de aprendizado do perceptron, definida como:

        Para cada amostra (x_i, y_i):
            1. Calcula a previsão: y_pred = φ(w·x_i + b)
            2. Calcula o erro: error = y_i - y_pred
            3. Atualiza os pesos: w = w + learning_rate * error * x_i
            4. Atualiza o viés: b = b + learning_rate * error

        Parâmetros:
            X: Array 2D de formato (num_samples, num_inputs)
            y: Array 1D de formato (num_samples,) contendo rótulos-alvo {0,1}
            epochs: Épocas; número de passagens pelo conjunto de dados.
        """
        X = np.array(X)
        y = np.array(y)
        losses: List[int] = []

        for epoch in range(epochs):
            for i in range(len(X)):
                x_i = X[i]
                target = y[i]

                pred = self.predict(x_i)
                error = target - pred
                if error != 0:
                    self.weights += self.learning_rate * error * x_i
                    self.bias += self.learning_rate * error

            preds = self.predict(X)
            epoch_loss = np.sum((y - preds) ** 2)
            losses.append(int(epoch_loss))

            print(
                f"Época {epoch+1}/{epochs}, Loss (classificações incorretas) = {epoch_loss}"
            )

        return losses

    def score(
        self, X: Union[List[List[float]], np.ndarray], y: Union[List[int], np.ndarray]
    ) -> float:
        """
        Calcula a acurácia em um determinado conjunto de dados, sendo:

        acurácia = número_de_previsões_corretas / número_total_de_amostras

        Parâmetros:
            X: Dados de entrada
            y: Rótulos verdadeiros
        """
        preds = self.predict(X)
        correct = np.sum(preds == y)
        return float(correct / len(y))

    def __repr__(self) -> str:
        """
        Retorna uma representação em string do perceptron.
        """
        return f"Perceptron(weights={self.weights}, bias={self.bias})"
