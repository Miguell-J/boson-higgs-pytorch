# Higgs Boson Neural Network

![image](https://github.com/user-attachments/assets/24ef6a96-eb98-44a9-aec2-186c17a315e9)


## Descrição
Este projeto implementa uma rede neural para classificação de eventos relacionados ao bóson de Higgs, utilizando um conjunto de dados fornecido pelo CERN. O objetivo é identificar eventos "s" (sinal) e "b" (fundo) com alta precisão, utilizando técnicas modernas de aprendizado de máquina e análise interpretável dos modelos com SHAP.

## Fundamentos Matemáticos e Físicos

### Contexto Físico
O bóson de Higgs é uma partícula fundamental prevista pelo Modelo Padrão da física de partículas. Ele é responsável por conferir massa a outras partículas através do mecanismo de Higgs. Os eventos que indicam a presença do bóson de Higgs são raros e frequentemente misturados com ruídos e eventos de fundo (background), tornando a sua identificação um desafio estatístico e computacional.

### Modelagem Matemática
A tarefa de classificação pode ser descrita matematicamente como um problema de minimização de uma função de custo. Neste caso:
**Função de Custo:** Binary Cross-Entropy (BCE)
  -
 $\( \text{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right] \)$
 
 Onde
$\( y_i \)$ é o rótulo verdadeiro (0 ou 1) e $\( \hat{y}_i \)$ é a probabilidade prevista pelo modelo.
A saída da rede neural é uma probabilidade, gerada pela função sigmoide:
$\[ \sigma(z) = \frac{1}{1 + e^{-z}} \]$
que transforma a soma ponderada das entradas em valores entre 0 e 1.

### Arquitetura e Generalização
A arquitetura da rede neural é composta por camadas densamente conectadas, cada uma aplicando:
1. Uma transformação linear: $\( z = W \cdot x + b \)$
2. Uma função de ativação não linear (ReLU): $\( f(z) = \max(0, z) \)$
3. Dropout para evitar overfitting.

A função de perda é minimizada utilizando o algoritmo de gradiente descendente otimizado pelo Adam, que ajusta a taxa de aprendizado dinamicamente com base nos momentos das gradientes.

## Tecnologias Utilizadas
- **Linguagem:** Python
- **Bibliotecas Principais:**
  - PyTorch: Construção e treinamento da rede neural
  - scikit-learn: Pré-processamento de dados e métricas
  - SHAP: Explicabilidade do modelo
  - Seaborn/Matplotlib: Visualização de dados
  - TensorBoard: Monitoramento do treinamento

## Estrutura do Modelo
- **Arquitetura da Rede Neural:**
  - Camada de entrada: 30 neurônios (características do dataset)
  - 1ª Camada Oculta: 128 neurônios, ReLU e Dropout (0.2)
  - 2ª Camada Oculta: 64 neurônios, ReLU e Dropout (0.2)
  - Camada de Saída: 1 neurônio, função sigmoide
- **Função de Perda:** Binary Cross-Entropy Loss
- **Otimizador:** Adam com Weight Decay
- **Scheduler:** StepLR para ajuste da taxa de aprendizado

## Conjunto de Dados
- **Tamanho:**
  - **Treino:** 70%
  - **Teste:** 30%
- **Pré-processamento:**
  - Normalização com StandardScaler
  - Separação de recursos e rótulos

## Métricas de Avaliação
- **Acurácia:** 99.55%
- **Precisão:** 99.19%
- **Recall:** 99.50%
- **F1-Score:** 99.34%
- **AUC-ROC:** 99.79%

## Resultados da Matriz de Confusão
```
[[49347   208]
 [  128 25317]]
```

## Visualizações de SHAP
O projeto inclui explicações interpretáveis para o modelo usando SHAP:
- **Background Data:** Primeiras 100 amostras de treino
- **Test Samples:** Primeiras 10 amostras do conjunto de teste
- **Saída:** Gráficos de importância das características gerados pelo SHAP para entender como cada atributo influencia as previsões.

## Como Executar o Projeto
### Pré-requisitos
1. Python 3.8+
2. Instale as dependências:
```bash
pip install torch scikit-learn matplotlib seaborn shap tensorboard
```

## Melhorias Futuras
- Expansão do conjunto de dados com técnicas de data augmentation.
- Avaliação com diferentes arquiteturas de rede neural.
- Inclusão de mais explicações interpretáveis usando LIME.

## Contribuições
Contribuições são bem-vindas! Por favor, envie um pull request ou abra uma issue para sugestões.

## Licença
Este projeto está licenciado sob a [MIT License](https://opensource.org/licenses/MIT).
