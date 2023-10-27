# Classificador
Um modelo de rede neural baseado em uma arquitetura de redes convolucionais de dupla camada. O dataset utilizado foi esse aqui https://www.kaggle.com/datasets/gpiosenka/100-bird-species

# Objetivo
Este trabalho teve como objetivo criar um classificador com base em qualquer conjunto de dados que contenha imagens. O cliente busca um classificador que seja capaz de separar grupos distintos de categorias ou espécies.

# Detalhes
Este código foi desenvolvido usando um conjunto de dados que contém aproximadamente 80 mil fotos de pássaros, os quais foram agrupados em cerca de 512 espécies. O conjunto de dados foi obtido no site Kaggle e consiste em três pastas: uma para treinamento, uma para validação e outra para testar a rede neural, permitindo a obtenção das métricas de treinamento.

# Metodologia
Com o objetivo de acelerar o treinamento e reduzir a necessidade de recursos computacionais, implementou-se um pré-processamento dos dados para redimensionar as imagens de 190x190 pixels e canais RGB para um único canal de pigmentação, variando de 0 a 1, em vez dos 256 canais que o RGB possui. Isso permitiu a execução mais rápida do treinamento da rede neural, embora tenha resultado em uma perda de acurácia no conjunto de teste. Essa perda ocorreu devido ao fato de que pássaros com a mesma forma podem ter nomes diferentes apenas com base na cor, como por exemplo, uma arara-azul e uma arara-vermelha. Essa abordagem resultou em uma diminuição da acurácia no conjunto de testes, mas, em contrapartida, melhorou a eficiência do modelo em termos de processamento de imagem e treinamento.

# Arquitetura
A arquitetura escolhida para este modelo combina redes neurais convolucionais 2D com camadas e tamanhos de lotes diferentes, a fim de evitar underfitting e overfitting. Foram realizados ajustes nessas camadas até alcançar um bom nível de acurácia, levando em consideração a transformação do espaço RGB realizada no pré-processamento dos dados.

# Conclusão
Este "experimento" ou trabalho que tive que realizar na faculdade me proporcionou um amplo conhecimento sobre como aplicar o aprendizado profundo no processamento de imagens, seja para criar um classificador ou construir uma IA capaz de gerar imagens a partir de texto, usando uma tecnologia chamada GAN.

#Como executar
Para rodar o programa, basta selecionar a base de dados correspondente ao treinamento, teste e validação e ajustar as métricas e por último mexer no código para que ele se adeque a sua necessidade para rodar é bem simples basta dar o comando ```python meu_programa.py``` lembrando de substituir o nome meu_programa.py pelo nome do seu código.
