import numpy as np
class RLDG():
  """Classe que define as instruções para encontrar um modelo preditivo que tenta
     mapear os atributos X de um exemplo à seu valor-chave real y, sendo o
     valor-chave real pertencente ao conjunto {0,1}, ou seja, um modelo preditivo
     para um problema de classificação binária, utilizando o algoritmo de regressão
     logística para encontrar uma previsão yhat a partir dos atributos X, e o algoritmo
     de descida gradiente para otimizar a função de previsão do algoritmo de regressão
     logística a partir da mudança dos valores de seus parâmetros."""
    
  def __init__(self,alpha):
    '''Inicializa uma instância da Classe RLDG que representa um determinado modelo
       que será encontrado após a execução do algoritmo de descida gradiente com o
       parâmetro de Taxa de Aprendizagem (alpha) sendo igual ao argumento fornecido.'''
    self.alpha=alpha
    self.treinado=0
    self.W=None
    self.b=None
  def __str__(self):
    'Representação da Instância em String'
    if self.treinado==0:
      return f'RLDG(alpha={self.alpha}),Não Treinado'
    else:
      return f'RLDG(alpha={self.alpha}),Treinado'

  def __repr__(self):
    'Representação da Instância em Conjuntos'
    return f'RLDG(alpha={self.alpha})'

  @staticmethod
  def sigmoide(arr):
    '''Método estático utilizado para calcular a função sigmoide(x)=1/1+(e^-x) de forma
       vetorizada para os'n' items de um array do numpy.'''
    return 1/(1+np.exp(-arr))
  
  def treinar(self,X_train,y_train,iter=1000):
    '''Método utilizado para realizar número 'iter' de iterações do algoritmo de descida
       gradiente para encontrar os valores dos parâmetros W e b os quais otimizam a função
       de custo J(W,b)=(-1/m)*soma[y_i*log(yhat_i)+(1-y_i)*log(1-yhat_i)], utilizando o
       argumento fornecido na criação da instância como o parâmetro 'alpha'(Taxa de 
       Aprendizagem) do algoritmo.
       Os valores de y e X correspondem, respectivamente, aos valores-chave reais do
       conjunto de dados para o qual se deseja encontrar um modelo e aos atributos dos
       exemplos deste.
       X_train=np.ndarray de shape (n_atributos,m)
       y_train=np.ndarray de shape (1,m)'''
    #Inicializando Parâmetros do Modelo como 0
    W=np.zeros((X_train.shape[0],1))
    b=0
    #Fazendo 'iter' iterações de descida gradiente
    for _ in range(iter):
      #Z=[[(WX_1_1+WX_1_2+...+WX_1_n_atributos)+b],...[(WX_m_1+...+WX_m_n_atributos)+b]](1,m)
      #Função Linear do Modelo
      Z=np.dot(W.T,X_train)+b
      #Yhat=[1/1+e^-Z_1,...,1/1+e^-Z_2](1,m)
      #Função Sigmoide aplicada à função linear do modelo-Previsão do Modelo para X_i
      Yhat=self.sigmoide(Z)
      #dZ=[(Yhat_1-Y_1),...,(Yhat_m-Y_m)](1,m)
      #derivada da função de perda L=soma[y_i*log(yhat_i)+(1-y_i)*log(1-yhat_i)]
      #quando o yhat é Yhat_i e y é Y_i
      dZ=Yhat-y_train
      #dW=[[X_1_1*dZ_1_1+...+X_m_1*dZ_m_1],...[X_1_n_atributos*dZ_1_n_atributos+...+X_m_n_atributos*dZ_m_n_atributos]].T(n_atributos,1)
      #derivada da função de custo J em relação a cada peso do modelo W_n_atributos
      dW=np.dot(X_train,dZ.T)/X_train.shape[1]
      #derivada da função de custo em relação ao parâmetro b
      db=dZ.mean()
      #Atualizando os parâmetros para o valor atual destes menos o produto das derivadas da função
      #de custo em relação a estes com a Taxa de Aprendizado alpha.
      W-=self.alpha*dW
      b-=self.alpha*db
    self.W=W
    self.b=b
    self.treinado=1
    
  def prever(self,X):
    '''Utiliza o modelo encontrado para prever o valor-chave real de um ,ou mais, exemplo(s) (y) 
       a partir de seus atributos (X), retornando uma previsão yhat que pertence ao conjunto 
       {0,1}. Será 1 caso yhat>=0.5 e 0 caso yhat<0.5.
       X=np.ndarray de shape (n_atributos vistos durante treino,numero de exs que se deseja prever)
    '''
    #Retornando aviso caso modelo não esteja treinado
    if self.treinado==0:
      return 'Modelo ainda não foi treinado.'
    #Calculando as previsões
    Yhat=self.sigmoide(np.dot(self.W.T,X)+self.b)
    return np.where(Yhat>=0.5,1,0).ravel()
  
  def acuracia(self,X,y):
    '''Calcula a porcentagem de valores-chave reais que o modelo preveu corretamente. 
       Primeiramente utiliza o método 'prever' da instância para calcular as previsões
       do modelo para o argumento de X(np.ndarray de shape (n_atributos,n exs que se
       deseja prever)) e depois calcula a porcentagem de valores do argumento y(
       np.ndarray de shape(n exs que se deseja prever,) que são iguais aos valores
       previstos pelo modelo,dado que estes estejam atrelados ao mesmo ex.'''
    #Retornando aviso caso modelo não esteja treinado
    if self.treinado==0:
      return 'Modelo ainda não foi treinado.'
    prev=self.prever(X)
    acertos= y==prev
    return len(acertos[acertos==True])/len(acertos)
  
  def prever_prob(self,X):
    '''Utiliza o modelo encontrado para prever o valor-chave real de um ,ou mais, exemplo(s) (y) 
       a partir de seus atributos (X), retornando uma previsão yhat que pertence ao intervalo [0,1]
       e representa a probabilidade de que, dados os atributos X o exemplo o valor-chave do(s)
       exemplo(s) é igual a 1,ou seja, P(y=1|X) para todos os exemplos de X.'''
    #Retornando aviso caso modelo não esteja treinado
    if self.treinado==0:
      return 'Modelo ainda não foi treinado.'
        #Calculando as previsões
    Yhat=self.sigmoide(np.dot(self.W.T,X)+self.b)
    return Yhat.ravel()
