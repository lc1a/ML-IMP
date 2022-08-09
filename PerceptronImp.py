import numpy as np

class Perceptron():
  """Classe que define as instruções para encontrar um modelo preditivo que tenta
     mapear os atributos X de um exemplo à seu valor-chave real y, sendo o
     valor-chave real pertencente ao conjunto {0,1}, ou seja, um modelo preditivo
     para um problema de classificação binária, utilizando uma rede neural
     (Perceptron) para encontrar uma previsão yhat a partir dos atributos X, e o algoritmo
     de descida gradiente para otimizar a função de previsão do algoritmo de regressão
     logística a partir da mudança dos valores de seus parâmetros."""
  
  def __init__(self,alpha,nos,func_ativ='relu'):
    '''Inicializa uma instância da Classe Percerptron que representa um determinado modelo
       que será encontrado após a execução do algoritmo de descida gradiente com o
       parâmetro de Taxa de Aprendizagem (alpha), o parâmetro de quantidade de nós na
       camada escondida (nos) sendo igual aos argumentos fornecidos.
       Utiliza por padrão a função não-linear de ativação da camada escondida como sendo
       a RELU por provar ser a melhor escolha para classificação binária, porém pode-se
       fornecer argumento "tanh" ou "sigmoide" no parâmetro "func_ativ" para utilizar qualquer uma 
       destas funções como a função de ativação da camada escondida.'''
    
    self.alpha=alpha
    self.nos=nos
    
    if func_ativ!='relu' and func_ativ!='tanh' and func_ativ!='sigmoide':
      raise ValueError(f'''Função de Ativação Não Implementada. As funcões de Ativação Implementadas são:
      RELU,tanh e Sigmoide.
      Foi fornecido como argumento : {func_ativ}.''')
      
    self.func_ativ=func_ativ
    self.W1=None
    self.W2=None
    self.b1=None
    self.b2=None
    self.treinado=0
    
  def __str__(self):
    '''Representação da Instância em String'''
    
    if self.treinado==0:
      return f'Perceptron(alpha={self.alpha},nos={self.nos},func_ativ={self.func_ativ}),Não Treinado'
    else:
      return f'Perceptron(alpha={self.alpha},nos={self.nos},func_ativ={self.func_ativ}),Treinado'
    
  def __repr__(self):
    '''Representação da Instância em Conjuntos'''
    
    return f'Perceptron(alpha={self.alpha},nos={self.nos},func_ativ={self.func_ativ})'
  
  @staticmethod
  def sigmoide(arr):
    '''Método estático utilizado para calcular a função sigmoide(x)=1/1+(e^-x) de forma
       vetorizada para os'n' items de um array do numpy.'''
    
    return 1/(1+np.exp(-arr))
  
  @staticmethod
  def tanh(arr):
    '''Método estático utilizado para calcular a função de Tangente Hiperbólica tanh(x)=(e^x-e^-x)/(e^x+e^-x) 
       de forma vetorizada para os'n' items de um array do numpy.'''
    
    return (np.exp(arr)-np.exp(-arr))/(np.exp(arr)+np.exp(-arr))
  
  @staticmethod
  def relu(arr):
    '''Método estático utilizado para calcular a função de Unidade Linear Retificada relu(x)=max(0,x)
       de forma vetorizada para os'n' items de um array do numpy.'''
    
    return np.where(arr>0,arr,0)
  
  @staticmethod
  def dsigmoide(arr):
    '''Método estático utilizado para calcular a derivada da função sigmoide(x)=1/1+(e^-x) de forma
       vetorizada para os'n' items de um array do numpy.'''
    
    return (1/(1+np.exp(-arr)))*(1-(1/(1+np.exp(-arr))))
  
  @staticmethod
  def dtanh(arr):
    '''Método estático utilizado para calcular a derivada da função tanh(x)=(e^x-e^-x)/(e^x+e^-x) de forma
       vetorizada para os'n' items de um array do numpy.'''
    
    return 1-np.square(((np.exp(arr)-np.exp(-arr))/(np.exp(arr)+np.exp(-arr))))
  
  @staticmethod
  def drelu(arr):
    '''Método estático utilizado para calcular a derivada da função relu(x)=max(0,x) de forma
       vetorizada para os'n' items de um array do numpy.'''
    
    return np.where(arr>0,1,0)
  
  def treinar(self,X_train,y_train,it=1000):
    '''Método utilizado para realizar número 'it' de iterações do algoritmo de descida
       gradiente para encontrar os valores dos parâmetros W1,W2 e b1,b2 os quais otimizam a função
       de custo J(W1,W2,b1,b2)=(-1/m)*soma[y_i*log(yhat_i)+(1-y_i)*log(1-yhat_i)], utilizando o
       argumento fornecido na criação da instância como o parâmetro 'alpha'(Taxa de 
       Aprendizagem) do algoritmo e o argumento fornecido na criação da instância como parâmetro
       'nos' como a quantidade de Nós na camada escondida do modelo Perceptron utilizado para o cálculo
       do valor chave previsto y_hat.
       Os valores de y e X correspondem, respectivamente, aos valores-chave reais do
       conjunto de dados para o qual se deseja encontrar um modelo e aos atributos dos
       exemplos deste.
       X_train=np.ndarray de shape (n_atributos,m)
       y_train=np.ndarray de shape (1,m)'''
    
    #Inicializando Pesos aleatoriamente utilizando modulo random do numpy
    
    #W1 são os pesos da camada escondida. W1(nós,n_atributos)
    W1=np.random.randn(self.nos,X_train.shape[0])
    #W2 são os pesos da camada de saída(Pois a rede possui somente 1 camada escondida). W2(1,nós)
    W2=np.random.randn(1,self.nos)
    #b1 são os termos independentes dos nós da camada escondida. b1(nós,1)
    b1=np.random.randn(self.nos,1)
    #b2 é o termo independente do nó da camada de saída. b2(1,1)
    b2=np.random.randn(1,1)
    
    #Definindo Função de Ativação
    if self.func_ativ=='relu':
      func_ativ=self.relu
      dfunc_ativ=self.drelu
    elif self.func_ativ=='tanh':
      func_ativ=self.tanh
      dfunc_ativ=self.dtanh
    else:
      func_ativ=self.sigmoide
      dfunc_ativ=self.dsigmoide
      
    
    #Realizando Número 'it' de iterações do algoritmo de descida gradiente
    
    for _ in range(it):
      
      #Calculando valores-chaves previstos (y_hat) do Perceptron (Propagação)
      
      #Z1(nós,m)=[np.dot(W1_1,X_train)+b1_1,...]
      Z1=np.dot(W1,X_train) + b1
      #A1(nós,m)=[func_ativ(Z1_1),...]
      A1=func_ativ(Z1)
      #Z2(1,m)=[np.dot(W2_1,A1_1)+b2,...]
      Z2=np.dot(W2,A1)+b2
      #A2(1,m)=[Sigmoide(Z2_1),...]
      A2=self.sigmoide(Z2)
      
      #Calculando Derivadas Parciais da Função de Custo J em relação aos parâmetros(Retropropagação)
      
      #dZ2(1,m)
      dZ2=A2-y_train
      #dW2(1,1)
      dW2=np.dot(dZ2,A1.T)/X_train.shape[1]
      #db2(1,1)
      db2=np.sum(dZ2,axis=1,keepdims=True)/X_train.shape[1]
      #dZ1(nós,m)
      dZ1=np.dot(W2.T,dZ2)*dfunc_ativ(Z1)
      #dW1(nós,n_atributos)
      dW1=np.dot(dZ1,X_train.T)/X_train.shape[1]
      #db1(nós,1)
      db1=np.sum(dZ1,axis=1,keepdims=True)/X_train.shape[1]
      
      #Atualizando Parâmetros
      W1=W1-self.alpha*dW1
      W2=W2-self.alpha*dW2
      b1=b1-self.alpha*db1
      b2=b2-self.alpha*db2
    
    #Salvando Valores Finais dos Parâmetros como atributos da instância e sinalizando treinado=1
    
    self.W1=W1
    self.W2=W2
    self.b1=b1
    self.b2=b2
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
    
    #Definindo Função de Ativação
    
    if self.func_ativ=='relu':
      func_ativ=self.relu
    elif self.func_ativ=='tanh':
      func_ativ=self.tanh
    else:
      func_ativ=self.sigmoide
      
    #Calculando Previsões:
    Z1=np.dot(self.W1,X) + self.b1
    A1= func_ativ(Z1)
    Z2=np.dot(self.W2,A1)+self.b2
    Yhat=self.sigmoide(Z2)
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
    
    prev=self.prever(X).reshape(1,-1)
    acertos= np.equal(y,prev).ravel()
    
    return len(acertos[acertos==True])/len(acertos)

  def prever_prob(self,X):
    '''Utiliza o modelo encontrado para prever o valor-chave real de um ,ou mais, exemplo(s) (y) 
       a partir de seus atributos (X), retornando uma previsão yhat que pertence ao intervalo [0,1]
       e representa a probabilidade de que, dados os atributos X o exemplo o valor-chave do(s)
       exemplo(s) é igual a 1,ou seja, P(y=1|X) para todos os exemplos de X.'''
    
    #Retornando aviso caso modelo não esteja treinado
    if self.treinado==0:
      return 'Modelo ainda não foi treinado.'
    
    #Definindo Função de Ativação
    
    if self.func_ativ=='relu':
      func_ativ=self.relu
    elif self.func_ativ=='tanh':
      func_ativ=self.tanh
    else:
      func_ativ=self.sigmoide
      
    
    #Calculando Previsões
    Z1=np.dot(self.W1,X_train) + self.b1
    A1= func_ativ(Z1)
    Z2=np.dot(self.W2,A1)+self.b2
    Yhat=self.sigmoide(Z2)
    return Yhat.ravel()
