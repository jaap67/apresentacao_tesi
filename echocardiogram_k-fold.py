import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
from collections import Counter

df = pd.read_csv('echocardiogram.csv')
df.drop('name', 1, inplace=True)
df.drop('group', 1, inplace=True)
df.drop('mult', 1, inplace=True)
df.drop('wall_motion_score', 1, inplace=True)
df.drop(df[df.survival == '?'].index, inplace=True)

df.loc[df['age_at_heart_attack'] == '0', 'age_at_heart_attack'] = int(0)
df.loc[df['age_at_heart_attack'] == '62.529', 'age_at_heart_attack'] = int(62)
df.drop(df[df.alive_at_1 == '?'].index, inplace=True)

X_df = df[['survival','still_alive','age_at_heart_attack','pericardial_effusion','fractional_shortening','epss','lvdd','wall_motion_index']]
Y_df = df['alive_at_1']

Xdummies_df = pd.get_dummies(X_df).astype(int)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_de_treino = 0.8

tamanho_de_treino = int(porcentagem_de_treino * len(Y))

treino_dados = X[0:tamanho_de_treino]
treino_marcacoes = Y[0:tamanho_de_treino]

validacao_dados = X[tamanho_de_treino:]
validacao_marcacoes = Y[tamanho_de_treino:]


def fit_and_predict_kf(nome, modelo, treino_dados, treino_marcacoes):
    k = 5
    scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv = k)
    taxa_de_acerto = np.mean(scores)
    msg = "Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto)
    print(msg)
    return taxa_de_acerto


resultados = {}

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state=0))
resultadoOneVsRest = fit_and_predict_kf("OneVsRest", modeloOneVsRest, treino_dados, treino_marcacoes)
resultados[resultadoOneVsRest] = modeloOneVsRest

from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state=0))
resultadoOneVsOne = fit_and_predict_kf("OneVsOne", modeloOneVsOne, treino_dados, treino_marcacoes)
resultados[resultadoOneVsOne] = modeloOneVsOne

from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict_kf("MultinomialNB",modeloMultinomial, treino_dados, treino_marcacoes)
resultados[resultadoMultinomial] = modeloMultinomial

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict_kf("AdaBoostClassifier",modeloAdaBoost, treino_dados, treino_marcacoes)
resultados[resultadoAdaBoost] = modeloAdaBoost

from sklearn.naive_bayes import GaussianNB
modeloGaussianNB = GaussianNB()
resultadoGaussianNB = fit_and_predict_kf("GaussianNB", modeloGaussianNB, treino_dados, treino_marcacoes)
resultados[resultadoGaussianNB] = modeloGaussianNB

from sklearn.ensemble import RandomForestClassifier
modeloRandomForestClassifier = RandomForestClassifier()
resultadoRandomForestClassifier = fit_and_predict_kf("RandomForestClassifier", modeloRandomForestClassifier, treino_dados, treino_marcacoes)
resultados[resultadoRandomForestClassifier] = modeloRandomForestClassifier
# print(resultados)


maximo = max(resultados)
vencedor = resultados[maximo]
print("========== V E N C E D O R ==========")
print(vencedor)

vencedor.fit(treino_dados, treino_marcacoes)
resultado = vencedor.predict(validacao_dados)
acertos = resultado == validacao_marcacoes

total_de_acertos = sum(acertos)
total_de_elementos = len(validacao_marcacoes)

taxa_de_acertos = 100.0 * total_de_acertos / total_de_elementos
print("Taxa de acerto do vencedor com dados de validação he: {0}".format(taxa_de_acertos))

acerto_base = max(Counter(validacao_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print("Taxa de acerto base: {}".format(taxa_de_acerto_base))

print("Total de elementos: {}".format(len(validacao_dados)))

num_casos_0 = len(df[df['alive_at_1'] == '0'])
num_casos_1 = len(df[df['alive_at_1'] == '1'])
total = num_casos_0 + num_casos_1
print('Número de casos 0 (pacientes mortos): {0} ({1:2.2f}%)'.format(num_casos_0,(num_casos_0/total) * 100))
print('Número de casos 1 (pacientes vivos): {0} ({1:2.2f}%)'.format(num_casos_1,(num_casos_1/total) * 100))