import pandas as pd
import numpy as np
import array
import random

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.style.use('seaborn-darkgrid')
plt.rcParams["figure.figsize"] = (7, 5)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM
from keras.metrics import RootMeanSquaredError
import datetime

import warnings
warnings.filterwarnings("ignore")
import os

"""
2-input XOR example -- this is most likely the simplest possible example.
"""
import os
import neat

import visualize

# 2-input XOR inputs and expected outputs.
# xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
# xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.feed_forward.FeedForwardNetwork.create(genome, config)
        # net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        list_real = []
        list_predicted = []
        df_results = pd.DataFrame(columns=['real', 'predicted'])
        for xi, xo in zip(xor_inputs, xor_outputs):
            # print(f'Input:{xi}')
            output_scaled = net.activate(xi)
            try:
                output = scaler.inverse_transform(np.array(output_scaled[0]).reshape(-1, 1))[0]
            except ValueError:
                output = [0.0]
            real = scaler.inverse_transform(np.array(xo[0]).reshape(-1, 1))[0]
            list_real.append(real[0])
            list_predicted.append(output[0])
            # print(f'Predicto:{output}')
            # print(f'Real:{real}')
        df_results.real = list_real
        df_results.predicted = list_predicted
        rmse = get_rmse(df_results)
        # print(rmse)
            # mse = np.square(real[0] - output[0]).mean(axis=None)
            # # print(f'Erro medio:{mse}')
            # rmse = np.sqrt(mse)
            # # print(f'Raiz do erro:{rmse}')
        # if rmse <= 0.50:
        #     genome.fitness += 100.0
        # elif rmse <= 1.0:
        #     genome.fitness += 50.0
        # else:
        #     genome.fitness -= 1.0
        genome.fitness -= rmse

# nw = neat.gru.GRUNetwork
# def avalia_genoma(genome, config):
#     # net = neat.nn.feed_forward.FeedForwardNetwork.create(genome, config)
#     # net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
#     net = nw.create(genome, config)
#     list_real = []
#     list_predicted = []
#     df_results = pd.DataFrame(columns=['real','predicted'])
#     for xi, xo in zip(xor_inputs, xor_outputs):
#         # print(f'Input:{xi}')
#         output_scaled = net.activate(xi)
#         output = scaler.inverse_transform(np.array(output_scaled[0]).reshape(-1, 1))[0]
#         real = scaler.inverse_transform(np.array(xo[0]).reshape(-1, 1))[0]
#         list_real.append(real[0])
#         list_predicted.append(output[0])
#         # print(f'Predicto:{output}')
#         # print(f'Real:{real}')
#     df_results.real = list_real
#     df_results.predicted = list_predicted
#     rmse = get_rmse(df_results)
#     return rmse
#
# def eval_genomes(genomes, config):
#     for genome_id, genome in genomes:
#         genome.fitness = avalia_genoma(genome, config)


def run(config_file, for_params=False):
    # Load Configuration
    # random.seed(42) # Fatores aleatótios excluídos, qlqr melhora fica vinculado a alteracao de param
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to  generations.
    winner = p.run(eval_genomes, 100)
    # nw = neat.gru.GRUNetwork
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    list_real = []
    list_predicted = []
    df_results = pd.DataFrame(columns=['real','predicted'])
    #Show output of the most fit genome against training data
    # print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    #winner_net = neat.nn.recurrent.RecurrentNetwork.create(winner, config)
    # winner_net = nw.create(winner, config)
    inicio_predicao = datetime.datetime.now()
    for xi, xo in zip(xor_inputs, xor_outputs):
        output_scaled = winner_net.activate(xi)
        output = scaler.inverse_transform(np.array(output_scaled[0]).reshape(-1, 1))[0]
        real = scaler.inverse_transform(np.array(xo[0]).reshape(-1, 1))[0]
        fim_predicao = datetime.datetime.now()
        list_real.append(real[0])
        list_predicted.append(output[0])
        # print("input {!r}, expected output {!r}, go {!r}".format(xi, xo, output))
    #print(f"Duracao da Previsao NEAT/RNN: {fim_predicao - inicio_predicao}")
    print(f"Duracao da Previsao NEAT/MLP: {fim_predicao - inicio_predicao}")
    df_results.real = list_real
    df_results.predicted = list_predicted
    #df_results.to_excel('df_results_NEAT_RNN.xlsx')
    #df_results.to_excel('df_results_NEAT_RNN_19.xlsx')
    #df_results.to_excel('df_results_NEAT_RNN_sem_selecao.xlsx')
    # print(df_results)
    print('\n')
    print(f'RMSE das Previsoes: {get_rmse(df_results):.2f} m^3/t')
    #plot_graf(df_results)
    # grafico_dispersao_histograma(df_results, 'NEAT/RNN')
    # node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    # visualize.draw_net(config, winner, True, node_names=node_names)
    # visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)


def series_to_supervised(df, n_out=1, dropnan=True):
    '''
    Do algoritmo de experimento de seleção de caracterisitcas.py
    :param n_in: Número de observaçoes defasadas(lag) do input
    :param n_out: Número de observaçoes defasadas(lag) do output
    :param dropnan:
    :return: Dataframe adequado para supervised learning
    '''
    df_sup = df
    for col in df_sup.columns:
        for j in range(0, n_out):
            col_name_2 = str(col+f'_(t-{j})')
            df_sup[col_name_2] = df_sup[col].shift(-j)
        df_sup = df_sup.drop(col, axis=1)
    if dropnan:
        df_sup = df_sup.dropna() # data cleaning
    return df_sup

def get_scaler(scaler):
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }
    return scalers.get(scaler.lower())()

def inverse_transform(scaler, df
                      , columns):
    df = df
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    # return df
    return df

def get_rmse(df):
    '''Gera a raiz quadrada do erro médio das previsoes'''
    mse = np.square(df.real - df.predicted).mean(axis=None)
    return np.sqrt(mse)

def grafico_dispersao_histograma(df, nome_modelo:str):
    erro = df.real.values - df.predicted.values
    fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)
    axs[1].hist(erro, bins=50)
    axs[1].set_title(f'Histograma de Residuos - {nome_modelo}')
    axs[0].scatter(df.real.values, df.predicted.values)
    #axs[0].set_ylim([5, 14])
    axs[0].set_title(f'Real x Predito - {nome_modelo}')
    return plt.show()

def plot_graf(df):
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.plot(df.real.values, label='Real')
    plt.plot(df.predicted.values, label='Modelo NEAT/RNN')
    plt.title('Real x Modelo NEAT/RNN')
    plt.legend(loc='best')
    return plt.show()

if __name__ == "__main__":
    start = datetime.datetime.now()
    # Base de Dados Crua
    df = pd.read_excel('dados_artigo_2 - cópia.xlsx', index_col=0).dropna()
    #df = pd.read_excel('dados_artigo_2019.xlsx', index_col=0)
    df = df.fillna(df.median())
    #print(df.head())

    # Base de Dados adaptada para um problema de aprendizado supervisionado
    df_sup = series_to_supervised(df, 6) # 6 Periodos de 4h = 24h
    #print(df_sup)

    # Características selecionadas no artigo para dados de entrada
    feat_selected = ['CEG_(t-1)', 'Prod_PQ_(t-0)', 'Cfix_(t-0)',
                     'pos_dp_03_(t-0)', 'Temp_02_(t-0)', 'Temp_05_(t-0)',
                     'Pres_01_(t-0)', 'Pres_04_(t-0)', 'rpm_03_(t-0)',
                     'Alt_Cam_(t-5)', 'Pres_01_(t-5)', 'rpm_06_(t-5)']
    X = df_sup[feat_selected]
    # X = df_sup.drop('CEG_(t-0)', axis=1)
    y = df_sup['CEG_(t-0)']
    xor_inputs_raw = X[:]
    xor_outputs_raw = y[:]
    scaler = get_scaler('minmax')
    xor_inputs = scaler.fit_transform(xor_inputs_raw).tolist()
    xor_outputs = scaler.fit_transform(xor_outputs_raw.values.reshape(-1, 1)).tolist()
    # print(xor_inputs)
    # print(xor_outputs)
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.ini')
    # print(config_path)
    tempo = []
    for _ in range(1):
        run(config_path)
        end = datetime.datetime.now()
        print('-'*40)
        print(f'Inicio: {start}')
        print(f'Fim: {end}')
        duracao = end - start
        tempo.append(duracao)
        print(f'Duração : {duracao}')
        print('-'*40)
        print()
    print(f'Tempos 5 Exec : {tempo}')












