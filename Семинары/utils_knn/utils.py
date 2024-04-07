import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def calc_recall(true_labels, pred_labels, k, exclude_self=False, return_mistakes=False):
    '''
    счиатет recall@k для приближенного поиска соседей
    
    true_labels: np.array (n_samples, k)
    pred_labels: np.array (n_samples, k)
    
    exclude_self: bool
        Если query_data была в трейне, считаем recall по k ближайшим соседям, не считая самого себя
    return_mistakes: bool
        Возвращать ли ошибки
    
    returns:
        recall@k
        mistakes: np.array (n_samples, ) с количеством ошибок
    '''
    n = true_labels.shape[0]
    n_success = []
    shift = int(exclude_self)
    
    for i in range(n):
        n_success.append(np.intersect1d(true_labels[i, shift:k+shift], pred_labels[i, shift:k+shift]).shape[0])
        
    recall = sum(n_success) / n / k
    if return_mistakes:
        mistakes = k - np.array(n_success)
        return recall, mistakes
    return recall





def plot_ann_performance(build_data, 
                         query_data, 
                         index_dict, 
                         k, 
                         flat_build_func, 
                         flat_search_func,
                         query_in_train,
                         qps_line,
                         recall_line,
                         title,
                         **kwargs):
    '''
    build_data: np.array(N, dim) данные для создания модели
    query_data: np.array(N, dim) данные для проверки качества модели
    index_dict: dict аргументы для моделей функций
    k: int для измерения точности
    flat_build_func: func - функция, которая строит Flat-индекс
    flat_search_func: func -  функция, которая ищет в Flat-индексе
    query_in_train: bool - флаг того, что query_data содержится в build_data.
    qps_line:  float. Если указано, нарисуем горизонтальную линию по этому значению
    recall_line: float. Если указано, нарисуем вертикальную линию по этому значению
    title: str название графика
    
    данная функция строит графики перформансов нескольких функций для поиска ближайший соседей.
    Внутри функции создается два пандас датасета с параметрами, определяющими все параметры модели. Мы запускаем все необходимые модели и подсчитываем время + реколл.
    '''
    
    fig, ax = plt.subplots(1, 2, figsize=(16,9))
    model_df = dict()
    model_indexes = dict()
    model_arr = []
    time_arr = []
    for model in index_dict:
        model_indexes[model], time = index_dict[model]['build_func'](build_data, index_dict[model]['fixed_params'])
        model_arr.append(model)
        time_arr.append(time)
    model_df['model'] = model_arr
    model_df['time'] = time_arr
    
    model_time = pd.DataFrame(model_df)
    sns.barplot(data = model_time, y = 'time', hue = 'model', x = 'model', ax = ax[0])
    ax[0].set_title('build_time')
    ax[1].set_title(title)
    points_recall = []
    point_time = []
    research_param_name = []
    point_model_type = []
    points_df = dict()
    index_flat, time_flat = flat_build_func(build_data)
    true_dist, true_labels, time_flat = flat_search_func(index_flat, query_data, k)
    for model in model_indexes:
        for research_param in index_dict[model]['search_param'][1]:
            distances, labels, time = index_dict[model]['search_func'](model_indexes[model],
                                             query_data, 
                                             k,
                                             research_param)
            recall = calc_recall(true_labels = true_labels, pred_labels = labels, k = k, exclude_self = query_in_train)
            points_recall.append(recall)
            point_time.append(query_data.shape[0] / time)
            research_param_name.append(index_dict[model]['search_param'][0] + ' ' + str(research_param))
            point_model_type.append(model)
    points_df['points_df'] = research_param_name
    points_df['recall@{k}'.format(k = k)] = points_recall
    points_df['quaries per second'] = point_time
    points_df['model'] = point_model_type
    points_df = pd.DataFrame(points_df)
    sns.lineplot(data = points_df, x = 'recall@{k}'.format(k = k), y = 'quaries per second', hue = 'model', ax = ax[1])
    for i in range(points_df.shape[0]):
        pos = points_df.iloc[i, [1, 2]].to_numpy()
        text = points_df.iloc[i, 0]
        ax[1].annotate(text, xy = pos)
    plt.axvline(x = recall_line, color = 'b', linestyle = '--')
    plt.axhline(y = qps_line, color = 'b', xmin=0, xmax=k, linestyle = '--')
    plt.axhline(y = query_data.shape[0] / time_flat, color = 'r', xmin=0, xmax=k, linestyle = '--', \
                label = 'flat: {qps:.3e}'.format(qps = query_data.shape[0] / time_flat))
    plt.yscale("log")
    plt.legend()
    
    

    
def analyze_ann_method(build_data, query_data, build_func, search_func, k, 
                      flat_build_func, flat_search_func, query_in_train, index_name):
    
    '''
    build_data - тренировочная данная
    query_data - данные для теста
    build_func - функция для построения приближенного индекса
    search_func - функция для поиска соседей
    k - количество соседей
    flat_build_func - функция, для построения 
    '''
    
    index_flat, time_flat = flat_build_func(build_data)
    true_dist, true_labels, time_flat = flat_search_func(index_flat, query_data, k)
    
    
    index_quant, time_quant = build_func(build_data)
    pred_dist, pred_labels, time_pred = search_func(index_quant, query_data, k)
    recall, mistakes = calc_recall(pred_labels, true_labels, k, query_in_train, True)
    true_unique = np.arange(k + 1)
    true_counts = np.zeros(k + 1)
    unique, counts = np.unique(mistakes, return_counts=True)
    true_counts[unique] = counts
    counts_to_plot = true_counts.copy()    
    counts_to_plot[counts_to_plot == 0] = -100
    fig, ax = plt.subplots(figsize=(8, 6))
    width = 0.6
    # сами бары
    true_unique -= query_in_train
    p = ax.bar(true_unique[1:], counts_to_plot[query_in_train:], width, label = str(true_counts[query_in_train:]))
    ax.bar_label(p, color = 'red', fontsize=15, labels = true_counts[query_in_train:])
    # приписать к ним значения весов
    
    # настройка ах'a
    ax.margins(0.2, 0.05)
    ax.set_title(index_name, fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    ax.set_ylabel('number', fontsize=15)
    ax.set_xlabel('mistakes', fontsize=15)
    build_line = 'build_time: {time_quant:.1e}'.format(time_quant = time_quant)
    qps_line = 'qps: {qps:.1e}'.format(qps = query_data.shape[0] / time_pred)
    recall_line = 'recall@{k}: {recall:.2e}'.format(k = k, recall = recall)
    legend_fin = build_line + '\n' + qps_line + '\n' + recall_line
    ax.legend([legend_fin])
    



# Для FASHION MNIST
def knn_predict_classification(neighbor_ids, tr_labels, n_classes, distances=None, weights='uniform'):
    '''
    по расстояниям и айдишникам получает ответ для задачи классификации
    
    distances: (n_samples, k) - расстояния до соседей
    neighbor_ids: (n_samples, k) - айдишники соседей
    tr_labels: (n_samples,) - метки трейна
    n_classes: кол-во классов
    
    returns:
        labels: (n_samples,) - предсказанные метки
    '''
    
    n, k = neighbor_ids.shape

    labels = np.take(tr_labels, neighbor_ids)
    labels = np.add(labels, np.arange(n).reshape(-1, 1) * n_classes, out=labels)

    if weights == 'uniform':
        w = np.ones(n * k)
    elif weights == 'distance' and distances is not None:
        w = 1. / (distances.ravel() + 1e-10)
    else:
        raise NotImplementedError()
        
    labels = np.bincount(labels.ravel(), weights=w, minlength=n * n_classes)
    labels = labels.reshape(n, n_classes).argmax(axis=1).ravel()
    return labels


# Для крабов!
def get_k_neighbors(distances, k):
    '''
    считает по матрице попарных расстояний метки k ближайших соседей
    
    distances: (n_queries, n_samples)
    k: кол-во соседей
    
    returns:
        labels: (n_queries, k) - метки соседей
    '''
    indices = np.argpartition(distances, k - 1, axis=1)[:, :k]
    lowest_distances = np.take_along_axis(distances, indices, axis=1)
    neighbors_idx = lowest_distances.argsort(axis=1)
    indices = np.take_along_axis(indices, neighbors_idx, axis=1) # sorted
    sorted_distances = np.take_along_axis(distances, indices, axis=1)
    return sorted_distances, indices


# Для крабов! Пишите сами...
def knn_predict_regression(labels, y, weights='uniform', distances=None):
    '''
    считаем регрессию KNN по меткам ближайших соседей, умножая на обратное расстояние
    '''
    return np.average(np.take(y, labels), weights = distances, axis = 1)





















