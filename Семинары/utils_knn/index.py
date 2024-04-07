from time import time as tm
import hnswlib
import faiss
import numpy as np


def timer(func):
    '''
    декоратор, замеряющий время работы функции
    '''
    def wrapper(*args, **kwargs):
        start_time = tm()
        result = func(*args, **kwargs)
        end_time = tm() - start_time
        if isinstance(result, tuple):
            return *result, end_time
        return result, end_time
    return wrapper


@timer
def build_IVFPQ(build_data, fixed_params):
    
    '''
    Данная функция строит модель IVFPQ по заданному набору параметров
    build_data - данные по которым строим
    fixed_params - параметры, по которому необходимо обучать модель
    '''
    dim = fixed_params['dim']
    coarse_index = fixed_params['coarse_index']
    nlist = fixed_params['nlist']
    m = fixed_params['m']
    nbits = fixed_params['nbits']
    metric = fixed_params['metric']
    
    num_threads = fixed_params.get('num_threads', 1)
    faiss.omp_set_num_threads(num_threads)
    
    index = faiss.IndexIVFPQ( # у faiss туго с именованными аргументами
        coarse_index, # индекс для поиска соседей-центроидов
        dim, # размерность исходных векторов
        nlist, # количество coarse-центроидов = ячеек таблицы
        m, # на какое кол-во подвекторов бить исходные для PQ
        nbits, # log2 k* - количество бит на один маленький (составной) PQ-центроид
        metric # метрика, по которой считается расстояние между остатком(q) и [pq-центроидом остатка](x)
    )
    index.train(build_data)
    index.add(build_data)
    return index # из-за декоратора ожидайте, что возвращается index, build_time


@timer
def build_hnsw(build_data, fixed_params):
    '''
    Данная функция строит модель HNSW по заданному набору параметров
    build_data - данные по которым строим
    fixed_params - параметры, по которому необходимо обучать модель
    '''
    dim = fixed_params['dim']
    space = fixed_params['space']
    M = fixed_params['M']
    ef_construction = fixed_params['ef_construction']
    index = hnswlib.Index(space = space, dim = dim) # possible options are l2, cosine or ip

    # Initing index - the maximum number of elements should be known beforehand
    index.init_index(max_elements = build_data.shape[0] * 2, ef_construction = ef_construction, M = M)
    index.add_items(build_data)
    return index


@timer
def build_IVF(build_data, fixed_params):
    '''
    Данная функция строит модель faiss по заданному набору параметров
    build_data - данные по которым строим
    fixed_params - параметры, по которому необходимо обучать модель
    '''
    dim = fixed_params['dim']
    coarse_index = fixed_params['coarse_index']
    nlist = fixed_params['nlist']
    metric = fixed_params['metric']
    
    num_threads = fixed_params.get('num_threads', 1)
    faiss.omp_set_num_threads(num_threads)
    
    index = faiss.IndexIVFFlat(
        coarse_index, 
        dim, 
        nlist, 
        metric
    )
    index.train(build_data)
    index.add(build_data)
    return index


@timer
def search_faiss(index, query_data, k, nprobe=1):
    '''
    функция для поиска ближайших соседей методом faiss
    index - фаиссовская модель
    query_data - тест данные
    k - количество соседей, на которые смотрим
    nprobe - количество ячеек таблицы, в которые мы заглядываем. Мы заглядываем в nprobe ближайших coarse-центроидов для q
    query_in_train - лежит ли тестовая часть данных в трейне
    '''
    
    index.nprobe = nprobe # количество ячеек таблицы, в которые мы заглядываем. Мы заглядываем в nprobe ближайших coarse-центроидов для q
    distances, labels = index.search(query_data, k)
    return distances, labels # из-за декоратора ожидайте, что возвращается distances, labels, search_time

@timer
def search_hnsw(index, query_data, k, efSearch =1):
    '''
    функция для поиска ближайших соседей методом hnsw
    index - hnsw модель
    query_data - тест данные
    k - количество соседей, на которые смотрим
    nprobe - количество ячеек таблицы, в которые мы заглядываем. Мы заглядываем в nprobe ближайших coarse-центроидов для q
    query_in_train - лежит ли тестовая часть данных в трейне
    '''
    
    index.set_ef(efSearch)
    labels, distances =  index.knn_query(query_data, k)
    return distances, labels # из-за декоратора ожидайте, что возвращается distances, labels, search_time



@timer
def build_flat_l2(build_data, dim):
    
    '''
    Данная функция строит обычный KNN индекс для l2 метрики
    
    '''
    index = faiss.IndexFlatL2(dim)
    index.train(build_data)
    index.add(build_data)
    return index

@timer
def build_flat_ip(build_data, dim):
    '''
    Данная функция строит обычный KNN индекс для cos метрики
    
    '''
    index = faiss.IndexFlatIP(dim)
    index.train(build_data)
    index.add(build_data)
    return index


@timer
def search_flat(index, query_data, k):
    '''
    Данная функция находит ближайших соседей для flat индекса
    
    '''
    distances, labels = index.search(query_data, k)
    return distances, labels
