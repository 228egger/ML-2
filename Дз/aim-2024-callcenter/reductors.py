#from umap.umap_ import UMAP
import openTSNE
from sklearn.decomposition import PCA
from time import time as tm
from umap import UMAP

# при желании можете использовать время работы в рисовалке.
# сейчас оно не используется

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = tm()
        result = func(*args, **kwargs)
        end_time = tm() - start_time
        if isinstance(result, tuple):
            return *result, end_time
        return result, end_time
    return wrapper


'''
Функции ниже должны
    принимать:
        data, params (остальное пихайте через partial извне)
    возвращать:
        mapper (если есть. если нет, возвращаем None)
            объект-обученный-reductor с методом transform (нужен только для того чтобы вернуть его пользователю для работы с тестом)
        embedding
            2D / 3D embedding для отрисовки. будем также возвращать пользователю, если попросит
'''

# при желании, UMAP можно изменить, чтобы он тоже принимал предпосчитанные affinities, init

@timer
def make_umap(data, params, y=None):
    '''
    можно вшить init, affinities через partial, чтобы не считать по сто раз,
        если вы не хотите их менять
    '''
    mapper = UMAP(**params).fit(data, y)
    return mapper, mapper.embedding_



@timer
def make_tsne(data, params, init=None, affinities=None):
    '''
    можно вшить init, affinities через partial, чтобы не считать по сто раз,
        если вы не хотите их менять
    '''
    rescaled_init = None
    if init is not None:
        rescaled_init = openTSNE.initialization.rescale(init, inplace=False, target_std=0.0001)
        
    # mapper_embedding - объект класса TSNEEmbedding - и маппер, и эмбеддинг в одном :)
    mapper_embedding = openTSNE.TSNE(**params).fit(data, initialization=rescaled_init, affinities=affinities)
    return None, mapper_embedding

@timer
def make_pca(data, params):
    mapper = PCA(**params).fit(data)
    embedding = mapper.transform(data)
    return mapper, embedding