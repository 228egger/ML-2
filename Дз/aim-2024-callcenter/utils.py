import numpy as np
from tqdm import tqdm
from ipywidgets import widgets
from IPython.display import display

import plotly.express as px
import plotly.graph_objects as go

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, FixedTicker, BooleanFilter, CDSView
from bokeh.io import output_notebook
from bokeh.layouts import gridplot
from bokeh.transform import linear_cmap
from bokeh.palettes import magma, tol



def plot_dim_reduction(data, mapper_dict, default_features=None, default_hue_info=None, 
                       row_width=950, row_height=500, plotly_marker_size=1.5, bokeh_marker_size=3, return_results=False):
    '''
    Функция принимает на вход данные и набор 2D/3D dimension-редукторов через mapper_dict.
    Отрисовывает эмбеддинги этих данных в наиболее удобных форматах: 3D - plotly, 2D - bokeh с CDS sharing'ом
    
    
    data - pd.DataFrame со всеми необходимыми данными - hue_cols, features
    
    mapper_dict - словарь знакомого вида :)
    
    default_features: array of strings - фичи которые будут использоваться для вычисления функции расстояния,
        если для reductor`а не указано иного
        
    default_hue_info: namedtuple - вида (hue-колонка-строка, is_categorical),
        инфа о hue-колонке, которая будет использоваться, если для reductor`а не указано иного
    
    row_width: int - ширина ряда из картинок
        узнать - рисуйте пустую bokeh.plotting.figure, увеличивая width,
        пока фигура не станет занимать все свободное место в ширину
        
    row_height: int
        желаемая высота ряда
        
    .._marker_size: размер точек на plotly и bokeh графиках
    
    return_results: bool - возвращать ли словарь {mapper_name: {'mapper': mapper, 'embedding': embedding}, ...}
    
    returns
        results: dict if return_results
        
        Note: для t-SNE mapper=embedding и лежит по ключу 'embedding'!
            Это объект класса TSNEEmbedding, это "обертка" над эмбеддингом.
            У него есть метод transform, а также его можно воспринимать как эмбеддинг и, например, слайсить и рисовать
    '''
    if default_hue_info is None:
        default_hue_info = None, None
    
    output_notebook() # bokeh render in notebook
    bokeh_first_time = True
    
    plotly_figs, bokeh_figs = [], []
    results = dict()
    for mapper_name in tqdm(mapper_dict):
        mapper_props = mapper_dict[mapper_name]
        params, features = mapper_props['params'], mapper_props.get('features', default_features)
        if features is None:
            raise ValueError(f'Мапперу {mapper_name} нужно указать фичи')
            
        mapper, embedding, time_passed = mapper_props['func'](data[features].values, params)
        results[mapper_name] = {
            'embedding': embedding,
            'mapper': mapper
        }
        
        # СБОР ИНФОРМАЦИИ ДЛЯ ОТРИСОВКИ
        
        x, y = embedding[:, 0], embedding[:, 1]
        hue_info = mapper_props.get('hue', default_hue_info)
        hue_field_name, hue_is_categorical = hue_info if hue_info is not None else (None, None)
        
        if embedding.shape[1] == 3: # plotly 3D render
            z = embedding[:, 2]
            plot_data = {'x': x, 'y': y, 'z': z}
            if hue_field_name is not None:
                if hue_is_categorical:
                    # простой способ показывать легенду вместо colorbar
                    plot_data[hue_field_name] = data[hue_field_name].astype(str)
                else:
                    # в этом случае будет показываться colorbar
                    plot_data[hue_field_name] = data[hue_field_name]
            
            plotly_fig = px.scatter_3d(plot_data, x='x', y='y', z='z', title=mapper_name, color=hue_field_name)
            plotly_figs.append(plotly_fig)
            
        else: # bokeh render with CDS sharing
            if bokeh_first_time:
                source = ColumnDataSource(data)
                bokeh_first_time = False
                
            x_name = f'{mapper_name}_x'
            y_name = f'{mapper_name}_y'
            source.data[x_name] = x
            source.data[y_name] = y
            
            # набор инструментов
            # можете добавить еще какие хотите
            bokeh_fig = figure(title=mapper_name, tools=['pan', 'wheel_zoom', 'box_select', 'lasso_select', 'reset', 'box_zoom'])
            
            if hue_is_categorical is None: # если не во что красить
                bokeh_fig.scatter(x=x_name, y=y_name, size=bokeh_marker_size, source=source)
                
            elif hue_is_categorical: # Если hue категориальный, у нас будет легенда с возможностью спрятать отдельные hue
                # scatter -> label_name требует строку. Поэтому делаем из числовых категорий строки
                # Сортируем числа, потом делаем строки для корректной сортировки
                uniques = np.sort(data[hue_field_name].unique()).astype(str)
                
                # Настраиваем палитры
                n_unique = uniques.shape[0]
                if n_unique == 2:
                    palette = tol['Bright'][3][:2]
                elif n_unique == 3:
                    palette = tol['HighContrast'][3]
                elif n_unique in tol['Bright']:
                    palette = tol['Bright'][n_unique]
                else:
                    palette = magma(n_unique)
                
                # Делаем через for чтобы поддерживать legend.click_policy = 'hide'
                for i, hue_val in enumerate(uniques):
                    # Будем рисовать только ту дату, где hue_col == hue_val
                    condition = (data[hue_field_name].astype(str) == hue_val).tolist()
                    view = CDSView(filter=BooleanFilter(condition))
                    
                    # Рисуем эмбеддинги
                    bokeh_fig.scatter(x=x_name, y=y_name, size=bokeh_marker_size,
                                      source=source, view=view, legend_label=hue_val, color=palette[i])
                
                # Добавляем легенде возможность спрятать по клику
                bokeh_fig.legend.click_policy = 'hide'
                
            else: # Если hue числовой, у нас будет colorbar
                # Настраиваем цветовую палитру
                min_val, max_val = data[hue_field_name].min(), data[hue_field_name].max()
                color = linear_cmap(
                    field_name=hue_field_name,
                    palette=magma(data[hue_field_name].nunique()),
                    low=min_val,
                    high=max_val
                )
                
                # Рисуем эмбеддинги
                plot = bokeh_fig.scatter(x=x_name, y=y_name, size=bokeh_marker_size, source=source, color=color)
                
                # Чуть настроим colorbar
                ticks = np.linspace(min_val, max_val, 5).round()
                ticker = FixedTicker(ticks=ticks)
                colorbar = plot.construct_color_bar(title=hue_field_name, title_text_font_size='20px', title_text_align='center',
                                                    ticker=ticker, major_label_text_font_size='15px')
                bokeh_fig.add_layout(colorbar, 'below')
            
            bokeh_fig.title.align = 'center'
            bokeh_figs.append(bokeh_fig)
    
    
    # ОТРИСОВКА
    # имеем запиханные в списки bokeh_figs и plotly_figs фигуры
    # теперь надо отрисовать в нормальной решетке...
    # но в этой реализации функции пихаются в один ряд :)
    # в ваших силах это исправить - желательно рисовать 2-3 графика на ряд
    
    n_bokeh = len(bokeh_figs)
    if n_bokeh > 0:
        plot_width = round(row_width / (n_bokeh + 0.1))
        grid = gridplot([bokeh_figs], width=plot_width, height=row_height)
        show(grid)
    
    n_plotly = len(plotly_figs)
    if n_plotly > 0:
        plot_width = round(row_width / (n_plotly + 0.05))
        
        # plotly удобнее всего запихнуть в строку с помощью ipywidgets.widgets.HBox
        # его нужно вернуть
        plotly_widgets = []
        for i in range(n_plotly):
            fig = plotly_figs[i]
            layout = fig.layout
            layout.update({'width': plot_width, 'height': row_height,
                           'title_x': 0.5, 'title_font_size': 13, 'legend_itemsizing': 'constant'})
            fig.update_layout(legend= {})
            new_fig = go.FigureWidget(fig.data, layout=layout)
            new_fig.update_traces(marker_size=plotly_marker_size)
            plotly_widgets.append(new_fig)
            
        display(widgets.HBox(plotly_widgets))
        
    if return_results:
        return results
