import gc

from pyecharts.charts import Bar, Page, Pie
from pyecharts import options as opts
from examination.toolkit import *

pd.options.display.max_columns = None  # 显示所有列
# pd.set_option('display.float_format', lambda x: '%.2f' % x)  # 取消科学计数法
pd.set_option("display.max_rows", None)


class DQReport(object):
    def __init__(self, data, Id, target, diff_limit=8, k=5):
        """
            data:           input data-frame
            target:         data-frame's target
            diff_limit:     depend on number of different values, to distinguish numeric or categorical attribute
            k:              for numeric -- equal width binning -- amount of binning on html-showing
        """
        self.data = data.copy()
        self.target = target
        self.diff_limit = diff_limit
        self.k = k
        self.colname = self.data.columns
        self.data_size = self.data.shape[0]

        list_kind = []
        data_copy = self.data.copy()
        for col in self.colname:
            temp, kind = data_kind(x=data_copy[col], num_limit=diff_limit)
            list_kind.append(kind)

        dict_data = {'col_name': self.colname, 'kinds': list_kind}
        self.list_kind = list_kind
        feature_df = pd.DataFrame(dict_data, columns=['col_name', 'kinds'])
        self.numeric_list = list(
            feature_df[(feature_df['kinds'] == ModeAnalyzer.Numeric) & (feature_df['col_name'] != target) & (
                    feature_df['col_name'] != Id)]['col_name'])
        self.categorical_list = list(
            feature_df[(feature_df['kinds'] == ModeAnalyzer.Categorical) & (feature_df['col_name'] != target) & (
                    feature_df['col_name'] != Id)]['col_name'])
        self.feature_df = feature_df

    def get_numeric_list(self):
        return self.numeric_list

    def get_categorical_list(self):
        return self.categorical_list

    def re_type(self):
        feature_df = self.feature_df
        for col in self.colname:
            if feature_df[feature_df['col_name'] == col]['kinds'].values == ModeAnalyzer.Categorical:
                self.data[col] = self.data[col].astype('object')
            else:
                self.data[col] = self.data[col].astype('float64')
        return self.data

    def SReport(self, save_path=None, top=5):
        """
            top:                get the n most frequent values,subset .value_counts() and grab the index
            save_path:          csv path
            list_na:            the count of NAN
            list_na_ratio:      the ratio of NAN
            list_value:         the count of non-null
            list_value_ratio    the ratio of non-null
            list_diff_count     the count of different value
            list_diff_value     show some different elements
        """
        list_na, list_na_ratio, list_value, list_value_ratio, list_diff_count, list_diff_value = [], [], [], [], [], []
        data = self.data.copy()
        for col in self.colname:
            diff_count = data[col].nunique()  # size of different number
            list_diff_count.append(diff_count)

            drop_na = data[col].dropna()
            dropna_size = drop_na.shape[0]  # size of after drop
            na_size = self.data_size - dropna_size  # size of nan
            list_na.append(na_size)
            na_ratio = str(round(na_size / self.data_size * 100, 4)) + '%'
            list_na_ratio.append(na_ratio)
            list_value.append(dropna_size)  # ratio of value
            value_ratio = str(round(dropna_size / self.data_size * 100, 4)) + '%'
            list_value_ratio.append(value_ratio)
            temp, kind = data_kind(x=data[col], num_limit=self.diff_limit)

            # if kind == 'numeric':
            #     a = list(temp.unique())
            #     b = list(drop_na.unique())
            #     if len(a) == len(b):
            #         if len(a) > 3:
            #             list_diff_value.append(a[:3] + ['...'])
            #         else:
            #             list_diff_value.append(a)
            #     else:
            #         list_diff_value.append(list(set(b).difference(set(a))) + a[:3] + ['...'])
            # else:
            #     llist = drop_na.value_counts()[:top].index.tolist()
            #     if diff_count > top:
            #         llist = llist + ['...']
            #     list_diff_value.append(llist)

        # release some RAM
        del data
        gc.collect()

        # make-up dataframe
        dict_data = {'col_name': self.colname, 'kinds': self.list_kind, 'null': list_na, 'null_ratio': list_na_ratio,
                     'value': list_value, 'value_ratio': list_value_ratio,
                     # 'count of different kinds': list_diff_count,
                     # 'value of different': list_diff_value
                     }
        data_quality_report_summary = pd.DataFrame(dict_data, columns=[
            'col_name', 'kinds', 'null', 'null_ratio', 'value', 'value_ratio',
            # 'count of different kinds',
            # 'value of different'
        ])
        if save_path is not None:
            data_quality_report_summary.to_csv(save_path, index=False)
        print(data_quality_report_summary)

    def create_graph(self, page_, data_, col, target, new_list_taget, pyecharts, col_type=None):
        import matplotlib.pyplot as plt

        """
        page_:      current interface
        col:        column name
        col_type:   numeric and categorical
        """
        file = data_.copy()
        file.sort_values(by=col, inplace=True)
        if col_type == ModeAnalyzer.Numeric:
            file[col] = pd.cut(file[col], self.k)
            file[col].replace(np.nan, 'unKnow', inplace=True)  # 数值属性切割完成后需要把空值转为unknown
        file[col] = file[col].astype('str')
        list_fea = list(file[col].unique())  # 类别字段-list
        df_cut = file[[col, target]].copy()
        df_cut.reset_index(inplace=True)
        col_ = df_cut.groupby([col, target], sort=False)['index'].count().unstack(fill_value=0).stack()
        print('col_\n', col_)
        df_cut.drop('index', inplace=True, axis=1)
        col_sum = col_.groupby(level=0, sort=False).sum()
        print('col_sum\n', col_sum)
        col_g = (col_ / col_sum).unstack()[new_list_taget]
        print('col_g\n', col_g)

        col_value = col_g.values

        """
        饼图 
        pie:    Pyechart
        plt:    Matplotlib
        """
        plt.pie(x=np.around((col_sum / self.data_size * 100), 2), autopct='%.2f%%', labels=list_fea)
        plt.title(col)
        plt.show()

        """
        条形图 
        bar0:   Pyechart
        plt:    Matplotlib
        """
        ax = col_sum.reset_index().plot(kind='bar', rot=1, figsize=(8, 8), title=col)  # 百分比堆叠图
        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy()
            ax.text(x + width / 2,
                    y + height / 2,
                    '{:.1f}'.format(height),
                    horizontalalignment='center',
                    verticalalignment='center')
        ax.set_xticklabels(labels=list_fea, rotation=30)
        plt.show()

        """
        百分比堆叠图 
        bar1:   Pyechart
        plt:    Matplotlib
        """
        ax = col_g.plot(kind='bar', stacked=True, rot=1, figsize=(16, 8), title=col)  # 百分比堆叠图
        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy()
            ax.text(x + width / 2,
                    y + height / 2,
                    '{:.1f} %'.format(height * 100),
                    horizontalalignment='center',
                    verticalalignment='center')
            ax.set_xticklabels(labels=list_fea, rotation=30)
        plt.show()
        if pyecharts:
            pie = Pie(init_opts=opts.InitOpts(width='1000px', height='600px'))
            pie.set_global_opts(legend_opts=opts.LegendOpts(orient='vertical', pos_top='15%', pos_left='2%'))
            pie.add('', [list(z) for z in zip(list_fea, np.around((col_sum / self.data_size * 100), 2))],
                    is_clockwise=True)
            pie.set_global_opts(title_opts=opts.TitleOpts(title='百分比对比图(%)'))
            page_.add(pie)

            bar0 = Bar(init_opts=opts.InitOpts(width='1000px', height='600px'))  # 图大小
            bar0.add_xaxis(list_fea)
            bar0.add_yaxis(col, list(col_sum))
            page_.add(bar0)

            bar1 = Bar(init_opts=opts.InitOpts(width='1000px', height='600px'))  # 图大小
            bar1.add_xaxis(list_fea)
            for tar_class, i in zip(new_list_taget, np.arange(0, len(new_list_taget))):
                list_class = list(np.around(col_value[:, i], 2))
                bar1.add_yaxis(tar_class, list_class, stack='stack1')

            bar1.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            bar1.set_global_opts(title_opts=opts.TitleOpts(title=col),
                                 datazoom_opts=opts.DataZoomOpts(range_start=0, range_end=100))  # 数据窗口范围的起始/终止百分比滑动条
            page_.add(bar1)

        del file
        gc.collect()
        return page_

    def NDataFrame(self, executor=None):
        data = self.data.copy()
        list_min = []  # 最小值
        list_max = []  # 最大值
        list_mean = []  # 均值
        list_std = []  # 标准差
        list_mean_sub_std = []  # 均值 - 3 x 标准差
        list_mean_add_std = []  # 均值 + 3 x 标准差
        list_Q1 = []  # 四分位数 Q1
        list_Q3 = []  # 四分位数 Q3
        list_quartile_min = []  # Q1 - 1.5 x IQR
        list_quartile_max = []  # Q3 + 1.5 x IQR

        for col in self.numeric_list:
            if executor is not None:
                executor(col)
            describe_ = data[col].describe()
            list_min.append(round(describe_['min'], 2))
            list_max.append(round(describe_['max'], 2))
            list_std.append(round(describe_['std'], 2))
            list_mean.append(round(describe_['mean'], 2))
            list_mean_sub_std.append(round(describe_['mean'] - 3 * describe_['std'], 2))
            list_mean_add_std.append(round(describe_['mean'] + 3 * describe_['std'], 2))
            list_Q1.append(round(describe_['25%'], 2))
            list_Q3.append(round(describe_['75%'], 2))
            IQR = round(describe_['75%'] - describe_['25%'], 2)
            list_quartile_min.append(round(describe_['25%'] - 1.5 * IQR, 2))
            list_quartile_max.append(round(describe_['75%'] + 1.5 * IQR, 2))

        dict_data = {'numeric_name': self.numeric_list, 'Min': list_min, 'Max': list_max, 'Mean': list_mean,
                     'StDev': list_std,
                     'M-3': list_mean_sub_std, 'M+3': list_mean_add_std, 'Q1': list_Q1, 'Q3': list_Q3,
                     'Q1-1.5*IQR': list_quartile_min, 'Q3+1.5*IQR': list_quartile_max}
        return pd.DataFrame(dict_data, columns=[
            'numeric_name',
            'Min',
            'Max',
            'Mean',
            'StDev',
            'M-3',
            'M+3',
            'Q1',
            'Q3',
            'Q1-1.5*IQR',
            'Q3+1.5*IQR']
                            )

    def NReport(self, csv_save_path=None, pyecharts=False, html_save_path='./html/numeric_analyse.html'):
        data = self.data.copy()

        page = Page()
        new_list_taget = data[self.target].value_counts(ascending=True).index  # 目标-list
        if pyecharts:
            page.page_title = 'numeric'  # html标签
            page.render(html_save_path)

        data_quality_report_numeric = self.NDataFrame(
            lambda col: page.add(self.create_graph(page_=page, data_=data, col=col, target=self.target,
                                                   new_list_taget=new_list_taget, pyecharts=pyecharts,
                                                   col_type=ModeAnalyzer.Numeric)
                                 )
        )

        if csv_save_path is not None:
            data_quality_report_numeric.to_csv(csv_save_path, index=False)
        print(data_quality_report_numeric)

        del data
        gc.collect()

    def CReport(self, xlsx_save_path=None, pyecharts=False, html_save_path='./html/categorical_analyse.html'):

        data = self.data.copy()
        page = Page()  # the new page
        new_list_taget = data[self.target].value_counts(ascending=True).index  # 目标-list
        if xlsx_save_path is not None:
            writer_ = pd.ExcelWriter(xlsx_save_path)
        for col in self.categorical_list:
            page = page.add(
                self.create_graph(page_=page, data_=data, col=col, target=self.target, pyecharts=pyecharts,
                                  new_list_taget=new_list_taget))
            if data[col].isnull().any():
                category_name = data[col].value_counts().index.tolist()
                category_name.append('NAN')
            else:
                category_name = data[col].value_counts().index.tolist()
            dict_data = {'category_name': category_name,
                         'number_feature': list(data[col].value_counts(dropna=False)),
                         'ratio_feature': list(data[col].value_counts(normalize=True, dropna=False))}
            data_quality_report_categorical = pd.DataFrame(dict_data,
                                                           columns=['category_name', 'number_feature', 'ratio_feature'])
            if xlsx_save_path is not None:
                data_quality_report_categorical.to_excel(writer_, sheet_name=col[:15], index=False)
        del data
        gc.collect()
        if xlsx_save_path is not None:
            writer_.close()
        if pyecharts:
            page.page_title = 'categorical'
            page.render(html_save_path)
