# -*- coding: utf-8 -*-
import numpy as np
'''
По требованиям из задания:
1. Много данных.
Читаем, обрабатываем и пишем поттрочно. Это долго и муторно, 
зато не должно упасть по памяти.
2. Дополнительные группы признаков.
Скрипт разбирает всю строку, обрабатывает каждую групу отдельно и хранит
статистики для каждой. Предполагается, что они будут в той же строчке через таб.
3. Дополнительные способы нормировки.
Дописываются методы _get_снужные_статистики и _способ_transform_row,
в fit добавляется ещё один elif, который определяет, что грузить и как считать.

Пример использования:
    
scaler = TestScaler(scaler='z_score')
scaler.fit(path="train.tsv")
scaler.transform(path="test.tsv")

Смотрим test_proc.tsv
'''

class TestScaler():
    def __init__(self, scaler):
        self.train_file_path = None
        self.result_path = "test_proc.tsv"
        self.scaler = scaler
        self.vec_len = 256
        self.mean_vec = None
        self.train_rows_count = None
        self.std_vec = None
        self.min_vec = None
        self.variance_vec = None
        self.trained = None
    # Обёртка для построчного чтения тренировочного файла.
    # Поскольку читаем два раза, будем декорировать для краткости кода
    def read_data(stat):
        def gener(self):
            with open(self.train_file_path) as f:
                next(f)
                for line in f:
                    yield self._parse_row(line)
        def proc(self):
            return stat(self, gener)
        return proc 
    # Лист листов [тип признаков [признаки]]
    # [[2 [256 значений]], [3 [256 значений]]]
    def _parse_row(self, row):
        l = row.split('\t')
        rid = l[0]
        dat = [n.rstrip().split(',') for n in l[1:]]
        dat = [[i[0], np.array(i[1:]).astype(int)] for i in dat]
        return rid, dat
    # Я думал, что придётся использовать scipy.sparse_matrix, или обрабатывать
    # случаи, когда у наблюдения какие-то группы признаков есть, а каких-то нету,
    # поэтому начал со словаря {тип признаков: [вектор средних]} но, видимо, 
    # ничего из вышеперечисленного не пригодится. А словарь пусть будет
    @read_data
    def _get_mean(self, data_generator):
        vec_sum = dict()
        count = 0
        for rid, data in data_generator(self):
            for fgroup, fval in data:
                if fgroup in vec_sum.keys():
                    vec_sum[fgroup] += fval
                else:
                    vec_sum[fgroup] = fval
            count += 1
        return {k: vec_sum[k] / count for k in vec_sum.keys()}, count
    # Поскольку мы исходим из того, что оперировать сразу всем файлом нам тяжело,
    # Придётся прочитать его ещё раз для стд. Долго. Зато надёжно. Но долго.
    @read_data
    def _get_std(self, data_generator):
        diff_sum = {k: np.zeros(self.vec_len) for k in self.mean_vec.keys()}
        for rid, data in data_generator(self):
            for fgroup, fval in data:
                diff_sum[fgroup] += (self.mean_vec[fgroup] - fval) ** 2
        return {k: np.sqrt(diff_sum[k] / (self.train_rows_count-1)) 
                for k in diff_sum.keys()}
    # Для иллюстрации добавил мин-макс нормировку.
    @read_data
    def _get_minmax(self, data_generator):
        min_vec = dict()
        max_vec = dict()
        for rid, data in data_generator(self):
            for fgroup, fval in data:
                if fgroup in min_vec.keys():
                    min_vec[fgroup] = np.min([min_vec[fgroup], fval], axis=0)
                    max_vec[fgroup] = np.max([max_vec[fgroup], fval], axis=0)
                else:
                    min_vec[fgroup] = fval
                    max_vec[fgroup] = fval
        return min_vec, {k: max_vec[k] - min_vec[k] for k in min_vec.keys()}
    # Обработка вектора из тестового файла
    # k - ключ к словарю из функций выше, vec - вектор признаков, которые мы процессим
    def _z_score_transform_row(self, k, vec):
        return '\t'.join(((vec - self.mean_vec[k]) / self.std_vec[k]
                          ).astype(str))
    def _minmax_transform_row(self, k, vec):
        return '\t'.join(((vec - self.min_vec[k]) / self.variance_vec[k]
                          ).astype(str))
    # Две последние колонки
    def _max_ind_and_diff(self, k, vec):
        ind = np.argmax(vec)
        return '\t' + str(ind) + '\t' + \
               str(int(vec[ind] - self.mean_vec[k][ind]))
    # Считаем статистики, которые нам понадобятся для выбранного способа нормировки 
    # И выбираем функцию трансформации
    # Сейчас для минмакса читаем два раза, очевидно, что можн всё собрать за один подход
    # Оптимизируем функции когда поймём, какие нормировки будут использоваться.
    # По задумке там будет вообще конструктор.
    def fit(self, path):
        self.train_file_path = path
        if self.scaler=='z_score':
            self.mean_vec, self.train_rows_count = self._get_mean()
            self.std_vec = self._get_std()
            self.trained = True
            self.transform_row = self._z_score_transform_row
        elif self.scaler=='minmax':
            self.mean_vec, self.train_rows_count = self._get_mean()
            self.min_vec, self.variance_vec = self._get_minmax()
            self.trained = True
            self.transform_row = self._minmax_transform_row
        else:
            print('set scaler first')
    # Открываем тестовый файл и одновременно файл с результатом.
    # Обработав одну строчку, тут же добавляем её к результатам
    def transform(self, path):
        self.test_file_path = path
        if not self.trained:
            return False
        with open(self.test_file_path) as f, open(self.result_path, "a+") as r:
            next(f)
            combinations = [[j, i] for j in self.mean_vec.keys() 
                                   for i in range(self.vec_len)]
            columns1 = 'id_job\t' + '\t'.join(["feature_{}_stand_{}" \
                                               .format(i, j) for i, j 
                                               in combinations]) + '\t'
            columns2 = '\t'.join(['max_feature_{}_index\tmax_feature_{}_abs_mean_diff' \
                        .format(i, i) for i in self.mean_vec.keys()])
            r.write(columns1+columns2+'\n')
            for line in f:
                rid, dat = self._parse_row(line)
                row1 = rid + '\t' + '\t'.join([self.transform_row(k, vec)
                                              for k, vec in dat])
                row2 = '\t'.join([self._max_ind_and_diff(k, vec) 
                                        for k, vec in dat])
                r.write(row1+'\t'+row2+'\n')