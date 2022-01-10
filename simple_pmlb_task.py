# A simple benchmark experiment for Evolutionary Forest
# pip install pmlb --upgrade
import operator
from functools import partial
from multiprocessing.pool import Pool

import pandas as pd
import pmlb
from evolutionary_forest.forest import EvolutionaryForestRegressor
from evolutionary_forest.multigene_gp import *
from pmlb.dataset_lists import df_summary
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pmlb.pmlb.GITHUB_URL = 'https://github.com.cnpmjs.org/EpistasisLab/penn-ml-benchmarks/raw/master/datasets'


def cxOnePoint_multiple_all_gene(ind1: MultipleGeneGP, ind2: MultipleGeneGP, probability):
    for a, b in zip(ind1.gene, ind2.gene):
        if random.random() < probability:
            cxOnePoint(a, b)
    return ind1, ind2


def mutate_all_gene(individual: MultipleGeneGP, expr, pset, probability):
    if random.random() < probability:
        mutUniform(individual.weight_select(), expr, pset)
    return individual,


class EvolutionaryForestRegressorPlus(EvolutionaryForestRegressor):

    def lazy_init(self, x):
        super().lazy_init(x)

        self.pset.addPrimitive(np.sin, 2)
        self.pset.addPrimitive(np.cos, 2)
        self.toolbox.register("mate", partial(cxOnePoint_multiple_all_gene, probability=self.cross_pb))
        self.toolbox.register("mutate", partial(mutate_all_gene,
                                                expr=self.toolbox.expr_mut, pset=self.pset,
                                                probability=self.mutation_pb))
        self.toolbox.decorate("mate", staticLimit_multiple_gene(key=operator.attrgetter("height"),
                                                                max_value=self.max_height))
        self.toolbox.decorate("mutate", staticLimit_multiple_gene(key=operator.attrgetter("height"),
                                                                  max_value=self.max_height))
        self.cross_pb = 1
        self.mutation_pb = 1


debug = True


def training(dataset):
    X, y = pmlb.fetch_data(dataset, local_cache_dir='./', return_X_y=True)
    X = np.array(X)
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    if debug:
        ef = EvolutionaryForestRegressorPlus(max_height=3, normalize=True, select='AutomaticLexicase',
                                             score_func='Spearman', gene_num=20, boost_size=30,
                                             n_gen=5, n_pop=5, cross_pb=0.5, verbose=True)
    else:
        ef = EvolutionaryForestRegressorPlus(max_height=3, normalize=True, select='AutomaticLexicase',
                                             score_func='Spearman', gene_num=20, boost_size=30,
                                             n_gen=50, n_pop=200, cross_pb=0.5, verbose=True)
    ef = Pipeline([
        ('Scaler', StandardScaler()),
        ('EF', ef)
    ])
    ef.fit(x_train, y_train)
    score = {
        'dataset': dataset,
        'train_score': r2_score(y_train, ef.predict(x_train)),
        'test_score': r2_score(y_test, ef.predict(x_test))
    }
    return score


if __name__ == '__main__':
    instance = 1000
    datasets = df_summary.query(f'task=="regression"&n_instances<={instance}')
    datasets = datasets.sort_values(['n_instances'], ascending=False)['dataset'].tolist()
    datasets = list(filter(lambda x: 'fri' not in x, datasets))
    print('Number of Datasets', len(datasets))
    # dataset = fetch_openml('195')
    if debug:
        all_exp_data = training(datasets[0])
    else:
        all_exp_data = Pool().map(training, datasets)
        all_exp_data = pd.DataFrame(all_exp_data)
    print(all_exp_data)
