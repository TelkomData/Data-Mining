from pyspark import SparkContext, SparkConf, HiveContext
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.types import StructType, StructField

from itertools import combinations
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:0.1f}'.format

from IPython.display import display
from scipy.stats import chi2_contingency

import timeit
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use('ggplot')

def load_parquet(database, table, sqlContext, sc):
    sqlContext.setConf('spark.sql.parquet.binaryAsString', 'True')
    return sqlContext.sql('Select * from parquet.`/user/hive/warehouse/{:s}.db/{:s}`'.format(database, table)), sc, sqlContext, table
    
path = '/var/lib/hadoop-hdfs/Jannes Test/dm_library/graphs'
http_path = 'http://cdh578egzp.telkom.co.za:8880/files/Jannes%20Test/dm_library/graphs'

def toPandas(df, n_partitions=None):
    if n_partitions is not None: 
        df = df.repartition(n_partitions)
    df.persist()
    def _map_to_pandas(rdds):
        return [pd.DataFrame(list(rdds))]
    df_pand = df.rdd.mapPartitions(_map_to_pandas).collect()
    df_pand = pd.concat(df_pand)
    df_pand.columns = df.columns
    df.unpersist()
    return df_pand

def g_barplot(df, cat, outlier_thres = 0.05): #, df_name=''):
    pd_df = toPandas(df.groupby(cat).count(),10)
    pd_df = pd_df[pd_df['count'] > outlier_thres*pd_df['count'].sum()]
    plt.figure(figsize=(8,5))
    try:
        cht = sns.barplot(data=pd_df.reset_index().sort_values(['count'], ascending=[False]), y=cat, x='count', color='b', orient='h')
    except:
        cht = 0
    cht_path = path + '/' + cat + '.png'
    plt.title(cat)
    plt.tight_layout()
    plt.savefig(cht_path)
    http_cht_path = http_path + '/' + cat + '.png'
    #plt.show()
    plt.clf()
    return cht, http_cht_path

def g_histogram(df, con, buckets=100, _min=-250, _max=250, kde=True): 
    pd_df = toPandas(df.select(con).dropna(),10)
    pd_df = pd_df[(pd_df[con] > _min) & (pd_df[con] < _max)]
    plt.figure(figsize=(8,5))
    try:
        cht = sns.distplot(a=pd_df.dropna(), bins=buckets, kde=kde)
    except:
        cht = 0
    cht_path = path + '/' + con + '.png'
    plt.title(con)
    plt.tight_layout()
    plt.savefig(cht_path)
    http_cht_path = http_path + '/' + con + '.png'
    #plt.show()
    plt.clf()
    return cht, http_cht_path

def g_scatter(df, con1, con2):
    pd_df = toPandas(df.select(con1, con2).dropna(),10)
    plt.figure(figsize=(8,5))
    try:
        cht = sns.lmplot(data=pd_df, x=con1, y=con2)
    except:
        cht = 0
    cht_path = path + '/' + con1 + '_vs_' + con2 + '.png'
    plt.title(con1 + ' vs ' + con2)
    plt.tight_layout()
    plt.savefig(cht_path)
    http_cht_path = http_path + '/' + con1 + '_vs_' + con2 + '.png'
    #plt.show()
    plt.clf()
    return cht, http_cht_path

def g_kde(df, cat, con, _min=-250, _max=250):
    pd_df = toPandas(df.select(cat, con).dropna(),10)
    pd_df = pd_df[(pd_df[con] > _min) & (pd_df[con] < _max)]
    categories = set(df.select(cat).rdd.flatMap(lambda x: x).collect())
    plt.figure(figsize=(8,5))
    for i in categories:
        try:
            cht = sns.kdeplot(pd_df[pd_df[cat]==i][con], label=i, shade=True)
        except:
            cht = 0
    cht_path = path + '/' + cat + '_vs_' + con + '.png'
    plt.title(cat + '_vs_' + con)
    plt.tight_layout()
    plt.savefig(cht_path)
    http_cht_path = http_path + '/' + cat + '_vs_' + con + '.png'
    #plt.show()
    plt.clf()
    return cht, http_cht_path

def g_chi2(df, cat1, cat2):
    pd_df = toPandas(df.crosstab(cat1,cat2),10)
    ctsum = pd_df.set_index(cat1+"_"+cat2)
    ctsum['sum'] = ctsum.sum(axis=1)
    ctsum = ctsum.sort_values('sum', ascending=False)
    ctsum.drop('sum', axis=1, inplace=True)

    chi2, p, dof, ex = chi2_contingency(ctsum, correction=False)
    diff_perc = (ctsum.fillna(0) - ex) / ex.sum()
    plt.figure(figsize=(8,5))
    try:
        cht = sns.heatmap(diff_perc, annot=(ctsum.values / 1000).round(0), fmt=',d',vmin=-0.02, vmax=0.02 )
    except:
        cht = 0
    cht_path = path + '/' + cat1 + '_vs_' + cat2 + '.png'
    plt.title(cat1 + '_vs_' + cat2)
    plt.tight_layout()
    plt.savefig(cht_path)
    http_cht_path = http_path + '/' + cat1 + '_vs_' + cat2 + '.png'
    #plt.show()
    plt.clf()
    return cht, http_cht_path

def get_rate_df(data, sum_fields, rate_field, d_field, rate_cond = ['1'], d_cond = ['FDC24']):
    # take in pyspark.dataframe, columns to summarise and return where a true_fdc24_redispatch was == 1 as a pandas dataframe
    sum_fields = list(set(sum_fields))   
    pd_data = data.groupBy(sum_fields).agg(
        F.count((F.col(d_field).isin(d_cond)).cast('int')).alias('Total'),
        F.sum((F.col(rate_field).isin(rate_cond)).cast('int')).alias(rate_field))
    print(type(pd_data))
    pd_data = pd_data.withColumn('rate', pd_data[rate_field] / pd_data['Total'])
    pd_data.fillna('No Data')
    return pd_data

def create_table(sc, sqlContext, table, df, size_limit = 20):
    
    df.persist()
    no_plot_cols = []
    output = []

    type_dict = {'float':'numeric','long':'numeric', 'integer':'numeric', 
                 'smallint':'numeric', 'int':'numeric', 'bigint':'numeric', 'string':'categorical', 
                 'timestamp':'date', 'binary':'indicator','decimal(9,2)':'numeric'}

    for c in df.columns:
        
        var_cols = ['colm', 'col_type', 'uniques', 'missing', 'mean', 'stddev', 'graph']
        col_graphs = ['graph_'+str(s) for s in list(set(df.columns)-set(c.split()))]
        var_cols.extend(col_graphs)
        
        #print("Producing graphs" + str(col_graphs))
       
        cols_complete = []
        cols_complete.append(c)
        rem_cols = list(set(df.columns) - set(cols_complete))
        
        #Initialize columns

        uniq = 0
        null = 0 
        mean = 0 
        std_dev = 0
        g = 0
        g_path = 0
        col_g = []
        col_g_paths = []
        col_g.extend(np.zeros(len(col_graphs)))
        
        print 'Getting {:s} data'.format(c)
        
        col_type = df.select(c).dtypes[0][1]
        col_type = type_dict[col_type]
        
        print(col_type)
    
        print 'Type Done'
        
        uniq = df.select(c).distinct().count()
        
        print(uniq)
        
        print 'Unique Count Done'
        
        null = df.where(F.col(c).isNull()).count()
        
        print 'Missing Count Done'
    
        if uniq == 2:
            col_type = 'indicator'
        
        if (uniq < size_limit) & (col_type == 'categorical'):
            g, g_path = g_barplot(df, c) 
        
        if col_type == 'numeric':   
            df_sum = df.select(c).agg(F.avg(F.col(c)),
                                   F.stddev(F.col(c))).take(1)
            mean = df_sum[0][0]
            std_dev = df_sum[0][1]
            
            g, g_path = g_histogram(df, c)
        
        print('Single Graph Done')
        
        if not (uniq > size_limit) & (col_type in ['categorical','date']): 
            
            for j in list(set(rem_cols)-set(no_plot_cols)):
                
                uniq_2 = df.select(j).distinct().count()
                col_2_type = df.select(j).dtypes[0][1]
                col_2_type = type_dict[col_2_type]
                
                if not (uniq_2 > size_limit) & (col_2_type in ['categorical', 'date']):
                    
                    print(c,j)  
                    
                    if uniq_2 == 2:
                        col_2_type = 'indicator'

                    if (col_type == 'categorical') & (col_2_type == 'categorical'):
                        if uniq < size_limit and uniq_2 < size_limit:
                            col_g.append(g_chi2(df, c, j)[0])
                            col_g_paths.append(g_chi2(df, c, j)[1])
                        elif uniq < size_limit and uniq_2 > size_limit:
                            no_plot_cols.append(j)
                        elif uniq > size_limit and uniq_2 < size_limit:
                            no_plot_cols.append(c)
                        else:
                            no_plot_cols.extend([c,j])    

                    if (col_type == 'categorical') & (col_2_type == 'numeric'):
                        if uniq < size_limit:
                            col_g.append(g_kde(df, c, j)[0])
                            col_g_paths.append(g_kde(df, c, j)[1])
                        else:
                            no_plot_cols.append(c)
                    if (col_type == 'numeric') & (col_2_type == 'categorical'):
                        if uniq_2 < size_limit:
                            col_g.append(g_kde(df, j, c)[0])
                            col_g_paths.append(g_kde(df, j, c)[1])
                        else:
                            no_plot_cols.append(j)    

                    if (col_type == 'numeric') & (col_2_type == 'numeric'):
                            col_g.append(g_scatter(df, c, j)[0])
                            col_g_paths.append(g_scatter(df, c, j)[1])
                else:
                    pass
                    
        else:
            pass                       
                    
                 
                
            
        var_list = [c, col_type, uniq, null, mean, std_dev, g_path]
        var_list.extend(col_g_paths)
        var_tuple = tuple(var_list)
        output.append(var_tuple)
        output_final = []

    
        for row in output:
            if len(row) < len(var_cols):
                row_list = list(row)
                row_list.extend(['Too large to plot' for x in range(len(var_cols) - len(row))])
                row_tuple = tuple(row_list)
                output_final.append(row_tuple)
            else:
                output_final.append(row)
        
        schema_list = [T.StructField("colm", T.StringType(), True),
    T.StructField("col_type", T.StringType(), True),
    T.StructField("uniques", T.IntegerType(), True),
    T.StructField("missing", T.IntegerType(), True),T.StructField("mean", T.FloatType(), True),
    T.StructField("stddev", T.FloatType(), True),T.StructField("graph", T.StringType(), True) ]
        graph_schema_list = [T.StructField(x, T.StringType(), True) for x in col_graphs]
        schema_list.extend(graph_schema_list)
        schema = T.StructType(schema_list) 
        
    rdd = sc.parallelize(output_final) 
    sqlContext.createDataFrame(rdd, schema=schema).write.mode('overwrite').saveAsTable('{:s}.{:s}'.format('datamining', table),format='parquet')
    
    df.unpersist()
    
    
    
    
    
def create_graphs(sc, sqlContext, table, df):
    
    df.persist()
    no_plot_cols = []
    output = []

    type_dict = {'float':'numeric','long':'numeric', 'integer':'numeric', 
                 'smallint':'numeric', 'int':'numeric', 'bigint':'numeric', 'string':'categorical', 
                 'timestamp':'date', 'binary':'indicator','decimal(9,2)':'numeric'}
  
    for c in df.columns:
        
        #print("Producing graphs" + str(col_graphs))
        col_graphs = ['graph_'+str(s) for s in list(set(df.columns)-set(c.split()))]
        cols_complete = []
        cols_complete.append(c)
        rem_cols = list(set(df.columns) - set(cols_complete))
        
        #Initialize columns

        g = 0
        g_path = 0
        col_g = []
        col_g_paths = []
        col_g.extend(np.zeros(len(col_graphs)))
        
        print 'Getting {:s} data'.format(c)
        
        col_type = df.select(c).dtypes[0][1]
        col_type = type_dict[col_type]
        
        print(col_type)
    
        print 'Type Done'
        
        if col_type == 'categorical':
            g, g_path = g_barplot(df, c) 
        
        if col_type == 'numeric':  
            g, g_path = g_histogram(df, c)
        
        print('Single Graph Done')
            
        for j in list(set(rem_cols)):
                
            col_2_type = df.select(j).dtypes[0][1]
            col_2_type = type_dict[col_2_type] 
                    
            print(c,j)  
           
            if (col_type == 'categorical') & (col_2_type == 'categorical'):
                col_g.append(g_chi2(df, c, j)[0])
                col_g_paths.append(g_chi2(df, c, j)[1])

            elif (col_type == 'categorical') & (col_2_type == 'numeric'):
                col_g.append(g_kde(df, c, j)[0])
                col_g_paths.append(g_kde(df, c, j)[1])
                        
            elif (col_type == 'numeric') & (col_2_type == 'categorical'):    
                col_g.append(g_kde(df, j, c)[0])
                col_g_paths.append(g_kde(df, j, c)[1])
                        
            elif (col_type == 'numeric') & (col_2_type == 'numeric'):
                col_g.append(g_scatter(df, c, j)[0])
                col_g_paths.append(g_scatter(df, c, j)[1])
            else:
                pass
    
    df.unpersist()