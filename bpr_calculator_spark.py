import pickle
import pandas as pd
import numpy as np
import pandas, numpy
from collections import defaultdict
from pyspark.sql.window import Window
from pyspark.sql import functions, DataFrame, SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark import SparkContext, StorageLevel
from pyspark.sql.types import * #StructType, StructField, LongType, StringType, ArrayType
import psycopg2
import time, datetime
from pyspark.sql.functions import *

# tag::cutLineage[]
def cutLineageCache(df):
    """
    Cut the lineage of a DataFrame - used for iterative algorithms
    .. Note: This uses internal members and may break between versions
    >>> df = rdd.toDF()
    >>> cutDf = cutLineage(df)
    >>> cutDf.count()
    3
    """
    jRDD = df._jdf.toJavaRDD()
    jSchema = df._jdf.schema()
    jRDD.cache()
    sqlCtx = df.sql_ctx
    try:
        javaSqlCtx = sqlCtx._jsqlContext
    except:
        javaSqlCtx = sqlCtx._ssql_ctx
    newJavaDF = javaSqlCtx.createDataFrame(jRDD, jSchema)
    newDF = DataFrame(newJavaDF, sqlCtx)
    return newDF
# end::cutLineage[]

# tag::cutLineage[]
def cutLineagePersist(df):
    """
    Cut the lineage of a DataFrame - used for iterative algorithms
    .. Note: This uses internal members and may break between versions
    >>> df = rdd.toDF()
    >>> cutDf = cutLineage(df)
    >>> cutDf.count()
    3
    """
    global rrr
    try:
        xx = rrr
    except:
        rrr = df._sc._jvm.org.apache.spark.storage.StorageLevel(True, True, False, True, 1)
    jRDD = df._jdf.toJavaRDD()
    jSchema = df._jdf.schema()
    jRDD.persist(rrr)
    sqlCtx = df.sql_ctx
    try:
        javaSqlCtx = sqlCtx._jsqlContext
    except:
        javaSqlCtx = sqlCtx._ssql_ctx
    newJavaDF = javaSqlCtx.createDataFrame(jRDD, jSchema)
    newDF = DataFrame(newJavaDF, sqlCtx)
    return newDF
# end::cutLineage[]

def deleteHDFSPathIfExists(ss, path):
    fs = ss._jvm.org.apache.hadoop.fs.FileSystem.get(ss._jsc.hadoopConfiguration())
    p = ss._jvm.org.apache.hadoop.fs.Path(path)
    if (fs.exists(p)):
        fs.delete(p)

#TODO:: this can be used when list is in spark data frame, please test it
@pandas_udf("float", PandasUDFType.SCALAR)
def fz_list(uv, iv, jv, ib, jb):
    niv = iv.apply(lambda x: numpy.asarray(x))
    njv = jv.apply(lambda x: numpy.asarray(x))
    nuv = uv.apply(lambda x: numpy.asarray(x))
    imjv = niv-njv
    u_dot_i = nuv.mul(imjv).apply(lambda x:x.sum())
    z = 1.0/((u_dot_i + ib - jb).apply(lambda x:numpy.exp(x)) + 1.0)
    return z

#TODO:: this cannot be achieved.... vector would be a tuple when transmitted in, lost all capabilities
@pandas_udf("float", PandasUDFType.SCALAR)
def fz_vector(uv, iv, jv, ib, jb):
    imjv = iv-jv
    return uv.dot(imjv)

#TODO:: this cannot be achieved.... numpy cannot be stored in dataframe
@pandas_udf("float", PandasUDFType.SCALAR)
def fz_numpy(uv, iv, jv, ib, jb):
    imjv = iv-jv
    return uv

def unionAll(*dfs):
    if not dfs:
        raise ValueError()
    first = dfs[0]
    return first.sql_ctx.createDataFrame(first._sc.union([df.rdd for df in dfs]), first.schema)

def cut(df):
    return df.sql_ctx.createDataFrame(df.rdd)

class BayesianPersonalizationRanking(object):
    #
    def __init__(self, save_path):
        self.save_path = save_path
    #
    def build(self, ratings, ss, params = None, minRating = 1, k=25, num_iterations=4000, batchSize = 1000, partitionNum = 4, learning_rate = 0.05, bias_regularization = 0.002, user_regularization = 0.005, positive_item_regularization = 0.003, negative_item_regularization = 0.0003):
        print(self)
        print(ratings)
        print(ss)
        #
        if params:
            k = params['k']
            num_iterations = params['num_iterations']
            batchSize = params['batchSize']
            partitionNum = params['partitionNum']
            learning_rate = params['learning_rate']
            bias_regularization = params['bias_regularization']
            user_regularization = params['user_regularization']
            positive_item_regularization = params['positive_item_regularization']
            negative_item_regularization = params['negative_item_regularization']
            minRating = params['minRating']
        #
        #remove duplicates(Or avarage them???)
        udata = ratings.select('user_id', 'movie_id', 'rating', 'type', 'rating_timestamp', functions.row_number().over(Window.partitionBy("user_id", "movie_id").orderBy(functions.desc("rating_timestamp"))).alias('seq')).where("seq = 1").select('user_id', 'movie_id', 'rating').repartition(partitionNum)
        #remove movies less than min ratings
        udata.persist(StorageLevel.MEMORY_AND_DISK)
        udata.createOrReplaceTempView("udata")
        umovie = udata.groupby('movie_id').count().where("count > {}".format(minRating)).select("movie_id")
        umovie.persist(StorageLevel.MEMORY_AND_DISK)
        umovie.createOrReplaceTempView("umovie")
        rdata = ss.sql("select * from udata where movie_id in (select movie_id from umovie)")
        rdata.persist(StorageLevel.MEMORY_AND_DISK)
        rdata.createOrReplaceTempView("rdata")
        #
        #create mappings, map movie id and user id into tables, id start from 1
        movieMapping = umovie.withColumn("id", functions.row_number().over(Window.orderBy('movie_id')))
        movieMapping.persist(StorageLevel.MEMORY_AND_DISK)
        movieMapping.createOrReplaceTempView("movieMapping")
        userMapping = rdata.select("user_id").distinct().withColumn("id", functions.row_number().over(Window.orderBy('user_id')))
        userMapping.persist(StorageLevel.MEMORY_AND_DISK)
        userMapping.createOrReplaceTempView("userMapping")
        mdata = ss.sql("select A.userID, movieMapping.id as movieID, A.rating from (select userMapping.id as userID, rdata.movie_id, rdata.rating from rdata join userMapping on rdata.user_id == userMapping.user_id) as A join movieMapping on A.movie_id = movieMapping.movie_id")
        #rdata is original user_movies, with userMapping and movieMapping
        userMovie = mdata.withColumn('id', functions.row_number().over(Window.orderBy('userID')))
        userMovie.persist(StorageLevel.MEMORY_AND_DISK)
        #
        self.userMovie = userMovie
        self.userMovieCT = userMovie.count()
        self.userMapping = userMapping
        self.userCT = userMapping.count()
        self.movieMapping = movieMapping
        self.movieCT = movieMapping.count()
        self.user_factors = self.appendVectorCol(userMapping.drop('user_id'), k, 'v')
        self.item_factors = self.appendVectorCol(movieMapping.drop('movie_id'), k, 'v').withColumn('b', functions.lit(0))
        #self.item_bias = movieMapping.drop('movie_id').withColumn('v', functions.lit(0))
        self.batchSize = batchSize
        self.partitionNum = partitionNum
        #self.lastSamples = None
        #self.userMovies = None
        self.k = k
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.bias_regularization = bias_regularization
        self.user_regularization = user_regularization
        self.positive_item_regularization = positive_item_regularization
        self.negative_item_regularization = negative_item_regularization
        self.lastSamples = None
        self.userMovies = None
        self.testSamples = None
        #
        #TODO:: this can be used when list is in spark data frame, please test it
        @pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR)
        def fuvu_list(uv, iv, jv, z):
            niv = iv.apply(lambda x: numpy.asarray(x))
            njv = jv.apply(lambda x: numpy.asarray(x))
            nuv = uv.apply(lambda x: numpy.asarray(x))
            uvu = learning_rate * ((niv-njv) * z - user_regularization * nuv)
            return uvu
        #
        @pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR)
        def fivu_list(uv, iv, z):
            niv = iv.apply(lambda x: numpy.asarray(x))
            nuv = uv.apply(lambda x: numpy.asarray(x))
            ivu = learning_rate * (nuv * z - niv * positive_item_regularization)
            return ivu
        #
        @pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR)
        def fjvu_list(uv, jv, z):
            njv = jv.apply(lambda x: numpy.asarray(x))
            nuv = uv.apply(lambda x: numpy.asarray(x))
            jvu = learning_rate * (-nuv * z - njv * negative_item_regularization)
            return jvu
        #
        #the below 2 are alternatives in pandas version
        @pandas_udf(ArrayType(FloatType()), PandasUDFType.GROUPED_AGG)
        def fvavg_list(v):
            nv = v.apply(lambda x: numpy.asarray(x))
            avgv = nv.mean()
            return avgv
        #
        @pandas_udf(ArrayType(FloatType()), PandasUDFType.SCALAR)
        def fvadd_list(v1, v2):
            nv1 = v1.apply(lambda x: numpy.asarray(0.0) if x is None else numpy.asarray(x))
            nv2 = v2.apply(lambda x: numpy.asarray(0.0) if x is None else numpy.asarray(x))
            nv = nv1 + nv2
            return nv
        #
        @pandas_udf(FloatType(), PandasUDFType.SCALAR)
        def fnadd(v1, v2):
            nv1 = v1.apply(lambda x: 0.0 if x is None else x)
            nv2 = v2.apply(lambda x: 0.0 if x is None else x)
            nv = nv1 + nv2
            return nv
        #
        self.pd_fuvu = fuvu_list
        self.pd_fivu = fivu_list
        self.pd_fjvu = fjvu_list
        #
        #This is the only way proved to be successful in doing sql multiplication
        ss.udf.register('ldot', lambda x, y: float(numpy.dot(x, y)))
        ss.udf.register('lsub', lambda x, y: numpy.subtract(x, y, dtype='float').tolist(), ArrayType(FloatType()))
        ss.udf.register('lavg', lambda x: numpy.average(x, axis=0).tolist(), ArrayType(FloatType()))
        ss.udf.register('ladd', lambda x, y: numpy.add(0 if x is None else x, 0 if y is None else y, dtype='float').tolist(), ArrayType(FloatType()))
        ss.udf.register('nadd', lambda x, y: float(0 if x is None else x + 0 if y is None else y), FloatType())
        #
        print("Initialization success")
        #
        #generate testSamples, align to batchSize for simplification
        numTestSamples = int(100 * self.userCT ** 0.5)
        self.numTestSamples = (numTestSamples + batchSize - 1) // batchSize * batchSize
        ct = 0
        testSamples = None
        while(ct < self.numTestSamples):
            samples = self.draw(ss)
            if testSamples is None:
                testSamples = samples
            else:
                testSamples = testSamples.unionAll(samples)
                #testSamples = unionAll([testSamples, samples])
            testSamples.persist(StorageLevel.MEMORY_AND_DISK)
            ct = testSamples.count()
            print(ct)
        testSamples.repartition(partitionNum)
        testSamples=cutLineagePersist(testSamples)
        testSamples.persist(StorageLevel.MEMORY_AND_DISK)
        testSamples.count()
        self.testSamples = testSamples
        print("Sampling success")
        #
        self.train(ss)
        print("training success")
    #
    #tempname must be a valid name with {} to fit in number formats
    def appendVectorCol(self, df, k, name, tempname="__t{}", min = 0.0, max = 1.0, seed = None):
        cols = []
        ndf = df
        for i in range(k):
            ndf = ndf.withColumn(tempname.format(i), functions.rand(seed)*(max - min) + min)
            cols.append(tempname.format(i))
        ndf = ndf.withColumn(name, functions.array(cols)).drop(*cols)
        return ndf
    #
    def loss(self, ss):
        ranking_loss = 0
        self.testSamples.createOrReplaceTempView("tts")
        self.user_factors.createOrReplaceTempView('uf')
        self.item_factors.createOrReplaceTempView('if')
        #append vector and bias for user(no bias) and item
        uij = ss.sql("""
            select A.*, uf.v as uv from uf join 
                (select A.*, if.v as jv, if.b as jb from if join 
                    (select A.*, if.v as iv, if.b as ib from tts as A join if on if.id = A.i) as A
                on if.id = A.j) as A
            on uf.id = A.u
            """).repartition(self.partitionNum)
        uij.persist(StorageLevel.MEMORY_AND_DISK)
        uij.createOrReplaceTempView("uij")
        zloss = ss.sql("select 1.0/(exp(ldot(uv, iv) - ldot(uv, jv) + ib - jb) + 1.0) as z from uij").agg(sum("z")).toPandas().values[0, 0]
        rloss = ss.sql("select {}*ldot(uv, uv) + {}*ldot(iv, iv) + {}*ldot(jv, jv) + {}*(ib*ib + jb*jb) as r from uij".format(
			self.user_regularization, self.positive_item_regularization, self.negative_item_regularization, self.bias_regularization)).agg(sum("r")).toPandas().values[0, 0]
        return zloss + 0.5 * rloss
    #        
    #this action would draw a batch size of data into samples, and remove the duplicates, if there are more than necessary items generated, then left them for next use
    def draw(self, ss, seed = None):
        lastSamples = self.lastSamples
        batchSize = self.batchSize
        #
        userMovie = self.userMovie
        userMovie.createOrReplaceTempView("userMovie")
        #
        if self.userMovies is None:
            self.userMovies = userMovie.select("userID", "movieID").groupby("userID").agg(functions.collect_set("movieID").alias('movieIDs'))
        userMovies = self.userMovies
        userMovies.createOrReplaceTempView("userMovies")
        #
        ec = self.userMovieCT
        mc = self.movieCT
        #if there is not enough sample, generate some
        while(lastSamples is None or lastSamples.count() < batchSize):
            #
            #generate index of event and movie, note the id starts from 1 because of row_number
            samples = ss.range(batchSize).withColumn('e', functions.floor(functions.rand(seed=seed)*(ec+2)) % (ec+1)).withColumn('j', functions.floor(functions.rand(seed=seed)*(mc+2)) % (mc+1))
            # 1. interpret e with u, i, 2. filter out lines with no overlaps finally get u, i, j
            #TODO:: use the equvialent to cut code short
            #samples = samples.join(userMovie, userMovie.id == samples.e).select(userMovie.userID.alias('u'), userMovie.movieID.alias('i'), samples.j)
            samples.createOrReplaceTempView("samples")
            samples = ss.sql("select userMovie.userID as u, userMovie.movieID as i, samples.j from samples join userMovie on samples.e = userMovie.id")
            # 3. link event with time
            samples.createOrReplaceTempView("samples")
            gsamples = ss.sql("select samples.*, now() as t from samples join userMovies on samples.u = userMovies.userID where not array_contains(userMovies.movieIDs, samples.j)")
            gsamples.persist(StorageLevel.MEMORY_AND_DISK)
            print(gsamples.count())
            if lastSamples is None:
                lastSamples = gsamples
            else:
                lastSamples = lastSamples.unionAll(gsamples)
                #lastSamples = unionAll([lastSamples,gsamples])
            lastSamples.persist(StorageLevel.MEMORY_AND_DISK)
        #
        #extract first batchSize of samples, and the rest to lastSamples
        sampleWithIndex = lastSamples.withColumn("id", functions.row_number().over(Window.orderBy("t")))
        #sampleWithIndex.persist(StorageLevel.MEMORY_AND_DISK)
        sampleWithIndex = cutLineagePersist(sampleWithIndex)
        #add this to cut too long sql expr which may boom spark context
        sampleWithIndex.persist(StorageLevel.MEMORY_AND_DISK)
        sampleWithIndex.count()
        #
        sampleForReturn = sampleWithIndex.filter(sampleWithIndex.id <= batchSize).drop('id', 't')
        lastSamples = sampleWithIndex.filter(sampleWithIndex.id > batchSize).drop('id')
        self.lastSamples = lastSamples
        return sampleForReturn
    #
    def train(self, ss):
        #
        lr = self.learning_rate
        br = self.bias_regularization
        #
        fuvu_list = self.pd_fuvu
        fivu_list = self.pd_fivu
        fjvu_list = self.pd_fjvu
        #
        num_iterations = self.num_iterations
        for iteration in range(num_iterations):
            print("iteration #{}".format(iteration))
            self.error = self.loss(ss)
            #
            print('iteration {} loss {}'.format(iteration, self.error))
            #
            samples = self.draw(ss).withColumn('id', functions.row_number().over(Window.orderBy('u'))).repartition(self.partitionNum)
            samples.createOrReplaceTempView("ts")
            samples.select("")
            self.user_factors.createOrReplaceTempView('uf')
            self.item_factors.createOrReplaceTempView('if')
            #append vector and bias for user(no bias) and item
            uij = ss.sql("""
                select A.*, uf.v as uv from uf join 
                    (select A.*, if.v as jv, if.b as jb from if join 
                        (select A.*, if.v as iv, if.b as ib from ts as A join if on if.id = A.i) as A
                    on if.id = A.j) as A
                on uf.id = A.u
                """).repartition(self.partitionNum)
            uij.createOrReplaceTempView("uij")
            uij.persist(StorageLevel.MEMORY_AND_DISK)
            #TODO:: this code can be used when lists are filled with dataframe
            #z = uij.select(fz_list('uv', 'iv', 'jv', 'ib', 'jb').alias('z'), 'id')
            z = ss.sql("""
                select 1/(exp(ib - jb + ldot(uv, imj)) + 1) as z, id from 
                    (select id, uv, lsub(iv, jv) as imj, ib, jb from uij) as A
                """)
            z.persist(StorageLevel.MEMORY_AND_DISK)
            #z.createOrReplaceTempView("z")
            #ib_update = ss.sql("select {} * (z.z - {} * uij.ib) as ibu, z.id, uij.i from z join uij on z.id = uij.id".format(lr, br))
            #jb_update = ss.sql("select {} * (-z.z - {} * uij.jb) as ibu, z.id, uij.j as i from z join uij on z.id = uij.id".format(lr, br))
            ib_update = uij.join(z, z.id == uij.id).select((lr*(z.z-br*uij.ib)).alias('ibu'), z.id, uij.i)
            jb_update = uij.join(z, z.id == uij.id).select((lr*(-z.z-br*uij.jb)).alias('ibu'), z.id, uij.j.alias('i'))
            #
            update_uv = uij.join(z, z.id == uij.id).select(fuvu_list("uv", "iv", "jv", "z").alias('uvu'), z.id, uij.u)
            update_iv = uij.join(z, z.id == uij.id).select(fivu_list("uv", "iv", "z").alias('ivu'), z.id, uij.i)
            update_jv = uij.join(z, z.id == uij.id).select(fjvu_list("uv", "jv", "z").alias('ivu'), z.id, uij.j.alias('i'))
            update_uv.createOrReplaceTempView("update_uv")
            update_iv.unionAll(update_jv).createOrReplaceTempView("update_iv")
            ib_update.unionAll(jb_update).createOrReplaceTempView("update_ib")
            update_uv = ss.sql("select lavg(collect_list(uvu)) as dv, u as id from update_uv group by u")
            update_iv = ss.sql("select lavg(collect_list(ivu)) as dv, i as id from update_iv group by i")
            update_ib = ss.sql("select avg(ibu) as db, i as id from update_ib group by i")
            #
            uv = self.user_factors
            uv.join(update_uv, uv.id == update_uv.id, "left").select(uv.id, uv.v, update_uv.dv).createOrReplaceTempView("uv")
            self.user_factors = ss.sql("select ladd(v, dv) as v, id from uv")
            iv = self.item_factors
            iv = iv.join(update_iv, iv.id == update_iv.id, "left").select(iv.id, iv.v, update_iv.dv, iv.b)
            iv.join(update_ib, iv.id == update_ib.id, "left").select(iv.id, iv.v, iv.dv, functions.when(update_ib.db.isNull(), iv.b).otherwise(iv.b + update_ib.db).alias('b')).createOrReplaceTempView("iv")
            self.item_factors = ss.sql("select id, ladd(v, dv) as v, b from iv")
            #
            oif = self.item_factors
            ouf = self.user_factors
            self.item_factors = cutLineagePersist(self.item_factors)
            self.user_factors = cutLineagePersist(self.user_factors)
            self.item_factors.persist(StorageLevel.MEMORY_AND_DISK)
            self.user_factors.persist(StorageLevel.MEMORY_AND_DISK)
            self.item_factors.count()
            self.user_factors.count()
            oif.unpersist()
            ouf.unpersist()
        #
        print("iteration #{}".format(iteration))
        self.error = self.loss(ss)
        print('iteration {} loss {}'.format(iteration, self.error))
        raw_input()
        self.save(iteration, iteration == num_iterations - 1, ss)
    #
    def save(self, ss):
        #
        save_path = self.save_path + '/model/'
        #
        #remove all data alread exists
        deleteHDFSPathIfExists(ss, save_path)
        #
        logger.info("saving factors in {}".format(save_path))
        #
        #with open(save_path + 'user_factors.json', 'w') as outfile:
        #    outfile.write(uf.to_json())
        um = self.userMapping
        uv = self.user_factors
        ufpd = uv.join(um, 'id').select(uv.v, um.user_id).toPandas()
        ufpd.index = list(ufpd.user_id)
        j = ufpd.v.apply(pandas.Series).to_json()
        ss.sparkContext.parallelize([j]).saveAsTextFile(save_path + 'user_factors.json')
        #
        #with open(save_path + 'item_factors.json', 'w') as outfile:
        #    outfile.write(it_f.to_json())
        mm = self.movieMapping
        iv = self.item_factors
        ifpd = iv.join(mm, 'id').select(iv.v, mm.movie_id).toPandas()
        ifpd.index = list(ifpd.movie_id)
        j = ifpd.v.apply(pandas.Series).to_json()
        ss.sparkContext.parallelize([j]).saveAsTextFile(save_path + 'item_factors.json')
        #
        #with open(save_path + 'item_bias.data', 'wb') as ub_file:
        #    pickle.dump(item_bias, ub_file)
        ibpd = iv.join(mm, 'id').select(mm.movie_id, iv.b).toPandas()
        d = {}
        for id, b in ibpd[['movie_id', 'b']].itertuples(index = False):
            d[str(id)] = b
        ss.sparkContext.parallelize([pickle.dumps(d)]).saveAsTextFile(save_path + 'item_bias.data')

def test(ss):
    number_of_factors = 25
    train_data = ss.read.jdbc("jdbc:postgresql://192.168.97.30:5432/moviegeek", "analytics_rating", properties = {"user":"postgres", "password":"123456"})
    bpr = BayesianPersonalizationRanking(save_path='/models/bpr/')
    bpr.build(train_data, ss, k = number_of_factors, num_iterations=80000)
    return

if __name__ == '__main__':

    number_of_factors = 25
    #create spark session
    ss = SparkSession.builder.appName("BPR").config("spark.default.parallelism", 20).config("spark.sql.shuffle.partitions", 20).getOrCreate()

    #TODO:: connect and load rating table
    train_data = ss.read.jdbc("jdbc:postgresql://192.168.97.30:5432/moviegeek", "analytics_rating", properties = {"user":"postgres", "password":"123456"})

    bpr = BayesianPersonalizationRanking(save_path='/models/bpr/')
    bpr.build(train_data, ss, k = number_of_factors, num_iterations=80000)
    
    user, item, itemBias = buildBPRModel(ss, data)



