package org.apache.spark.ml.made

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec._
import org.scalatest.matchers._

import breeze.linalg.{*, DenseMatrix, DenseVector}

import com.google.common.io.Files

class LinearRegressionTest
    extends AnyFlatSpec
    with should.Matchers
    with WithSpark {
  val delta = 0.01
  val w: DenseVector[Double] = LinearRegressionTest._w
  val b: Double = LinearRegressionTest._b
  val real: DenseVector[Double] = LinearRegressionTest._y
  val df: DataFrame = LinearRegressionTest._df

  private def validateModel(model: LinearRegressionModel): Unit = {
    model.w.size should be(w.size)

    (0 until w.size).foreach(i => model.w(i) should be(w(i) +- delta))

    model.b should be(b +- delta)

  }

  private def validateModelAndData(model: LinearRegressionModel, data: DataFrame): Unit = {
    validateModel(model)

    val pred = data.collect().map(_.getAs[Double](1))
    pred.length should be(LinearRegressionTest.DATA_SIZE)
    for (i <- pred.indices) {
      pred(i) should be(real(i) +- delta)
    }
  }

  "Estimator" should "create model" in {
    val estimator = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMaxIter(100)
      .setStepSize(1.0)

    val model = estimator.fit(df)

    validateModel(model)
  }

  "Model" should "predict" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      w = Vectors.fromBreeze(w).toDense,
      b = b
    ).setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")

    validateModelAndData(model, model.transform(df))
  }

  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(
      Array(
        new LinearRegression()
          .setFeaturesCol("features")
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .setMaxIter(100)
          .setStepSize(1.0)
      )
    )

    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = Pipeline.load(tmpFolder.getAbsolutePath)

    val model = reRead.fit(df).stages(0).asInstanceOf[LinearRegressionModel]

    validateModelAndData(model, model.transform(df))
  }

  "Model" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(
      Array(
        new LinearRegression()
          .setFeaturesCol("features")
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .setMaxIter(100)
          .setStepSize(1.0)
      )
    )

    val model = pipeline.fit(df)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tmpFolder.getAbsolutePath)

    validateModelAndData(
      reRead.stages(0).asInstanceOf[LinearRegressionModel],
      reRead.transform(df)
    )
  }
}

object LinearRegressionTest extends WithSpark {
  val DATA_SIZE = 1000
  val VECTOR_SIZE = 2

  lazy val _X: DenseMatrix[Double] =
    DenseMatrix.rand[Double](DATA_SIZE, VECTOR_SIZE)
  lazy val _w: DenseVector[Double] = DenseVector(1.5, -0.7)
  lazy val _b: Double = 1.0
  lazy val _y: DenseVector[Double] = _X * _w + _b
  lazy val _df: DataFrame = createDataFrame(_X, _y)

  def createDataFrame(X: DenseMatrix[Double], y: DenseVector[Double]): DataFrame = {
    import sqlc.implicits._

    lazy val data: DenseMatrix[Double] =
      DenseMatrix.horzcat(X, y.asDenseMatrix.t)

    lazy val df = data(*, ::).iterator
      .map(x => (x(0), x(1), x(2)))
      .toSeq
      .toDF("x1", "x2", "label")

    lazy val assembler = new VectorAssembler()
      .setInputCols(Array("x1", "x2"))
      .setOutputCol("features")

    lazy val _df: DataFrame = assembler
      .transform(df)
      .select("features", "label")

    _df
  }
}
