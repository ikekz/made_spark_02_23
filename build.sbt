name := "linear_regression"

scalaVersion := "2.12.18"

//fork := true
//javaOptions += "--illegal-access=warn"

val sparkVersion = "3.0.1"
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % sparkVersion withSources (),
  "org.apache.spark" %% "spark-mllib" % sparkVersion withSources ()
)

libraryDependencies += ("org.scalatest" %% "scalatest" % "3.2.2" % "test" withSources ())
