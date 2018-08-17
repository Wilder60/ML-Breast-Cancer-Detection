import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.rdd._
import org.apache.spark.mllib.clustering._

//importing the text file
val file = sc.textFile("/home/andrew/Documents/Programs/sparkcode/BreastCancer/wdbc.txt")

//parsing the data to find the correct output
val line = file.map (x=>x.split(','))
val patientId = line.map(x=>(x(0),(x(1),Vectors.dense(x.drop(2).map(x=>x.toDouble)))))
val patientlist = patientId.mapValues(x=> if (x._1 == "M") (1.0, x._2) else (0.0, x._2))

//Creating the k-means Cluster and passing in data to learn from
val parsedData = file.map(x=> Vectors.dense(x.split(',').drop(2).map(x=> x.toDouble)))
val kmeans = new KMeans()
kmeans.setK(2)
val model = kmeans.run(parsedData) //parsedData is an RDD vector
model.clusterCenters.foreach(println)

//checking to see the correctness of the output of the cluster
val test = patientlist.map(x=>(x._1, (x._2._1.toInt, model.predict(x._2._2))))
val misdiagnosed = test.filter{case(a,b) => b._1 != b._2}
misdiagnosed.collect.foreach(println)
