import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.json4s._
import org.json4s.jackson.Json
import org.json4s.jackson.JsonMethods.{parse, pretty}

import java.io.{File, FileWriter}


object task3 {
  def main(args: Array[String]): Unit = {
    val input_review_path = args(0)
    val input_business_path = args(1)
    val output_path_a = args(2)
    val output_path_b = args(3)
    val conf = new SparkConf().setAppName("task2").setMaster("local[*]")
    val sc = new SparkContext(conf)
    var input_lines_reivew_rdd = sc.textFile(input_review_path).map(x => parse(x)).map(x => (x \ "business_id", x \ "stars"))
    var input_lines_business_rdd = sc.textFile(input_business_path).map(x => parse(x)).map(x => (x\"business_id",x\"city"))

    var star_rdd = input_lines_reivew_rdd.join(input_lines_business_rdd).map(x => (x._2._2,x._2._1))
    val star_average_rdd = star_average(star_rdd.map(x => (x._1,x._2.values.toString.toFloat)))

    var file = new File(output_path_a)
    var fp = new FileWriter(file)
    fp.write("city,stars\n")
    for(i <- 0 until star_average_rdd.length){
      fp.write(star_average_rdd(i)(0) + "," + star_average_rdd(i)(1) + "\n")
    }
    fp.close()

    var default_start = System.nanoTime
    input_lines_reivew_rdd = sc.textFile(input_review_path).map(x => parse(x)).map(x => (x\"business_id",x\"stars"))
    input_lines_business_rdd = sc.textFile(input_business_path).map(x => parse(x)).map(x => (x\"business_id",x\"city"))
    star_rdd = input_lines_reivew_rdd.join(input_lines_business_rdd).map(x => (x._2._2,x._2._1))
    val star_average_rdd_spark = star_average(star_rdd.map(x => (x._1,x._2.values.toString.toFloat))).take(10)
    println(Json(DefaultFormats).write(star_average_rdd_spark))
    val default_exe_time_spark = (System.nanoTime - default_start) / 1e9d

    default_start = System.nanoTime
    input_lines_reivew_rdd = sc.textFile(input_review_path).map(x => parse(x)).map(x => (x\"business_id",x\"stars"))
    input_lines_business_rdd = sc.textFile(input_business_path).map(x => parse(x)).map(x => (x\"business_id",x\"city"))
    star_rdd = input_lines_reivew_rdd.join(input_lines_business_rdd).map(x => (x._2._2,x._2._1))
    val star_average_rdd_java = star_average(star_rdd.map(x => (x._1,x._2.values.toString.toFloat))).take(10)
    println(Json(DefaultFormats).write(star_average_rdd_java))
    val default_exe_time_java = (System.nanoTime - default_start) / 1e9d

    val results = Map[String, Any]("m1" -> default_exe_time_java, "m2" -> default_exe_time_spark,
                                  "reason" -> "Python sort is faster since Spark sorting requires shuffling, which is computational expensive.")
    file = new File(output_path_b)
    fp = new FileWriter(file)
    fp.write(Json(DefaultFormats).write(results))
    fp.close()

  }

  def star_average(rdd : RDD[(JValue,Float)]): Array[List[Any]] = {
      return rdd.aggregateByKey((0.0f, 0.0f))(
        (p1,p2) => (p1._1 + p2, p1._2 + 1),
        (t1,t2) => (t1._1 + t2._1, t1._2 + t2._2)
      ).map(x => (pretty(x._1),(x._2._1/x._2._2)))
        .sortBy(x => (-x._2,x._1),ascending = true)
        .map(x => List(x._1.slice(1,x._1.length() - 1),x._2))
        .collect()
  }

}

