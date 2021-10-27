import org.apache.spark.rdd.RDD
import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}
import org.json4s._
import org.json4s.jackson.Json
import org.json4s.jackson.JsonMethods.{parse, pretty, render}

import java.io.{File, FileWriter}
import scala.math.Numeric.IntIsIntegral

object task2 {
  def main(args: Array[String]): Unit = {
    val input_path = args(0)
    val output_path = args(1)
    val n_partition = args(2).toInt
    val conf = new SparkConf().setAppName("task2").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val input_lines = sc.textFile(input_path).map(x => parse(x))

    val default_partition_items = partion_by_items(input_lines.map(x => (x\"business_id")))
    var customize_rdd = customized_partition(n_partition, input_lines.map(x => (x\"business_id")))

    val default_start = System.nanoTime
    top_10_default(input_lines.map(x => x\"business_id"))
    val default_exe_time = (System.nanoTime - default_start) / 1e9d

    val custome_start = System.nanoTime
    top_10_custome(customize_rdd)
    val custome_exe_time = (System.nanoTime - custome_start) / 1e9d

    val default = Map[String, Any]("n_partition" -> input_lines.partitions.size,
      "n_items" -> default_partition_items, "exe_time" -> default_exe_time)
    val custome = Map[String, Any]("n_partition" -> n_partition,
      "n_items" -> customize_rdd.glom().map(_.size).collect(), "exe_time" -> custome_exe_time)
    val results = Map[String, Any]("default" -> default, "customized" -> custome)
    println(Json(DefaultFormats).write(results))

    val file = new File(output_path)
    val fp = new FileWriter(file)
    fp.write(Json(DefaultFormats).write(results))
    fp.close()
  }

  def top_10_custome(ids: RDD[String]): Array[Any] = {
    return ids.map(x => (x,1))
      .reduceByKey(_ + _)
      .sortBy(x => (-x._2, x._1), true)
      .take(10)
      .map(x => List(x._1.slice(1,x._1.length() - 1),x._2))
  }

  def top_10_default(ids: RDD[JValue]): Array[Any] = {
    return ids.map(x => (pretty(render(x)), 1))
      .reduceByKey(_ + _)
      .sortBy(x => (-x._2, x._1), true)
      .take(10)
      .map(x => List(x._1.slice(1,x._1.length() - 1),x._2))
  }

  def partion_by_items(ids: RDD[JValue]): Array[Int] = {
    return ids.mapPartitionsWithIndex{case (i,rows) => Iterator((i,rows.size))}
      .map(x => x._2)
      .collect()
  }

  def customized_partition(n_partitions: Int, rdd: RDD[JValue]): RDD[String] = {
    return rdd.map(x => (pretty(render(x)), 1))
      .partitionBy(new HashPartitioner(n_partitions))
      .map(x => x._1)
  }


}

