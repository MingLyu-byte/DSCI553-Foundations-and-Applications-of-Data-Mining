import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.json4s._
import org.json4s.jackson.Json
import org.json4s.jackson.JsonMethods.{parse, pretty, render}

import java.io.{File, FileWriter}
import scala.math.Numeric.IntIsIntegral

object task1 {
  def main(args: Array[String]): Unit = {
    val input_path = args(0)
    val output_path = args(1)
    val conf = new SparkConf().setAppName("task1").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val input_lines = sc.textFile(input_path).map(x => parse(x))

    val n_review = total_num_reviews(input_lines.map(x => (x\"review_id")))
    val n_review_2018 = total_num_reviews_2018(input_lines.map(x => (x\"date")))
    val n_user = total_num_distinct(input_lines.map(x => (x\"user_id")))
    val top10_user = top_10(input_lines.map(x => (x\"user_id")))
    val n_business = total_num_distinct(input_lines.map(x => (x\"business_id")))
    val top10_business = top_10(input_lines.map(x => (x\"business_id")))
    val results = Map[String, Any]("n_review" -> n_review,
      "n_review_2018" -> n_review_2018, "n_user" -> n_user,
      "top10_user" -> top10_user.toList, "n_business" -> n_business,
      "top10_business" -> top10_business)

    println(Json(DefaultFormats).write(results))
    val file = new File(output_path)
    val fp = new FileWriter(file)
    fp.write(Json(DefaultFormats).write(results))
    fp.close()
    println("ASDASDASDAS")
  }

  def total_num_reviews(review_ids: RDD[JValue]): Long = {
    return review_ids.count()
  }

  def total_num_reviews_2018(review_ids_date: RDD[JValue]): Long = {
    return review_ids_date.map(x => pretty(x))
      .filter(x => x.contains("2018"))
      .count()
  }

  def total_num_distinct(ids: RDD[JValue]): Long = {
    return ids.distinct().count()
  }

  def top_10(ids: RDD[JValue]): Array[Any] = {
    return ids.map(x => (pretty(render(x)), 1))
      .reduceByKey(_ + _)
      .sortBy(x => (-x._2, x._1), true)
      .take(10)
      .map(x => List(x._1.slice(1,x._1.length() - 1),x._2))
  }

}
