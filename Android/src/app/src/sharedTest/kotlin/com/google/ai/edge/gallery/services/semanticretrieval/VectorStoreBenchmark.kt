/*
 * Copyright 2026 ObjectBox Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.gallery.services.semanticretrieval

import com.google.mediapipe.tasks.core.TextPart
import com.google.mediapipe.tasks.retrieval.components.VectorStore
import com.google.mediapipe.tasks.retrieval.model.RetrievalRecord
import java.io.File
import java.time.Instant
import java.util.Locale
import kotlin.math.ceil
import kotlin.system.measureNanoTime
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.FixMethodOrder
import org.junit.Test
import org.junit.runners.MethodSorters

/**
 * Search quality and performance of a [VectorStore] on real embeddings ([VectorDataset]).
 *
 * For each corpus size, a new store gets the first N vectors of the data set. The benchmark then
 * measures insert time, search latency and recall@10 against the exact neighbors for the reference
 * queries at several top-K values, a search filtered by metadata, getting records by id (also
 * checking that the store returns the embeddings unchanged), reopening the store (if it persists)
 * and its size on disk.
 *
 * Opt-in: the tests only run when [dataDir] points to the data set and are skipped otherwise.
 * A summary is printed as lines starting with [TAG]. If [outputDir] is set, every measurement is
 * also written there as CSV, three files per store and corpus size:
 * - `<store>-<size>-run.csv`: one row with the run's totals.
 * - `<store>-<size>-searches.csv`: one row per search, with its latency and all returned ids.
 * - `<store>-<size>-inserts.csv`: one row per inserted batch.
 */
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
abstract class VectorStoreBenchmark {
  /** Directory with the data set files, or null to skip the benchmark. */
  protected abstract val dataDir: File?

  /** Creates the store under test, backed by storage named [name]. */
  protected abstract fun createStore(name: String): VectorStore

  /** Directory for the CSV files, or null to only print a summary. */
  protected open val outputDir: File? = System.getProperty("benchmarkOutput")?.let(::File)

  /** Describes the machine the benchmark runs on; recorded with the results. */
  protected open val device: String =
    "${System.getProperty("os.name")} ${System.getProperty("os.arch")}, " +
      "${Runtime.getRuntime().availableProcessors()} CPUs, " +
      "Java ${System.getProperty("java.version")}"

  /** Whether a store created again with the same name has the records of the closed one. */
  protected open val persistent: Boolean = false

  /** Bytes the store named [name] occupies on disk, or null if it is not stored in files. */
  protected open fun storageBytes(name: String): Long? = null

  /** Removes all storage of the store named [name]. */
  protected open fun deleteStorage(name: String) {}

  @Test fun size001k() = benchmark(1_000)

  @Test fun size010k() = benchmark(10_000)

  @Test fun size100k() = benchmark(100_000)

  private fun benchmark(size: Int) {
    val dir = dataDir
    assumeTrue("No vector data set given; see VectorStoreBenchmark subclasses", dir != null)
    val data = dataset(dir!!)
    assumeTrue("Data set has only ${data.corpusSize} vectors", size <= data.corpusSize)
    val storeName = javaClass.simpleName.removeSuffix("Benchmark")
    val name = "benchmark-$storeName-$size"
    val run = linkedMapOf<String, Any>(
      "timestamp" to Instant.now().toString(),
      "device" to device,
      "store" to storeName,
      "corpus" to size,
      "dimension" to data.dimension,
    )
    val summary = linkedMapOf<String, Any>()
    val inserts = mutableListOf<List<Any>>()
    val searches = mutableListOf<List<Any>>()
    deleteStorage(name)
    var store = createStore(name)
    try {
      var insertNanos = 0L
      for (from in 0 until size step INSERT_BATCH) {
        val vectors = data.corpus(from, minOf(INSERT_BATCH, size - from))
        val records = vectors.mapIndexed { i, vector -> record(data, from + i, vector) }
        val nanos = measureNanoTime { store.upsert(records) }
        inserts += listOf(from, records.size, nanos)
        insertNanos += nanos
      }
      run["insert_ns"] = insertNanos
      summary["insert_ms"] = insertNanos / 1e6
      summary["insert_per_s"] = size / (insertNanos / 1e9)

      val truth = data.groundTruth(size)
      repeat(WARMUP_QUERIES) { store.search(data.queries[it % data.queries.size], RECALL_AT) }
      for (topK in TOP_KS) {
        summary["search_k$topK"] =
          measureSearches("k$topK", data.queries, truth, minOf(topK, size), searches) {
            store.search(it, topK)
          }
      }

      val filter = mapOf(BUCKET_KEY to FILTER_BUCKET.toString())
      val filteredTruth = exactNeighbors(data, size) { row -> row % BUCKETS == FILTER_BUCKET }
      summary["search_k${RECALL_AT}_filtered"] =
        measureSearches(
          "k${RECALL_AT}_filtered",
          data.queries,
          filteredTruth,
          minOf(RECALL_AT, size / BUCKETS),
          searches,
        ) {
          store.search(it, RECALL_AT, filter)
        }

      val ids = List(GET_IDS) { (it.toLong() * size / GET_IDS).toString() }
      lateinit var fetched: List<RetrievalRecord>
      val getNanos = measureNanoTime { fetched = store.get(ids) }
      assertEquals(ids.toSet(), fetched.map { it.id }.toSet())
      run["get_${GET_IDS}_ids_ns"] = getNanos
      summary["get_${GET_IDS}_ids_ms"] = getNanos / 1e6
      // Bit-exact, full-length embeddings: no truncation, quantization or other loss.
      val exact = fetched.all { it.embeddings.contentEquals(data.corpus(it.id.toInt(), 1)[0]) }
      run["embeddings_exact"] = exact
      summary["embeddings_exact"] = exact

      if (persistent) {
        store.close()
        lateinit var allIds: List<String>
        val reopenNanos = measureNanoTime {
          store = createStore(name)
          allIds = store.getAllRecordIds()
        }
        assertEquals(size, allIds.size)
        run["reopen_and_get_all_ids_ns"] = reopenNanos
        summary["reopen_and_get_all_ids_ms"] = reopenNanos / 1e6
      }
    } finally {
      store.close()
    }
    storageBytes(name)?.let {
      run["disk_bytes"] = it
      summary["disk_mb"] = it / 1e6
    }
    deleteStorage(name)
    print(storeName, size, summary)
    outputDir?.let { out ->
      out.mkdirs()
      val prefix = "$storeName-$size"
      writeCsv(File(out, "$prefix-run.csv"), run.keys.toList(), listOf(run.values.toList()))
      val key = listOf(storeName, size)
      writeCsv(
        File(out, "$prefix-searches.csv"),
        listOf(
          "store",
          "corpus",
          "search",
          "query",
          "latency_ns",
          "returned",
          "hits_at_$RECALL_AT",
          "ids",
        ),
        searches.map { key + it },
      )
      writeCsv(
        File(out, "$prefix-inserts.csv"),
        listOf("store", "corpus", "first_row", "rows", "latency_ns"),
        inserts.map { key + it },
      )
    }
  }

  private fun record(data: VectorDataset, row: Int, vector: FloatArray) =
    RetrievalRecord(
      row.toString(),
      listOf(TextPart(data.titles[row])),
      vector,
      mapOf(BUCKET_KEY to (row % BUCKETS).toString()),
    )

  /**
   * Runs [search] for every query, adds a row per search to [rows], and returns latency
   * percentiles in ms and recall@10 against [truth]. Checks that each search returns [expected]
   * records.
   *
   * A result counts as a hit if it is one of the true top 10, or if it ties with the 10th: its
   * similarity is within [TIE_TOLERANCE], below which float rounding decides the order.
   */
  private fun measureSearches(
    name: String,
    queries: List<FloatArray>,
    truth: List<VectorDataset.Neighbors>,
    expected: Int,
    rows: MutableList<List<Any>>,
    search: (FloatArray) -> List<RetrievalRecord>,
  ): Map<String, Double> {
    val nanos = LongArray(queries.size)
    var hits = 0
    queries.forEachIndexed { q, query ->
      lateinit var found: List<RetrievalRecord>
      nanos[q] = measureNanoTime { found = search(query) }
      assertEquals("results for query $q", expected, found.size)
      val relevant = truth[q].rows.take(RECALL_AT).toSet()
      val tenthScore = truth[q].scores[minOf(RECALL_AT, truth[q].scores.size) - 1]
      val queryHits =
        found.take(RECALL_AT).count {
          it.id.toInt() in relevant ||
            (it.embeddings.size == query.size &&
              dot(query, it.embeddings) >= tenthScore - TIE_TOLERANCE)
        }
      hits += queryHits
      rows += listOf(name, q, nanos[q], found.size, queryHits, found.joinToString(" ") { it.id })
    }
    nanos.sort()
    fun percentileMs(p: Double) = nanos[ceil(p * nanos.size).toInt() - 1] / 1e6
    return mapOf(
      "p50_ms" to percentileMs(0.5),
      "p90_ms" to percentileMs(0.9),
      "p99_ms" to percentileMs(0.99),
      "recall_at_$RECALL_AT" to hits.toDouble() / (queries.size * RECALL_AT),
    )
  }

  private fun print(store: String, size: Int, summary: Map<String, Any>) {
    println("$TAG $store, corpus $size")
    for ((key, value) in summary) {
      val text =
        when (value) {
          is Map<*, *> -> value.entries.joinToString(", ") { (k, v) -> "$k ${format(v)}" }
          else -> format(value)
        }
      println("$TAG   $key: $text")
    }
  }

  companion object {
    /** Prefix of all output lines, e.g. for `adb logcat | grep VectorStoreBenchmark`. */
    const val TAG = "VectorStoreBenchmark"

    private const val INSERT_BATCH = 1_000
    private const val WARMUP_QUERIES = 10
    private const val RECALL_AT = 10
    private val TOP_KS = listOf(10, 100, 1000)
    private const val GET_IDS = 100
    private const val TIE_TOLERANCE = 1e-5f

    /** Every record gets a metadata bucket (row % [BUCKETS]) to measure filtered search. */
    private const val BUCKET_KEY = "bucket"
    private const val BUCKETS = 10
    private const val FILTER_BUCKET = 0

    private val datasets = mutableMapOf<File, VectorDataset>()

    private fun dataset(dir: File) = datasets.getOrPut(dir) { VectorDataset(dir) }

    /** Exact top [RECALL_AT] rows per query among the first [size] rows that match [include]. */
    private fun exactNeighbors(
      data: VectorDataset,
      size: Int,
      include: (Int) -> Boolean,
    ): List<VectorDataset.Neighbors> {
      val best = data.queries.map { TopK(RECALL_AT) }
      for (from in 0 until size step 10_000) {
        data.corpus(from, minOf(10_000, size - from)).forEachIndexed { i, vector ->
          val row = from + i
          if (include(row)) {
            data.queries.forEachIndexed { q, query -> best[q].offer(row, dot(query, vector)) }
          }
        }
      }
      return best.map { VectorDataset.Neighbors(it.rows(), it.scores()) }
    }

    /** Vectors in the data set are normalized, so the dot product is the cosine similarity. */
    private fun dot(a: FloatArray, b: FloatArray): Float {
      var sum = 0f
      for (i in a.indices) sum += a[i] * b[i]
      return sum
    }

    private class TopK(private val k: Int) {
      private val rows = IntArray(k)
      private val scores = FloatArray(k) { Float.NEGATIVE_INFINITY }

      fun offer(row: Int, score: Float) {
        if (score <= scores[k - 1]) return
        var i = k - 1
        while (i > 0 && scores[i - 1] < score) {
          scores[i] = scores[i - 1]
          rows[i] = rows[i - 1]
          i--
        }
        scores[i] = score
        rows[i] = row
      }

      fun rows() = rows.copyOf()

      fun scores() = scores.copyOf()
    }

    private fun format(value: Any?): String =
      when (value) {
        is Double -> String.format(Locale.ROOT, if (value < 10) "%.3f" else "%.1f", value)
        else -> value.toString()
      }

    /** Writes an RFC 4180 CSV file; text fields are quoted when needed. */
    private fun writeCsv(file: File, header: List<String>, rows: List<List<Any>>) {
      fun field(value: Any): String {
        val text = value.toString()
        val quote = text.any { it == ',' || it == '"' || it == '\n' }
        return if (quote) "\"${text.replace("\"", "\"\"")}\"" else text
      }
      file.bufferedWriter().use { out ->
        out.write(header.joinToString(",") + "\n")
        for (row in rows) out.write(row.joinToString(",") { field(it) } + "\n")
      }
    }
  }
}
