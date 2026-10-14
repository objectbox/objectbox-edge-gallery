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

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Real-world vectors for benchmarks: the Android export of the `wikipedia-embedding-gemma-2`
 * dataset (popular Wikipedia pages embedded with EmbeddingGemma 2, see AGENTS.md).
 *
 * Corpus rows are in popularity order, so a corpus of size N is the first N rows. Files are read
 * on demand, so the corpus never has to fit in memory at once.
 */
class VectorDataset(private val dir: File) {
  private val corpusFile = File(dir, "corpus.fvecs")

  /** Query embeddings, one per reference query. */
  val queries: List<FloatArray> = readVecs(File(dir, "queries.fvecs")).map { it.asFloats() }

  val dimension: Int = queries.first().size

  private val rowBytes = 4L * (dimension + 1)

  /** Number of corpus vectors. */
  val corpusSize: Int = (corpusFile.length() / rowBytes).toInt()

  /** Page titles by corpus row. */
  val titles: List<String> by lazy {
    File(dir, "corpus.tsv").useLines { lines ->
      lines.drop(1).map { it.split('\t', limit = 3)[2] }.toList()
    }
  }

  /** Reads [count] corpus vectors starting at row [from]. */
  fun corpus(from: Int, count: Int): List<FloatArray> {
    require(from >= 0 && from + count <= corpusSize) { "rows $from+$count out of 0..$corpusSize" }
    val buffer = ByteBuffer.allocate((rowBytes * count).toInt()).order(ByteOrder.LITTLE_ENDIAN)
    RandomAccessFile(corpusFile, "r").use { file ->
      while (buffer.hasRemaining()) {
        check(file.channel.read(buffer, from * rowBytes + buffer.position()) > 0) { "$corpusFile ended early" }
      }
    }
    buffer.flip()
    return List(count) {
      check(buffer.getInt() == dimension) { "corrupt row ${from + it} in $corpusFile" }
      FloatArray(dimension) { buffer.getFloat() }
    }
  }

  /**
   * Exact nearest neighbors of each query within the first [corpusSize] rows, as computed by the
   * data set's export.
   */
  fun groundTruth(corpusSize: Int): List<Neighbors> {
    val file = File(dir, "groundtruth-$corpusSize.ivecs")
    check(file.exists()) { "No ground truth for corpus size $corpusSize in $dir" }
    val rows = readVecs(file).map { buffer -> IntArray(buffer.remaining() / 4) { buffer.getInt() } }
    val scores = readVecs(File(dir, "groundtruth-$corpusSize-scores.fvecs")).map { it.asFloats() }
    return rows.zip(scores) { r, s -> Neighbors(r, s) }
  }

  /** Corpus rows of a query's nearest neighbors, best first, and their cosine similarities. */
  class Neighbors(val rows: IntArray, val scores: FloatArray)

  companion object {
    /** Returns each row of an fvecs/ivecs file as a buffer positioned after its length prefix. */
    private fun readVecs(file: File): List<ByteBuffer> {
      val buffer = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
      val rows = mutableListOf<ByteBuffer>()
      while (buffer.hasRemaining()) {
        val length = buffer.getInt()
        val row = buffer.slice().order(ByteOrder.LITTLE_ENDIAN)
        row.limit(4 * length)
        rows += row
        buffer.position(buffer.position() + 4 * length)
      }
      return rows
    }

    private fun ByteBuffer.asFloats() = FloatArray(remaining() / 4) { getFloat() }
  }
}
