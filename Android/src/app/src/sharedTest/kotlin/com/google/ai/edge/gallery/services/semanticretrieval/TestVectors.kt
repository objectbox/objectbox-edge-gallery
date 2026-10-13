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

import kotlin.math.sqrt

/** Embedding dimension used by tests; matches what the app configures for its vector stores. */
const val TEST_DIMENSION = GemmaEmbeddingModelStore.DEFAULT_EMBEDDING_DIMENSION

/** Returns a unit vector pointing along [axis], so vectors on different axes are orthogonal. */
fun axisVector(axis: Int, dimension: Int = TEST_DIMENSION): FloatArray {
  require(axis in 0 until dimension) { "axis $axis out of range for dimension $dimension" }
  return FloatArray(dimension).also { it[axis] = 1f }
}

/**
 * Returns a normalized vector whose leading components are [components] (rest zero).
 *
 * Useful for building vectors with a known cosine similarity to [axisVector]s, e.g.
 * `vectorOf(1f, 1f)` is equally similar (≈0.707) to axes 0 and 1.
 */
fun vectorOf(vararg components: Float, dimension: Int = TEST_DIMENSION): FloatArray {
  require(components.size <= dimension)
  val norm = sqrt(components.sumOf { (it * it).toDouble() }).toFloat()
  require(norm > 0f) { "vector must not be all zeros" }
  return FloatArray(dimension).also { v -> components.forEachIndexed { i, c -> v[i] = c / norm } }
}

/**
 * Deterministic [OnDeviceEmbedder] for tests: maps inputs to preset vectors instead of running a
 * model.
 * - Text: looked up in [textVectors]; unknown texts fail the test.
 * - Image: [axisVector] along the first byte of the image data.
 * - Multimodal: the embedding of the first part.
 */
class FakeEmbedder(
  private val textVectors: Map<String, FloatArray> = emptyMap(),
  private val dimension: Int = TEST_DIMENSION,
) : OnDeviceEmbedder {
  override fun isAvailable() = true

  override fun generateTextEmbedding(text: String): FloatArray =
    checkNotNull(textVectors[text]) { "FakeEmbedder has no vector for text \"$text\"" }

  override fun generateImageEmbedding(imageData: ByteArray): FloatArray =
    axisVector(imageData.first().toInt(), dimension)

  override fun generateImagesEmbedding(imageDataList: List<ByteArray>): FloatArray =
    generateImageEmbedding(imageDataList.first())

  override fun generateMultimodalEmbedding(parts: List<EmbedPart>): FloatArray =
    when (val first = parts.first()) {
      is EmbedPart.Text -> generateTextEmbedding(first.text)
      is EmbedPart.Image -> generateImageEmbedding(first.imageData)
      is EmbedPart.Audio -> error("FakeEmbedder does not support audio")
    }
}
