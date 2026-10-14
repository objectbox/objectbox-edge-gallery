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

import android.net.Uri
import com.google.mediapipe.tasks.core.AudioPart
import com.google.mediapipe.tasks.core.ImagePart
import com.google.mediapipe.tasks.core.Part
import com.google.mediapipe.tasks.core.TextPart
import com.google.mediapipe.tasks.retrieval.components.VectorStore
import com.google.mediapipe.tasks.retrieval.model.RetrievalRecord
import io.objectbox.Box
import io.objectbox.BoxStore
import io.objectbox.annotation.ConflictStrategy
import io.objectbox.annotation.Entity
import io.objectbox.annotation.HnswIndex
import io.objectbox.annotation.Id
import io.objectbox.annotation.Unique
import io.objectbox.annotation.VectorDistanceType
import io.objectbox.query.QueryBuilder.StringOrder
import io.objectbox.query.QueryCondition
import java.io.File

/** A [RetrievalRecord] as stored by [ObjectBoxVectorStore]. */
@Entity
class VectorRecordEntity(
  @Id var id: Long = 0,
  /** The [RetrievalRecord] id; putting a record with an existing id replaces the old record. */
  @Unique(onConflict = ConflictStrategy.REPLACE) var recordId: String = "",
  // Must match GemmaEmbeddingModelStore.DEFAULT_EMBEDDING_DIMENSION (annotations need a literal).
  @HnswIndex(dimensions = 768, distanceType = VectorDistanceType.COSINE)
  var embedding: FloatArray? = null,
  /** Stored as an ObjectBox flex map, so queries can match key-value pairs. */
  var metadata: MutableMap<String, String>? = null,
  /** Type of each content part ([CONTENT_TEXT], [CONTENT_IMAGE] or [CONTENT_AUDIO]). */
  var contentTypes: MutableList<String>? = null,
  /** Per content part: the text of a text part, the file URI of an image or audio part. */
  var contentValues: MutableList<String>? = null,
  var parentId: String? = null,
  var childIds: MutableList<String>? = null,
) {
  companion object {
    const val CONTENT_TEXT = "text"
    const val CONTENT_IMAGE = "image"
    const val CONTENT_AUDIO = "audio"
  }
}

/**
 * [VectorStore] backed by an ObjectBox database in [directory], searched with ObjectBox's HNSW
 * vector index (approximate nearest neighbors, cosine distance).
 *
 * Differences to MediaPipe's `SqliteVectorStore`:
 * - [upsert] replaces a record with an existing id instead of adding a duplicate.
 * - Image and audio content parts keep only their file URI, not the image or audio data. The app
 *   never reads content back; it uses ids, embeddings and metadata.
 *
 * @param minSearchCount HNSW searches at least this many candidates, then returns the best
 *   `topK`. The search is only as wide as the number of results it is asked for, so a small
 *   `topK` alone can miss close neighbors; 0 searches exactly `topK`.
 */
class ObjectBoxVectorStore(
  directory: File,
  dimension: Int = GemmaEmbeddingModelStore.DEFAULT_EMBEDDING_DIMENSION,
  private val minSearchCount: Int = 0,
  maxSizeInKByte: Long = 4L * 1024 * 1024,
) : VectorStore {
  init {
    require(dimension == GemmaEmbeddingModelStore.DEFAULT_EMBEDDING_DIMENSION) {
      "The vector index is built for " +
        "${GemmaEmbeddingModelStore.DEFAULT_EMBEDDING_DIMENSION} dimensions, not $dimension"
    }
  }

  private val store: BoxStore =
    MyObjectBox.builder().directory(directory).maxSizeInKByte(maxSizeInKByte).build()
  private val box: Box<VectorRecordEntity> = store.boxFor(VectorRecordEntity::class.java)

  override fun upsert(records: List<RetrievalRecord>) {
    box.put(records.map { it.toEntity() })
  }

  override fun delete(ids: List<String>) {
    if (ids.isEmpty()) return
    box.query(idIn(ids)).build().use { it.remove() }
  }

  override fun delete(metadataFilter: Map<String, String>) {
    val condition = metadataCondition(metadataFilter) ?: return
    box.query(condition).build().use { it.remove() }
  }

  override fun search(queryEmbedding: FloatArray, topK: Int): List<RetrievalRecord> =
    search(queryEmbedding, topK, emptyMap())

  override fun search(
    queryEmbedding: FloatArray,
    topK: Int,
    metadataFilter: Map<String, String>,
  ): List<RetrievalRecord> {
    if (topK <= 0) return emptyList()
    val filter = metadataCondition(metadataFilter)
    var candidates = maxOf(topK, minSearchCount)
    while (true) {
      val nearest = VectorRecordEntity_.embedding.nearestNeighbors(queryEmbedding, candidates)
      // Results are ordered by distance, closest first.
      val found =
        box.query(filter?.let { nearest.and(it) } ?: nearest).build().use { query ->
          query.findWithScores().take(topK).map { it.get() }
        }
      if (filter == null || found.size >= topK) return found.map { it.toRecord() }
      // The filter applies to the candidates the HNSW search found, so a selective filter can
      // leave fewer than topK although more records match: search more candidates, and once that
      // would cover all records, search the matching records exactly.
      candidates *= FILTERED_SEARCH_GROWTH
      if (candidates >= recordCount()) return exactSearch(queryEmbedding, topK, filter)
    }
  }

  /** Brute-force search over the records matching [filter], by cosine similarity. */
  private fun exactSearch(
    queryEmbedding: FloatArray,
    topK: Int,
    filter: QueryCondition<VectorRecordEntity>,
  ): List<RetrievalRecord> {
    val candidates = box.query(filter).build().use { it.find() }
    return candidates
      .filter { it.embedding?.size == queryEmbedding.size }
      .sortedByDescending { cosineSimilarity(queryEmbedding, it.embedding!!) }
      .take(topK)
      .map { it.toRecord() }
  }

  override fun get(ids: List<String>): List<RetrievalRecord> {
    if (ids.isEmpty()) return emptyList()
    return box.query(idIn(ids)).build().use { query -> query.find().map { it.toRecord() } }
  }

  override fun getAllRecordIds(): List<String> =
    box.query().build().use { query ->
      query.property(VectorRecordEntity_.recordId).findStrings()?.toList() ?: emptyList()
    }

  override fun deleteAll() {
    box.removeAll()
  }

  override fun close() {
    store.close()
  }

  /**
   * Counting keeps a read transaction cached in the calling thread, unlike finding objects. The
   * app calls the store from its IO thread pool and closes it from another thread, which would
   * then have to abort that transaction, so release it right away.
   */
  private fun recordCount(): Long =
    try {
      box.count()
    } finally {
      store.closeThreadResources()
    }

  private fun idIn(ids: List<String>): QueryCondition<VectorRecordEntity> =
    VectorRecordEntity_.recordId.oneOf(ids.toTypedArray(), StringOrder.CASE_SENSITIVE)

  /** A condition matching records whose metadata has every entry of [filter]; null if empty. */
  private fun metadataCondition(filter: Map<String, String>): QueryCondition<VectorRecordEntity>? =
    filter.entries
      .map<_, QueryCondition<VectorRecordEntity>> { (key, value) ->
        VectorRecordEntity_.metadata.equalKeyValue(key, value, StringOrder.CASE_SENSITIVE)
      }
      .reduceOrNull { all, condition -> all.and(condition) }

  private fun RetrievalRecord.toEntity(): VectorRecordEntity {
    val parts = content.orEmpty()
    return VectorRecordEntity(
      recordId = id,
      embedding = embeddings,
      metadata = metadata.orEmpty().toMutableMap(),
      contentTypes = parts.map { it.contentType() }.toMutableList(),
      contentValues = parts.map { it.contentValue() }.toMutableList(),
      parentId = parentId,
      childIds = childIds?.toMutableList(),
    )
  }

  private fun VectorRecordEntity.toRecord(): RetrievalRecord {
    val types = contentTypes.orEmpty()
    val values = contentValues.orEmpty()
    val parts = types.zip(values) { type, value -> toPart(type, value) }
    return RetrievalRecord(
      recordId,
      parts,
      embedding ?: FloatArray(0),
      metadata.orEmpty(),
      parentId,
      childIds,
    )
  }

  private companion object {
    /** Factor by which a filtered search widens when the filter left too few results. */
    const val FILTERED_SEARCH_GROWTH = 4

    fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
      var dot = 0f
      var normA = 0f
      var normB = 0f
      for (i in a.indices) {
        dot += a[i] * b[i]
        normA += a[i] * a[i]
        normB += b[i] * b[i]
      }
      return if (normA == 0f || normB == 0f) 0f else dot / kotlin.math.sqrt(normA * normB)
    }

    fun Part.contentType(): String =
      when (this) {
        is ImagePart -> VectorRecordEntity.CONTENT_IMAGE
        is AudioPart -> VectorRecordEntity.CONTENT_AUDIO
        else -> VectorRecordEntity.CONTENT_TEXT
      }

    fun Part.contentValue(): String =
      when (this) {
        is TextPart -> text
        is ImagePart -> filePath()?.toString().orEmpty()
        is AudioPart -> filePath()?.toString().orEmpty()
        else -> ""
      }

    fun toPart(type: String, value: String): Part =
      when (type) {
        VectorRecordEntity.CONTENT_IMAGE -> ImagePart(Uri.parse(value))
        VectorRecordEntity.CONTENT_AUDIO -> AudioPart(Uri.parse(value))
        else -> TextPart(value)
      }
  }
}
