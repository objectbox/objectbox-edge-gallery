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

import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test

/**
 * End-to-end behavior of [SemanticRetrievalService] (indexing, text and image search, metadata,
 * deletion), independent of which vector store backs it.
 *
 * Embeddings come from [FakeEmbedder], so no model is needed: image data `byteArrayOf(n)` embeds
 * to [axisVector] `n`, and the text queries below map to fixed vectors.
 */
abstract class SemanticRetrievalServiceContractTest {
  /** Creates a service storing its records under [databaseName] and embedding with [embedder]. */
  protected abstract fun createService(
    databaseName: String,
    embedder: OnDeviceEmbedder,
  ): SemanticRetrievalService

  /** Whether records survive closing the service and creating a new one on the same database. */
  protected open val persistsAcrossReopen: Boolean = true

  private val embedder =
    FakeEmbedder(
      textVectors =
        mapOf(
          "cat" to axisVector(0),
          "dog" to axisVector(1),
          "mostly cat" to vectorOf(3f, 2f, 1f),
          "mostly dog" to vectorOf(1f, 3f, 2f),
        )
    )

  private lateinit var databaseName: String
  private lateinit var service: SemanticRetrievalService

  @Before
  fun setUpService() {
    databaseName = "contract-${javaClass.simpleName}-${System.nanoTime()}.db"
    service = createService(databaseName, embedder)
  }

  @After
  fun closeService() {
    service.close()
  }

  private fun image(axis: Int) = byteArrayOf(axis.toByte())

  @Test
  fun emptyService_findsNothing() = runBlocking {
    assertEquals(emptySet<String>(), service.fetchRecordIdentifiers())
    assertEquals(emptyList<SemanticRetrievalResult>(), service.retrieveEntities("cat"))
    assertNull(service.fetchRecord("missing"))
  }

  @Test
  fun textSearch_ranksRecordsBySimilarity() = runBlocking {
    service.addRecord("cat.jpg", image(0))
    service.addRecord("dog.jpg", image(1))
    service.addRecord("bird.jpg", image(2))

    assertEquals(
      listOf("cat.jpg", "dog.jpg", "bird.jpg"),
      service.retrieveEntities("mostly cat").map { it.id },
    )
    assertEquals(
      listOf("dog.jpg", "bird.jpg", "cat.jpg"),
      service.retrieveEntities("mostly dog").map { it.id },
    )
  }

  @Test
  fun textSearch_reportsCosineSimilarityScores() = runBlocking {
    service.addRecord("cat.jpg", image(0))
    service.addRecord("dog.jpg", image(1))

    val scores = service.retrieveEntities("cat").associate { it.id to it.similarityScore }

    assertEquals(1f, scores.getValue("cat.jpg")!!, 1e-3f)
    assertEquals(0f, scores.getValue("dog.jpg")!!, 1e-3f)
  }

  @Test
  fun textSearch_respectsLimit() = runBlocking {
    service.addRecord("cat.jpg", image(0))
    service.addRecord("dog.jpg", image(1))
    service.addRecord("bird.jpg", image(2))

    assertEquals(listOf("cat.jpg"), service.retrieveEntities("mostly cat", limit = 1).map { it.id })
  }

  @Test
  fun textSearch_returnsStoredEmbeddings() = runBlocking {
    service.addRecord("dog.jpg", image(1))

    val hit = service.retrieveEntities("dog").single()

    assertArrayEquals(axisVector(1), hit.embeddings, 1e-6f)
  }

  /** Keys like `text` and `parent_id` collide with SQLite columns; the service must escape them. */
  @Test
  fun metadata_roundTripsIncludingReservedKeys() = runBlocking {
    val metadata =
      mapOf(
        "album" to "holidays",
        "text" to "caption",
        "parent_id" to "video-1",
        "metadata" to "nested",
        "_user_custom" to "prefixed",
      )
    service.addRecord("cat.jpg", image(0), metadata)

    assertEquals(metadata, service.retrieveEntities("cat").single().metadata)
    assertEquals(metadata, service.fetchRecord("cat.jpg")?.metadata)
  }

  @Test
  fun textSearch_withMetadataFilter_scopesResults() = runBlocking {
    service.addRecord("v1-frame0", image(0), mapOf("parent_id" to "video-1"))
    service.addRecord("v1-frame1", image(1), mapOf("parent_id" to "video-1"))
    service.addRecord("v2-frame0", image(0), mapOf("parent_id" to "video-2"))

    val hits = service.retrieveEntities("cat", mapOf("parent_id" to "video-1"))

    assertEquals(listOf("v1-frame0", "v1-frame1"), hits.map { it.id })
  }

  @Test
  fun imageSearch_ranksBySimilarityAndExcludesSeed() = runBlocking {
    service.addRecord("cat.jpg", image(0))
    service.addRecord("cat2.jpg", image(0))
    service.addRecord("dog.jpg", image(1))

    val hits = service.retrieveEntities(image(0), seedId = "cat.jpg")

    assertEquals(listOf("cat2.jpg", "dog.jpg"), hits.map { it.id })
    assertEquals(1f, hits.first().similarityScore!!, 1e-3f)
  }

  @Test
  fun multimodalRecord_isSearchable() = runBlocking {
    service.addMultimodalRecord("clip", listOf(EmbedPart.Text("dog"), EmbedPart.Image(image(0))))

    assertEquals("clip", service.retrieveEntities("dog").first().id)
  }

  @Test
  fun fetchRecord_returnsEmbeddingAndMetadata() = runBlocking {
    service.addRecord("dog.jpg", image(1), mapOf("album" to "pets"))

    val record = checkNotNull(service.fetchRecord("dog.jpg"))

    assertArrayEquals(axisVector(1), record.embeddings, 1e-6f)
    assertEquals(mapOf("album" to "pets"), record.metadata)
  }

  @Test
  fun addRecord_sameId_replacesPreviousRecord() = runBlocking {
    service.addRecord("photo.jpg", image(0), mapOf("version" to "1"))
    service.addRecord("photo.jpg", image(1), mapOf("version" to "2"))

    assertEquals(setOf("photo.jpg"), service.fetchRecordIdentifiers())
    val hits = service.retrieveEntities("cat")
    assertEquals(listOf("photo.jpg"), hits.map { it.id })
    assertEquals(mapOf("version" to "2"), hits.single().metadata)
    assertEquals(0f, hits.single().similarityScore!!, 1e-3f)
    assertArrayEquals(axisVector(1), checkNotNull(service.fetchRecord("photo.jpg")).embeddings, 1e-6f)
  }

  @Test
  fun indexedIds_trackAddsAndDeletes() = runBlocking {
    service.addRecord("cat.jpg", image(0))
    service.addRecord("dog.jpg", image(1))
    assertEquals(setOf("cat.jpg", "dog.jpg"), service.fetchRecordIdentifiers())
    assertTrue(service.isRecordIndexed("cat.jpg"))

    service.deleteRecord("cat.jpg")

    assertEquals(setOf("dog.jpg"), service.fetchRecordIdentifiers())
    assertEquals(1, service.getIndexedRecordCount())
    assertNull(service.fetchRecord("cat.jpg"))
    assertEquals(listOf("dog.jpg"), service.retrieveEntities("cat").map { it.id })
  }

  @Test
  fun deleteRecordsWithIdPrefix_removesMatchingRecordsAndReturnsCount() = runBlocking {
    service.addRecord("video-1/0", image(0))
    service.addRecord("video-1/1", image(1))
    service.addRecord("video-2/0", image(2))

    assertEquals(2, service.deleteRecordsWithIdPrefix("video-1/"))

    assertEquals(setOf("video-2/0"), service.fetchRecordIdentifiers())
    assertEquals(listOf("video-2/0"), service.retrieveEntities("cat").map { it.id })
  }

  @Test
  fun deleteAllRecords_emptiesTheIndex() = runBlocking {
    service.addRecord("cat.jpg", image(0))
    service.addRecord("dog.jpg", image(1))

    service.deleteAllRecords()

    assertEquals(emptySet<String>(), service.fetchRecordIdentifiers())
    assertEquals(emptyList<SemanticRetrievalResult>(), service.retrieveEntities("cat"))
  }

  @Test
  fun newServiceOnSameDatabase_seesExistingRecords() = runBlocking {
    assumeTrue("store is not persistent", persistsAcrossReopen)
    service.addRecord("cat.jpg", image(0), mapOf("album" to "pets"))
    service.close()

    service = createService(databaseName, embedder)

    assertEquals(setOf("cat.jpg"), service.fetchRecordIdentifiers())
    val hit = service.retrieveEntities("cat").single()
    assertEquals("cat.jpg", hit.id)
    assertEquals(mapOf("album" to "pets"), hit.metadata)
  }
}
